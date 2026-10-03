"""Step 7: confidence-based escalation, analysed from the cache.

CAVR routes each query to path p. An uncertainty signal is calibrated on the val split into
P(answer of p is good, ANLS >= 0.5) by logistic regression; if P < tau the query escalates to hybrid.

Signals and when they are available:
  pre-generation   conf_head, path_entropy, quality_margin, bm25_top/margin, colpali_top/margin
  post-generation  answer_logprob_mean/min of the routed path's answer (requires running p first)
Escalation cost/latency:
  pre-generation signal -> re-route: pay only hybrid
  post-generation, serial      -> pay p + hybrid, latency p + hybrid
  post-generation, speculative -> launch hybrid alongside p for every non-hybrid query and keep it only
                                  when escalating: pay p + hybrid on all of them, latency max(p, hybrid)
                                  (assumes two concurrent GPU workers; measured in step 8)
Output: results/escalation.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from common import RESULTS, hardware_info, write_json
from routing_data import ANLS_OK, LAMBDA_DEFAULT, PATHS, load

HYB = PATHS.index("hybrid")
TAUS = [round(t, 2) for t in np.arange(0.3, 0.91, 0.05)]
PRE = ["conf_head", "path_entropy", "quality_margin", "bm25_top", "bm25_margin", "colpali_top", "colpali_margin"]
POST = ["answer_logprob_mean", "answer_logprob_min"]


def retrieval_feats(path, qids):
    r = {x["qid"]: x for x in map(json.loads, open(path, encoding="utf-8"))}
    top = np.array([r[q]["scores"][0] for q in qids])
    margin = np.array([r[q]["scores"][0] - r[q]["scores"][1] for q in qids])
    return top, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(RESULTS / "path_outputs.jsonl"))
    ap.add_argument("--runs", default=str(RESULTS / "router_runs"))
    ap.add_argument("--out", default=str(RESULTS / "escalation.json"))
    args = ap.parse_args()

    qids, Q, A, C, rows = load(args.outputs)
    # Per-stage costs, so each retrieval is paid once per query however many paths/signals use it.
    G = np.array([[rows[q][p]["generate_s"] for p in PATHS] for q in qids])          # generator GPU-s
    GE = np.array([[rows[q][p]["gen_energy_j"] for p in PATHS] for q in qids])
    GL = np.array([[rows[q][p]["preprocess_s"] + rows[q][p]["generate_s"] for p in PATHS] for q in qids])
    VR = np.array([rows[q]["hybrid"]["visual_retrieval_gpu_s"] for q in qids])      # ColPali GPU-s (= latency)
    VRE = np.array([rows[q]["hybrid"]["visual_retrieval_energy_j"] for q in qids])
    TR = np.array([rows[q]["hybrid"]["text_retrieval_cpu_s"] for q in qids])        # BM25 CPU latency
    USES_V = np.array([p != "text" for p in PATHS])
    USES_T = np.array([p != "visual" for p in PATHS])
    LPm = np.array([[rows[q][p]["answer_logprob_mean"] for p in PATHS] for q in qids])
    LPn = np.array([[rows[q][p]["answer_logprob_min"] for p in PATHS] for q in qids])
    bm25_top, bm25_margin = retrieval_feats(RESULTS / "retrieval_text.jsonl", qids)
    col_top, col_margin = retrieval_feats(RESULTS / "retrieval_visual.jsonl", qids)

    zero = {"gpu_s": 0.0, "energy_j": 0.0, "latency_s": 0.0}
    def load_overhead(name):
        p = RESULTS / name
        return json.load(open(p))["per_query_batch1"] if p.exists() else zero
    ovh_cavr, ovh_ce = load_overhead("router_cost.json"), load_overhead("router_ce_cost.json")
    ev_path = RESULTS / "eval_main.json"
    ce_lams = ({s["run"]: s["policies"]["cavr_ce_valsel"].get("lambda") for s in json.load(open(ev_path))["per_seed"]
                if "cavr_ce_valsel" in s["policies"]} if ev_path.exists() else {})

    per_seed = []
    for f in sorted(Path(args.runs).glob("seed*.npz")):
        R = np.load(f, allow_pickle=True)
        assert list(R["qids"]) == qids
        split, cost = R["split"], R["cost"]
        idx = np.arange(len(qids))
        va, te = split == "val", split == "test"
        shared = {"bm25_top": bm25_top, "bm25_margin": bm25_margin, "colpali_top": col_top, "colpali_margin": col_margin}

        def analyse(route, feats, base_needs_colpali, ovh):
            good = (A[idx, route] >= ANLS_OK).astype(int)

            def calibrate(names):
                X = np.column_stack([feats[n] for n in names])
                mu, sd = X[va].mean(0), X[va].std(0) + 1e-9
                m = LogisticRegression(max_iter=2000).fit((X[va] - mu) / sd, good[va])
                return m.predict_proba((X - mu) / sd)[:, 1]

            def outcome(p_good, tau, post, mode, sig_v=False, sig_t=False):
                """sig_v / sig_t: the signal (or the router) needs ColPali / BM25 scores up front."""
                sig_v = sig_v or base_needs_colpali
                esc = te & (route != HYB) & (p_good < tau)
                final = np.where(esc, HYB, route)
                t = np.where(te)[0]
                r, fnl, e = route[t], final[t], esc[t]
                anls = A[t, fnl]
                pre_v = sig_v | USES_V[r] | (not post and USES_V[fnl])
                pre_t = sig_t | USES_T[r] | (not post and USES_T[fnl])
                pre_lat = pre_v * VR[t] + pre_t * TR[t]
                if not post:      # re-route before generating: one generation, of the final path
                    gpu = G[t, fnl] + pre_v * VR[t]
                    en = GE[t, fnl] + pre_v * VRE[t]
                    lat = pre_lat + GL[t, fnl]
                else:
                    spec = (r != HYB) if mode == "speculative" else e
                    late_v = spec & ~pre_v
                    late_t = spec & ~pre_t
                    gpu = G[t, r] + spec * G[t, HYB] + (pre_v | late_v) * VR[t]
                    en = GE[t, r] + spec * GE[t, HYB] + (pre_v | late_v) * VRE[t]
                    if mode == "serial":
                        lat = pre_lat + GL[t, r] + e * (late_v * VR[t] + late_t * TR[t] + GL[t, HYB])
                    else:
                        lat = (pre_v | late_v) * VR[t] + (pre_t | late_t) * TR[t] +                               np.where(spec, np.maximum(GL[t, r], GL[t, HYB]), GL[t, r])
                gpu, en, lat = gpu + ovh["gpu_s"], en + ovh["energy_j"], lat + ovh["latency_s"]
                return {"anls": float(anls.mean()), "gpu_s": float(gpu.mean()), "energy_j": float(en.mean()),
                        "latency_p50_s": float(np.percentile(lat, 50)), "latency_p95_s": float(np.percentile(lat, 95)),
                        "escalation_rate": float(e.mean())}

            pre = [n for n in PRE if n in feats]
            signals = {n: [n] for n in pre + POST}
            signals["all_pre"] = pre
            signals["all"] = pre + POST
            res = {"none": outcome(np.ones(len(qids)), 0.0, False, "serial")}
            for name, cols in signals.items():
                p_good = calibrate(cols)
                post = any(c in POST for c in cols)
                sig_v = any(c.startswith("colpali") for c in cols)
                sig_t = any(c.startswith("bm25") for c in cols)
                modes = ("serial", "speculative") if post else ("reroute",)
                res[name] = {m: {str(tau): outcome(p_good, tau, post, m, sig_v, sig_t) for tau in TAUS} for m in modes}
            return res

        # Base router 1: CAVR (question embedding), lambda = LAMBDA_DEFAULT
        util = R["cavr_qhat"] - LAMBDA_DEFAULT * cost
        route = np.argmax(util, 1)
        srt = np.sort(util, 1)
        probs = R["cavr_probs"]
        feats = {"conf_head": R["cavr_conf"], "path_entropy": -(probs * np.log(probs + 1e-12)).sum(1),
                 "quality_margin": srt[:, -1] - srt[:, -2], **shared,
                 "answer_logprob_mean": LPm[idx, route], "answer_logprob_min": LPn[idx, route]}
        entry = {"run": f.name, "results": analyse(route, feats, False, ovh_cavr)}

        # Base router 2: content-aware CAVR-CE, lambda chosen on val in evaluate.py
        ce_file = RESULTS / "router_ce" / f.name
        if ce_file.exists() and ce_lams.get(f.name) is not None:
            qce = np.load(ce_file, allow_pickle=True)["qhat_ce_full"]
            lam = ce_lams[f.name]
            util = qce - lam * cost
            route = np.argmax(util, 1)
            srt = np.sort(util, 1)
            feats = {"conf_head": qce[idx, route], "quality_margin": srt[:, -1] - srt[:, -2], **shared,
                     "answer_logprob_mean": LPm[idx, route], "answer_logprob_min": LPn[idx, route]}
            entry["results_ce"] = analyse(route, feats, True, ovh_ce)
            entry["ce_lambda"] = lam
        per_seed.append(entry)
        print(f.name, "cavr none", round(entry["results"]["none"]["anls"], 3),
              "| ce none", round(entry.get("results_ce", {}).get("none", {}).get("anls", float("nan")), 3), flush=True)

    write_json(args.out, {"taus": TAUS, "escalation_target": "hybrid", "calibration": "logistic regression on val",
                          "good_answer": f"ANLS >= {ANLS_OK}", "per_seed": per_seed, "hardware": hardware_info()})


if __name__ == "__main__":
    main()
