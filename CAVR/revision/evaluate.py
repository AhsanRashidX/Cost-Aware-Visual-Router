"""Step 6: evaluate routing policies from the cache (no model is re-run).

For each seed's test split: ANLS, EM, measured GPU-s and energy, p50/p95 latency, path mix,
bootstrap 95% CIs, McNemar (EM) and paired-bootstrap (ANLS) tests of CAVR vs every policy,
and a lambda sweep for the cost-quality frontier.
Router overhead (embedding + MLP) comes from results/router_cost.json (bench_router.py) when present.
Output: results/eval_main.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from common import RESULTS, SEED, hardware_info, write_json
from routing_data import LAMBDA_DEFAULT, LAMBDA_GRID, PATHS, keyword_regex, keyword_submitted, load, utility_labels

N_BOOT = 1000
N_RANDOM = 200
LEARNED = {"logreg", "knn", "cavr", "cavr_cls_head", "cavr_bm25", "cavr_rf"}
CE = {"cavr_ce", "cavr_ce_valsel"}          # content-aware router (router_ce.py); own measured overhead
VALSEL_LAMBDAS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
VALSEL_DELTA = 0.005                         # cheapest lambda whose val ANLS >= val(always_hybrid) - delta
# Router variants whose features need a retrieval to run before routing:
# cavr_rf needs ColPali scores (paid extra when the chosen path is text), cavr_bm25 needs BM25 (paid when visual).
PRE_RETRIEVAL = {"cavr_rf": "visual", "cavr_bm25": "text", "cavr_ce": "visual", "cavr_ce_valsel": "visual"}


def oracle(A, C, lam):
    """Per-query best path using the true per-query cost; ties to the cheaper path."""
    u = A - lam * C
    cheap = np.argsort(C, axis=1)
    hit = np.take_along_axis(u, cheap, 1) >= u.max(1, keepdims=True) - 1e-12
    return np.take_along_axis(cheap, np.argmax(hit, 1)[:, None], 1)[:, 0]


def metrics(choice, M, overhead, rng, pre=None):
    idx = np.arange(len(choice))
    anls, em = M["A"][idx, choice], M["EM"][idx, choice]
    xg = xe = xl = np.zeros(len(choice))
    if pre == "visual":   # ColPali ran for routing; paths that don't use it still paid for it
        miss = choice == 0
        xg, xe, xl = miss * M["VR"], miss * M["VRE"], miss * M["VR"]
    elif pre == "text":   # BM25 ran for routing (CPU only): latency for the visual path
        xl = (choice == 1) * M["TR"]
    lat = M["W"][idx, choice] + overhead["latency_s"] + xl
    boot = rng.integers(0, len(choice), (N_BOOT, len(choice)))
    return {
        "anls": float(anls.mean()), "anls_ci95": [float(x) for x in np.percentile(anls[boot].mean(1), [2.5, 97.5])],
        "em": float(em.mean()),
        "gpu_s": float((M["C"][idx, choice] + xg).mean() + overhead["gpu_s"]),
        "energy_j": float((M["E"][idx, choice] + xe).mean() + overhead["energy_j"]),
        "latency_p50_s": float(np.percentile(lat, 50)), "latency_p95_s": float(np.percentile(lat, 95)),
        "path_mix": {p: float((choice == k).mean()) for k, p in enumerate(PATHS)},
    }


def mcnemar(em_a, em_b):
    b, c = int(((em_a == 1) & (em_b == 0)).sum()), int(((em_a == 0) & (em_b == 1)).sum())
    return {"cavr_only_correct": b, "other_only_correct": c,
            "p_value": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def paired_boot(x, y, rng):
    d = x - y
    boot = rng.integers(0, len(d), (N_BOOT, len(d)))
    return {"mean_diff": float(d.mean()), "ci95": [float(v) for v in np.percentile(d[boot].mean(1), [2.5, 97.5])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(RESULTS / "path_outputs.jsonl"))
    ap.add_argument("--runs", default=str(RESULTS / "router_runs"))
    ap.add_argument("--out", default=str(RESULTS / "eval_main.json"))
    args = ap.parse_args()

    qids, Q, A, C, rows = load(args.outputs)
    pos = {q: i for i, q in enumerate(qids)}
    full = {"A": A, "C": C,
            "EM": np.array([[rows[q][p]["em"] for p in PATHS] for q in qids]),
            "E": np.array([[rows[q][p]["energy_j"] for p in PATHS] for q in qids]),
            "W": np.array([[rows[q][p]["wall_s"] for p in PATHS] for q in qids]),
            "VR": np.array([rows[q]["hybrid"]["visual_retrieval_gpu_s"] for q in qids]),
            "VRE": np.array([rows[q]["hybrid"]["visual_retrieval_energy_j"] for q in qids]),
            "TR": np.array([rows[q]["hybrid"]["text_retrieval_cpu_s"] for q in qids])}
    kw_sub = np.array([PATHS.index(keyword_submitted(q["question"])) for q in Q])
    kw_re = np.array([PATHS.index(keyword_regex(q["question"])) for q in Q])

    rc_path = RESULTS / "router_cost.json"
    rc = json.load(open(rc_path))["per_query_batch1"] if rc_path.exists() else None
    zero = {"gpu_s": 0.0, "energy_j": 0.0, "latency_s": 0.0}
    router_overhead = rc or zero
    ce_path = RESULTS / "router_ce_cost.json"
    ce_overhead = json.load(open(ce_path))["per_query_batch1"] if ce_path.exists() else None
    ce_dir = RESULTS / "router_ce"

    def overhead(n):
        if n in CE:
            return ce_overhead or zero
        return router_overhead if n in LEARNED else zero

    per_seed = []
    for f in sorted(Path(args.runs).glob("seed*.npz")):
        R = np.load(f, allow_pickle=True)
        assert list(R["qids"]) == qids, "router run was made from a different outputs file"
        te = R["split"] == "test"
        M = {k: v[te] for k, v in full.items()}
        cost = R["cost"]
        rng = np.random.default_rng(SEED)
        lam_i = list(R["lr_lambdas"]).index(LAMBDA_DEFAULT)
        ce_qhat, ce_lam = None, None
        ce_file = ce_dir / f.name
        if ce_file.exists():
            RC = np.load(ce_file, allow_pickle=True)
            assert list(RC["qids"]) == qids and (RC["split"] == R["split"]).all(), "router_ce split mismatch"
            ce_qhat = RC["qhat_ce_full"]
            va = R["split"] == "val"
            Av = A[va]
            target = Av[:, 2].mean() - VALSEL_DELTA
            val_pts = []
            for lam in VALSEL_LAMBDAS:
                ch = np.argmax(ce_qhat[va] - lam * cost, 1)
                g = C[va][np.arange(va.sum()), ch] + (ch == 0) * full["VR"][va]
                val_pts.append((lam, float(Av[np.arange(va.sum()), ch].mean()), float(g.mean())))
            ok = [p for p in val_pts if p[1] >= target]
            ce_lam = min(ok, key=lambda p: p[2])[0] if ok else max(val_pts, key=lambda p: p[1])[0]

        def choose(name, lam=LAMBDA_DEFAULT):
            return {
                "always_text": np.zeros(te.sum(), int), "always_visual": np.ones(te.sum(), int),
                "always_hybrid": np.full(te.sum(), 2),
                "keyword_submitted": kw_sub[te], "keyword_regex": kw_re[te],
                "logreg": R["lr_pred"][list(R["lr_lambdas"]).index(lam)][te],
                "knn": np.argmax(R["knn_qhat"][te] - lam * cost, 1),
                "cavr": np.argmax(R["cavr_qhat"][te] - lam * cost, 1),
                "cavr_cls_head": np.argmax(R["cavr_probs"][te], 1),
                "cavr_bm25": np.argmax(R["cavr_bm25_qhat"][te] - lam * cost, 1),
                "cavr_rf": np.argmax(R["cavr_rf_qhat"][te] - lam * cost, 1),
                "cavr_ce": np.argmax(ce_qhat[te] - lam * cost, 1) if ce_qhat is not None else None,
                "cavr_ce_valsel": np.argmax(ce_qhat[te] - ce_lam * cost, 1) if ce_qhat is not None else None,
                "oracle": oracle(M["A"], M["C"], lam),
            }[name]

        names = ["always_text", "always_visual", "always_hybrid", "keyword_submitted", "keyword_regex",
                 "logreg", "knn", "cavr", "cavr_cls_head", "cavr_bm25", "cavr_rf", "oracle"]
        if ce_qhat is not None:
            names[-1:-1] = ["cavr_ce", "cavr_ce_valsel"]
        res = {n: metrics(choose(n), M, overhead(n), rng, PRE_RETRIEVAL.get(n)) for n in names}
        if ce_qhat is not None:
            res["cavr_ce_valsel"]["lambda"] = ce_lam

        # Random routing at CAVR's budget: iid paths drawn with CAVR's path mix.
        mix = np.array([res["cavr"]["path_mix"][p] for p in PATHS])
        draws = [metrics(rng.choice(3, te.sum(), p=mix), M, zero, rng) for _ in range(N_RANDOM)]
        res["random_equal_budget"] = {k: float(np.mean([d[k] for d in draws]))
                                      for k in ("anls", "em", "gpu_s", "energy_j", "latency_p50_s", "latency_p95_s")}

        cavr_choice = choose("cavr")
        idx = np.arange(te.sum())
        tests = {n: {"mcnemar_em": mcnemar(M["EM"][idx, cavr_choice], M["EM"][idx, choose(n)]),
                     "paired_boot_anls": paired_boot(M["A"][idx, cavr_choice], M["A"][idx, choose(n)], rng)}
                 for n in names if n != "cavr"}
        tests_ce = {}
        if ce_qhat is not None:
            ce_choice = choose("cavr_ce_valsel")
            tests_ce = {n: {"mcnemar_em": mcnemar(M["EM"][idx, ce_choice], M["EM"][idx, choose(n)]),
                            "paired_boot_anls": paired_boot(M["A"][idx, ce_choice], M["A"][idx, choose(n)], rng)}
                        for n in ("always_hybrid", "always_visual", "cavr", "cavr_rf")}

        sweep_names = ["logreg", "knn", "cavr", "cavr_bm25", "cavr_rf", "oracle"] + (["cavr_ce"] if ce_qhat is not None else [])
        sweep = {str(lam): {n: {k: metrics(choose(n, lam), M, overhead(n), rng, PRE_RETRIEVAL.get(n))[k]
                                for k in ("anls", "gpu_s", "energy_j")}
                            for n in sweep_names}
                 for lam in LAMBDA_GRID}

        conf = R["cavr_conf"][te]
        good = (M["A"][idx, cavr_choice] >= 0.5).astype(float)
        bins = np.clip((conf * 10).astype(int), 0, 9)
        ece = float(sum(abs(conf[bins == b].mean() - good[bins == b].mean()) * (bins == b).mean()
                        for b in range(10) if (bins == b).any()))
        per_seed.append({"run": f.name, "n_test": int(te.sum()), "policies": res, "tests_vs_cavr": tests,
                         "tests_cavr_ce_valsel": tests_ce,
                         "lambda_sweep": sweep, "confidence_ece": ece})

    agg = {}
    for n in per_seed[0]["policies"]:
        agg[n] = {k: {"mean": float(np.mean([s["policies"][n][k] for s in per_seed])),
                      "std": float(np.std([s["policies"][n][k] for s in per_seed]))}
                  for k in ("anls", "em", "gpu_s", "energy_j", "latency_p50_s", "latency_p95_s")}
    write_json(args.out, {"lambda_default": LAMBDA_DEFAULT, "n_seeds": len(per_seed),
                          "router_overhead_included": rc is not None, "router_overhead": router_overhead,
                          "router_ce_overhead": ce_overhead, "valsel_delta": VALSEL_DELTA,
                          "aggregate": agg, "per_seed": per_seed, "hardware": hardware_info()})
    print(f"{'policy':22s} {'ANLS':>6s} {'EM':>6s} {'GPU-s':>7s} {'J':>7s} {'p95 s':>6s}")
    for n, m in agg.items():
        print(f"{n:22s} {m['anls']['mean']:6.3f} {m['em']['mean']:6.3f} {m['gpu_s']['mean']:7.3f} "
              f"{m['energy_j']['mean']:7.1f} {m['latency_p95_s']['mean']:6.2f}")


if __name__ == "__main__":
    main()
