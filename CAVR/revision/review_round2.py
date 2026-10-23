"""Analyses requested by the second review round, computed from cached results only (no model is re-run).

1. Gradient-boosting router as a policy (R20). router_ce.py trained HistGradientBoosting on the same tabular
   features as CAVR-CE (question embedding, dataset, BM25/ColPali retrieval scores, content features, with and
   without ColPali token statistics) and saved its per-path quality predictions. Here they go through exactly the
   CAVR-CE decision rule (evaluate.py: cheapest validation lambda within 0.005 ANLS of always-hybrid), with ColPali
   charged to text-routed queries and no router overhead (CPU inference, well under a millisecond).
2. Pooled estimates of the headline effects (R11).
   - Main splits: mean over the 5 splits with a t interval over splits. The splits overlap (a question can be in
     several test sets), so the interval is optimistic; it is reported next to the per-split range.
   - Confirmation set: one page-cluster bootstrap (1,000 resamples of whole pages) of the effect averaged over the
     five frozen routers, so router seed and question sampling are covered by a single interval.
3. Training-split exposure of the confirmation set (R10). Per path and dataset: ANLS on the validation questions vs
   the confirmation (training-split) questions, reweighted to the validation mix of question types, split by
   whether the gold page reached the generator. Memorised training answers would show as higher ANLS on training-
   split questions when the gold page was NOT in the context.
4. Compressed ColPali index footprints (R16), analytic: per-vector bytes for bf16, ColBERTv2-style residual
   compression (centroid id + 2-bit or 1-bit residuals) and binarisation, at 1,031 vectors of 128 dimensions per
   page. Retrieval quality is measured only for token pooling (token_pooling.json), not for these codecs.
Output: results/review_round2.json
"""
import json

import numpy as np
from scipy import stats

from common import RESULTS, ROOT, SEED, write_json
from evaluate import VALSEL_DELTA, VALSEL_LAMBDAS, metrics, paired_boot
from routing_data import PATHS, load

TXT, VIS, HYB = 0, 1, 2
N_BOOT = 1000
GBM = ("gbm_tab_tok", "gbm_tab")


def t_interval(xs):
    xs = np.asarray(xs, float)
    h = stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs))
    return {"mean": float(xs.mean()), "ci95_t_over_splits": [float(xs.mean() - h), float(xs.mean() + h)],
            "min": float(xs.min()), "max": float(xs.max()), "per_split": [float(x) for x in xs]}


def valsel_lambda(qhat, A, C, VR, va, cost):
    target = A[va][:, HYB].mean() - VALSEL_DELTA
    pts = []
    for lam in VALSEL_LAMBDAS:
        ch = np.argmax(qhat[va] - lam * cost, 1)
        g = C[va][np.arange(va.sum()), ch] + (ch == TXT) * VR[va]
        pts.append((lam, float(A[va][np.arange(va.sum()), ch].mean()), float(g.mean())))
    ok = [p for p in pts if p[1] >= target]
    return (min(ok, key=lambda p: p[2])[0], False) if ok else (max(pts, key=lambda p: p[1])[0], True)


def gbm_policies():
    qids, Q, A, C, rows = load()
    full = {"A": A, "C": C,
            "EM": np.array([[rows[q][p]["em"] for p in PATHS] for q in qids]),
            "E": np.array([[rows[q][p]["energy_j"] for p in PATHS] for q in qids]),
            "W": np.array([[rows[q][p]["wall_s"] for p in PATHS] for q in qids]),
            "VR": np.array([rows[q]["hybrid"]["visual_retrieval_gpu_s"] for q in qids]),
            "VRE": np.array([rows[q]["hybrid"]["visual_retrieval_energy_j"] for q in qids]),
            "TR": np.array([rows[q]["hybrid"]["text_retrieval_cpu_s"] for q in qids]),
            "PG": np.array([f"{q['dataset']}/{q['page_id']}" for q in Q])}
    ev = json.load(open(RESULTS / "eval_main.json"))
    zero = {"gpu_s": 0.0, "energy_j": 0.0, "latency_s": 0.0}
    per = []
    for k in range(5):
        R = np.load(RESULTS / "router_runs" / f"seed{k}.npz", allow_pickle=True)
        G = np.load(RESULTS / "router_ce" / f"seed{k}.npz", allow_pickle=True)
        assert list(R["qids"]) == qids and list(G["qids"]) == qids
        te, va, cost = R["split"] == "test", R["split"] == "val", R["cost"]
        M = {kk: v[te] for kk, v in full.items()}
        rng = np.random.default_rng(SEED)
        ce = ev["per_seed"][k]["policies"]["cavr_ce_valsel"]
        ce_choice = np.argmax(G["qhat_ce_full"][te] - ce["lambda"] * cost, 1)
        row = {"seed_split": k}
        for name in GBM:
            qhat = G[f"qhat_{name}"]
            lam, fb = valsel_lambda(qhat, full["A"], full["C"], full["VR"], va, cost)
            ch = np.argmax(qhat[te] - lam * cost, 1)
            m = metrics(ch, M, zero, rng, "visual")
            idx = np.arange(te.sum())
            m.update({"lambda": lam, "lambda_fallback": fb,
                      "vs_cavr_ce_anls": paired_boot(M["A"][idx, ch], M["A"][idx, ce_choice], rng, M["PG"]),
                      "vs_hybrid_anls": paired_boot(M["A"][idx, ch], M["A"][:, HYB], rng, M["PG"])})
            row[name] = m
        row["cavr_ce_valsel"] = {kk: ce[kk] for kk in ("anls", "gpu_s", "lambda")}
        row["always_hybrid"] = {kk: ev["per_seed"][k]["policies"]["always_hybrid"][kk] for kk in ("anls", "gpu_s")}
        row["always_visual"] = {kk: ev["per_seed"][k]["policies"]["always_visual"][kk] for kk in ("anls", "gpu_s")}
        per.append(row)
    agg = {}
    for name in GBM + ("cavr_ce_valsel", "always_hybrid", "always_visual"):
        agg[name] = {kk: t_interval([r[name][kk] for r in per]) for kk in ("anls", "gpu_s")}
        agg[name]["saving_vs_hybrid"] = t_interval([1 - r[name]["gpu_s"] / r["always_hybrid"]["gpu_s"] for r in per])
    for name in GBM:
        agg[name]["anls_diff_vs_cavr_ce"] = t_interval([r[name]["vs_cavr_ce_anls"]["mean_diff"] for r in per])
    return {"per_split": per, "aggregate": agg}


def pooled_main():
    ev = json.load(open(RESULTS / "eval_main.json"))
    ra = json.load(open(RESULTS / "review_analyses.json"))
    S = ev["per_seed"]
    ce = [s["policies"]["cavr_ce_valsel"] for s in S]
    hy = [s["policies"]["always_hybrid"] for s in S]
    vi = [s["policies"]["always_visual"] for s in S]
    return {"note": "t interval over the 5 page-grouped splits; splits overlap, so the interval is optimistic",
            "anls_diff_vs_hybrid": t_interval([c["anls"] - h["anls"] for c, h in zip(ce, hy)]),
            "anls_diff_vs_visual": t_interval([c["anls"] - v["anls"] for c, v in zip(ce, vi)]),
            "saving_vs_hybrid": t_interval([1 - c["gpu_s"] / h["gpu_s"] for c, h in zip(ce, hy)]),
            "extra_gpu_vs_visual": t_interval([c["gpu_s"] / v["gpu_s"] - 1 for c, v in zip(ce, vi)]),
            "saving_vs_blind_mix": t_interval(ra["blind_mix"]["saving_vs_mix"]["per_seed"])
            }


def confirm_data():
    out = ROOT / "results_confirm"
    rows = {}
    for r in map(json.loads, open(out / "path_outputs.jsonl", encoding="utf-8")):
        rows.setdefault(r["qid"], {})[r["path"]] = r
    Q = {q["qid"]: q for q in map(json.loads, open(ROOT / "data_confirm" / "queries.jsonl", encoding="utf-8"))}
    return rows, Q


def pooled_confirmation():
    rows, Qd = confirm_data()
    qh = np.load(ROOT / "results_confirm" / "confirm_ce_qhat.npz", allow_pickle=True)
    meta = json.load(open(ROOT / "results_confirm" / "confirm_ce_meta.json"))
    qids = list(qh["qids"])
    A = np.array([[rows[q][p]["anls"] for p in PATHS] for q in qids])
    C = np.array([[rows[q][p]["gpu_seconds"] for p in PATHS] for q in qids])
    VR = np.array([rows[q]["hybrid"]["visual_retrieval_gpu_s"] for q in qids])
    ce_over = json.load(open(RESULTS / "router_ce_cost.json"))["per_query_batch1"]["gpu_s"]
    i = np.arange(len(qids))
    ch = [np.argmax(qh[f"qhat_seed{k}"] - s["lambda_valsel"] * qh[f"cost_seed{k}"], 1) for k, s in enumerate(meta["seeds"])]
    a = np.mean([A[i, c] for c in ch], 0)                                   # per query, averaged over routers
    g = np.mean([C[i, c] + (c == TXT) * VR + ce_over for c in ch], 0)
    pages = np.array([f"{Qd[q]['dataset']}/{Qd[q]['page_id']}" for q in qids])
    up, pidx = np.unique(pages, return_inverse=True)
    pick = np.random.default_rng(SEED).integers(0, len(up), (N_BOOT, len(up)))
    cnt = np.bincount(pidx, minlength=len(up)).astype(float)

    def boot(x):
        s = np.bincount(pidx, x, len(up))
        return s[pick].sum(1) / cnt[pick].sum(1)

    def ci(v):
        return [float(x) for x in np.percentile(v, [2.5, 97.5])]
    d_h, d_v = boot(a - A[:, HYB]), boot(a - A[:, VIS])
    sav_h = 1 - boot(g) / boot(C[:, HYB])
    extra_v = boot(g) / boot(C[:, VIS]) - 1
    return {"n_queries": len(qids), "pages": len(up), "routers_averaged": len(ch),
            "anls_diff_vs_hybrid": {"mean": float((a - A[:, HYB]).mean()), "ci95": ci(d_h)},
            "anls_diff_vs_visual": {"mean": float((a - A[:, VIS]).mean()), "ci95": ci(d_v)},
            "saving_vs_hybrid": {"mean": float(1 - g.mean() / C[:, HYB].mean()), "ci95": ci(sav_h)},
            "extra_gpu_vs_visual": {"mean": float(g.mean() / C[:, VIS].mean() - 1), "ci95": ci(extra_v)}}


def exposure():
    """Per path/dataset ANLS: validation vs training-split (confirmation) questions, by gold page in context."""
    _, Qv, Av, _, rows_v = load()
    rows_c, Qc = confirm_data()
    qc = [q for q in Qc if q in rows_c and all(p in rows_c[q] for p in PATHS)]
    out = {}
    for ds in ("docvqa", "infovqa"):
        V = [(q, rows_v[q["qid"]]) for q in Qv if q["dataset"] == ds]
        T = [(Qc[q], rows_c[q]) for q in qc if Qc[q]["dataset"] == ds]
        types = sorted({q["qtype"] for q, _ in V})
        wv = {t: sum(q["qtype"] == t for q, _ in V) / len(V) for t in types}
        # InfographicVQA's training split carries no question types ("unknown"): compared without reweighting
        can_reweight = any(q["qtype"] in wv for q, _ in T)
        res = {"n_validation": len(V), "n_training_split": len(T), "reweighted_by_question_type": can_reweight}
        for p in PATHS:
            r = {}
            for gold in (True, False, None):
                def mean_w(items, reweight):
                    sel = [(q, rr[p]) for q, rr in items if gold is None or bool(rr[p]["gold_in_context"]) == gold]
                    if not sel:
                        return None, 0
                    if not reweight:
                        return float(np.mean([x["anls"] for _, x in sel])), len(sel)
                    num = den = 0.0
                    for t in types:                     # reweight training-split questions to the validation type mix
                        xs = [x["anls"] for q, x in sel if q["qtype"] == t]
                        if xs:
                            num, den = num + wv[t] * np.mean(xs), den + wv[t]
                    return float(num / den), len(sel)
                key = {True: "gold_in_context", False: "gold_not_in_context", None: "all"}[gold]
                v, nv = mean_w(V, False)
                t, nt = mean_w(T, can_reweight)
                r[key] = {"validation": v, "training_split": t, "n_validation": nv, "n_training_split": nt,
                          "diff": None if v is None or t is None else t - v}
            r["gold_in_context_rate"] = {"validation": float(np.mean([bool(rr[p]["gold_in_context"]) for _, rr in V])),
                                         "training_split": float(np.mean([bool(rr[p]["gold_in_context"]) for _, rr in T]))}
            res[p] = r
        out[ds] = res
    return out


def compressed_index():
    ix = json.load(open(RESULTS / "index_scaling.json"))
    tp = json.load(open(RESULTS / "token_pooling.json"))
    vec, dim, pages = 1031, 128, 10**6
    codecs = {"bf16 (measured layout)": 2 * dim,
              "residual, 2-bit (ColBERTv2-style: 4-byte centroid id + 2 bits/dim)": 4 + dim * 2 // 8,
              "residual, 1-bit (4-byte centroid id + 1 bit/dim)": 4 + dim // 8,
              "binary (1 bit/dim)": dim // 8}
    return {"vectors_per_page": vec, "dim": dim, "pages": pages,
            "gib_1m_pages": {k: vec * b * pages / 2**30 for k, b in codecs.items()},
            "bytes_per_vector": codecs,
            "check_bf16_vs_index_scaling_1m_gib": 10 * ix["scaling"]["100000"]["index_gb_bf16"],
            "token_pooling_vectors_per_page_full": tp["vectors_per_page_full"],
            "note": "analytic footprints; retrieval quality of residual/binary codecs not measured on this corpus"}


def main():
    out = {"gbm_router": gbm_policies(), "pooled_main_splits": pooled_main(),
           "pooled_confirmation": pooled_confirmation(), "training_split_exposure": exposure(),
           "compressed_index": compressed_index()}
    write_json(RESULTS / "review_round2.json", out)
    a = out["gbm_router"]["aggregate"]
    for n in GBM + ("cavr_ce_valsel", "always_hybrid", "always_visual"):
        print(f"{n:16s} ANLS {a[n]['anls']['mean']:.3f}  GPU-s {a[n]['gpu_s']['mean']:.3f}  "
              f"saving vs hybrid {100 * a[n]['saving_vs_hybrid']['mean']:.0f}%")
    print(json.dumps(out["pooled_confirmation"], indent=1))
    print(json.dumps({k: v for k, v in out["pooled_main_splits"].items() if k != "note"}, indent=1)[:1500])


if __name__ == "__main__":
    main()
