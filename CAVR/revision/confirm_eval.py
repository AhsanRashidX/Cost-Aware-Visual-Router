"""Evaluate the frozen CAVR-CE routers on the confirmation set (run with CAVR_DATA=data_confirm
CAVR_RESULTS=results_confirm after confirm_ce.py).

Per seed's router: its validation-chosen lambda and its training-split mean path costs (the decision rule is not
re-tuned); GPU cost is measured per query on the confirmation run (A10 node), including ColPali for text-routed
queries and the cross-encoder overhead measured on the workstation. Compared with always-text, always-visual,
always-hybrid, a quality-blind visual/hybrid mix at the router's cost (in expectation), and the per-query oracle.
Paired page-cluster bootstrap (1000 resamples) of the ANLS difference to always-hybrid; McNemar on exact match.
Output: results/confirmation.json (main results folder, read by the paper generators)
"""
import json

import numpy as np
from scipy.stats import binomtest

from common import RESULTS, ROOT, SEED, hardware_info, write_json
from routing_data import PATHS, load

TXT, VIS, HYB = 0, 1, 2
MAIN = ROOT / "results"


def main():
    qids, Q, A, C, rows = load()
    EM = np.array([[rows[q][p]["em"] for p in PATHS] for q in qids])
    VR = np.array([rows[q]["hybrid"]["visual_retrieval_gpu_s"] for q in qids])
    qh = np.load(ROOT / "results_confirm" / "confirm_ce_qhat.npz", allow_pickle=True)
    meta = json.load(open(ROOT / "results_confirm" / "confirm_ce_meta.json"))
    assert list(qh["qids"]) == list(qids), "query order differs"
    ce_over = json.load(open(MAIN / "router_ce_cost.json"))["per_query_batch1"]["gpu_s"]
    pages = np.array([f"{q['dataset']}/{q['page_id']}" for q in Q])
    upages, pidx = np.unique(pages, return_inverse=True)
    rng = np.random.default_rng(SEED)
    boot_pages = rng.integers(0, len(upages), (1000, len(upages)))

    def cluster_boot(d):
        sums = np.bincount(pidx, d, len(upages))
        cnts = np.bincount(pidx, None, len(upages))
        means = sums[boot_pages].sum(1) / cnts[boot_pages].sum(1)
        return [float(x) for x in np.percentile(means, [2.5, 97.5])]

    fixed = {p: {"anls": float(A[:, i].mean()), "em": float(EM[:, i].mean()), "gpu_s": float(C[:, i].mean())}
             for i, p in enumerate(PATHS)}
    i = np.arange(len(qids))
    best = A.max(1)
    orc = np.array([min((C[j, p], p) for p in range(3) if A[j, p] >= best[j] - 1e-12)[1] for j in i])
    seeds = []
    for k, sm in enumerate(meta["seeds"]):
        qhat, cost, lam = qh[f"qhat_seed{k}"], qh[f"cost_seed{k}"], sm["lambda_valsel"]
        ch = np.argmax(qhat - lam * cost, 1)
        gpu = C[i, ch] + (ch == TXT) * VR + ce_over
        a, g = A[i, ch], gpu
        v, h = fixed["visual"], fixed["hybrid"]
        f_cost = (g.mean() - v["gpu_s"]) / (h["gpu_s"] - v["gpu_s"])
        f_anls = (a.mean() - v["anls"]) / (h["anls"] - v["anls"])
        d = a - A[:, HYB]
        b = int(((EM[i, ch] == 1) & (EM[:, HYB] == 0)).sum())
        c = int(((EM[i, ch] == 0) & (EM[:, HYB] == 1)).sum())
        seeds.append({"seed": sm["seed"], "lambda": lam, "anls": float(a.mean()), "em": float(EM[i, ch].mean()),
                      "gpu_s": float(g.mean()), "path_mix": {p: float((ch == j).mean()) for j, p in enumerate(PATHS)},
                      "saving_vs_hybrid": float(1 - g.mean() / h["gpu_s"]),
                      "anls_diff_vs_hybrid": float(d.mean()), "anls_diff_vs_hybrid_ci95": cluster_boot(d),
                      "anls_diff_vs_visual": float((a - A[:, VIS]).mean()),
                      "anls_diff_vs_visual_ci95": cluster_boot(a - A[:, VIS]),
                      "mcnemar_vs_hybrid": {"ce_only": b, "hybrid_only": c,
                                            "p": float(binomtest(b, b + c).pvalue) if b + c else 1.0},
                      "blind_mix_anls_at_ce_cost": float(v["anls"] + f_cost * (h["anls"] - v["anls"])),
                      "blind_mix_gpu_at_ce_anls": float(v["gpu_s"] + f_anls * (h["gpu_s"] - v["gpu_s"])),
                      "saving_vs_blind_mix": float(1 - g.mean() / (v["gpu_s"] + f_anls * (h["gpu_s"] - v["gpu_s"])))})
    keys = ["anls", "gpu_s", "saving_vs_hybrid", "anls_diff_vs_hybrid", "anls_diff_vs_visual", "saving_vs_blind_mix",
            "blind_mix_anls_at_ce_cost"]
    summ = {kk: {"mean": float(np.mean([s[kk] for s in seeds])), "min": float(min(s[kk] for s in seeds)),
                 "max": float(max(s[kk] for s in seeds))} for kk in keys}
    out = {"n_queries": len(qids), "by_dataset": {ds: int(sum(q["dataset"] == ds for q in Q)) for ds in ("docvqa", "infovqa")},
           "pages": len(upages), "fixed": fixed,
           "oracle": {"anls": float(A[i, orc].mean()), "gpu_s": float(C[i, orc].mean())},
           "all_fail_share": float((best == 0).mean()), "ce_overhead_gpu_s": ce_over,
           "seeds": seeds, "summary": summ,
           "reproducibility": [{k: s[k] for k in ("seed", "max_abs_diff_vs_saved_qhat", "mean_abs_diff_vs_saved_qhat",
                                                  "test_decision_agreement", "test_anls_saved", "test_anls_retrained",
                                                  "test_gpu_s_saved", "test_gpu_s_retrained")} for s in meta["seeds"]],
           "excluded_oom": [json.loads(l) for l in open(ROOT / "results_confirm" / "path_outputs_oom.jsonl")],
           "note": "costs measured on the A10 node (generation) and the workstation (cross-encoder overhead)",
           "hardware": hardware_info()}
    write_json(MAIN / "confirmation.json", out)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
