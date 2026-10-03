"""Phase 0, probes 4-5 (from cache).

Probe 4 (option D preview): restrict to queries where every path already had the gold page in context
  (BM25 top-3 contains it and ColPali top-1 is it) - a proxy for the given-page setting - and compare
  per-path ANLS, oracle headroom and the router's cost-quality curve there.
Probe 5 (how good must a router be?): blend the router's predicted quality with the true per-path ANLS,
  qhat_a = a*ANLS + (1-a)*qhat, and record the success AUC and the cost-quality curve for each a.
  This shows what prediction quality would be needed for "hybrid quality at >=30% less GPU".
Output: results/phase0/probe45.json
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from common import DATA, RESULTS, SEED, write_json
from phase0 import N_SEEDS, TXT, VIS, HYB, build_features, stage_costs
from routing_data import ANLS_OK, PATHS, group_split, load, path_costs

LAMBDAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
BUDGETS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]


def gbm(seed):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)


def curve(qhat, A, C, VR, cost, mask):
    pts = []
    for lam in LAMBDAS:
        ch = np.argmax(qhat[mask] - lam * cost, 1)
        i = np.arange(mask.sum())
        pts.append((float((C[mask][i, ch] + (ch == TXT) * VR[mask]).mean()), float(A[mask][i, ch].mean())))
    pts.sort()
    xs, ys = np.array([p[0] for p in pts]), np.maximum.accumulate([p[1] for p in pts])
    return [float(np.interp(b, xs, ys)) if b >= xs[0] else None for b in BUDGETS], pts


def mean_cols(rows):
    return [float(np.mean([v for v in c if v is not None])) if any(v is not None for v in c) else None
            for c in zip(*rows)]


def main():
    qids, Q, A, C, rows = load()
    G, VR = stage_costs(qids, rows)
    F_retr, F_cont, _, _, _ = build_features(qids, Q, rows)
    E = np.load(DATA / "embeddings.npz")
    emb = dict(zip(E["qids"], E["emb"]))
    ds1h = np.array([[q["dataset"] == "docvqa", q["dataset"] == "infovqa"] for q in Q], np.float32)
    X = np.hstack([np.stack([emb[q] for q in qids]), ds1h, F_retr, F_cont])
    gold_all = np.array([rows[q]["text"]["gold_in_context"] and rows[q]["visual"]["gold_in_context"] for q in qids])

    p4 = {"router": [], "fixed": [], "oracle": []}
    p5 = {a: {"curve": [], "auc": []} for a in ALPHAS}
    for k in range(N_SEEDS):
        seed = SEED + k
        split = group_split(Q, seed)
        tr, te = split == "train", split == "test"
        cost = path_costs(C, tr)
        qhat = np.column_stack([np.clip(gbm(seed).fit(X[tr], A[tr, p]).predict(X), 0, 1) for p in range(3)])

        # Probe 5: blend toward the truth.
        for a in ALPHAS:
            qa = a * A + (1 - a) * qhat
            p5[a]["curve"].append(curve(qa, A, C, VR, cost, te)[0])
            p5[a]["auc"].append([float(roc_auc_score(A[te, p] >= ANLS_OK, qa[te, p])) for p in range(3)])

        # Probe 4: gold page available to every path; router retrained on that subset.
        trg, teg = tr & gold_all, te & gold_all
        cost_g = path_costs(C, trg)
        qg = np.column_stack([np.clip(gbm(seed).fit(X[trg], A[trg, p]).predict(X), 0, 1) for p in range(3)])
        p4["router"].append(curve(qg, A, C, VR, cost_g, teg)[0])
        p4["fixed"].append([[float(A[teg, p].mean()), float(C[teg, p].mean())] for p in range(3)])
        orc = np.argmax(A[teg] - 1e-9 * C[teg], 1)
        i = np.arange(teg.sum())
        p4["oracle"].append([float(A[teg][i, orc].mean()), float(C[teg][i, orc].mean())])
        print(f"seed {seed} done", flush=True)

    out = {"budgets": BUDGETS,
           "probe4_gold_subset": {"share_of_queries": float(gold_all.mean()),
                                  "router_curve": mean_cols(p4["router"]),
                                  "fixed": {p: np.mean([f[i] for f in p4["fixed"]], 0).tolist() for i, p in enumerate(PATHS)},
                                  "oracle": np.mean(p4["oracle"], 0).tolist()},
           "probe5_blend": {str(a): {"curve": mean_cols(v["curve"]),
                                     "auc_success": dict(zip(PATHS, np.mean(v["auc"], 0).tolist()))}
                            for a, v in p5.items()}}
    write_json(RESULTS / "phase0" / "probe45.json", out)

    g = out["probe4_gold_subset"]
    print(f"\nProbe 4: gold page in every path's context for {100 * g['share_of_queries']:.1f}% of queries")
    for p in PATHS:
        print(f"  always {p:7s} ANLS {g['fixed'][p][0]:.3f} @ {g['fixed'][p][1]:.3f} GPU-s")
    print(f"  oracle         ANLS {g['oracle'][0]:.3f} @ {g['oracle'][1]:.3f} GPU-s")
    print("  router at budgets", dict(zip(BUDGETS, [round(v, 3) if v else None for v in g["router_curve"]])))
    print("\nProbe 5: ANLS at GPU budget", BUDGETS)
    for a, v in out["probe5_blend"].items():
        au = v["auc_success"]
        print(f"  alpha={a:4s} AUC text/vis/hyb {au['text']:.3f}/{au['visual']:.3f}/{au['hybrid']:.3f} | "
              + " ".join(f"{x:.3f}" if x else "  -  " for x in v["curve"]))


if __name__ == "__main__":
    main()
