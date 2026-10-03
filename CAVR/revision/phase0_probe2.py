"""Phase 0, probe 2 redone with full cost-quality curves (option A: a cheaper text path).

HYPOTHETICAL: text-path GPU cost scaled by s and a fraction (1-r) of its correct answers dropped at random;
no smaller model was run. Router = gradient-boosted per-path quality on question + retrieval + content features
(ColPali up front, charged to text-routed queries). Reports test ANLS at fixed GPU budgets (mean over seeds)
and the oracle. Output: results/phase0/probe2_curves.json
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from common import DATA, RESULTS, SEED, write_json
from phase0 import N_SEEDS, TXT, VIS, HYB, build_features, stage_costs
from routing_data import group_split, load, path_costs

LAMBDAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
BUDGETS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SCEN_S = [1.0, 0.5, 0.33, 0.2]
SCEN_R = [1.0, 0.9, 0.8, 0.7]


def gbm(seed):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)


def curve_at(pts, budgets):
    pts = sorted(pts)
    xs, ys = np.array([p[0] for p in pts]), np.maximum.accumulate([p[1] for p in pts])
    return [float(np.interp(b, xs, ys)) if b >= xs[0] else None for b in budgets]


def main():
    qids, Q, A, C, rows = load()
    G, VR = stage_costs(qids, rows)
    F_retr, F_cont, _, _, _ = build_features(qids, Q, rows)
    E = np.load(DATA / "embeddings.npz")
    emb = dict(zip(E["qids"], E["emb"]))
    ds1h = np.array([[q["dataset"] == "docvqa", q["dataset"] == "infovqa"] for q in Q], np.float32)
    X = np.hstack([np.stack([emb[q] for q in qids]), ds1h, F_retr, F_cont])

    out = {"budgets": BUDGETS, "scenarios": {}}
    per = {f"s={s},r={r}": {"router": [], "oracle": [], "visual": [], "hybrid": []} for s in SCEN_S for r in SCEN_R}
    for k in range(N_SEEDS):
        seed = SEED + k
        split = group_split(Q, seed)
        tr, te = split == "train", split == "test"
        q_vh = {p: np.clip(gbm(seed).fit(X[tr], A[tr, p]).predict(X), 0, 1) for p in (VIS, HYB)}
        rng = np.random.default_rng(seed)
        u = rng.random(len(A))                       # same random draw for every r, so scenarios are nested
        for s in SCEN_S:
            for r in SCEN_R:
                A2, C2 = A.copy(), C.copy()
                C2[:, TXT] = G[:, TXT] * s
                A2[(A2[:, TXT] > 0) & (u > r), TXT] = 0.0
                cost = path_costs(C2, tr)
                qt = np.clip(gbm(seed).fit(X[tr], A2[tr, TXT]).predict(X), 0, 1)
                qh = np.column_stack([qt, q_vh[VIS], q_vh[HYB]])
                pts = []
                for lam in LAMBDAS:
                    ch = np.argmax(qh[te] - lam * cost, 1)
                    g = C2[te][np.arange(te.sum()), ch] + (ch == TXT) * VR[te]   # ColPali ran for routing
                    pts.append((float(g.mean()), float(A2[te][np.arange(te.sum()), ch].mean())))
                opts = []
                for lam in LAMBDAS:                   # oracle curve: per-query best of ANLS - lam*cost
                    orc = np.argmax(A2[te] - lam * C2[te] - 1e-9 * C2[te], 1)
                    opts.append((float(C2[te][np.arange(te.sum()), orc].mean()),
                                 float(A2[te][np.arange(te.sum()), orc].mean())))
                key = f"s={s},r={r}"
                per[key]["router"].append(curve_at(pts, BUDGETS))
                per[key]["oracle"].append(curve_at(opts, BUDGETS))
        print(f"seed {seed} done", flush=True)

    for key, d in per.items():
        out["scenarios"][key] = {
            m: [float(np.mean([v for v in col if v is not None])) if any(v is not None for v in col) else None
                for col in zip(*d[m])] for m in ("router", "oracle")}
    write_json(RESULTS / "phase0" / "probe2_curves.json", out)

    print(f"{'scenario':14s} {'':7s}" + "".join(f"{b:7.2f}" for b in BUDGETS))
    for key, d in out["scenarios"].items():
        for m in ("router", "oracle"):
            print(f"{key:14s} {m:7s}" + "".join(f"{v:7.3f}" if v is not None else "     - " for v in d[m]))


if __name__ == "__main__":
    main()
