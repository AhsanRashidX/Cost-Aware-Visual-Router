"""Step 4: outcome-based route labels from the cached path outputs.

  utility label   = argmax_p ANLS(q,p) - lambda * cost_p     (cost_p = mean GPU-s on train split)
  threshold label = cheapest path with ANLS >= 0.5 (else cheapest)
Also: how often the submitted keyword rule agrees with the outcome labels.
Output: results/labels.json
"""
import argparse
from collections import Counter

import numpy as np
from sklearn.metrics import cohen_kappa_score

from common import RESULTS, SEED, write_json
from routing_data import (ANLS_OK, LAMBDA_DEFAULT, LAMBDA_GRID, PATHS, group_split, keyword_regex,
                          keyword_submitted, load, path_costs, threshold_labels, utility_labels)


def counts(lbl, split):
    return {s: {PATHS[k]: int(v) for k, v in sorted(Counter(lbl[split == s]).items())}
            for s in ("train", "val", "test")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(RESULTS / "path_outputs.jsonl"))
    ap.add_argument("--out", default=str(RESULTS / "labels.json"))
    args = ap.parse_args()

    qids, Q, A, C, _ = load(args.outputs)
    split = group_split(Q, SEED)
    cost = path_costs(C, split == "train")

    by_lambda = {}
    for lam in LAMBDA_GRID:
        lbl = utility_labels(A, cost, lam)
        by_lambda[str(lam)] = {"counts": counts(lbl, split),
                               "oracle_anls": float(A[np.arange(len(A)), lbl].mean()),
                               "oracle_cost": float(C[np.arange(len(C)), lbl].mean())}
    thr = threshold_labels(A, cost)
    default = utility_labels(A, cost, LAMBDA_DEFAULT)

    kw = {}
    for name, fn in (("keyword_submitted", keyword_submitted), ("keyword_regex", keyword_regex)):
        pred = np.array([PATHS.index(fn(q["question"])) for q in Q])
        kw[name] = {"distribution": {PATHS[k]: int(v) for k, v in sorted(Counter(pred).items())},
                    "agreement_with_utility_label": float((pred == default).mean()),
                    "kappa_with_utility_label": float(cohen_kappa_score(pred, default)),
                    "agreement_with_threshold_label": float((pred == thr).mean())}

    # How often each path is uniquely best, tied, or every path fails.
    best = A.max(axis=1, keepdims=True)
    n_best = (A >= best - 1e-12).sum(axis=1)
    write_json(args.out, {
        "n_queries": len(qids), "split_seed": SEED,
        "split_sizes": {s: int((split == s).sum()) for s in ("train", "val", "test")},
        "path_cost_gpu_s_train": dict(zip(PATHS, map(float, cost))),
        "mean_anls_per_path": dict(zip(PATHS, map(float, A.mean(axis=0)))),
        "lambda_default": LAMBDA_DEFAULT,
        "utility_labels_by_lambda": by_lambda,
        "threshold_labels": {"tau": ANLS_OK, "counts": counts(thr, split)},
        "ties": {"all_paths_fail(anls=0)": int((best[:, 0] == 0).sum()),
                 "single_best_path": int(((n_best == 1) & (best[:, 0] > 0)).sum()),
                 "tied_best": int(((n_best > 1) & (best[:, 0] > 0)).sum())},
        "keyword_rule_vs_outcome": kw,
    })
    print(f"{len(qids)} queries; default-lambda label counts:", by_lambda[str(LAMBDA_DEFAULT)]["counts"])


if __name__ == "__main__":
    main()
