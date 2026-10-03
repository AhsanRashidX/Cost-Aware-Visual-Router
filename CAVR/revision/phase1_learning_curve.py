"""Phase 1c: learning curve - does the content-aware router improve with more training data?

Train on 25/50/75/100% of the training *pages* (val/test fixed) and record mean test success AUC, for the
cross-encoder (2 epochs: val AUC peaks at epoch 1-2 in the full run) and the GBM baseline.
A clear upward trend is evidence that labelling the DocVQA/InfoVQA train splits would raise AUC further.
Output: results/router_ce/learning_curve.json
"""
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

import router_ce as R
from common import DATA, RESULTS, SEED, write_json
from phase0 import build_features
from routing_data import ANLS_OK, group_split, load

FRACS = [0.25, 0.5, 0.75, 1.0]
SEEDS = [SEED, SEED + 1]


def main():
    from transformers import AutoTokenizer
    R.HP["epochs"] = 2
    qids, Q, A, C, rows = load()
    F_retr, F_cont, _, _, _ = build_features(qids, Q, rows)
    T = np.load(DATA / "colpali_token_feats.npz")
    tpos = {q: i for i, q in enumerate(T["qids"])}
    F_tok = T["feats"][[tpos[q] for q in qids]]
    ds1h = np.array([[q["dataset"] == "docvqa", q["dataset"] == "infovqa"] for q in Q], np.float32)
    E = np.load(DATA / "embeddings.npz")
    emb = dict(zip(E["qids"], E["emb"]))
    Xq = np.stack([emb[q] for q in qids])
    tok = AutoTokenizer.from_pretrained(R.BACKBONE)
    enc = tok([q["question"] for q in Q], R.build_passages(qids, Q), truncation="only_second",
              max_length=R.MAX_LEN, padding="max_length", return_tensors="pt")
    pages = np.array([f"{q['dataset']}/{q['page_id']}" for q in Q])

    out = {"fracs": FRACS, "seeds": SEEDS, "ce_epochs": R.HP["epochs"], "runs": []}
    for seed in SEEDS:
        split = group_split(Q, seed)
        tr_all, va, te = split == "train", split == "val", split == "test"
        tr_pages = np.unique(pages[tr_all])
        np.random.default_rng(seed).shuffle(tr_pages)
        for f in FRACS:
            keep = set(tr_pages[:int(round(f * len(tr_pages)))])
            tr = tr_all & np.array([p in keep for p in pages])
            mu = lambda F: (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-6)
            tab = np.hstack([mu(np.hstack([F_retr, F_cont, F_tok])), ds1h]).astype(np.float32)
            t0 = time.time()
            qhat, _, _ = R.train_ce(enc, tab, A, tr, va, seed)
            ce_auc = float(np.mean([roc_auc_score(A[te, p] >= ANLS_OK, qhat[te, p]) for p in range(3)]))
            X = np.hstack([Xq, ds1h, F_retr, F_cont, F_tok])
            g = np.column_stack([HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                 l2_regularization=1.0, random_state=seed).fit(X[tr], A[tr, p]).predict(X) for p in range(3)])
            gbm_auc = float(np.mean([roc_auc_score(A[te, p] >= ANLS_OK, g[te, p]) for p in range(3)]))
            out["runs"].append({"seed": seed, "frac": f, "n_train": int(tr.sum()), "ce_auc": ce_auc, "gbm_auc": gbm_auc})
            print(f"seed {seed} frac {f}: n={tr.sum()} CE AUC {ce_auc:.3f} GBM AUC {gbm_auc:.3f} ({time.time() - t0:.0f}s)", flush=True)
            write_json(RESULTS / "router_ce" / "learning_curve.json", out)


if __name__ == "__main__":
    main()
