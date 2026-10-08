"""Step 5: train the CAVR router and learned baselines on 5 seeds (each seed = new page-grouped split).

CAVR: MLP on [mpnet embedding, dataset one-hot] with three heads
  path logits (CE on the utility label at LAMBDA_DEFAULT)
  per-path predicted quality (MSE on ANLS per path)
  confidence (BCE: does the path chosen by the quality head reach ANLS >= 0.5)
Inference decision: argmax_p qhat_p - lambda * cost_p, lambda chosen at run time.

Baselines: logistic regression on the utility label (one model per lambda),
           kNN regression of per-path ANLS (decision as CAVR).
Output: results/router_runs/seed<k>.npz (+ router_meta.json)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

from common import DATA, RESULTS, SEED, hardware_info, set_seed, write_json
from routing_data import (ANLS_OK, LAMBDA_DEFAULT, LAMBDA_GRID, PATHS, group_split, load, path_costs,
                          utility_labels)

N_SEEDS = 5
HP = {"hidden": [256, 128], "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "batch": 128,
      "max_epochs": 100, "patience": 10, "w_cls": 1.0, "w_quality": 1.0, "w_conf": 0.5}


class CAVR(nn.Module):
    def __init__(self, d_in, n_paths=len(PATHS), hidden=HP["hidden"], dropout=HP["dropout"]):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.logits = nn.Linear(d, n_paths)
        self.quality = nn.Linear(d, n_paths)
        self.conf = nn.Linear(d, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.logits(h), torch.sigmoid(self.quality(h)), self.conf(h).squeeze(-1)


def decide(qhat, cost, lam):
    return np.argmax(qhat - lam * cost[None, :], axis=1)


def utility(A, C, choice, lam):
    idx = np.arange(len(A))
    return float((A[idx, choice] - lam * C[idx, choice]).mean())


def train_cavr(X, A, C, y, cost, tr, va, seed, device, log=None):
    """log: optional dict; if given, per-epoch train/val losses and val utility are appended to it."""
    set_seed(seed)
    model = CAVR(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    Xt = torch.tensor(X, device=device)
    At = torch.tensor(A, dtype=torch.float32, device=device)
    yt = torch.tensor(y, device=device)
    cost_t = torch.tensor(cost, dtype=torch.float32, device=device)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    tr_idx = torch.tensor(np.where(tr)[0])
    best, best_state, bad, history = -np.inf, None, 0, []
    va_t = torch.tensor(np.where(va)[0], device=device)

    def losses(logits, q, c, idx):
        with torch.no_grad():
            chosen = torch.argmax(q - LAMBDA_DEFAULT * cost_t, dim=1)
            conf_target = (At[idx].gather(1, chosen[:, None]).squeeze(1) >= ANLS_OK).float()
        parts = (HP["w_cls"] * ce(logits, yt[idx]), HP["w_quality"] * ((q - At[idx]) ** 2).mean(),
                 HP["w_conf"] * bce(c, conf_target))
        return parts[0] + parts[1] + parts[2], parts

    for epoch in range(HP["max_epochs"]):
        model.train()
        ep_sum, ep_n = np.zeros(4), 0
        for b in tr_idx[torch.randperm(len(tr_idx), generator=gen)].split(HP["batch"]):
            b = b.to(device)
            logits, q, c = model(Xt[b])
            with torch.no_grad():
                chosen = torch.argmax(q - LAMBDA_DEFAULT * cost_t, dim=1)
                conf_target = (At[b].gather(1, chosen[:, None]).squeeze(1) >= ANLS_OK).float()
            loss = (HP["w_cls"] * ce(logits, yt[b]) + HP["w_quality"] * ((q - At[b]) ** 2).mean()
                    + HP["w_conf"] * bce(c, conf_target))
            if log is not None:
                with torch.no_grad():
                    _, parts = losses(logits, q, c, b)
                ep_sum += len(b) * np.array([loss.item()] + [p.item() for p in parts])
                ep_n += len(b)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            qv = model(Xt[torch.tensor(np.where(va)[0], device=device)])[1].cpu().numpy()
        u = utility(A[va], C[va], decide(qv, cost, LAMBDA_DEFAULT), LAMBDA_DEFAULT)
        history.append(u)
        if log is not None:
            with torch.no_grad():
                vl, vparts = losses(*model(Xt[va_t]), va_t)
            for k, v in zip(("train_total", "train_cls", "train_quality", "train_conf"), ep_sum / max(ep_n, 1)):
                log.setdefault(k, []).append(float(v))
            for k, v in zip(("val_total", "val_cls", "val_quality", "val_conf"), [vl] + list(vparts)):
                log.setdefault(k, []).append(float(v.item()))
            log.setdefault("val_utility", []).append(float(u))
        if u > best + 1e-6:
            best, bad = u, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    model.load_state_dict(best_state)
    if log is not None:
        log["best_epoch"] = int(np.argmax(history))   # 0-based; the epoch whose weights are kept
    model.eval()
    with torch.no_grad():
        logits, q, c = model(Xt)
    return {"probs": torch.softmax(logits, 1).cpu().numpy(), "qhat": q.cpu().numpy(),
            "conf": torch.sigmoid(c).cpu().numpy(), "epochs": len(history), "val_utility": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(RESULTS / "path_outputs.jsonl"))
    ap.add_argument("--out-dir", default=str(RESULTS / "router_runs"))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"  # tiny model; keeps the GPU free and results bit-reproducible

    qids, Q, A, C, _ = load(args.outputs)
    E = np.load(DATA / "embeddings.npz")
    emb = dict(zip(E["qids"], E["emb"]))
    ds_onehot = np.array([[q["dataset"] == "docvqa", q["dataset"] == "infovqa"] for q in Q], dtype=np.float32)
    X = np.hstack([np.stack([emb[q] for q in qids]), ds_onehot]).astype(np.float32)

    # Retrieval-score features for the CAVR variants (top score and top-1 minus top-2 margin).
    def ret_feats(path):
        r = {x["qid"]: x for x in map(json.loads, open(path, encoding="utf-8"))}
        return np.array([[r[q]["scores"][0], r[q]["scores"][0] - r[q]["scores"][1]] for q in qids], dtype=np.float32)
    F_BM25 = ret_feats(RESULTS / "retrieval_text.jsonl")
    F_COL = ret_feats(RESULTS / "retrieval_visual.jsonl")

    summary = []
    for s in range(args.seeds):
        seed = SEED + s
        split = group_split(Q, seed)
        tr, va, te = split == "train", split == "val", split == "test"
        cost = path_costs(C, tr)
        y = utility_labels(A, cost, LAMBDA_DEFAULT)

        cavr = train_cavr(X, A, C, y, cost, tr, va, seed, device)

        def with_feats(F):  # standardize retrieval scores with train-split statistics
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
            return np.hstack([X, (F - mu) / sd]).astype(np.float32)
        cavr_bm25 = train_cavr(with_feats(F_BM25), A, C, y, cost, tr, va, seed, device)
        cavr_rf = train_cavr(with_feats(np.hstack([F_BM25, F_COL])), A, C, y, cost, tr, va, seed, device)

        # Logistic regression: one classifier per lambda (labels change with lambda); C picked on val.
        lr_pred = {}
        for lam in LAMBDA_GRID:
            yl = utility_labels(A, cost, lam)
            if len(np.unique(yl[tr])) < 2:
                lr_pred[lam] = np.full(len(A), yl[tr][0])
                continue
            best = None
            for c_reg in (0.1, 1.0, 10.0):
                m = LogisticRegression(C=c_reg, max_iter=3000, random_state=seed).fit(X[tr], yl[tr])
                u = utility(A[va], C[va], m.predict(X[va]), lam)
                if best is None or u > best[0]:
                    best = (u, m)
            lr_pred[lam] = best[1].predict(X)

        # kNN regression of per-path ANLS; k picked on val utility at the default lambda.
        best = None
        for k in (5, 10, 25, 50, 100):
            m = KNeighborsRegressor(n_neighbors=min(k, int(tr.sum())), metric="cosine").fit(X[tr], A[tr])
            qv = m.predict(X[va])
            u = utility(A[va], C[va], decide(qv, cost, LAMBDA_DEFAULT), LAMBDA_DEFAULT)
            if best is None or u > best[0]:
                best = (u, k, m)
        knn_qhat = best[2].predict(X)

        np.savez(out_dir / f"seed{s}.npz", qids=np.array(qids), split=split, cost=cost, A=A, C=C,
                 cavr_probs=cavr["probs"], cavr_qhat=cavr["qhat"], cavr_conf=cavr["conf"],
                 cavr_bm25_qhat=cavr_bm25["qhat"], cavr_bm25_conf=cavr_bm25["conf"],
                 cavr_rf_qhat=cavr_rf["qhat"], cavr_rf_conf=cavr_rf["conf"],
                 knn_qhat=knn_qhat, lr_lambdas=np.array(LAMBDA_GRID),
                 lr_pred=np.stack([lr_pred[l] for l in LAMBDA_GRID]))
        summary.append({"seed": seed, "sizes": {k: int(v.sum()) for k, v in (("train", tr), ("val", va), ("test", te))},
                        "cavr_epochs": cavr["epochs"], "cavr_val_utility": cavr["val_utility"],
                        "cavr_bm25_val_utility": cavr_bm25["val_utility"], "cavr_rf_val_utility": cavr_rf["val_utility"],
                        "knn_k": best[1]})
        print(summary[-1], flush=True)

    write_json(out_dir / "router_meta.json", {"hyperparameters": HP, "lambda_default": LAMBDA_DEFAULT,
                                              "features": {"cavr": "all-mpnet-base-v2 (normalized) + dataset one-hot",
                                                           "cavr_bm25": "cavr features + BM25 top score and margin (standardized)",
                                                           "cavr_rf": "cavr_bm25 features + ColPali top score and margin"},
                                              "seeds": summary, "hardware": hardware_info()})


if __name__ == "__main__":
    main()
