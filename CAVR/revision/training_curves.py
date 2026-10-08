"""Training and validation loss curves for CAVR and CAVR-CE.

Re-trains both routers on every split with exactly the settings and seeds of router.py and router_ce.py, with
per-epoch loss logging switched on. Only the curves are saved; the evaluated predictions in results/router_runs and
results/router_ce are not touched. CAVR trains on the CPU and is deterministic; CAVR-CE trains on the GPU with bf16,
so its curves can differ slightly from the original run in the last digits.
Output: results/training_curves.json
"""
import json
import time

import numpy as np

import router as RC
import router_ce as RE
from common import DATA, RESULTS, SEED, hardware_info, write_json
from phase0 import build_features
from routing_data import LAMBDA_DEFAULT, group_split, load, path_costs, utility_labels


def main():
    from transformers import AutoTokenizer
    qids, Q, A, C, rows = load()
    E = np.load(DATA / "embeddings.npz")
    emb = dict(zip(E["qids"], E["emb"]))
    ds1h = np.array([[q["dataset"] == "docvqa", q["dataset"] == "infovqa"] for q in Q], np.float32)
    X = np.hstack([np.stack([emb[q] for q in qids]), ds1h]).astype(np.float32)

    F_retr, F_cont, _, _, _ = build_features(qids, Q, rows)
    T = np.load(DATA / "colpali_token_feats.npz")
    tpos = {q: i for i, q in enumerate(T["qids"])}
    F_tok = T["feats"][[tpos[q] for q in qids]]
    tok = AutoTokenizer.from_pretrained(RE.BACKBONE)
    enc = tok([q["question"] for q in Q], RE.build_passages(qids, Q), truncation="only_second",
              max_length=RE.MAX_LEN, padding="max_length", return_tensors="pt")

    out = {"note": "re-trained with the settings and seeds of router.py / router_ce.py, for the loss figure only",
           "cavr_hp": RC.HP, "ce_hp": RE.HP, "cavr": [], "cavr_ce": []}
    for k in range(5):
        seed = SEED + k
        split = group_split(Q, seed)
        tr, va = split == "train", split == "val"
        cost = path_costs(C, tr)
        y = utility_labels(A, cost, LAMBDA_DEFAULT)

        log = {}
        RC.train_cavr(X, A, C, y, cost, tr, va, seed, "cpu", log=log)
        out["cavr"].append({"seed": seed, **log})

        mu, sd = lambda F: F[tr].mean(0), lambda F: F[tr].std(0) + 1e-6
        Fall = np.hstack([F_retr, F_cont, F_tok])
        tab = np.hstack([(Fall - mu(Fall)) / sd(Fall), ds1h]).astype(np.float32)
        log = {}
        t0 = time.time()
        RE.train_ce(enc, tab, A, tr, va, seed, log=log)
        out["cavr_ce"].append({"seed": seed, **log})
        print(f"seed {seed}: CAVR {len(out['cavr'][-1]['train_total'])} epochs (best {out['cavr'][-1]['best_epoch'] + 1}); "
              f"CAVR-CE best epoch {log['best_epoch'] + 1} ({time.time() - t0:.0f}s)", flush=True)
        write_json(RESULTS / "training_curves.json", out)
    out["hardware"] = hardware_info()
    write_json(RESULTS / "training_curves.json", out)


if __name__ == "__main__":
    main()
