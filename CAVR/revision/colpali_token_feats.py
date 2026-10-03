"""Phase 1a: ColPali per-query-token match statistics (router features).

For every query: encode it with ColPali (the visual path does this anyway) and, for the ColPali top-1 and
top-2 pages and the BM25 top-1 page, take each question token's best-matching page vector (MaxSim per token).
Summaries say whether *every* part of the question finds a match on the page, which the plain page score hides.
Output: data/colpali_token_feats.npz (qids, feats, names, per-query feature time)
"""
import json
import time

import numpy as np
import torch

from common import DATA, RESULTS, set_seed
from retrieve_visual import load_model

N_AUG = 10   # ColPali appends 10 query-augmentation tokens


def stats(x):
    return [float(x.mean()), float(x.min()), float(np.percentile(x, 25)), float(x.std())]


def main():
    set_seed()
    queries = [json.loads(l) for l in open(DATA / "queries.jsonl", encoding="utf-8")]
    rt = {r["qid"]: r for r in map(json.loads, open(RESULTS / "retrieval_text.jsonl", encoding="utf-8"))}
    rv = {r["qid"]: r for r in map(json.loads, open(RESULTS / "retrieval_visual.jsonl", encoding="utf-8"))}
    saved = torch.load(DATA / "colpali_index.pt")
    index = saved["index"].cuda()
    pos = {k: i for i, k in enumerate(saved["keys"])}
    model, proc = load_model()
    tok = proc.tokenizer
    n_prefix = len(tok(tok.bos_token + proc.query_prefix, add_special_tokens=False)["input_ids"])

    feats, secs = [], []
    with torch.inference_mode():
        for q in queries:
            ds = q["dataset"]
            pages = [rv[q["qid"]]["top"][0], rv[q["qid"]]["top"][1], rt[q["qid"]]["top"][0]]
            t0 = time.perf_counter()
            inp = proc.process_queries([q["question"]]).to(model.device)
            emb = model(**inp)[0][inp["attention_mask"][0].bool()]          # (Lq, 128)
            real = emb[n_prefix:len(emb) - N_AUG] if len(emb) - N_AUG > n_prefix else emb
            per_tok = torch.stack([(real @ index[pos[(ds, p)]].T).amax(1) for p in pages]).float().cpu().numpy()
            torch.cuda.synchronize()
            secs.append(time.perf_counter() - t0)
            c1, c2, b1 = per_tok
            d = c1 - c2
            feats.append(stats(c1) + stats(c2) + stats(b1) + [float(d.mean()), float(d.min()), float((d > 0).mean()),
                                                              float((b1 - c1).mean()), float(len(real))])
    names = [f"{p}_{s}" for p in ("col1", "col2", "bm1") for s in ("mean", "min", "p25", "std")] + \
            ["col1_minus_col2_mean", "col1_minus_col2_min", "col1_beats_col2_frac", "bm1_minus_col1_mean", "n_tokens"]
    np.savez(DATA / "colpali_token_feats.npz", qids=np.array([q["qid"] for q in queries]),
             feats=np.array(feats, np.float32), names=np.array(names), seconds=np.array(secs))
    print(f"{len(feats)} queries; median {1000 * np.median(secs):.1f} ms/query (includes ColPali query encoding)")


if __name__ == "__main__":
    main()
