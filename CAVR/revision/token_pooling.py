"""Compressed ColPali index: hierarchical token pooling against full and 2x2 spatial pooling (ARS review request).

For each page, the 1,024 patch vectors are clustered by Ward linkage (one linkage per page, cut at every pool
factor), each cluster is mean-pooled and re-normalised, and the prompt-token vectors are kept unpooled, as in the
2x2 pooling of bench_index.py. Retrieval quality is recall@1 and recall@5 of the gold page over each dataset's page
pool for all questions, with the same batched query embeddings (data/colpali_query_emb.pt) for every variant.
Scoring cost and index memory scale with the vectors per page.
Output: results/token_pooling.json
"""
import math
import time
from multiprocessing import Pool

import numpy as np
import torch

from bench_index import N_PATCH, maxsim_chunked, pool2x2
from common import DATA, RESULTS, hardware_info, write_json

FACTORS = [2, 3, 4]


def pool_page(page):
    """page: (Lp, d) float32. Returns {factor: (ceil(N_PATCH / f) + Lp - N_PATCH, d)} pooled vectors."""
    from scipy.cluster.hierarchy import fcluster, linkage
    patches, prompt = page[:N_PATCH], page[N_PATCH:]
    Z = linkage(patches, method="ward")
    out = {}
    for f in FACTORS:
        k = math.ceil(N_PATCH / f)
        lab = fcluster(Z, k, criterion="maxclust")
        cents = np.stack([patches[lab == c].mean(0) for c in np.unique(lab)])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12
        if len(cents) < k:                                 # ties can give fewer clusters: pad with zero vectors
            cents = np.vstack([cents, np.zeros((k - len(cents), page.shape[1]), np.float32)])
        out[f] = np.vstack([cents, prompt]).astype(np.float32)
    return out


def recall(index, keys, queries, qemb, ks=(1, 5)):
    hits = {k: 0 for k in ks}
    for ds in ("docvqa", "infovqa"):
        rows = [i for i, k in enumerate(keys) if k[0] == ds]
        corpus = index[rows].cuda()
        pid = np.array([keys[i][1] for i in rows])
        qs = [(q, e) for q, e in zip(queries, qemb) if q["dataset"] == ds]
        for q, e in qs:
            s = maxsim_chunked(e[None].cuda().to(corpus.dtype), corpus)[0]
            top = pid[torch.topk(s, max(ks)).indices.cpu().numpy()]
            for k in ks:
                hits[k] += q["page_id"] in top[:k]
        del corpus
        torch.cuda.empty_cache()
    return {f"recall@{k}": hits[k] / len(queries) for k in ks}


def main():
    import json
    saved = torch.load(DATA / "colpali_index.pt")
    real, keys = saved["index"], [tuple(k) for k in saved["keys"]]
    qe = torch.load(DATA / "colpali_query_emb.pt")
    qmap = {q["qid"]: q for q in map(json.loads, open(DATA / "queries.jsonl", encoding="utf-8"))}
    queries, qemb = [qmap[q] for q in qe["qids"]], qe["emb"]

    t0 = time.perf_counter()
    pages = [real[i].float().numpy() for i in range(real.shape[0])]
    with Pool(32) as pool:
        pooled = pool.map(pool_page, pages, chunksize=8)
    cluster_s = time.perf_counter() - t0
    variants = {"full": real, "pooled2x2": pool2x2(real)}
    for f in FACTORS:
        variants[f"hier{f}"] = torch.tensor(np.stack([p[f] for p in pooled])).to(real.dtype)

    gib_page_full = real[0].numel() * 2 / 2**30
    out = {"vectors_per_page_full": real.shape[1], "clustering_s_total": cluster_s, "methods": {}}
    for name, idx in variants.items():
        t1 = time.perf_counter()
        r = recall(idx, keys, queries, qemb)
        out["methods"][name] = {"vectors_per_page": idx.shape[1], **r,
                                "index_gib_1m_pages": idx.shape[1] / real.shape[1] * gib_page_full * 1e6,
                                "relative_scoring_cost": idx.shape[1] / real.shape[1], "eval_s": time.perf_counter() - t1}
        print(name, out["methods"][name], flush=True)
    out["hardware"] = hardware_info()
    write_json(RESULTS / "token_pooling.json", out)


if __name__ == "__main__":
    main()
