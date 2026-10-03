"""Step 5a: all-mpnet-base-v2 question embeddings (router input) -> data/embeddings.npz

Embeddings are deterministic, so they are computed once. Router latency/throughput is
measured separately (bench_router.py, step 8) on an otherwise idle GPU.
"""
import argparse
import json

import numpy as np
import torch

from common import DATA, set_seed

MODEL = "sentence-transformers/all-mpnet-base-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu while the generator run owns the GPU")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    set_seed()
    torch.set_num_threads(args.threads)
    from sentence_transformers import SentenceTransformer

    queries = [json.loads(l) for l in open(DATA / "queries.jsonl", encoding="utf-8")]
    model = SentenceTransformer(MODEL, device=args.device)
    emb = model.encode([q["question"] for q in queries], batch_size=64, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    np.savez(DATA / "embeddings.npz", qids=np.array([q["qid"] for q in queries]), emb=emb.astype(np.float32),
             model=MODEL)
    print(emb.shape)


if __name__ == "__main__":
    main()
