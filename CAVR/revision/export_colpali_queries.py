"""Export ColPali query embeddings (per-token, fp16) for every question, for GPU benchmarks on other machines.

Together with data/colpali_index.pt this lets the multi-GPU index benchmark (p40/shard_bench.py) run without loading
any model. Output: data/colpali_query_emb.pt  {"qids": [...], "emb": [Tensor(L_i, 128) fp16, ...]}
"""
import json

import torch

from common import DATA, set_seed
from retrieve_visual import load_model


def main():
    set_seed()
    queries = [json.loads(l) for l in open(DATA / "queries.jsonl", encoding="utf-8")]
    model, proc = load_model()
    out = []
    with torch.inference_mode():
        for i in range(0, len(queries), 64):
            texts = [q["question"] for q in queries[i:i + 64]]
            batch = proc.process_queries(texts)
            emb = model(**batch.to(model.device))
            mask = batch["attention_mask"].bool()
            out += [emb[j][mask[j]].to(torch.float16).cpu() for j in range(len(texts))]
    torch.save({"qids": [q["qid"] for q in queries], "emb": out}, DATA / "colpali_query_emb.pt")
    print(f"{len(out)} query embeddings, mean {sum(e.shape[0] for e in out) / len(out):.1f} tokens")


if __name__ == "__main__":
    main()
