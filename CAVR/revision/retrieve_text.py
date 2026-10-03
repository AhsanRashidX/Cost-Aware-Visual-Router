"""Step 3a: BM25 text retrieval over docTR OCR text, one corpus per dataset.

Outputs: results/retrieval_text.jsonl (top-K pages per query, gold rank, CPU time),
         results/index_cost_text.json (index build time, recall@k).
"""
import json
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi

from common import DATA, RESULTS, hardware_info, write_json

TOP_K = 10
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(s):
    return TOKEN.findall(s.lower())


def main():
    ocr = [json.loads(l) for l in open(DATA / "ocr.jsonl", encoding="utf-8")]
    queries = [json.loads(l) for l in open(DATA / "queries.jsonl", encoding="utf-8")]

    index, build = {}, {}
    for ds in ("docvqa", "infovqa"):
        docs = [o for o in ocr if o["dataset"] == ds]
        t0 = time.perf_counter()
        bm25 = BM25Okapi([tokenize(o["text"]) or ["<empty>"] for o in docs])
        build[ds] = {"pages": len(docs), "seconds": time.perf_counter() - t0}
        index[ds] = (bm25, [o["page_id"] for o in docs])

    ranks = []
    with open(RESULTS / "retrieval_text.jsonl", "w", encoding="utf-8") as out:
        for q in queries:
            bm25, page_ids = index[q["dataset"]]
            t0 = time.perf_counter()
            scores = bm25.get_scores(tokenize(q["question"]))
            order = np.argsort(-scores)
            dt = time.perf_counter() - t0
            ranked = [page_ids[i] for i in order]
            gold_rank = ranked.index(q["page_id"]) + 1
            ranks.append(gold_rank)
            out.write(json.dumps({"qid": q["qid"], "top": ranked[:TOP_K],
                                  "scores": [float(scores[i]) for i in order[:TOP_K]],
                                  "gold_rank": gold_rank, "cpu_seconds": dt}) + "\n")

    ranks = np.array(ranks)
    write_json(RESULTS / "index_cost_text.json", {
        "method": "BM25Okapi (rank-bm25 defaults) over docTR text, per-dataset corpus",
        "build": build,
        "recall": {f"@{k}": float((ranks <= k).mean()) for k in (1, 3, 5, 10)},
        "mrr": float((1 / ranks).mean()),
        "hardware": hardware_info(),
    })
    print({f"R@{k}": round(float((ranks <= k).mean()), 3) for k in (1, 3, 5, 10)})


if __name__ == "__main__":
    main()
