"""Step 3b: ColPali v1.3 visual retrieval over page images, one corpus per dataset.

Offline: encode every page (timed, energy-metered) -> data/colpali_index.pt
Online:  per query, encode the question and MaxSim-score against its dataset's pages (batch 1).
Outputs: results/retrieval_visual.jsonl, results/index_cost_visual.json
"""
import argparse
import json
import time

import numpy as np
import pynvml
import torch
from PIL import Image

from common import DATA, RESULTS, hardware_info, set_seed, write_json

MODEL = "vidore/colpali-v1.3"
TOP_K = 10
PAGE_BATCH = 8


def maxsim(q, pages):
    """q: (Lq, d); pages: (N, Lp, d) -> (N,) late-interaction scores."""
    return torch.einsum("qd,npd->nqp", q, pages).amax(dim=2).sum(dim=1)


def load_model():
    from colpali_engine.models import ColPali, ColPaliProcessor
    model = ColPali.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    processor = ColPaliProcessor.from_pretrained(MODEL)
    return model, processor


def build_index(model, processor, pages, gpu):
    embs, per_batch = [], []
    torch.cuda.reset_peak_memory_stats()
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
    t_all = time.perf_counter()
    for i in range(0, len(pages), PAGE_BATCH):
        batch = pages[i:i + PAGE_BATCH]
        t_load = time.perf_counter()
        imgs = [Image.open(DATA / p["path"]).convert("RGB") for p in batch]
        inputs = processor.process_images(imgs).to(model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(**inputs)
        torch.cuda.synchronize()
        per_batch.append({"n": len(batch), "preprocess_s": t0 - t_load, "gpu_s": time.perf_counter() - t0})
        embs.append(out.cpu())
        if i % (PAGE_BATCH * 25) == 0:
            print(f"index {i}/{len(pages)}", flush=True)
    total = time.perf_counter() - t_all
    energy = (pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu) - e0) / 1000
    lens = {e.shape[1] for e in embs}
    assert len(lens) == 1, f"variable page embedding lengths: {lens}"
    index = torch.cat(embs)
    cost = {"pages": len(pages), "page_batch": PAGE_BATCH, "wall_s": total,
            "gpu_s": sum(b["gpu_s"] for b in per_batch),
            "preprocess_s": sum(b["preprocess_s"] for b in per_batch),
            "energy_j": energy, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
            "vectors_per_page": index.shape[1], "dim": index.shape[2],
            "index_bytes_bf16": index.numel() * 2}
    return index, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries (pilot)")
    args = ap.parse_args()
    set_seed()
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

    pages = [json.loads(l) for l in open(DATA / "pages.jsonl", encoding="utf-8")]
    queries = [json.loads(l) for l in open(DATA / "queries.jsonl", encoding="utf-8")]
    if args.limit:
        queries = queries[:args.limit]
    model, processor = load_model()

    index_path = DATA / "colpali_index.pt"
    if index_path.exists():
        saved = torch.load(index_path)
        index, page_keys, index_cost = saved["index"], saved["keys"], saved["cost"]
    else:
        index, index_cost = build_index(model, processor, pages, gpu)
        page_keys = [(p["dataset"], p["page_id"]) for p in pages]
        torch.save({"index": index, "keys": page_keys, "cost": index_cost}, index_path)

    # Per-dataset corpora resident on the GPU.
    corpora = {}
    for ds in ("docvqa", "infovqa"):
        rows = [i for i, k in enumerate(page_keys) if k[0] == ds]
        corpora[ds] = (index[rows].cuda(), [page_keys[i][1] for i in rows])

    # warm-up
    with torch.inference_mode():
        q = model(**processor.process_queries(["warm up"]).to(model.device))[0]
        maxsim(q, corpora["docvqa"][0])
    torch.cuda.synchronize()

    ranks, enc_s, score_s = [], [], []
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
    with open(RESULTS / "retrieval_visual.jsonl", "w", encoding="utf-8") as out, torch.inference_mode():
        for n, qr in enumerate(queries):
            corpus, page_ids = corpora[qr["dataset"]]
            ea = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
            t0 = time.perf_counter()
            inputs = processor.process_queries([qr["question"]]).to(model.device)
            q = model(**inputs)[0]
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            scores = maxsim(q, corpus).float()
            top = torch.topk(scores, k=len(page_ids))
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            eb = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
            ranked = [page_ids[i] for i in top.indices.tolist()]
            gold_rank = ranked.index(qr["page_id"]) + 1
            ranks.append(gold_rank)
            enc_s.append(t1 - t0)
            score_s.append(t2 - t1)
            out.write(json.dumps({"qid": qr["qid"], "top": ranked[:TOP_K],
                                  "scores": top.values[:TOP_K].tolist(), "gold_rank": gold_rank,
                                  "encode_s": t1 - t0, "score_s": t2 - t1,
                                  "energy_j": (eb - ea) / 1000}) + "\n")
            if n % 500 == 0:
                print(f"query {n}/{len(queries)}", flush=True)
    energy = (pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu) - e0) / 1000

    ranks = np.array(ranks)
    write_json(RESULTS / "index_cost_visual.json", {
        "model": MODEL, "index": index_cost,
        "query": {"n": len(queries), "encode_p50_s": float(np.median(enc_s)),
                  "score_p50_s": float(np.median(score_s)),
                  "total_s": float(sum(enc_s) + sum(score_s)), "energy_j": energy},
        "recall": {f"@{k}": float((ranks <= k).mean()) for k in (1, 3, 5, 10)},
        "mrr": float((1 / ranks).mean()),
        "hardware": hardware_info(),
    })
    print({f"R@{k}": round(float((ranks <= k).mean()), 3) for k in (1, 3, 5, 10)})


if __name__ == "__main__":
    main()
