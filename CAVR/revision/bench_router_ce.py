"""Phase 1d: measured cost of the content-aware router (cavr_ce) per query.

Batch-1 chain on the GPU: mpnet question embedding -> best OCR windows (dot products against the precomputed
window index) -> tokenization -> cross-encoder forward. ColPali query encoding is not included here: it is the
visual retrieval itself, charged separately to text-routed queries. Also throughput at larger batches.
Weights do not affect timing, so an untrained head is used. Output: results/router_ce_cost.json
"""
import json
import time

import numpy as np
import pynvml
import torch

import router_ce as R
from common import DATA, RESULTS, hardware_info, set_seed, write_json
from routing_data import load

N_B1 = 300
BATCHES = [1, 8, 32, 128, 256]


def main():
    set_seed()
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    qids, Q, A, C, rows = load()
    passages = R.build_passages(qids, Q)      # window selection is re-timed below with live embeddings
    z = np.load(DATA / "ocr_windows.npz", allow_pickle=True)
    wemb = torch.tensor(z["emb"]).cuda()
    spans = dict(z["spans"].item())
    rt = {r["qid"]: r for r in map(json.loads, open(RESULTS / "retrieval_text.jsonl", encoding="utf-8"))}
    rv = {r["qid"]: r for r in map(json.loads, open(RESULTS / "retrieval_visual.jsonl", encoding="utf-8"))}

    enc = SentenceTransformer(R.BACKBONE, device="cuda").half()
    tok = AutoTokenizer.from_pretrained(R.BACKBONE)
    model = R.CrossRouter(36).cuda().eval()   # 34 standardized tabular features + 2 dataset dims
    tab = torch.zeros(1, 36, device="cuda")
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

    def route_one(i):
        q = Q[i]
        qv = enc.encode([q["question"]], convert_to_tensor=True, normalize_embeddings=True)[0].float()
        ds = q["dataset"]
        for p in rt[qids[i]]["top"][:3] + rv[qids[i]]["top"][:1]:     # window scoring for the 4 pages used
            a, b = spans[(ds, p)]
            if b > a:
                torch.topk(wemb[a:b] @ qv, min(2, b - a))
        x = tok([q["question"]], [passages[i]], truncation="only_second", max_length=R.MAX_LEN, return_tensors="pt")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x["input_ids"].cuda(), x["attention_mask"].cuda(), tab)
        torch.cuda.synchronize()

    for i in range(20):
        route_one(i)
    lat = []
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
    for i in range(N_B1):
        t0 = time.perf_counter()
        route_one(i)
        lat.append(time.perf_counter() - t0)
    energy = (pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu) - e0) / 1000 / N_B1

    thr = {}
    for b in BATCHES:
        x = tok([q["question"] for q in Q[:b]], passages[:b], truncation="only_second", max_length=R.MAX_LEN,
                padding=True, return_tensors="pt")
        ids, mask, tb = x["input_ids"].cuda(), x["attention_mask"].cuda(), tab.expand(b, -1)
        reps = max(3, 512 // b)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(ids, mask, tb)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                model(ids, mask, tb)
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        thr[str(b)] = {"batch_latency_s": dt, "queries_per_s": b / dt}
    p50 = float(np.median(lat))
    out = {"per_query_batch1": {"gpu_s": p50, "latency_s": p50, "energy_j": energy},
           "batch1_p95_s": float(np.percentile(lat, 95)), "cross_encoder_throughput": thr,
           "note": "mpnet embedding + window scoring + tokenization + cross-encoder; ColPali charged separately",
           "hardware": hardware_info()}
    write_json(RESULTS / "router_ce_cost.json", out)
    print(f"batch-1 p50 {1000 * p50:.1f} ms, p95 {1000 * out['batch1_p95_s']:.1f} ms, {energy:.2f} J/query;",
          {b: round(v["queries_per_s"]) for b, v in thr.items()})


if __name__ == "__main__":
    main()
