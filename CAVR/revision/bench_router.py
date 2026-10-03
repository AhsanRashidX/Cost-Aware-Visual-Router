"""Step 8a: router cost - mpnet embedding + CAVR MLP - latency at batch 1 and throughput at batch 1..1024.

Run on an idle GPU. Writes results/router_cost.json; evaluate.py adds per_query_batch1 to every
learned policy's cost.
"""
import json
import time

import numpy as np
import pynvml
import torch

from common import DATA, RESULTS, hardware_info, set_seed, write_json
from router import CAVR

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
REPEATS_B1 = 500


def main():
    set_seed()
    from sentence_transformers import SentenceTransformer
    queries = [json.loads(l)["question"] for l in open(DATA / "queries.jsonl", encoding="utf-8")]
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    out = {"hardware": hardware_info()}

    for device in ("cuda", "cpu"):
        enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
        if device == "cuda":
            enc.half()
        mlp = CAVR(770).to(device).eval()  # weights don't affect timing
        onehot = torch.zeros(1, 2, device=device)

        def route(batch):
            e = enc.encode(batch, batch_size=len(batch), convert_to_tensor=True, normalize_embeddings=True)
            with torch.inference_mode():
                x = torch.cat([e.float(), onehot.expand(len(batch), 2)], 1)
                mlp(x)
            if device == "cuda":
                torch.cuda.synchronize()

        for _ in range(20):
            route(queries[:8])
        # batch-1 latency over many distinct queries
        n1 = REPEATS_B1 if device == "cuda" else 100
        lat = []
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
        for q in queries[:n1]:
            t0 = time.perf_counter()
            route([q])
            lat.append(time.perf_counter() - t0)
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
        res = {"batch1_p50_s": float(np.median(lat)), "batch1_p95_s": float(np.percentile(lat, 95)),
               "batch1_energy_j_per_query": (e1 - e0) / 1000 / n1 if device == "cuda" else None,
               "throughput": {}}
        for b in BATCHES if device == "cuda" else BATCHES[:8]:
            batch = (queries * (b // len(queries) + 1))[:b]
            reps = max(3, 2048 // b)
            t0 = time.perf_counter()
            for _ in range(reps):
                route(batch)
            dt = (time.perf_counter() - t0) / reps
            res["throughput"][str(b)] = {"latency_s": dt, "queries_per_s": b / dt}
        out[device] = res
        print(device, {k: v for k, v in res.items() if k != "throughput"},
              {b: round(v["queries_per_s"]) for b, v in res["throughput"].items()}, flush=True)
        del enc, mlp
        torch.cuda.empty_cache()

    g = out["cuda"]
    out["per_query_batch1"] = {"gpu_s": g["batch1_p50_s"], "latency_s": g["batch1_p50_s"],
                               "energy_j": g["batch1_energy_j_per_query"]}
    write_json(RESULTS / "router_cost.json", out)


if __name__ == "__main__":
    main()
