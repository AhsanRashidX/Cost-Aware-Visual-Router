"""Step 8b: ColPali index scaling - size, build throughput and query latency at 1k..100k pages.

Larger corpora are built by tiling the 1,786 real page embeddings (with small noise); MaxSim cost
depends only on corpus size, not content. Configurations:
  gpu_bf16       index resident on the GPU (as long as it fits)
  cpu_stream     index in pinned host RAM, streamed to the GPU in chunks (PCIe-bound)
  gpu_pooled2x2  2x2 spatial pooling of patch vectors (4x smaller); recall impact measured on the real corpus
  sharded        index split into shards that each fit one GPU; shards are timed one after another on this
                 GPU, and the parallel latency on S GPUs is reported as max shard latency (emulated, 1 GPU here)
Output: results/index_scaling.json
"""
import json
import math
import time

import numpy as np
import pynvml
import torch
from PIL import Image

from common import DATA, RESULTS, hardware_info, set_seed, write_json

SIZES = [1_000, 10_000, 50_000, 100_000]
QUERY_BATCHES = [1, 4, 16, 64]
CHUNK = 4096           # pages per scoring chunk
N_LAT = 50             # batch-1 latency samples
GPU_BUDGET_GB = 18.0   # index memory allowed on the 24 GB card (rest: activations, context, desktop)
N_PATCH, GRID = 1024, 32


def maxsim_chunked(qs, index, device="cuda"):
    """qs: (B, Lq, d) on GPU; index: (N, Lp, d) on GPU or pinned CPU. Returns (B, N) scores."""
    out = []
    step = max(32, CHUNK // qs.shape[0])  # keep the (B, chunk, Lq, Lp) intermediate ~200 MB
    for i in range(0, index.shape[0], step):
        chunk = index[i:i + step].to(device, non_blocking=True)
        out.append(torch.einsum("bqd,npd->bnqp", qs, chunk).amax(3).sum(2))
    return torch.cat(out, 1)


def pool2x2(index, step=2048):
    """Mean-pool the 32x32 patch grid 2x2 (1024 -> 256 vectors); keep the trailing prompt tokens.
    Done in chunks so a 100k-page index never needs a float32 copy in RAM."""
    n, lp, d = index.shape
    out = torch.empty((n, N_PATCH // 4 + lp - N_PATCH, d), dtype=index.dtype)
    for i in range(0, n, step):
        x = index[i:i + step]
        g = x[:, :N_PATCH].reshape(-1, GRID // 2, 2, GRID // 2, 2, d).float().mean((2, 4)).reshape(len(x), -1, d)
        out[i:i + step, :N_PATCH // 4] = torch.nn.functional.normalize(g, dim=-1).to(index.dtype)
        out[i:i + step, N_PATCH // 4:] = x[:, N_PATCH:]
    return out


def tile(index, n, pin=False):
    """n-page corpus made of repeated real pages, allocated once (optionally pinned) and filled in place.
    Repeats score exactly like distinct pages, so latency/memory are representative."""
    try:
        big = torch.empty((n,) + tuple(index.shape[1:]), dtype=index.dtype, pin_memory=pin)
    except RuntimeError:  # Windows can refuse very large pinned allocations
        big = torch.empty((n,) + tuple(index.shape[1:]), dtype=index.dtype)
    for i in range(0, n, index.shape[0]):
        big[i:i + index.shape[0]] = index[:min(index.shape[0], n - i)]
    return big


def time_queries(qs_list, index, batches):
    torch.cuda.synchronize()
    lat = []
    for q in qs_list[:N_LAT]:
        t0 = time.perf_counter()
        s = maxsim_chunked(q[None].cuda(), index)
        torch.topk(s, 10, dim=1)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
    thr = {}
    lq = max(q.shape[0] for q in qs_list)
    for b in batches:
        qb = torch.stack([torch.nn.functional.pad(q, (0, 0, 0, lq - q.shape[0])) for q in qs_list[:b]]).cuda()
        t0 = time.perf_counter()
        reps = max(2, 16 // b)
        for _ in range(reps):
            torch.topk(maxsim_chunked(qb, index), 10, dim=1)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        thr[str(b)] = {"batch_latency_s": dt, "queries_per_s": b / dt}
    return {"p50_s": float(np.median(lat)), "p95_s": float(np.percentile(lat, 95)), "throughput": thr}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--merge", action="store_true",
                    help="reuse build/recall from an existing index_scaling.json and add these sizes to it")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    set_seed()
    from retrieve_visual import load_model
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    pages = [json.loads(l) for l in open(DATA / "pages.jsonl", encoding="utf-8")]
    queries = [json.loads(l) for l in open(DATA / "queries.jsonl", encoding="utf-8")]
    saved = torch.load(DATA / "colpali_index.pt")
    real, keys = saved["index"], saved["keys"]
    out = {"hardware": hardware_info(), "vectors_per_page": real.shape[1], "dim": real.shape[2]}
    prev = json.load(open(RESULTS / "index_scaling.json")) if args.merge else None

    # 1) Page-encoding throughput at batch sizes that stay inside VRAM, and all query embeddings.
    model, proc = load_model()
    build = {}
    for b in (() if prev else (1, 2, 4)):
        imgs = [Image.open(DATA / p["path"]).convert("RGB") for p in pages[:64]]
        torch.cuda.reset_peak_memory_stats()
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for i in range(0, 64, b):
                model(**proc.process_images(imgs[i:i + b]).to(model.device))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        build[str(b)] = {"pages_per_s": 64 / dt, "energy_j_per_page": (pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu) - e0) / 1000 / 64,
                         "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    if prev:
        out["build"] = prev["build"]
    else:
      best_b = max(build, key=lambda k: build[k]["pages_per_s"] if build[k]["peak_mem_gb"] < 22 else 0)
      out["build"] = {"by_batch": build, "best_batch": int(best_b),
                      "extrapolated_build_hours": {str(n): n / build[best_b]["pages_per_s"] / 3600 for n in SIZES}}
      print("build", {k: round(v["pages_per_s"], 1) for k, v in build.items()}, flush=True)

    qemb = []
    with torch.inference_mode():
        for i in range(0, len(queries), 64):
            e = model(**proc.process_queries([q["question"] for q in queries[i:i + 64]]).to(model.device))
            mask = proc.process_queries([q["question"] for q in queries[i:i + 64]])["attention_mask"].bool()
            qemb += [e[j][mask[j]].cpu() for j in range(len(e))]
    del model
    torch.cuda.empty_cache()

    # 2) Recall of the pooled index on the real corpus (per-dataset corpora, as in retrieve_visual.py).
    def recall_at1(index):
        hits = 0
        for ds in ("docvqa", "infovqa"):
            rows = [i for i, k in enumerate(keys) if k[0] == ds]
            corpus = index[rows].cuda()
            pid = [keys[i][1] for i in rows]
            for q, qe in zip(queries, qemb):
                if q["dataset"] == ds:
                    s = maxsim_chunked(qe[None].cuda(), corpus)[0]
                    hits += pid[int(s.argmax())] == q["page_id"]
            del corpus
        return hits / len(queries)

    if prev:
        out["pooling_recall_at1"] = prev["pooling_recall_at1"]
    else:
        pooled_real = pool2x2(real)
        out["pooling_recall_at1"] = {"full": recall_at1(real), "pooled2x2": recall_at1(pooled_real),
                                     "vectors_per_page_pooled": pooled_real.shape[1]}
    print("recall", out["pooling_recall_at1"], flush=True)

    # 3) Scaling.
    qs = qemb[:max(N_LAT, max(QUERY_BATCHES))]
    scaling = dict(prev["scaling"]) if prev else {}
    for n in sizes:
        res = {}
        full = tile(real, n, pin=n >= 10_000)
        gb = full.numel() * 2 / 2**30
        res["index_gb_bf16"] = gb
        res["host_index_pinned"] = bool(full.is_pinned())
        if gb <= GPU_BUDGET_GB:
            g = full.cuda()
            res["gpu_bf16"] = time_queries(qs, g, QUERY_BATCHES)
            del g
            torch.cuda.empty_cache()
        if n >= 10_000:
            res["cpu_stream"] = time_queries(qs, full, QUERY_BATCHES[:3])  # already pinned
        pooled = pool2x2(full)
        res["index_gb_pooled2x2"] = pooled.numel() * 2 / 2**30
        if res["index_gb_pooled2x2"] <= GPU_BUDGET_GB:
            g = pooled.cuda()
            res["gpu_pooled2x2"] = time_queries(qs, g, QUERY_BATCHES)
            del g
        del pooled
        torch.cuda.empty_cache()
        n_shards = math.ceil(gb / GPU_BUDGET_GB)
        if n_shards > 1:
            shard_lat = []
            for s in range(n_shards):
                g = full[s * math.ceil(n / n_shards):(s + 1) * math.ceil(n / n_shards)].cuda()
                shard_lat.append(time_queries(qs, g, [1])["p50_s"])
                del g
                torch.cuda.empty_cache()
            res["sharded"] = {"gpus_needed": n_shards, "per_shard_p50_s": shard_lat,
                              "emulated_parallel_p50_s": max(shard_lat)}
        scaling[str(n)] = res
        del full
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()  # release cached pinned host memory before the next size
        print(n, {k: (v["p50_s"] if isinstance(v, dict) and "p50_s" in v else v) for k, v in res.items()}, flush=True)
        out["scaling"] = dict(sorted(scaling.items(), key=lambda kv: int(kv[0])))
        write_json(RESULTS / "index_scaling.json", out)  # save after every size


if __name__ == "__main__":
    main()
