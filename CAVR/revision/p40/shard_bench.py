"""Measured multi-GPU sharding of the ColPali index (replaces the emulated sharding of the single-GPU study).

Standalone: needs torch, numpy, psutil, nvidia-ml-py, and two files exported on the main workstation:
  colpali_index.pt       page embeddings of the 1,786 evaluation pages (1,031 x 128 per page)
  colpali_query_emb.pt   per-token query embeddings of the 8,150 questions
Larger corpora are built by repeating the real pages (as in the single-GPU study). The index is stored in fp16 on
each GPU and scored chunk by chunk in fp16 on GPUs with fast fp16 (compute capability >= 7.0, e.g. T4) or in fp32 on
older GPUs (e.g. Pascal P40, whose fp16 arithmetic is very slow). A query is sent to every shard at once (one thread per
GPU); each shard returns its top-k, and the host merges them, so the measured latency includes the merge.
Before timing, sharded top-1 results on the real corpus are checked against a single GPU.

Usage:  python shard_bench.py --data <folder with the two .pt files> [--gpus 1,2,3] [--sizes 1000,...]
Output: p40_shard_scaling.json next to this script.
"""
import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

TOPK = 10


class PowerSampler:
    """Energy per GPU: the NVML energy counter if supported, else integrated power samples (every 20 ms)."""

    def __init__(self, n):
        self.n, self.handles, self.mode = n, [], None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handles[0])
                self.mode = "counter"
            except Exception:
                self.mode = "power_sampling"
        except Exception:
            self.mode = None

    def __enter__(self):
        self.joules = 0.0
        if self.mode == "counter":
            self.e0 = [self.nv.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles]
        elif self.mode == "power_sampling":
            self.stop = threading.Event()
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()
        return self

    def _sample(self):
        last = time.perf_counter()
        while not self.stop.is_set():
            time.sleep(0.02)
            now = time.perf_counter()
            watts = sum(self.nv.nvmlDeviceGetPowerUsage(h) for h in self.handles) / 1000
            self.joules += watts * (now - last)
            last = now

    def __exit__(self, *exc):
        if self.mode == "counter":
            self.joules = sum(self.nv.nvmlDeviceGetTotalEnergyConsumption(h) - e
                              for h, e in zip(self.handles, self.e0)) / 1000
        elif self.mode == "power_sampling":
            self.stop.set()
            self.thread.join()


SAME_DEVICE = False   # test only: put every shard on cuda:0 to check the merge logic on a single-GPU machine


def build_shards(base, n_pages, n_gpus):
    """Split an n_pages corpus of repeated real pages into n_gpus contiguous fp16 shards, built on each GPU."""
    per = math.ceil(n_pages / n_gpus)
    shards, offset = [], 0
    for g in range(n_gpus):
        n = min(per, n_pages - offset)
        dev = torch.device("cuda:0" if SAME_DEVICE else f"cuda:{g}")
        b = base.to(dev)
        # corpus page j is real page j mod N, whatever the number of shards (shard g covers [offset, offset + n))
        idx = torch.arange(offset, offset + n, device=dev) % b.shape[0]
        s = b.index_select(0, idx)
        del b, idx
        shards.append((s, offset))
        offset += n
    return shards


COMPUTE_DTYPE = torch.float32


def shard_topk(q, shard, offset, chunk):
    """q: (B, Lq, d) on the shard's GPU; returns (scores, global page ids) of the shard's top-k, on the CPU."""
    s, outs = shard, []
    step = max(16, chunk // q.shape[0])
    with torch.cuda.device(s.device):
        q = q.to(COMPUTE_DTYPE)
        for i in range(0, s.shape[0], step):
            c = s[i:i + step].to(COMPUTE_DTYPE)
            outs.append(torch.einsum("bqd,npd->bnqp", q, c).amax(3).float().sum(2))
        scores = torch.cat(outs, 1)
        k = min(TOPK, scores.shape[1])
        v, idx = torch.topk(scores, k, dim=1)
        return v.cpu(), (idx + offset).cpu()


def search(qs_cpu, shards, pool, chunk, scores=False):
    """Query all shards in parallel and merge their top-k on the host (optionally also return the scores)."""
    lq = max(q.shape[0] for q in qs_cpu)
    qb = torch.stack([torch.nn.functional.pad(q.float(), (0, 0, 0, lq - q.shape[0])) for q in qs_cpu])
    futs = [pool.submit(shard_topk, qb.to(s.device, non_blocking=True), s, off, chunk) for s, off in shards]
    parts = [f.result() for f in futs]
    v = torch.cat([p[0] for p in parts], 1)
    i = torch.cat([p[1] for p in parts], 1)
    top = torch.topk(v, min(TOPK, v.shape[1]), dim=1)
    ids = torch.gather(i, 1, top.indices)
    return (ids, top.values) if scores else ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gpus", default=None, help="comma list of GPU counts (default: 1..all)")
    ap.add_argument("--sizes", default="1000,10000,50000,100000,150000,200000")
    ap.add_argument("--n-lat", type=int, default=50)
    ap.add_argument("--batches", default="1,4,16,64")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--budget-gib", type=float, default=None,
                    help="max index GiB per GPU (default: 70%% of the smallest GPU's memory)")
    ap.add_argument("--compute-dtype", choices=["auto", "fp16", "fp32"], default="auto",
                    help="auto: fp16 if every GPU has compute capability >= 7.0, else fp32")
    ap.add_argument("--out", default=str(Path(__file__).with_name("p40_shard_scaling.json")))
    ap.add_argument("--same-device", action="store_true", help="test only: all shards on cuda:0")
    args = ap.parse_args()
    global SAME_DEVICE, COMPUTE_DTYPE
    SAME_DEVICE = args.same_device
    torch.manual_seed(1234)
    n_dev = torch.cuda.device_count()
    gpu_counts = [int(x) for x in args.gpus.split(",")] if args.gpus else list(range(1, n_dev + 1))
    if not SAME_DEVICE and max(gpu_counts) > n_dev:
        raise SystemExit(f"asked for {max(gpu_counts)} GPUs but only {n_dev} are visible")
    sizes = [int(x) for x in args.sizes.split(",")]
    caps = [torch.cuda.get_device_capability(i) for i in range(n_dev)]
    if args.compute_dtype == "auto":
        COMPUTE_DTYPE = torch.float16 if all(c >= (7, 0) for c in caps) else torch.float32
    else:
        COMPUTE_DTYPE = torch.float16 if args.compute_dtype == "fp16" else torch.float32
    if args.budget_gib is None:
        args.budget_gib = 0.7 * min(torch.cuda.get_device_properties(i).total_memory for i in range(n_dev)) / 2**30
    batches = [int(x) for x in args.batches.split(",")]

    saved = torch.load(Path(args.data) / "colpali_index.pt")
    base = saved["index"].to(torch.float16)
    qe = torch.load(Path(args.data) / "colpali_query_emb.pt")["emb"]
    qs = qe[:max(args.n_lat, max(batches))]          # same query set as the single-GPU study (first questions)
    page_gib = base[0].numel() * 2 / 2**30

    out = {"torch": torch.__version__, "devices": [torch.cuda.get_device_name(i) for i in range(n_dev)],
           "capability": [".".join(map(str, torch.cuda.get_device_capability(i))) for i in range(n_dev)],
           "store_dtype": "fp16", "compute_dtype": str(COMPUTE_DTYPE).replace("torch.", ""),
           "vectors_per_page": base.shape[1], "topk": TOPK,
           "n_lat": args.n_lat, "chunk": args.chunk, "budget_gib_per_gpu": args.budget_gib, "checks": {}, "runs": []}
    sampler = PowerSampler(n_dev)
    out["energy_mode"] = sampler.mode

    # correctness: sharded top-1 on the real corpus equals single-GPU top-1
    ref_shards = build_shards(base, base.shape[0], 1)
    with ThreadPoolExecutor(max_workers=max(gpu_counts)) as pool:
        check_q = qe[:500]

        def top1(sh):
            res = [search(check_q[i:i + 50], sh, pool, args.chunk, scores=True) for i in range(0, 500, 50)]
            return torch.cat([r[0][:, 0] for r in res]), torch.cat([r[1][:, 0] for r in res])
        ref, ref_s = top1(ref_shards)
        del ref_shards
        for g in gpu_counts:
            if g == 1:
                continue
            sh = build_shards(base, base.shape[0], g)
            got, got_s = top1(sh)
            same = got == ref
            tie = ~same & torch.isclose(got_s, ref_s, rtol=1e-6, atol=1e-5)   # different page, identical score
            out["checks"][str(g)] = {"exact": float(same.float().mean()), "exact_or_tie": float((same | tie).float().mean())}
            del sh
            torch.cuda.empty_cache()
        print("top-1 agreement with 1 GPU:", out["checks"], flush=True)

        for n in sizes:
            for g in gpu_counts:
                shard_gib = math.ceil(n / g) * page_gib
                if shard_gib > args.budget_gib:
                    print(f"{n:>7} pages on {g} GPU(s): {shard_gib:.1f} GiB per GPU exceeds budget, skipped", flush=True)
                    continue
                shards = build_shards(base, n, g)
                for d in range(1 if SAME_DEVICE else g):
                    torch.cuda.synchronize(d)
                search(qs[:2], shards, pool, args.chunk)            # warm-up
                lat = []
                for q in qs[:args.n_lat]:
                    t0 = time.perf_counter()
                    search([q], shards, pool, args.chunk)
                    lat.append(time.perf_counter() - t0)
                thr = {}
                for b in batches:
                    reps = max(3, 32 // b)
                    search(qs[:b], shards, pool, args.chunk)
                    with sampler as e:
                        t0 = time.perf_counter()
                        for _ in range(reps):
                            search(qs[:b], shards, pool, args.chunk)
                        dt = (time.perf_counter() - t0) / reps
                    thr[str(b)] = {"batch_latency_s": dt, "queries_per_s": b / dt,
                                   "energy_j_per_query": (e.joules / (reps * b)) if sampler.mode else None}
                run = {"pages": n, "gpus": g, "index_gib_total": n * page_gib, "index_gib_per_gpu": shard_gib,
                       "p50_s": float(np.median(lat)), "p95_s": float(np.percentile(lat, 95)), "throughput": thr}
                out["runs"].append(run)
                print(f"{n:>7} pages on {g} GPU(s): p50 {1000 * run['p50_s']:.1f} ms, p95 {1000 * run['p95_s']:.1f} ms, "
                      f"batch-64 {thr[str(batches[-1])]['queries_per_s']:.0f} q/s", flush=True)
                Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
                del shards
                torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
