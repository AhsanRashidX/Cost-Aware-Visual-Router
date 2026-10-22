"""ColPali index too large for GPU memory (up to 1M pages, 264 GB) held in host RAM and streamed to the GPUs.

Measured on the 2x A10 node (1 TB RAM). The index is one host tensor of repeated real pages (page j is real page j mod
N, as in p40/shard_bench.py), page-locked with cudaHostRegister in blocks: Windows refuses pinned allocations above
about 75 GiB per process, but registering ordinary memory works and streams at full PCIe speed. An n-page corpus is the
first n pages; with g GPUs it is split into g contiguous shards. For every query batch each GPU streams its whole shard
over PCIe in chunks into two device buffers (the copy of one chunk overlaps scoring of the other) and keeps its top-k;
the host merges them, so the measured latency includes the merge. Scoring is exhaustive MaxSim in fp16, as in the
GPU-resident runs on this node (results/a10/a10_shard_scaling.json).

Also measured: the PCIe copy rate alone (one GPU, and both GPUs at once), and, on a 100k-page corpus held GPU-resident
across all GPUs (shard_bench.search, whose sharded result matches one GPU), that streamed search returns the same top-1
page and score.

Usage:  python a10/host_stream_bench.py [--sizes 100000,250000,500000,1000000] [--gpus 1,2] [--batches 1,16,64]
Output: results/a10/a10_host_stream.json
"""
import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "p40")]
import shard_bench as sb  # noqa: E402
from common import DATA, RESULTS, hardware_info, write_json  # noqa: E402

CHUNK = 4068          # pages per streamed chunk: 1,073,691,648 bytes, just under 1 GiB
REG_PAGES = 16 * CHUNK  # pages per cudaHostRegister call (about 16 GiB)
REG_PORTABLE = 1      # cudaHostRegisterPortable: pinned for every GPU's context
N_CHECK_PAGES = 100000  # corpus for the streamed-vs-resident check: crosses registration blocks, fits two GPUs
N_CHECK_Q = 100


def score(q, c, step):
    """Exhaustive MaxSim of q (B, Lq, d) against pages c (n, Lp, d), both fp16 on one GPU; returns (B, n) fp32."""
    return torch.cat([torch.einsum("bqd,npd->bnqp", q, c[i:i + step]).amax(3).float().sum(2)
                      for i in range(0, c.shape[0], step)], 1)


class Streamer:
    """One GPU: two chunk buffers, a copy stream and a compute stream."""

    def __init__(self, d, page_shape):
        self.dev = torch.device(f"cuda:{d}")
        with torch.cuda.device(self.dev):
            self.bufs = [torch.empty((CHUNK,) + page_shape, dtype=torch.float16, device=self.dev) for _ in range(2)]
            self.copy_s, self.comp_s = torch.cuda.Stream(self.dev), torch.cuda.Stream(self.dev)
            self.free = [torch.cuda.Event() for _ in range(2)]
            self.ready = [torch.cuda.Event() for _ in range(2)]

    def topk(self, q_cpu, host, a, b, compute=True):
        """Stream pages [a, b) of host; return the top-k (scores, global page ids) of q over them, on the CPU."""
        with torch.cuda.device(self.dev):
            q = q_cpu.to(self.dev, torch.float16)
            torch.cuda.current_stream().synchronize()
            step = max(16, 4096 // q.shape[0])     # same scoring step as shard_bench.shard_topk
            vs, ids = [], []
            for e in self.free:
                e.record(self.comp_s)
            s, k = a, 0
            while s < b:
                # one copy may not span two cudaHostRegister blocks (cudaErrorInvalidValue), so cut at their edges
                n, j = min(CHUNK, b - s, REG_PAGES - s % REG_PAGES), k % 2
                buf = self.bufs[j][:n]
                self.copy_s.wait_event(self.free[j])
                with torch.cuda.stream(self.copy_s):
                    buf.copy_(host[s:s + n], non_blocking=True)
                    self.ready[j].record(self.copy_s)
                self.comp_s.wait_event(self.ready[j])
                with torch.cuda.stream(self.comp_s):
                    if compute:
                        v, i = torch.topk(score(q, buf, step), min(sb.TOPK, n), dim=1)
                        vs.append(v)
                        ids.append(i + s)
                    self.free[j].record(self.comp_s)
                s, k = s + n, k + 1
            self.copy_s.synchronize()
            self.comp_s.synchronize()
            if not compute:
                return None
            v, i = torch.cat(vs, 1), torch.cat(ids, 1)
            top = torch.topk(v, min(sb.TOPK, v.shape[1]), dim=1)
            return top.values.cpu(), torch.gather(i, 1, top.indices).cpu()


def search(qs_cpu, host, n, g, streamers, pool, compute=True):
    """Query an n-page corpus split over g GPUs; merge the per-GPU top-k on the host. Returns (ids, scores)."""
    lq = max(q.shape[0] for q in qs_cpu)
    qb = torch.stack([torch.nn.functional.pad(q, (0, 0, 0, lq - q.shape[0])) for q in qs_cpu])
    per = math.ceil(n / g)
    futs = [pool.submit(streamers[d].topk, qb, host, d * per, min(n, (d + 1) * per), compute) for d in range(g)]
    parts = [f.result() for f in futs]
    if not compute:
        return None
    v, i = torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1)
    top = torch.topk(v, min(sb.TOPK, v.shape[1]), dim=1)
    return torch.gather(i, 1, top.indices), top.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100000,250000,500000,1000000")
    ap.add_argument("--gpus", default=None, help="comma list of GPU counts (default: 1..all)")
    ap.add_argument("--batches", default="1,16,64")
    ap.add_argument("--n-lat", type=int, default=5)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default=str(RESULTS / "a10" / "a10_host_stream.json"))
    args = ap.parse_args()
    torch.manual_seed(1234)
    sb.COMPUTE_DTYPE = torch.float16          # resident reference scores in fp16 too
    n_dev = torch.cuda.device_count()
    gpu_counts = [int(x) for x in args.gpus.split(",")] if args.gpus else list(range(1, n_dev + 1))
    sizes, batches = [int(x) for x in args.sizes.split(",")], [int(x) for x in args.batches.split(",")]

    base = torch.load(DATA / "colpali_index.pt")["index"].to(torch.float16)
    qe = torch.load(DATA / "colpali_query_emb.pt")["emb"]
    qs = qe[:max(args.n_lat, max(batches))]       # first questions, as in the other index benchmarks
    page_shape, page_bytes = tuple(base.shape[1:]), base[0].numel() * 2
    out = {"hardware": hardware_info(), "devices": [torch.cuda.get_device_name(i) for i in range(n_dev)],
           "store_dtype": "fp16", "compute_dtype": "fp16", "vectors_per_page": page_shape[0], "topk": sb.TOPK,
           "chunk_pages": CHUNK, "chunk_bytes": CHUNK * page_bytes, "n_lat": args.n_lat, "reps": args.reps,
           "batches": batches, "runs": []}

    # host index: ordinary memory, filled with repeated real pages, then page-locked block by block
    nmax = max(sizes)
    t0 = time.perf_counter()
    host = torch.empty((nmax,) + page_shape, dtype=torch.float16)
    for i in range(0, nmax, base.shape[0]):
        host[i:i + base.shape[0]] = base[:min(base.shape[0], nmax - i)]
    out["host_index"] = {"pages": nmax, "gib": nmax * page_bytes / 2**30, "fill_s": time.perf_counter() - t0}
    cr = torch.cuda.cudart()
    t0, blocks = time.perf_counter(), []
    for s in range(0, nmax, REG_PAGES):
        n = min(REG_PAGES, nmax - s)
        r = cr.cudaHostRegister(host.data_ptr() + s * page_bytes, n * page_bytes, REG_PORTABLE)
        if int(r) != 0:
            raise SystemExit(f"cudaHostRegister failed at page {s}: {r}")
        blocks.append(s)
    out["host_index"].update({"registered": True, "register_s": time.perf_counter() - t0,
                              "register_blocks": len(blocks), "register_block_pages": REG_PAGES})
    print(f"host index: {nmax:,} pages, {out['host_index']['gib']:.1f} GiB, filled in "
          f"{out['host_index']['fill_s']:.0f} s, registered in {out['host_index']['register_s']:.0f} s", flush=True)

    streamers = [Streamer(d, page_shape) for d in range(max(gpu_counts))]
    sampler = sb.PowerSampler(n_dev)
    out["energy_mode"] = sampler.mode
    with ThreadPoolExecutor(max_workers=max(gpu_counts)) as pool:
        # correctness: streamed top-1 (page mod N and score) equals GPU-resident top-1
        cq = qe[:N_CHECK_Q]
        ref_sh = sb.build_shards(base, N_CHECK_PAGES, n_dev)
        ref = [sb.search(cq[i:i + 50], ref_sh, pool, 4096, scores=True) for i in range(0, N_CHECK_Q, 50)]
        ref_i, ref_s = torch.cat([r[0][:, 0] for r in ref]), torch.cat([r[1][:, 0] for r in ref])
        del ref_sh
        torch.cuda.empty_cache()
        out["checks"] = {}
        for g in gpu_counts:
            got = [search(cq[i:i + 50], host, N_CHECK_PAGES, g, streamers, pool) for i in range(0, N_CHECK_Q, 50)]
            got_i, got_s = torch.cat([r[0][:, 0] for r in got]), torch.cat([r[1][:, 0] for r in got])
            out["checks"][str(g)] = {
                "pages": N_CHECK_PAGES, "queries": N_CHECK_Q, "reference_gpus": n_dev,
                "same_page": float((got_i % base.shape[0] == ref_i % base.shape[0]).float().mean()),
                "same_score": float((got_s == ref_s).float().mean()),
                "max_abs_score_diff": float((got_s - ref_s).abs().max())}
        print("streamed vs resident top-1:", out["checks"], flush=True)

        # PCIe copy rate alone: every GPU streams its shard of the largest corpus, no scoring
        out["copy_only"] = {}
        for g in gpu_counts:
            search(qs[:1], host, nmax, g, streamers, pool, compute=False)
            t0 = time.perf_counter()
            search(qs[:1], host, nmax, g, streamers, pool, compute=False)
            dt = time.perf_counter() - t0
            out["copy_only"][str(g)] = {"pages": nmax, "seconds": dt, "gb_per_s_total": nmax * page_bytes / dt / 1e9,
                                        "gb_per_s_per_gpu": nmax * page_bytes / dt / 1e9 / g}
            print(f"copy only, {g} GPU(s): {out['copy_only'][str(g)]['gb_per_s_total']:.1f} GB/s total", flush=True)

        for n in sizes:
            for g in gpu_counts:
                per_gpu_bytes = math.ceil(n / g) * page_bytes
                search(qs[:1], host, n, g, streamers, pool)          # warm-up
                lat = []
                for q in qs[:args.n_lat]:
                    t0 = time.perf_counter()
                    search([q], host, n, g, streamers, pool)
                    lat.append(time.perf_counter() - t0)
                thr = {}
                for b in batches:
                    with sampler as e:
                        t0 = time.perf_counter()
                        for _ in range(args.reps):
                            search(qs[:b], host, n, g, streamers, pool)
                        dt = (time.perf_counter() - t0) / args.reps
                    thr[str(b)] = {"batch_latency_s": dt, "queries_per_s": b / dt,
                                   "stream_gb_per_s_per_gpu": per_gpu_bytes / dt / 1e9,
                                   "energy_j_per_query": (e.joules / (args.reps * b)) if sampler.mode else None}
                run = {"pages": n, "gpus": g, "index_gib_total": n * page_bytes / 2**30,
                       "index_gib_per_gpu": per_gpu_bytes / 2**30, "p50_s": float(np.median(lat)),
                       "p95_s": float(np.percentile(lat, 95)), "lat_s": lat, "throughput": thr}
                out["runs"].append(run)
                print(f"{n:>9,} pages on {g} GPU(s): p50 {run['p50_s']:.2f} s, "
                      + ", ".join(f"batch {b} {v['queries_per_s']:.2f} q/s" for b, v in thr.items()), flush=True)
                write_json(args.out, out)
    for s in blocks:
        cr.cudaHostUnregister(host.data_ptr() + s * page_bytes)
    write_json(args.out, out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
