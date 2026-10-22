"""Host-memory placement of a ColPali index too large for GPU memory: analysis of the 2x A10 run.

Inputs (raw outputs of a10/host_stream_bench.py and p40/shard_bench.py on the same node):
  results/a10/a10_host_stream.json, results/a10/a10_shard_scaling.json
Output: results/a10/host_stream_scaling.json
  per run          batch-1 latency, batch-64 throughput, PCIe rate achieved, energy per query
  bound            batch 1 is set by PCIe (latency vs copy-only time for the same bytes); large batches by scoring,
                   measured as GPU-seconds per query (GPUs x batch latency / batch). The crossover batch is the batch at
                   which scoring time equals the streaming time of one pass.
  vs resident      the same corpus held GPU-resident (100k pages on two A10s, measured): throughput and energy ratio
  1M resident      what a GPU-resident 1M-page index would need: GPUs to hold it within the same per-GPU budget, and
                   its batch-1 latency from a linear fit of resident p50 against pages per GPU (projected, not measured)
"""
import json
import math

import numpy as np

from common import RESULTS, write_json

SRC = RESULTS / "a10"
B_BIG = "64"     # batch for throughput and energy (as in the sharding runs)


def main():
    hs = json.load(open(SRC / "a10_host_stream.json", encoding="utf-8"))
    rs = json.load(open(SRC / "a10_shard_scaling.json", encoding="utf-8"))
    page_bytes = hs["vectors_per_page"] * 128 * 2
    copy = hs["copy_only"]
    runs = []
    for r in hs["runs"]:
        g, t = r["gpus"], r["throughput"]
        per_gpu = r["index_gib_per_gpu"] * 2**30
        t_copy = per_gpu / (copy[str(g)]["gb_per_s_per_gpu"] * 1e9)
        b = int(B_BIG)
        t_q = t[B_BIG]["batch_latency_s"] / b            # wall time per query at large batch
        runs.append({"pages": r["pages"], "gpus": g, "index_gib_total": r["index_gib_total"],
                     "p50_s": r["p50_s"], "p95_s": r["p95_s"], "copy_only_s": t_copy,
                     "p50_over_copy": r["p50_s"] / t_copy,
                     "stream_gb_per_s_per_gpu_b1": t["1"]["stream_gb_per_s_per_gpu"],
                     "qps": {k: v["queries_per_s"] for k, v in t.items()},
                     "energy_j_per_query": {k: v["energy_j_per_query"] for k, v in t.items()},
                     "stream_gb_per_s_per_gpu_big": t[B_BIG]["stream_gb_per_s_per_gpu"],
                     "gpu_s_per_query_big": g * t_q, "gpu_s_per_query_per_100k": g * t_q * 1e5 / r["pages"],
                     "crossover_batch": r["p50_s"] / t_q})
    R = {(r["pages"], r["gpus"]): r for r in runs}
    sizes = sorted({r["pages"] for r in runs})
    two = max(r["gpus"] for r in runs)
    speedup = {str(n): {"latency": R[(n, 1)]["p50_s"] / R[(n, two)]["p50_s"],
                        "throughput_big": R[(n, two)]["qps"][B_BIG] / R[(n, 1)]["qps"][B_BIG]}
               for n in sizes if (n, 1) in R and (n, two) in R}

    res = {(r["pages"], r["gpus"]): r for r in rs["runs"]}
    n_ref = max(n for n, g in res if g == two and (n, two) in R) if any((n, two) in R for n, g in res) else None
    vs_res = None
    if n_ref:
        a, s = res[(n_ref, two)], R[(n_ref, two)]
        vs_res = {"pages": n_ref, "gpus": two, "resident_p50_s": a["p50_s"], "streamed_p50_s": s["p50_s"],
                  "resident_qps_big": a["throughput"][B_BIG]["queries_per_s"], "streamed_qps_big": s["qps"][B_BIG],
                  "throughput_ratio": s["qps"][B_BIG] / a["throughput"][B_BIG]["queries_per_s"],
                  "resident_j_per_query_big": a["throughput"][B_BIG]["energy_j_per_query"],
                  "streamed_j_per_query_big": s["energy_j_per_query"][B_BIG]}

    # GPU-resident 1M-page index: GPUs needed within the per-GPU budget, latency projected from the resident runs
    nmax = max(sizes)
    pts = [(math.ceil(n / g), r["p50_s"]) for (n, g), r in res.items() if n >= 10000]
    slope, icpt = np.polyfit([p for p, _ in pts], [t for _, t in pts], 1)
    g_need = math.ceil(nmax * page_bytes / 2**30 / rs["budget_gib_per_gpu"])
    gpu_s_res = two / res[(n_ref, two)]["throughput"][B_BIG]["queries_per_s"] * nmax / n_ref
    resident_1m = {"pages": nmax, "gpus_needed": g_need, "budget_gib_per_gpu": rs["budget_gib_per_gpu"],
                   "pages_per_gpu": math.ceil(nmax / g_need),
                   "projected_p50_s": float(slope * math.ceil(nmax / g_need) + icpt),
                   "fit": {"slope_s_per_page": float(slope), "intercept_s": float(icpt), "points": len(pts)},
                   "gpu_s_per_query_big": gpu_s_res,
                   "projected_qps_big": g_need / gpu_s_res}
    s1m = R[(nmax, two)]
    out = {"source": {"devices": hs["devices"], "compute_dtype": hs["compute_dtype"], "chunk_pages": hs["chunk_pages"],
                      "n_lat": hs["n_lat"], "reps": hs["reps"], "batches": hs["batches"], "big_batch": int(B_BIG),
                      "host_index": hs["host_index"], "checks": hs["checks"], "energy_mode": hs["energy_mode"]},
           "copy_only": copy, "pcie_two_gpu_share": copy[str(two)]["gb_per_s_total"] / (two * copy["1"]["gb_per_s_total"]),
           "runs": runs, "speedup": speedup, "vs_resident": vs_res, "resident_1m": resident_1m,
           "summary": {"max_pages": nmax, "max_gib": hs["host_index"]["gib"],
                       "p50_1m_by_gpus": {str(g): R[(nmax, g)]["p50_s"] for g in (1, two) if (nmax, g) in R},
                       "qps_big_1m_by_gpus": {str(g): R[(nmax, g)]["qps"][B_BIG] for g in (1, two) if (nmax, g) in R},
                       "gpu_s_per_query_1m": s1m["gpu_s_per_query_big"],
                       "max_p50_over_copy": max(r["p50_over_copy"] for r in runs),
                       "min_check_same": min(min(c["same_page"], c["same_score"]) for c in hs["checks"].values())}}
    write_json(SRC / "host_stream_scaling.json", out)
    for r in runs:
        print(f"{r['pages']:>9,} pages, {r['gpus']} GPU: p50 {r['p50_s']:.2f} s ({r['p50_over_copy']:.2f}x copy time), "
              f"batch {B_BIG} {r['qps'][B_BIG]:.2f} q/s, {r['gpu_s_per_query_big']:.3f} GPU-s/query, "
              f"crossover batch {r['crossover_batch']:.0f}")
    print("speedup 1->2 GPUs:", {k: {a: round(b, 2) for a, b in v.items()} for k, v in speedup.items()})
    print("vs resident:", vs_res)
    print("resident 1M:", resident_1m)
    print("PCIe two-GPU share:", round(out["pcie_two_gpu_share"], 3))
    print(f"wrote {SRC / 'host_stream_scaling.json'}")


if __name__ == "__main__":
    main()
