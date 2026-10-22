"""Measured multi-GPU sharding of the ColPali index: scaling analysis of the run on the multi-GPU node.

Inputs, the raw outputs of p40/check_env.py and p40/shard_bench.py, copied unchanged from the node:
  results/multi_gpu/p40_env.json, results/multi_gpu/p40_shard_scaling.json
Output: results/multi_gpu_scaling.json
Other nodes (e.g. the 2x A10 server): --env results/a10/a10_env.json --scaling results/a10/a10_shard_scaling.json
  --out results/a10/multi_gpu_scaling.json
  strong scaling  per corpus size: speedup and parallel efficiency of p50 latency and of batch-64 throughput against
                  the fewest GPUs that hold the corpus, and the dispatch + merge overhead over ideal division
  weak scaling    runs with the same pages per GPU
  energy          the NVML counters of every GPU on the node are read, so a run on g of G GPUs includes the idle
                  draw of the G - g unused ones. If the active energy per query does not depend on g (the work is
                  split evenly), the idle power per GPU is (E_g - E_G) / ((G - g) t_g), with t_g the time per query;
                  it is estimated on corpora of at least MIN_PAGES pages and used to separate active from idle energy.
"""
import argparse
import json
import math

import numpy as np

from common import RESULTS, write_json

SRC = RESULTS / "multi_gpu"
N_CHECK = 500        # queries in the top-1 check (check_q = qe[:500] in p40/shard_bench.py)
THR_BATCH = "64"     # batch size for throughput and energy
MIN_PAGES = 10000    # smaller corpora are dominated by fixed per-query overheads


def node_hardware(env):
    head, *rows = [l.split(", ") for l in env["smi_gpus"].strip().splitlines()]
    smi = [dict(zip(head, r)) for r in rows]
    return {"gpus": [{**g, "power_limit_w": float(s["power.limit [W]"].split()[0]),
                      "pcie_width": int(s["pcie.link.width.current"])} for g, s in zip(env["gpus"], smi)],
            "driver": smi[0]["driver_version"], "driver_model": smi[0]["driver_model.current"],
            "cpu": env["cpu"], "cpu_threads": env["cpu_threads"], "ram_gib": env["ram_gb"], "os": env["os"],
            "python": env["python"], "torch": env["torch"], "cuda_build": env["cuda_build"],
            "topology_recorded": not env["smi_topology"].startswith("nvidia-smi failed")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=str(SRC / "p40_env.json"))
    ap.add_argument("--scaling", default=str(SRC / "p40_shard_scaling.json"))
    ap.add_argument("--out", default=str(RESULTS / "multi_gpu_scaling.json"))
    args = ap.parse_args()
    env = json.load(open(args.env, encoding="utf-8"))
    sb = json.load(open(args.scaling, encoding="utf-8"))
    G = len(sb["devices"])
    R = {(r["pages"], r["gpus"]): r for r in sb["runs"]}

    def per_query_s(r):
        return r["throughput"][THR_BATCH]["batch_latency_s"] / int(THR_BATCH)

    def energy(r):
        return r["throughput"][THR_BATCH]["energy_j_per_query"]

    idle = [(energy(R[(n, g)]) - energy(R[(n, G)])) / ((G - g) * per_query_s(R[(n, g)]))
            for n, g in R if n >= MIN_PAGES and g < G and (n, G) in R]
    p_idle = float(np.median(idle))

    strong = {}
    for n in sorted({n for n, _ in R}):
        gs = sorted(g for m, g in R if m == n)
        g0, b = gs[0], R[(n, gs[0])]
        by = {}
        for g in gs:
            r = R[(n, g)]
            sp = b["p50_s"] / r["p50_s"]
            qps = r["throughput"][THR_BATCH]["queries_per_s"]
            tsp = qps / b["throughput"][THR_BATCH]["queries_per_s"]
            by[str(g)] = {"index_gib_per_gpu": r["index_gib_per_gpu"], "p50_s": r["p50_s"], "p95_s": r["p95_s"],
                          "latency_speedup": sp, "latency_efficiency": sp * g0 / g,
                          "overhead_s": r["p50_s"] - b["p50_s"] * g0 / g,
                          "throughput_qps": qps, "throughput_speedup": tsp, "throughput_efficiency": tsp * g0 / g,
                          "energy_j_per_query_node": energy(r),
                          "energy_j_per_query_active": energy(r) - p_idle * (G - g) * per_query_s(r),
                          "node_power_w": energy(r) / per_query_s(r)}
        strong[str(n)] = {"base_gpus": g0, "index_gib_total": b["index_gib_total"], "by_gpus": by}

    groups = {}
    for (n, g), r in R.items():
        groups.setdefault(math.ceil(n / g), {})[g] = (n, r)
    weak = {}
    for per, d in sorted(groups.items()):
        if len(d) < 2:
            continue
        g0 = min(d)
        weak[str(per)] = {str(g): {"pages": n, "index_gib_total": r["index_gib_total"], "p50_s": r["p50_s"],
                                   "efficiency": d[g0][1]["p50_s"] / r["p50_s"]} for g, (n, r) in sorted(d.items())}

    def runs(min_pages, cond=lambda g, g0: True):
        return [v for n, s in strong.items() if int(n) >= min_pages
                for g, v in s["by_gpus"].items() if cond(int(g), s["base_gpus"])]
    summary = {
        "check_exact_min": min(c["exact"] for c in sb["checks"].values()),
        "check_exact_or_tie_min": min(c["exact_or_tie"] for c in sb["checks"].values()),
        "max_overhead_s_ge50k": max(v["overhead_s"] for v in runs(50000, lambda g, g0: g > g0)),
        "min_latency_efficiency_ge50k": min(v["latency_efficiency"] for v in runs(50000)),
        "min_throughput_efficiency_ge10k": min(v["throughput_efficiency"] for v in runs(MIN_PAGES)),
        "idle_w_per_gpu": p_idle, "idle_w_range": [float(min(idle)), float(max(idle))],
        "max_pages_by_gpus": {str(g): max(n for n, h in R if h == g) for g in range(1, G + 1)},
    }
    out = {"source": {"node": node_hardware(env), "devices": sb["devices"], "capability": sb["capability"],
                      "store_dtype": sb["store_dtype"], "compute_dtype": sb["compute_dtype"],
                      "vectors_per_page": sb["vectors_per_page"], "topk": sb["topk"], "n_lat": sb["n_lat"],
                      "n_check": N_CHECK, "batches": list(sb["runs"][0]["throughput"]), "chunk": sb["chunk"],
                      "budget_gib_per_gpu": sb["budget_gib_per_gpu"], "energy_mode": sb["energy_mode"]},
           "n_gpus": G, "throughput_batch": int(THR_BATCH), "min_pages_fit": MIN_PAGES,
           "strong": strong, "weak": weak, "summary": summary}
    write_json(args.out, out)
    for n, s in strong.items():
        print(f"{int(n):>7} pages: " + ", ".join(
            f"{g} GPU {1000 * v['p50_s']:.1f} ms (eff {100 * v['latency_efficiency']:.0f}%, "
            f"thr eff {100 * v['throughput_efficiency']:.0f}%)" for g, v in s["by_gpus"].items()))
    print("weak:", {p: {g: round(1000 * v["p50_s"], 1) for g, v in d.items()} for p, d in weak.items()})
    print("summary:", json.dumps(summary))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
