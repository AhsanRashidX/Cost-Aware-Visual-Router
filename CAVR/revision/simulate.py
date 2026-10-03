"""Step 8d: Poisson load test - discrete-event simulation of a GPU serving pool with dynamic batching.

Service times are the measured per-query times from the cache (path_outputs.jsonl); batching speed-up
per path comes from bench_batching.py. Each GPU runs one batch at a time; a batch holds requests of one
path, up to the largest batch size measured for that path. CPU-side work (OCR-text retrieval,
preprocessing, router) runs on the host in parallel and only adds latency.

Outputs (results/load_test.json):
  latency vs arrival rate on 1 GPU, per policy
  max sustainable rate on 1 GPU at p95 <= target
  GPUs needed to serve target loads at p95 <= target
"""
import heapq
import json
from collections import deque
from pathlib import Path

import numpy as np

from common import RESULTS, SEED, hardware_info, write_json
from routing_data import LAMBDA_DEFAULT, PATHS, keyword_submitted, load

RATES = [round(0.25 * i, 2) for i in range(1, 25)] + [7, 8, 10, 12, 15, 20]
P95_TARGETS = [2.0, 5.0]
TARGET_LOADS = [5, 10, 25, 50, 100]
N_REQ, WARMUP = 6000, 500
POLICIES = ["always_text", "always_visual", "always_hybrid", "keyword_submitted", "cavr", "oracle"]
OPERATING_LAMBDAS = [0.0, 0.02, 0.05, 0.1, 0.2]   # CAVR operating points (lambda is a runtime knob)


def simulate(reqs, rate, n_gpus, batch_curve, bmax, rng):
    """reqs: list of (path, gen_s, gpu_retrieval_s, cpu_s). Returns latencies (s) after warm-up."""
    arrivals = np.cumsum(rng.exponential(1 / rate, N_REQ))
    picks = rng.integers(0, len(reqs), N_REQ)
    ready = sorted((arrivals[i] + reqs[picks[i]][3], i) for i in range(N_REQ))
    queues = {p: deque() for p in PATHS}
    free = [(0.0, g) for g in range(n_gpus)]
    heapq.heapify(free)
    done = np.zeros(N_REQ)
    k = 0
    while k < len(ready) or any(queues.values()):
        t_free, g = heapq.heappop(free)
        while k < len(ready) and ready[k][0] <= t_free:
            i = ready[k][1]
            queues[reqs[picks[i]][0]].append((ready[k][0], i))
            k += 1
        if not any(queues.values()):                 # idle until the next request is ready
            t_next, i = ready[k]
            queues[reqs[picks[i]][0]].append((t_next, i))
            k += 1
            t_free = t_next
        path = min((q[0][0], p) for p, q in queues.items() if q)[1]   # oldest waiting request's path
        batch = [queues[path].popleft() for _ in range(min(bmax[path], len(queues[path])))]
        members = [reqs[picks[i]] for _, i in batch]
        service = sum(m[2] for m in members) + np.mean([m[1] for m in members]) * batch_curve[path][len(batch)]
        t_end = t_free + service
        for _, i in batch:
            done[i] = t_end
        heapq.heappush(free, (t_end, g))
    return (done - arrivals)[WARMUP:]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(RESULTS / "path_outputs.jsonl"))
    ap.add_argument("--runs", default=str(RESULTS / "router_runs"))
    ap.add_argument("--batching", default=str(RESULTS / "batching.json"))
    ap.add_argument("--out", default=str(RESULTS / "load_test.json"))
    args = ap.parse_args()
    qids, Q, A, C, rows = load(args.outputs)
    R = np.load(sorted(Path(args.runs).glob("seed*.npz"))[0], allow_pickle=True)
    te = np.where(R["split"] == "test")[0]
    cost = R["cost"]
    rc_path = RESULTS / "router_cost.json"
    router_lat = json.load(open(rc_path))["per_query_batch1"]["latency_s"] if rc_path.exists() else 0.0

    bj = json.load(open(args.batching))["paths"]
    batch_curve, bmax = {}, {}
    for p in PATHS:
        # drop batch sizes the benchmark flagged as near the VRAM limit (WDDM sysmem spill distorts them)
        meas = {int(b): v["mean_batch_s"] for b, v in bj[p].items() if "note" not in v}
        bs = sorted(meas)
        bmax[p] = bs[-1]
        # ratio T(B)/T(1), interpolated for unmeasured B
        batch_curve[p] = {b: float(np.interp(b, bs, [meas[x] / meas[1] for x in bs])) for b in range(1, bs[-1] + 1)}

    choices = {
        "always_text": np.zeros(len(qids), int), "always_visual": np.ones(len(qids), int),
        "always_hybrid": np.full(len(qids), 2),
        "keyword_submitted": np.array([PATHS.index(keyword_submitted(q["question"])) for q in Q]),
        "cavr": np.argmax(R["cavr_qhat"] - LAMBDA_DEFAULT * cost, 1),
    }
    u = A - LAMBDA_DEFAULT * C
    cheap = np.argsort(C, 1)
    for lam in OPERATING_LAMBDAS:
        choices[f"cavr@{lam}"] = np.argmax(R["cavr_qhat"] - lam * cost, 1)
        choices[f"cavr_rf@{lam}"] = np.argmax(R["cavr_rf_qhat"] - lam * cost, 1)
    ce_file = RESULTS / "router_ce" / sorted(Path(args.runs).glob("seed*.npz"))[0].name
    ce_cost_path = RESULTS / "router_ce_cost.json"
    ce_gpu = json.load(open(ce_cost_path))["per_query_batch1"]["gpu_s"] if ce_cost_path.exists() else 0.0
    ce_names = []
    if ce_file.exists():
        qce = np.load(ce_file, allow_pickle=True)["qhat_ce_full"]
        for lam in OPERATING_LAMBDAS:
            choices[f"cavr_ce@{lam}"] = np.argmax(qce - lam * cost, 1)
            ce_names.append(f"cavr_ce@{lam}")
        ev = json.load(open(RESULTS / "eval_main.json"))["per_seed"][0]["policies"]
        if "cavr_ce_valsel" in ev:
            choices["cavr_ce_valsel"] = np.argmax(qce - ev["cavr_ce_valsel"]["lambda"] * cost, 1)
            ce_names.append("cavr_ce_valsel")
    choices["oracle"] = np.take_along_axis(cheap, np.argmax(np.take_along_axis(u, cheap, 1) >= u.max(1, keepdims=True) - 1e-12, 1)[:, None], 1)[:, 0]

    out = {"service_model": "measured per-query times + measured batching curves", "batch_curve": batch_curve,
           "bmax": bmax, "router_latency_s": router_lat, "policies": {}, "hardware": hardware_info()}
    names = POLICIES + [f"{m}@{lam}" for m in ("cavr", "cavr_rf") for lam in OPERATING_LAMBDAS] + ce_names
    for name in names:
        reqs = []
        for i in te:
            p = PATHS[choices[name][i]]
            r = rows[qids[i]][p]
            learned = name.startswith("cavr") and not name.startswith("cavr_ce")
            cpu = r["preprocess_s"] + r["text_retrieval_cpu_s"] + (router_lat if learned else 0.0)
            gpu_ret = r["visual_retrieval_gpu_s"]
            if (name.startswith("cavr_rf") or name.startswith("cavr_ce")) and p == "text":   # ColPali ran for routing
                gpu_ret = rows[qids[i]]["hybrid"]["visual_retrieval_gpu_s"]
            if name.startswith("cavr_ce"):        # cross-encoder runs on the GPU, so it occupies the GPU
                gpu_ret = gpu_ret + ce_gpu
            reqs.append((p, r["generate_s"], gpu_ret, cpu))
        rng = np.random.default_rng(SEED)
        curve = {}
        for rate in RATES:
            lat = simulate(reqs, rate, 1, batch_curve, bmax, rng)
            curve[str(rate)] = {"p50_s": float(np.percentile(lat, 50)), "p95_s": float(np.percentile(lat, 95))}
            if curve[str(rate)]["p95_s"] > 60:
                break
        cap = {str(t): max([float(r) for r, v in curve.items() if v["p95_s"] <= t], default=0.0) for t in P95_TARGETS}
        gpus = {}
        for t in P95_TARGETS:
            gpus[str(t)] = {}
            for load_qps in TARGET_LOADS:
                def ok(n):
                    lat = simulate(reqs, load_qps, n, batch_curve, bmax, np.random.default_rng(SEED))
                    return np.percentile(lat, 95) <= t
                hi = 1
                while hi <= 256 and not ok(hi):
                    hi *= 2
                if hi > 256:
                    n = 257
                else:
                    lo = hi // 2 + 1 if hi > 1 else 1
                    while lo < hi:          # smallest n in [lo, hi] that meets the target
                        mid = (lo + hi) // 2
                        if ok(mid):
                            hi = mid
                        else:
                            lo = mid + 1
                    n = hi
                gpus[str(t)][str(load_qps)] = n if n <= 256 else None
        out["policies"][name] = {"anls_test": float(A[te, choices[name][te]].mean()),
                                 "latency_vs_rate_1gpu": curve, "max_rate_1gpu": cap, "gpus_needed": gpus,
                                 "path_mix": {p: float((choices[name][te] == k).mean()) for k, p in enumerate(PATHS)}}
        print(name, "max QPS@p95<=2s:", cap["2.0"], "| GPUs for 25 QPS:", gpus["2.0"]["25"], flush=True)
    write_json(args.out, out)


if __name__ == "__main__":
    main()
