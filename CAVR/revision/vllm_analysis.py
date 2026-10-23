"""Experiment 3: live vLLM serving on the 2x A10 node, and a check of the discrete-event simulator against it.

Inputs (results/a10/vllm/, copied from the WSL run of t4/run_a10_wsl.sh):
  pilot.json            correctness (prompt-token equality, ANLS vs the 4090 run), closed-loop sweep per path,
                        and the summary of the per-query service stage
  service_times.jsonl   every split-1 test query x path at concurrency 1 (A10 vLLM service times)
  load/<policy>_r<N>/   open-loop Poisson runs per policy and replica count: summary.json, requests.jsonl
  serve_info.txt        launch commands, vLLM and torch versions, driver
Request set: t4_package/requests.jsonl (the policies' path choice per query).

Simulator check. Each measured arrival trace (scheduled time, query, path of every sent request) is replayed through
simulate.py's GPU-pool model (one batch of one path per GPU at a time, batch time = mean service time x T(B)/T(1),
batches up to the largest measured B), with two sets of inputs:
  a10  service times = the A10 vLLM concurrency-1 latencies; T(B)/T(1) = the pilot sweep's latency ratio (B <= 16)
  ws   service times = the 4090 generate_s of the paper's cache; T(B)/T(1) = batching.json (the paper's simulator)
The live test serves generation only (retrieval and the router ran beforehand), so both replays use generation
time only. Predicted and measured p95 are compared rate by rate (aborted rates excluded), and the capacity at
p95 <= 2 s and <= 5 s is compared: measured = linear interpolation of p95 between the measured rates; predicted =
the same model on fresh Poisson traces (loadgen's draw order and seed, 6,000 arrivals, 500 warm-up as in
simulate.py). Load directories with a suffix after _r<N> (a run cut off when the WSL VM shut down) are skipped.
Output: results/vllm_a10.json
"""
import heapq
import json
from collections import deque
from pathlib import Path

import numpy as np

from common import RESULTS, ROOT, SEED, write_json

PATHS = ["text", "visual", "hybrid"]
POLICIES = ["always_visual", "always_hybrid", "cavr_ce_valsel"]
TARGETS = [2.0, 5.0]   # p95 targets (s), as simulate.py's P95_TARGETS
N_SIM, WARMUP_SIM = 6000, 500
VDIR = RESULTS / "a10" / "vllm"


def replay(arrivals, paths, svc, curve, bmax, n_gpus):
    """simulate.simulate's queueing model on a given trace. arrivals: times (s); paths: path per request;
    svc: service time at batch 1 per request. Returns completion - arrival per request (trace order)."""
    n = len(arrivals)
    ready = sorted((arrivals[i], i) for i in range(n))
    queues = {p: deque() for p in PATHS}
    free = [(0.0, g) for g in range(n_gpus)]
    heapq.heapify(free)
    done = np.zeros(n)
    k = 0
    while k < n or any(queues.values()):
        t_free, g = heapq.heappop(free)
        while k < n and ready[k][0] <= t_free:
            i = ready[k][1]
            queues[paths[i]].append((ready[k][0], i))
            k += 1
        if not any(queues.values()):
            t_next, i = ready[k]
            queues[paths[i]].append((t_next, i))
            k += 1
            t_free = t_next
        path = min((q[0][0], p) for p, q in queues.items() if q)[1]
        batch = [queues[path].popleft() for _ in range(min(bmax[path], len(queues[path])))]
        t_end = t_free + np.mean([svc[i] for _, i in batch]) * curve[path][len(batch)]
        for _, i in batch:
            done[i] = t_end
        heapq.heappush(free, (t_end, g))
    return done - np.asarray(arrivals)


def interp_curve(points):
    """points {B: T(B)/T(1)} -> ratio for every B in 1..max (linear in B), and max B."""
    bs = sorted(points)
    return {b: float(np.interp(b, bs, [points[x] for x in bs])) for b in range(1, bs[-1] + 1)}, bs[-1]


def capacity(rates, p95s, target):
    """Largest rate with p95 <= target, linearly interpolated to the crossing; 0 if the first rate misses it."""
    pts = [(r, p) for r, p in sorted(zip(rates, p95s)) if p is not None]
    if not pts or pts[0][1] > target:
        return 0.0
    for (r0, p0), (r1, p1) in zip(pts, pts[1:]):
        if p1 > target:
            return float(r0 + (target - p0) * (r1 - r0) / (p1 - p0))
    return float(pts[-1][0])   # never crossed: lower bound


def main():
    pilot = json.load(open(VDIR / "pilot.json", encoding="utf-8"))
    req_file = ROOT / "t4_package" / "requests.jsonl"
    reqs = [json.loads(line) for line in open(req_file, encoding="utf-8")]
    svc_a10 = {}
    for line in open(VDIR / "service_times.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["error"] is None:
            svc_a10[(r["i"], r["path"])] = r["latency_s"]
    svc_ws = {(e["i"], p): e["paths"][p]["ref_4090"]["generate_s"] for e in reqs for p in PATHS}

    sweep = pilot["sweep"]
    curve_a10, bmax_a10 = {}, {}
    for p in PATHS:
        pts = {int(c): v["latency_ratio_vs_first_level"] for c, v in sweep[p].items() if c.isdigit()}
        curve_a10[p], bmax_a10[p] = interp_curve(pts)
    bj = json.load(open(RESULTS / "batching.json", encoding="utf-8"))["paths"]
    curve_ws, bmax_ws = {}, {}
    for p in PATHS:
        meas = {int(b): v["mean_batch_s"] for b, v in bj[p].items() if "note" not in v}
        curve_ws[p], bmax_ws[p] = interp_curve({b: meas[b] / meas[1] for b in meas})
    models = {"a10": (svc_a10, curve_a10, bmax_a10), "ws": (svc_ws, curve_ws, bmax_ws)}

    out = {"description": "live vLLM serving on 1-2 A10 replicas vs simulate.py's queueing model",
           "p95_targets_s": TARGETS, "pilot_correctness": pilot["correctness"],
           "service_stage": pilot.get("service"),
           "sweep": {p: {c: {k: v[k] for k in ("steady_throughput_rps", "latency_ratio_vs_first_level")}
                         | {"p95_s": v["latency_s"]["p95"], "mean_s": v["latency_s"]["mean"]}
                         for c, v in sweep[p].items() if c.isdigit()}
                     | {"max_steady_throughput_rps": sweep[p]["max_steady_throughput_rps"]} for p in PATHS},
           "batch_curve_a10": curve_a10, "bmax_a10": bmax_a10,
           "request_tokens_mean": {p: {k: float(np.mean([e["paths"][p]["ref_4090"][k] for e in reqs]))
                                       for k in ("input_tokens", "output_tokens")} for p in PATHS},
           "run_metadata": pilot["runs"][0],
           "serve_info": (VDIR / "serve_info.txt").read_text(encoding="utf-8", errors="replace")
           if (VDIR / "serve_info.txt").exists() else None,
           "sampler_note": "VLLM_USE_FLASHINFER_SAMPLER=0 (FlashInfer's sampler needs nvcc, absent in WSL); "
                           "greedy decoding, so the tokens are those of the default sampler",
           "runs": {}}

    for d in sorted((VDIR / "load").glob("*_r[0-9]*")):
        if not (d / "summary.json").exists():   # run still in progress (summary is written after its first rate)
            continue
        summ = json.load(open(d / "summary.json", encoding="utf-8"))
        policy, n_rep = summ["policy"], len(summ["replicas"])
        if d.name != f"{policy}_r{n_rep}":   # e.g. cavr_ce_valsel_r1_interrupted_2014: cut off when WSL shut down
            continue
        rows = [json.loads(line) for line in open(d / "requests.jsonl", encoding="utf-8")]
        rates_out = {}
        for key, s in summ["rates"].items():
            rate = float(key)
            rr = sorted((r for r in rows if r["rate"] == rate), key=lambda r: r["k"])
            arr = [r["scheduled_s"] for r in rr]
            paths = [r["path"] for r in rr]
            meas_idx = [j for j, r in enumerate(rr) if not r["warmup"]]
            pred = {}
            for name, (svc, curve, bmax) in models.items():
                lat = replay(arr, paths, [svc[(r["i"], r["path"])] for r in rr], curve, bmax, n_rep)
                pred[name] = {"p50_s": float(np.percentile(lat[meas_idx], 50)),
                              "p95_s": float(np.percentile(lat[meas_idx], 95))}
            # requests that found their replica idle (nothing in flight there at the scheduled time): latency over
            # the same query's back-to-back service time. Isolates idle-to-busy effects (clocks, caches) from queueing.
            spans = {}
            for r in rr:
                spans.setdefault(r["replica"], []).append((r["scheduled_s"], r["end_s"], r["k"]))
            idle_ratio = [r["latency_s"] / svc_a10[(r["i"], r["path"])] for r in rr
                          if not r["warmup"] and r["error"] is None
                          and not any(s0 < r["scheduled_s"] < e0 for s0, e0, k0 in spans[r["replica"]] if k0 != r["k"])]
            rates_out[key] = {
                "offered_rps": rate,
                "idle_arrivals": len(idle_ratio),
                "idle_arrival_latency_over_service_median": float(np.median(idle_ratio)) if idle_ratio else None, "realised_rps": s["realised_arrival_rate_rps"], "throughput_rps": s["throughput_rps"],
                "n_measured_ok": s["n_measured_ok"], "n_errors": s["n_errors"], "aborted": s["aborted"],
                "p50_s": s["latency_s"]["p50"], "p95_s": s["latency_s"]["p95"], "p99_s": s["latency_s"]["p99"],
                "ttft_p95_s": s["ttft_s"]["p95"], "max_outstanding": s["max_outstanding"],
                "anls": s["quality"]["anls"], "anls_4090_same_requests": s["quality"]["anls_4090_same_requests"],
                "answer_identical_to_4090": s["quality"]["answer_identical_to_4090"],
                "energy_j_per_request": (s["energy"] or {}).get("j_per_completed_request"),
                "energy_j_per_request_above_idle": (s["energy"] or {}).get("j_per_completed_request_above_idle"),
                "predicted": pred}
        ok = {k: v for k, v in rates_out.items() if not v["aborted"] and v["n_errors"] == 0}
        cap = {str(t): {"measured": capacity([v["offered_rps"] for v in ok.values()],
                                             [v["p95_s"] for v in ok.values()], t)} for t in TARGETS}
        hi_rate = max(float(k) for k in summ["rates"])
        for name, (svc, curve, bmax) in models.items():
            grid = np.round(np.arange(0.05, 2.0 * max(hi_rate, 0.5) + 1e-9, 0.05), 2)
            p95s = []
            for rate in grid:
                rng = np.random.default_rng(SEED)
                arr = np.cumsum(rng.exponential(1 / rate, N_SIM))
                picks = rng.integers(0, len(reqs), N_SIM)
                paths = [reqs[i]["policy"][policy] for i in picks]
                lat = replay(arr, paths, [svc[(int(i), p)] for i, p in zip(picks, paths)], curve, bmax, n_rep)
                p95s.append(float(np.percentile(lat[WARMUP_SIM:], 95)))
                if p95s[-1] > 4 * max(TARGETS):
                    break
            for t in TARGETS:
                cap[str(t)][f"predicted_{name}"] = capacity(list(grid[:len(p95s)]), p95s, t)
        err = {name: [abs(v["predicted"][name]["p95_s"] - v["p95_s"]) / v["p95_s"]
                      for v in ok.values() if v["p95_s"] and v["p95_s"] <= 4 * max(TARGETS)] for name in models}
        out["runs"][f"{policy}_r{n_rep}"] = {
            "policy": policy, "replicas": n_rep, "path_share": summ["path_share"], "stop_reason": summ.get("stop_reason"),
            "idle_power_w": summ.get("idle_power_w"), "rates": rates_out,
            "capacity_rps_at_p95_target": cap,
            "measured_capacity_is_lower_bound": {str(t): all(v["p95_s"] <= t for v in ok.values()) for t in TARGETS},
            "p95_rel_error_median": {k: float(np.median(v)) if v else None for k, v in err.items()},
            "anls_mean_over_rates": float(np.mean([v["anls"] for v in ok.values()])) if ok else None}
        for t in TARGETS:
            c = cap[str(t)]
            print(f"{policy:15s} r{n_rep}: capacity@p95<={t:g}s measured {c['measured']:.2f} | "
                  f"sim a10 {c['predicted_a10']:.2f} | sim ws {c['predicted_ws']:.2f} rps", flush=True)

    runs = out["runs"]
    cmp = {}
    for t in map(str, TARGETS):
        for n_rep in (1, 2):
            get = lambda p: runs.get(f"{p}_r{n_rep}", {}).get("capacity_rps_at_p95_target", {}).get(t)
            h, v, ce = get("always_hybrid"), get("always_visual"), get("cavr_ce_valsel")
            if h and ce:
                c = cmp.setdefault(t, {})[f"r{n_rep}"] = {f"ce_over_hybrid_{k}": (ce[k] / h[k] if h[k] else None) for k in ce}
                if v:
                    c.update({f"ce_over_visual_{k}": (ce[k] / v[k] if v[k] else None) for k in ce})
        for p in POLICIES:
            a, b = runs.get(f"{p}_r1"), runs.get(f"{p}_r2")
            if a and b and a["capacity_rps_at_p95_target"][t]["measured"]:
                cmp.setdefault(t, {}).setdefault("scaling_r2_over_r1", {})[p] = (
                    b["capacity_rps_at_p95_target"][t]["measured"] / a["capacity_rps_at_p95_target"][t]["measured"])
    out["comparisons"] = cmp
    write_json(RESULTS / "vllm_a10.json", out)
    print("wrote", RESULTS / "vllm_a10.json")


if __name__ == "__main__":
    main()
