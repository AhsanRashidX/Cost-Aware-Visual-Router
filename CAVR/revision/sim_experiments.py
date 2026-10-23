"""Load-test experiments requested in the ARS panel review, run with the discrete-event simulator of simulate.py.

  slo_sweep        GPUs needed for 100 queries/s at p95 <= T for T in P95_SWEEP, on every split, for the main policies
  batch_model      the simulator charges a batch the mean member time times the measured batching curve; as a
                   conservative bound it is re-run charging the longest member instead (p95 <= 2 s and 5 s, all splits)
  joint            GPUs for generation (simulated, 100 queries/s, p95 <= 2 s) plus a retrieval tier for a large ColPali
                   index (projected, not simulated): ColPali GPU-s per query at batch 64 on the workstation GPU, scaled
                   linearly from the largest GPU-resident size measured, times the share of queries that need ColPali,
                   at full utilisation; at least as many GPUs as are needed to hold the index (bench_index budget)
  adaptive         a fixed pool (the GPUs CAVR-CE needs for 100 queries/s at p95 <= 2 s) under bursty Poisson load;
                   fixed policies vs a controller that raises lambda with the waiting time of the oldest queued request
                   (thresholds t0, 2 t0, 4 t0); t0 is chosen on split 1's validation queries, results are on its test
                   queries. (Backlog per GPU is a poor signal here: overload shows as larger batches, not long queues.)
Inputs: results/path_outputs.jsonl, router_runs/, router_ce/, batching.json, eval_main.json, router_ce_cost.json,
        index_scaling.json, load_test.json
Output: results/sim_experiments.json
"""
import heapq
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

import simulate as S
from common import RESULTS, SEED, hardware_info, write_json
from routing_data import PATHS, load

P95_SWEEP = [1.5, 2.0, 3.0, 5.0, 7.5, 10.0]
LOAD = 100
SWEEP_POLICIES = ["always_visual", "always_hybrid", "cavr_ce_valsel"]
INDEX_PAGES = [100_000, 250_000, 1_000_000]
BURST = {"base_qps": 80.0, "burst_qps": 140.0, "base_s": 60.0, "burst_s": 20.0, "n_req": 24000, "warmup": 2000}
ADAPT_LEVELS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]   # CAVR-CE operating points (lambda), the paper's sweep range
ADAPT_T0 = [0.05, 0.1, 0.2, 0.3, 0.5]               # candidate base thresholds (s) of head-of-line waiting time
ADAPT_SLO_MIN = 0.95                                # validation rule: highest ANLS with >= 95% of requests within 2 s


def batching():
    bj = json.load(open(RESULTS / "batching.json"))["paths"]
    curve, bmax = {}, {}
    for p in PATHS:
        meas = {int(b): v["mean_batch_s"] for b, v in bj[p].items() if "note" not in v}
        bs = sorted(meas)
        bmax[p] = bs[-1]
        curve[p] = {b: float(np.interp(b, bs, [meas[x] / meas[1] for x in bs])) for b in range(1, bs[-1] + 1)}
    return curve, bmax


def simulate_longest(reqs, rate, n_gpus, batch_curve, bmax, rng):
    """simulate.simulate with a batch charged its longest member's generation time (conservative bound)."""
    arrivals = np.cumsum(rng.exponential(1 / rate, S.N_REQ))
    picks = rng.integers(0, len(reqs), S.N_REQ)
    ready = sorted((arrivals[i] + reqs[picks[i]][3], i) for i in range(S.N_REQ))
    queues = {p: deque() for p in PATHS}
    free = [(0.0, g) for g in range(n_gpus)]
    heapq.heapify(free)
    done = np.zeros(S.N_REQ)
    k = 0
    while k < len(ready) or any(queues.values()):
        t_free, g = heapq.heappop(free)
        while k < len(ready) and ready[k][0] <= t_free:
            queues[reqs[picks[ready[k][1]]][0]].append(ready[k])
            k += 1
        if not any(queues.values()):
            t_free = ready[k][0]
            queues[reqs[picks[ready[k][1]]][0]].append(ready[k])
            k += 1
        path = min((q[0][0], p) for p, q in queues.items() if q)[1]
        batch = [queues[path].popleft() for _ in range(min(bmax[path], len(queues[path])))]
        members = [reqs[picks[i]] for _, i in batch]
        service = sum(m[2] for m in members) + max(m[1] for m in members) * batch_curve[path][len(batch)]
        for _, i in batch:
            done[i] = t_free + service
        heapq.heappush(free, (t_free + service, g))
    return (done - arrivals)[S.WARMUP:]


def gpus_needed(reqs, load_qps, target, sim, curve, bmax):
    def ok(n):
        return np.percentile(sim(reqs, load_qps, n, curve, bmax, np.random.default_rng(SEED)), 95) <= target
    hi = 1
    while hi <= 256 and not ok(hi):
        hi *= 2
    if hi > 256:
        return None
    lo = hi // 2 + 1 if hi > 1 else 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return hi


class Setup:
    """Per-split choices and request builder, as in simulate.main."""

    def __init__(self):
        self.qids, self.Q, self.A, self.C, self.rows = load(str(RESULTS / "path_outputs.jsonl"))
        self.runs = sorted((RESULTS / "router_runs").glob("seed*.npz"))
        self.ev = json.load(open(RESULTS / "eval_main.json"))["per_seed"]
        self.ce_gpu = json.load(open(RESULTS / "router_ce_cost.json"))["per_query_batch1"]["gpu_s"]
        self.router_lat = json.load(open(RESULTS / "router_cost.json"))["per_query_batch1"]["latency_s"]

    def split(self, k):
        R = np.load(self.runs[k], allow_pickle=True)
        te, cost = np.where(R["split"] == "test")[0], R["cost"]
        qce = np.load(RESULTS / "router_ce" / self.runs[k].name, allow_pickle=True)["qhat_ce_full"]
        n = len(self.qids)
        ch = {"always_text": np.zeros(n, int), "always_visual": np.ones(n, int), "always_hybrid": np.full(n, 2),
              "cavr": np.argmax(R["cavr_qhat"] - 0.1 * cost, 1),
              "cavr_ce_valsel": np.argmax(qce - self.ev[k]["policies"]["cavr_ce_valsel"]["lambda"] * cost, 1)}
        for lam in ADAPT_LEVELS:
            ch[f"cavr_ce@{lam}"] = np.argmax(qce - lam * cost, 1)
        return te, ch, np.where(R["split"] == "val")[0]

    def req(self, name, i, p):
        r = self.rows[self.qids[i]][p]
        learned = name.startswith("cavr") and not name.startswith("cavr_ce")
        cpu = r["preprocess_s"] + r["text_retrieval_cpu_s"] + (self.router_lat if learned else 0.0)
        gpu_ret = r["visual_retrieval_gpu_s"]
        if name.startswith("cavr_ce"):
            if p == "text":
                gpu_ret = self.rows[self.qids[i]]["hybrid"]["visual_retrieval_gpu_s"]
            gpu_ret += self.ce_gpu
        return (p, r["generate_s"], gpu_ret, cpu)

    def reqs(self, name, choice, te):
        return [self.req(name, i, PATHS[choice[i]]) for i in te]


def simulate_adaptive(levels, rate_segments, n_gpus, curve, bmax, rng, controller):
    """levels: list over lambda levels of per-request tuples (path, gen_s, gpu_ret_s, cpu_s, anls).
    controller(head-of-line wait in s) -> level index, applied when a request is queued (at the next scheduling event)."""
    n_req, t, arr = BURST["n_req"], 0.0, []
    seg = 0
    while len(arr) < n_req:
        qps, dur = rate_segments[seg % 2]
        t_end = t + dur
        while True:
            t += rng.exponential(1 / qps)
            if t > t_end or len(arr) >= n_req:
                break
            arr.append(t)
        t = t_end
        seg += 1
    arr = np.array(arr[:n_req])
    picks = rng.integers(0, len(levels[0]), n_req)
    ready = sorted((arr[i] + levels[0][picks[i]][3], i) for i in range(n_req))
    queues = {p: deque() for p in PATHS}
    free = [(0.0, g) for g in range(n_gpus)]
    heapq.heapify(free)
    done, lvl = np.zeros(n_req), np.zeros(n_req, int)
    k, waiting = 0, 0
    while k < len(ready) or waiting:
        t_free, g = heapq.heappop(free)
        while k < len(ready) and (ready[k][0] <= t_free or waiting == 0):
            if ready[k][0] > t_free:
                t_free = ready[k][0]
            i = ready[k][1]
            hol = t_free - min(q[0][0] for q in queues.values() if q) if waiting else 0.0
            lvl[i] = controller(hol)
            queues[levels[lvl[i]][picks[i]][0]].append((ready[k][0], i))
            waiting += 1
            k += 1
        path = min((q[0][0], p) for p, q in queues.items() if q)[1]
        batch = [queues[path].popleft() for _ in range(min(bmax[path], len(queues[path])))]
        waiting -= len(batch)
        members = [levels[lvl[i]][picks[i]] for _, i in batch]
        service = sum(m[2] for m in members) + np.mean([m[1] for m in members]) * curve[path][len(batch)]
        for _, i in batch:
            done[i] = t_free + service
        heapq.heappush(free, (t_free + service, g))
    w = BURST["warmup"]
    lat = (done - arr)[w:]
    anls = np.array([levels[lvl[i]][picks[i]][4] for i in range(n_req)])[w:]
    gpu = np.array([levels[lvl[i]][picks[i]][1] + levels[lvl[i]][picks[i]][2] for i in range(n_req)])[w:]
    return {"p95_s": float(np.percentile(lat, 95)), "p50_s": float(np.percentile(lat, 50)),
            "slo_2s": float((lat <= 2.0).mean()), "anls": float(anls.mean()), "gpu_s_per_query": float(gpu.mean()),
            "level_share": [float((lvl[w:] == j).mean()) for j in range(len(levels))]}


def main():
    curve, bmax = batching()
    st = Setup()
    out = {"load_qps": LOAD, "hardware": hardware_info()}

    # 1. latency-target sweep and 2. batch-time model, all splits
    sweep = {p: {str(T): [] for T in P95_SWEEP} for p in SWEEP_POLICIES}
    bm = {m: {p: {"2.0": [], "5.0": []} for p in SWEEP_POLICIES} for m in ("mean_member", "longest_member")}
    for k in range(len(st.runs)):
        te, ch, _ = st.split(k)
        for p in SWEEP_POLICIES:
            reqs = st.reqs(p, ch[p], te)
            for T in P95_SWEEP:
                sweep[p][str(T)].append(gpus_needed(reqs, LOAD, T, S.simulate, curve, bmax))
            for T in (2.0, 5.0):
                bm["mean_member"][p][str(T)].append(sweep[p][str(T)][-1])
                bm["longest_member"][p][str(T)].append(gpus_needed(reqs, LOAD, T, simulate_longest, curve, bmax))
        print(f"split {k + 1}: sweep done", flush=True)

    def ms(xs):
        xs = [x for x in xs if x is not None]
        return {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, "per_seed": xs}
    out["slo_sweep"] = {p: {T: ms(v) for T, v in d.items()} for p, d in sweep.items()}
    out["slo_sweep_saving_vs_hybrid"] = {
        T: float(1 - out["slo_sweep"]["cavr_ce_valsel"][T]["mean"] / out["slo_sweep"]["always_hybrid"][T]["mean"])
        for T in out["slo_sweep"]["always_hybrid"]}
    out["batch_model"] = {m: {p: {T: ms(v) for T, v in d.items()} for p, d in pd.items()} for m, pd in bm.items()}
    out["batch_model_saving_vs_hybrid"] = {
        m: {T: float(1 - out["batch_model"][m]["cavr_ce_valsel"][T]["mean"] / out["batch_model"][m]["always_hybrid"][T]["mean"])
            for T in ("2.0", "5.0")} for m in bm}

    # 3. joint generation + retrieval provisioning (retrieval projected)
    ix = json.load(open(RESULTS / "index_scaling.json"))
    lt = json.load(open(RESULTS / "load_test.json"))["seeds_summary"]["policies"]
    import re
    budget = float(re.search(r"^GPU_BUDGET_GB = ([\d.]+)", (Path(__file__).with_name("bench_index.py")).read_text(),
                             re.M).group(1))   # GiB of index per workstation GPU, as in bench_index.py
    n_ref = max(int(n) for n, v in ix["scaling"].items() if "gpu_bf16" in v)
    ref = ix["scaling"][str(n_ref)]
    gpu_s_ref = 1 / ref["gpu_bf16"]["throughput"]["64"]["queries_per_s"]
    gib_page = ref["index_gb_bf16"] / n_ref
    te0, ch0, va0 = st.split(0)
    share = {"always_text": 0.0, "always_visual": 1.0, "always_hybrid": 1.0, "cavr_ce_valsel": 1.0,
             "cavr": float((ch0["cavr"][te0] != 0).mean())}
    joint = {"reference_pages": n_ref, "gpu_s_per_query_at_reference": gpu_s_ref, "index_budget_gib_per_gpu": budget,
             "colpali_share": share, "by_pages": {}}
    for n in INDEX_PAGES:
        gs = gpu_s_ref * n / n_ref
        floor = math.ceil(n * gib_page / budget)
        row = {"retrieval_gpu_s_per_query": gs, "gpus_to_hold_index": floor, "policies": {}}
        for p in share:
            gen = lt[p]["gpus_needed"]["100"]["mean"]
            ret = 0 if share[p] == 0 else max(floor, math.ceil(LOAD * share[p] * gs))
            row["policies"][p] = {"generation_gpus": gen, "retrieval_gpus": ret, "total": gen + ret,
                                  "anls": lt[p]["anls_test"]["mean"]}
        joint["by_pages"][str(n)] = row
    out["joint"] = joint
    print("joint done", flush=True)

    # 4. load-adaptive lambda under bursts (split 1), pool sized for CAVR-CE at 100 q/s, p95 <= 2 s
    pool = lt["cavr_ce_valsel"]["gpus_needed"]["100"]["per_seed"][0]
    segs = [(BURST["base_qps"], BURST["base_s"]), (BURST["burst_qps"], BURST["burst_s"])]

    def levels_for(names, idx=te0):
        return [[st.req(nm, i, PATHS[ch0[nm][i]]) + (float(st.A[i, ch0[nm][i]]),) for i in idx] for nm in names]
    lam_val = st.ev[0]["policies"]["cavr_ce_valsel"]["lambda"]
    adapt_levels = ["cavr_ce_valsel"] + [f"cavr_ce@{l}" for l in ADAPT_LEVELS if l > lam_val]
    def ctl_for(t0):
        th = [t0 * 2 ** j for j in range(len(adapt_levels) - 1)]
        return lambda w: min(sum(w >= x for x in th), len(adapt_levels) - 1)
    val = {}
    for t0 in ADAPT_T0:
        val[str(t0)] = simulate_adaptive(levels_for(adapt_levels, va0), segs, pool, curve, bmax,
                                         np.random.default_rng(SEED), ctl_for(t0))
    ok = [t for t in ADAPT_T0 if val[str(t)]["slo_2s"] >= ADAPT_SLO_MIN]
    t0_sel = max(ok, key=lambda t: val[str(t)]["anls"]) if ok else max(ADAPT_T0, key=lambda t: val[str(t)]["slo_2s"])
    configs = {"always_hybrid": (["always_hybrid"], lambda w: 0),
               "always_visual": (["always_visual"], lambda w: 0),
               "cavr_ce_valsel": (["cavr_ce_valsel"], lambda w: 0),
               "cavr_ce@0.2": (["cavr_ce@0.2"], lambda w: 0),
               "cavr_ce_adaptive": (adapt_levels, ctl_for(t0_sel))}
    adaptive = {"pool_gpus": pool, "burst": BURST, "levels": adapt_levels, "signal": "head-of-line waiting time (s)",
                "t0_candidates": ADAPT_T0, "t0_selected": t0_sel, "validation": val,
                "lambda_valsel": lam_val, "policies": {}}
    for name, (names, ctl) in configs.items():
        adaptive["policies"][name] = simulate_adaptive(levels_for(names), segs, pool, curve, bmax,
                                                       np.random.default_rng(SEED), ctl)
        print(name, adaptive["policies"][name], flush=True)
    out["adaptive"] = adaptive
    write_json(RESULTS / "sim_experiments.json", out)
    print("sweep saving vs hybrid:", {k: round(v, 3) for k, v in out["slo_sweep_saving_vs_hybrid"].items()})
    print("batch model saving:", out["batch_model_saving_vs_hybrid"])
    print("wrote results/sim_experiments.json")


if __name__ == "__main__":
    main()
