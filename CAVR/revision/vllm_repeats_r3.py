"""Live vLLM serving with repeated arrival traces (third review round, R6 concern 3).

The original live matrix (results/a10/vllm/load/<policy>_r<N>/, trace seed 1234) is repeated with trace seeds
1235-1238 (results/a10/vllm_r3/load_r3/<policy>_r<N>_s<seed>/, t4/run_repeats_r3.sh): same server settings, rate
grid, policies and request set; only the Poisson arrivals and query picks change. Per run, the capacity at a p95 target
is measured exactly as in vllm_analysis.py (linear interpolation between tested rates; 0 if the lowest rate misses).
Over the 5 runs: mean, standard deviation and a t interval of capacity per policy and replica count; ratios between
policies paired by trace seed (every policy replays the same trace for a given seed); and the simulator's error as
predicted capacity (vllm_a10.json, A10 service times and batching curve) over measured capacity, per run.
A pilot (results/a10/vllm_r3/load_r3_pilot/) re-ran seed 1234 at the two lowest always-hybrid rates on 1 GPU; its p95
is compared with the original run to check that the setup reproduces it.
Live pipeline runs (live_pipeline_r3.py: retrieval and routing in the loop) are summarised against the
generation-only run of the same seed and rates.
Output: results/vllm_repeats_r3.json
"""
import json

import numpy as np
from scipy import stats

from common import RESULTS, write_json
from vllm_analysis import TARGETS, capacity

V0 = RESULTS / "a10" / "vllm" / "load"
V3 = RESULTS / "a10" / "vllm_r3"
POLICIES = ["always_visual", "always_hybrid", "cavr_ce_valsel"]


def run_stats(d):
    s = json.load(open(d / "summary.json", encoding="utf-8"))
    ok = {k: v for k, v in s["rates"].items() if not v["aborted"] and v["n_errors"] == 0}
    rates = [float(k) for k in ok]
    p95 = [v["latency_s"]["p95"] for v in ok.values()]
    return {"seed": s["seed"], "stop_reason": s.get("stop_reason"),
            "p95_by_rate": {k: v["latency_s"]["p95"] for k, v in s["rates"].items()},
            "anls_by_rate": {k: v["quality"]["anls"] for k, v in ok.items()},
            "capacity": {str(t): capacity(rates, p95, t) for t in TARGETS},
            "never_crossed": {str(t): all(p <= t for p in p95) for t in TARGETS}}


def summarise(xs):
    xs = np.asarray([x for x in xs if x is not None], float)
    if len(xs) < 2:
        return {"n": len(xs), "mean": float(xs.mean()) if len(xs) else None}
    h = stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs))
    return {"n": len(xs), "mean": float(xs.mean()), "std": float(xs.std(ddof=1)),
            "ci95_t": [float(xs.mean() - h), float(xs.mean() + h)], "min": float(xs.min()), "max": float(xs.max()),
            "values": [float(x) for x in xs]}


def main():
    va = json.load(open(RESULTS / "vllm_a10.json", encoding="utf-8"))
    out = {"note": "5 live traces per rate (seed 1234 = original run, 1235-1238 = repeats); t intervals over traces",
           "runs": {}, "aggregate": {}, "ratios": {}, "simulator_error": {}}
    for pol in POLICIES:
        for n in (1, 2):
            runs = []
            if (V0 / f"{pol}_r{n}" / "summary.json").exists():
                runs.append(run_stats(V0 / f"{pol}_r{n}"))
            for d in sorted((V3 / "load_r3").glob(f"{pol}_r{n}_s*")):
                if d.name.split("_s")[-1].isdigit() and (d / "summary.json").exists():
                    r = run_stats(d)
                    if r["stop_reason"]:
                        runs.append(r)
            key = f"{pol}_r{n}"
            out["runs"][key] = runs
            agg = {"n_traces": len(runs)}
            for t in map(str, TARGETS):
                agg[f"capacity_{t}"] = summarise([r["capacity"][t] for r in runs])
                agg[f"zero_capacity_traces_{t}"] = sum(r["capacity"][t] == 0 for r in runs)
            rates = sorted({k for r in runs for k in r["p95_by_rate"]}, key=float)
            agg["p95_by_rate"] = {k: summarise([r["p95_by_rate"].get(k) for r in runs]) for k in rates}
            agg["anls"] = summarise([np.mean(list(r["anls_by_rate"].values())) for r in runs if r["anls_by_rate"]])
            out["aggregate"][key] = agg
            pred = va["runs"].get(key, {}).get("capacity_rps_at_p95_target", {})
            out["simulator_error"][key] = {
                t: summarise([pred[t]["predicted_a10"] / r["capacity"][t] for r in runs if r["capacity"][t] > 0])
                | {"predicted_a10": pred.get(t, {}).get("predicted_a10")} for t in map(str, TARGETS) if t in pred}
    for n in (1, 2):
        by = {p: {r["seed"]: r for r in out["runs"][f"{p}_r{n}"]} for p in POLICIES}
        seeds = sorted(set.intersection(*(set(v) for v in by.values())))
        for t in map(str, TARGETS):
            for a, b in (("cavr_ce_valsel", "always_hybrid"), ("always_visual", "cavr_ce_valsel")):
                xs = [by[a][s]["capacity"][t] / by[b][s]["capacity"][t] for s in seeds if by[b][s]["capacity"][t] > 0]
                out["ratios"][f"{a}_over_{b}_r{n}_{t}"] = summarise(xs) | {"paired_seeds": len(seeds)}
    # pilot: same seed and rates as the original always_hybrid r1 run
    pdir = V3 / "load_r3_pilot" / "always_hybrid_r1_pilot"
    if (pdir / "summary.json").exists():
        p, o = run_stats(pdir), run_stats(V0 / "always_hybrid_r1")
        out["pilot_reproduction"] = {k: {"pilot_p95": v, "original_p95": o["p95_by_rate"].get(k),
                                         "rel_diff": v / o["p95_by_rate"][k] - 1 if o["p95_by_rate"].get(k) else None}
                                     for k, v in p["p95_by_rate"].items()}
    # live pipeline (retrieval + routing in the loop) against generation-only runs of the same seed
    pl = {}
    for d in sorted((RESULTS / "a10" / "live_pipeline_r3").glob("*_r1_s*")):
        if not (d / "summary.json").exists():
            continue
        s = json.load(open(d / "summary.json", encoding="utf-8"))
        base = {"cavr_ce": "cavr_ce_valsel", "always_visual": "always_visual"}[s["policy"]]
        g = json.load(open(V0 / f"{base}_r1" / "summary.json", encoding="utf-8"))
        pl[d.name] = {k: {"pipeline_p95": v["latency_s"]["p95"], "pipeline_p50": v["latency_s"]["p50"],
                          "generation_only_p95": (g["rates"].get(k) or {}).get("latency_s", {}).get("p95"),
                          "generation_only_p50": (g["rates"].get(k) or {}).get("latency_s", {}).get("p50"),
                          "frontend_p50": v["frontend_s"]["p50"], "frontend_p95": v["frontend_s"]["p95"],
                          "frontend_wait_p95": v["frontend_wait_s"]["p95"], "anls": v["anls"],
                          "path_share": v["path_share"], "agree_cached_router": v["agree_cached_router"],
                          "agree_offline_retrained": v["agree_offline_retrained"]}
                      for k, v in s["rates"].items()}
        pl[d.name]["capacity"] = {str(t): capacity([float(k) for k in s["rates"]],
                                                   [v["latency_s"]["p95"] for v in s["rates"].values()], t)
                                  for t in TARGETS}
    out["live_pipeline"] = pl
    write_json(RESULTS / "vllm_repeats_r3.json", out)
    for key, agg in out["aggregate"].items():
        c5, c2 = agg["capacity_5.0"], agg["capacity_2.0"]
        print(f"{key:18s} traces {agg['n_traces']}  cap@5s {c5.get('mean')} {c5.get('ci95_t', '')}  "
              f"cap@2s {c2.get('mean')} zero@2s {agg['zero_capacity_traces_2.0']}")
    for k, v in out["ratios"].items():
        print(k, v.get("mean"), v.get("ci95_t"))
    print("pilot:", out.get("pilot_reproduction"))


if __name__ == "__main__":
    main()
