"""Open-loop Poisson load test of one routing policy on 1-3 vLLM replicas (the measured counterpart of simulate.py).

Arrivals, per rate: rng = numpy default_rng(--seed), fresh for every rate (as simulate.py's GPUs-needed search);
N exponential gaps of mean 1/rate are drawn first, then N query picks with replacement from the split-1 test
queries, exactly the draw order of simulate.simulate. The gaps are scaled standard exponentials, so every rate
replays the same query sequence (common random numbers across rates and policies). Each request goes out on the path
the policy chose for that query. Dispatch is client-side, to the replica with the fewest outstanding requests
(ties: fewest sent so far). Responses are streamed, giving TTFT and completion time; latency is measured from the
scheduled arrival time, so client lag would count against the server (it is logged as send_lag_s).
The first --warmup-frac of each rate's arrivals is excluded from the statistics (simulate.py drops 500 of 6000).
Rates run in increasing order; the run stops after the first rate whose p95 exceeds --p95-cap, that has more than
--max-errors errors, or that was aborted (a request older than --abort-latency: remaining arrivals are not sent
and in-flight requests are cancelled, which makes vLLM drop them).
Rates come from --rates (req/s, all replicas together) or from --rate-fractions x replicas x capacity per replica,
where capacity = 1 / sum_p share_p / thr_p with thr_p the path's best closed-loop throughput in pilot.json.
Energy per rate window (first send to last completion) from the NVML energy counters of the metered GPUs, with
fallbacks (client_common.EnergyMeter); idle power is measured once before the first rate.
Outputs (results/load/<policy>_r<N>[_tag]/):
  requests.jsonl        one row per request and rate (trace: arrival index, query index, path, scheduled time ...)
  summary.json          metadata, idle power, per-rate summaries, stop reason (rewritten after every rate)
  server_metrics.jsonl  vLLM Prometheus /metrics of every replica at the start and end of each rate
Usage:  python loadgen.py --policy cavr_ce_valsel --replicas 2 --rate-fractions 0.3,0.5,0.7,0.8,0.9,1.0,1.1
        python loadgen.py --policy always_hybrid --replicas 1 --rates 0.2,0.4,0.6
"""
import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np

import client_common as cc
from vqa_metrics import anls, exact_match, normalize


def plan_rates(args, reqs, paths_share):
    if args.rates:
        return [float(r) for r in args.rates.split(",")], None
    pilot = json.load(open(args.pilot, encoding="utf-8"))
    sw = pilot.get("sweep") or {}
    thr = {p: sw[p]["max_steady_throughput_rps"] for p in paths_share if paths_share[p] > 0}
    missing = [p for p, v in thr.items() if not v]
    if missing:
        raise SystemExit(f"pilot.json has no sweep throughput for {missing}; run pilot.py first or pass --rates")
    cap = 1.0 / sum(paths_share[p] / thr[p] for p in thr)
    rates = [round(f * args.replicas * cap, 3) for f in (float(x) for x in args.rate_fractions.split(","))]
    return rates, {"capacity_per_replica_rps": cap, "path_throughput_rps": thr, "fractions": args.rate_fractions}


def n_for_rate(args, rate):
    return int(min(args.n_requests, max(args.min_requests, math.ceil(rate * args.max_rate_seconds))))


async def scrape(session, urls, f, rate, when):
    if f is None:
        return
    for r, u in enumerate(urls):
        text = await cc.get_text(session, f"{u}/metrics")
        lines = [ln for ln in (text or "").splitlines() if ln.startswith("vllm")]
        f.write(json.dumps({"rate": rate, "when": when, "replica": r, "t": time.time(), "metrics": lines}) + "\n")
    f.flush()


async def run_rate(session, urls, reqs, policy, rate, n, warmup, args, ctx):
    rng = np.random.default_rng(args.seed)
    arrivals = np.cumsum(rng.exponential(1 / rate, n))      # same draw order as simulate.simulate
    picks = rng.integers(0, len(reqs), n)
    R = len(urls)
    outstanding, sent = [0] * R, [0] * R
    inflight = {}                                           # k -> scheduled time
    rows = []
    abort = asyncio.Event()
    max_out = [0]
    t0 = time.perf_counter() + 0.5

    async def one(k, r, t_sched):
        e = reqs[picks[k]]
        path = e["policy"][policy]
        try:
            res = await cc.chat(session, urls[r], ctx["body"](e, path), args.timeout)
        except asyncio.CancelledError:
            res = {"t_send": None, "t_first_chunk": None, "t_first_token": None, "t_end": time.perf_counter(),
                   "answer": "", "finish_reason": None, "prompt_tokens": None, "completion_tokens": None,
                   "error": "cancelled (rate aborted)"}
        outstanding[r] -= 1
        inflight.pop(k, None)
        ok = res["error"] is None
        ref = e["paths"][path]["ref_4090"]
        rows.append({
            "rate": rate, "k": k, "warmup": k < warmup, "i": int(picks[k]), "qid": e["qid"], "path": path, "replica": r,
            "scheduled_s": t_sched - t0, "send_s": (res["t_send"] - t0) if res["t_send"] else None,
            "send_lag_s": (res["t_send"] - t_sched) if res["t_send"] else None,
            "first_token_s": (res["t_first_token"] - t0) if res["t_first_token"] else None,
            "end_s": res["t_end"] - t0, "latency_s": res["t_end"] - t_sched,
            "ttft_s": (res["t_first_token"] - t_sched) if res["t_first_token"] else None,
            "prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
            "answer": res["answer"], "finish_reason": res["finish_reason"], "error": res["error"],
            "anls": anls(res["answer"], e["answers"]) if ok else None,
            "em": exact_match(res["answer"], e["answers"]) if ok else None,
            "same_as_4090": (normalize(res["answer"]) == normalize(ref["answer"])) if ok else None,
            "ref_anls": ref["anls"], "ref_generate_s": ref["generate_s"]})

    async def watchdog():
        while not abort.is_set():
            await asyncio.sleep(0.5)
            now = time.perf_counter()
            if inflight and now - min(inflight.values()) > args.abort_latency:
                abort.set()

    wd = asyncio.create_task(watchdog())
    tasks = []
    for k in range(n):
        t_sched = t0 + arrivals[k]
        delay = t_sched - time.perf_counter()
        if delay > 0:
            try:
                await asyncio.wait_for(abort.wait(), delay)
            except asyncio.TimeoutError:
                pass
        if abort.is_set():
            break
        r = min(range(R), key=lambda j: (outstanding[j], sent[j]))
        outstanding[r] += 1
        sent[r] += 1
        inflight[k] = t_sched
        max_out[0] = max(max_out[0], sum(outstanding))
        tasks.append(asyncio.create_task(one(k, r, t_sched)))
    pending = set(tasks)
    while pending:
        _, pending = await asyncio.wait(pending, timeout=0.5)
        if abort.is_set() and pending:
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            break
    wd.cancel()
    rows.sort(key=lambda x: x["k"])
    return rows, {"t0": t0, "n_sent": len(tasks), "aborted": abort.is_set(), "max_outstanding": max_out[0],
                  "arrival_span_s": float(arrivals[len(tasks) - 1] - arrivals[0]) if len(tasks) > 1 else 0.0}


def summarize(rows, info, rate, n, warmup, de, idle_w, args):
    meas = [r for r in rows if not r["warmup"]]
    good = [r for r in meas if r["error"] is None]
    errs = [r for r in rows if r["error"] is not None]
    t_end = max((r["end_s"] for r in rows), default=0.0)
    t_first = min((r["send_s"] for r in rows if r["send_s"] is not None), default=0.0)
    duration = max(t_end - t_first, 1e-9)
    n_ok_all = sum(r["error"] is None for r in rows)
    lat = cc.lat_stats([r["latency_s"] for r in good])
    s = {
        "rate": rate, "n_planned": n, "warmup": warmup, "n_sent": info["n_sent"], "n_measured": len(meas),
        "n_measured_ok": len(good), "n_errors": len(errs), "error_examples": [r["error"] for r in errs][:5],
        "aborted": info["aborted"],
        "latency_s": lat, "ttft_s": cc.lat_stats([r["ttft_s"] for r in good]),
        "send_lag_s": {"p95": cc.pct([r["send_lag_s"] for r in rows], 95), "max": cc.pct([r["send_lag_s"] for r in rows], 100)},
        "per_path": {p: {"n": sum(r["path"] == p for r in good),
                         **{k: v for k, v in cc.lat_stats([r["latency_s"] for r in good if r["path"] == p]).items()
                            if k in ("mean", "p50", "p95")}} for p in cc.PATHS},
        "per_replica": {str(rep): {"n": sum(r["replica"] == rep for r in rows),
                                   "p95": cc.pct([r["latency_s"] for r in good if r["replica"] == rep], 95)}
                        for rep in sorted({r["replica"] for r in rows})},
        "offered_rate_rps": rate,
        "realised_arrival_rate_rps": (info["n_sent"] - 1) / info["arrival_span_s"] if info["arrival_span_s"] else None,
        "duration_s": duration, "throughput_rps": n_ok_all / duration, "max_outstanding": info["max_outstanding"],
        "tokens": {"prompt_mean": cc.mean([r["prompt_tokens"] for r in good]),
                   "completion_mean": cc.mean([r["completion_tokens"] for r in good])},
        "quality": {"anls": cc.mean([r["anls"] for r in good]), "em": cc.mean([r["em"] for r in good]),
                    "anls_4090_same_requests": cc.mean([r["ref_anls"] for r in good]),
                    "answer_identical_to_4090": cc.mean([float(r["same_as_4090"]) for r in good])},
        "energy": None,
    }
    if de is not None:
        tot = float(sum(de))
        s["energy"] = {"per_gpu_j": de, "total_j": tot, "avg_power_w": tot / duration,
                       "j_per_completed_request": tot / n_ok_all if n_ok_all else None}
        if idle_w is not None:
            above = tot - sum(idle_w) * duration
            s["energy"]["j_per_completed_request_above_idle"] = above / n_ok_all if n_ok_all else None
    s["p95_cap_exceeded"] = lat["p95"] is None or lat["p95"] > args.p95_cap
    return s


async def main_async(args):
    reqs, meta = cc.load_requests(args.requests)
    if args.policy not in meta["policies"]:
        raise SystemExit(f"unknown policy {args.policy}; choose from {meta['policies']}")
    urls = cc.replica_urls(args)
    share = {p: float(np.mean([e["policy"][args.policy] == p for e in reqs])) for p in cc.PATHS}
    rates, rate_plan = plan_rates(args, reqs, share)
    out_dir = Path(args.out_dir) / (f"{args.policy}_r{len(urls)}" + (f"_{args.tag}" if args.tag else ""))
    if (out_dir / "summary.json").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already has results; use --tag or --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    images = cc.Images(Path(args.requests).parent, args.image_transport, args.server_image_dir)
    for e in reqs:   # encode every image this policy can send before the clock starts
        p = e["paths"][e["policy"][args.policy]]
        if p["image"]:
            images.url(p["image"])
    gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else list(range(len(urls)))
    meter = cc.EnergyMeter(gpu_ids)

    async with cc.make_session() as session:
        servers = await cc.server_info(session, urls)
        model = args.model or cc.served_model(servers)
        ctx = {"body": lambda e, p: cc.build_body(e["paths"][p]["messages"], images, model, meta["sampling"])}
        summary = {"policy": args.policy, "replicas": urls, "path_share": share, "rates_planned": rates,
                   "rate_plan": rate_plan, "seed": args.seed, "p95_cap_s": args.p95_cap,
                   "meta": cc.run_metadata(args, servers, meter.mode), "rates": {}, "stop_reason": None}
        print(f"{args.policy} on {len(urls)} replica(s), model {model}, energy {meter.mode}; rates {rates}", flush=True)

        # warm every replica on every path this policy uses (CUDA graphs, allocator), not recorded
        for u in urls:
            for p in [p for p in cc.PATHS if share[p] > 0]:
                e = next(x for x in reqs if x["policy"][args.policy] == p)
                for _ in range(args.server_warmup):
                    await cc.chat(session, u, ctx["body"](e, p), args.timeout)
        idle_w = None
        if args.idle_s > 0 and meter.mode != "none":
            e0 = meter.read()
            await asyncio.sleep(args.idle_s)
            idle_w = [x / args.idle_s for x in cc.energy_window(e0, meter.read())]
        summary["idle_power_w"] = idle_w

        mf = open(out_dir / "server_metrics.jsonl", "w", encoding="utf-8") if args.metrics else None
        with open(out_dir / "requests.jsonl", "w", encoding="utf-8") as rf:
            for rate in rates:
                n = n_for_rate(args, rate)
                warmup = max(args.min_warmup, int(round(args.warmup_frac * n)))
                await scrape(session, urls, mf, rate, "start")
                e0 = meter.read()
                rows, info = await run_rate(session, urls, reqs, args.policy, rate, n, warmup, args, ctx)
                de = cc.energy_window(e0, meter.read())
                await scrape(session, urls, mf, rate, "end")
                for r in rows:
                    rf.write(json.dumps(r, ensure_ascii=False) + "\n")
                rf.flush()
                s = summarize(rows, info, rate, n, warmup, de, idle_w, args)
                summary["rates"][str(rate)] = s
                print(f"rate {rate:7.3f}/s n={n:4d} p50={s['latency_s']['p50'] or 0:7.3f}s "
                      f"p95={s['latency_s']['p95'] or 0:7.3f}s p99={s['latency_s']['p99'] or 0:7.3f}s "
                      f"thr={s['throughput_rps']:.3f}/s err={s['n_errors']} aborted={s['aborted']} "
                      f"replicas={[v['n'] for v in s['per_replica'].values()]}", flush=True)
                reason = ("aborted" if s["aborted"] else "errors" if s["n_errors"] > args.max_errors
                          else "p95_cap" if s["p95_cap_exceeded"] else None)
                if reason:
                    summary["stop_reason"] = f"{reason} at rate {rate}"
                cc.write_json(out_dir / "summary.json", summary)
                if reason:
                    break
                await asyncio.sleep(args.pause)
        if mf:
            mf.close()
        summary["stop_reason"] = summary["stop_reason"] or "all rates done"
        cc.write_json(out_dir / "summary.json", summary)
    meter.close()
    print(f"wrote {out_dir}  ({summary['stop_reason']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cc.add_server_args(ap)
    ap.add_argument("--policy", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rates", help="comma-separated arrival rates (req/s over all replicas)")
    g.add_argument("--rate-fractions", help="comma-separated fractions of the pilot-estimated capacity x replicas")
    ap.add_argument("--pilot", default=str(cc.HERE / "results" / "pilot.json"))
    ap.add_argument("--seed", type=int, default=cc.SEED)
    ap.add_argument("--n-requests", type=int, default=400, help="arrivals per rate (upper bound)")
    ap.add_argument("--min-requests", type=int, default=150)
    ap.add_argument("--max-rate-seconds", type=float, default=900, help="n = clamp(rate * this, min, n-requests)")
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--min-warmup", type=int, default=10)
    ap.add_argument("--p95-cap", type=float, default=30.0)
    ap.add_argument("--max-errors", type=int, default=0)
    ap.add_argument("--abort-latency", type=float, default=120.0)
    ap.add_argument("--server-warmup", type=int, default=2, help="unrecorded requests per replica and path")
    ap.add_argument("--idle-s", type=float, default=10.0, help="idle-power measurement before the first rate")
    ap.add_argument("--pause", type=float, default=5.0, help="seconds between rates")
    ap.add_argument("--no-metrics", dest="metrics", action="store_false", help="do not scrape vLLM /metrics")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", default=str(cc.HERE / "results" / "load"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
