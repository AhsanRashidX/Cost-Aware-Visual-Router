"""Pilot on the serving setup: correctness, service time vs concurrency, and (optionally) per-query service times.

Stages (--stages, comma-separated):
  correctness  per path, the first --n-correct test queries (requests.jsonl order) one at a time on replica 0,
               temperature 0, 32 new tokens: answer, latency, TTFT, prompt/completion tokens; ANLS/EM vs gold,
               agreement with the 4090 answers, and prompt_tokens vs the 4090 input_tokens (equal when the server
               rendered generate.py's prompt and the same image grid)
  sweep        per path, closed loop at each --concurrency level on replica 0 (each finished request is replaced
               at once): throughput and latency per level, the serving counterpart of bench_batching.py's batching
               curve (simulator input). Every level uses the same seeded query order; the first `c` requests of a
               level are warm-up. steady_throughput = (measured sends - 1) / (last send - first measured send),
               valid because in a closed loop every send after warm-up follows a completion.
  service      every test query x path at concurrency 1, spread over all replicas (one request in flight per
               replica): per-query service times, the counterpart of path_outputs' generate_s. Resumable.
Outputs (results/): pilot.json (summaries per stage, merged across invocations, plus run metadata),
pilot_requests.jsonl (one row per correctness/sweep request), service_times.jsonl (service stage).
Usage:  python pilot.py --replicas 1                       (correctness + sweep)
        python pilot.py --replicas 3 --stages service      (per-query service times on 3 replicas)
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

import client_common as cc
from vqa_metrics import anls, exact_match, normalize


def row_from(res, e, path, stage, **extra):
    ref = e["paths"][path]["ref_4090"]
    ok = res["error"] is None
    return {"stage": stage, **extra, "i": e["i"], "qid": e["qid"], "path": path,
            "latency_s": res["t_end"] - res["t_send"],
            "ttft_s": (res["t_first_token"] - res["t_send"]) if res["t_first_token"] else None,
            "t_send": res["t_send"], "t_end": res["t_end"],
            "prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
            "answer": res["answer"], "finish_reason": res["finish_reason"], "error": res["error"],
            "anls": anls(res["answer"], e["answers"]) if ok else None,
            "em": exact_match(res["answer"], e["answers"]) if ok else None,
            "same_as_4090": (normalize(res["answer"]) == normalize(ref["answer"])) if ok else None,
            "anls_vs_4090_answer": anls(res["answer"], [ref["answer"]]) if ok else None,
            "ref_input_tokens": ref["input_tokens"], "ref_generate_s": ref["generate_s"], "ref_anls": ref["anls"]}


async def warm_up(session, urls, reqs, paths, ctx, n=2):
    for u in urls:
        for p in paths:
            for e in reqs[:n]:
                await cc.chat(session, u, ctx["body"](e, p), ctx["timeout"])


async def stage_correctness(session, url, reqs, paths, n, ctx, log):
    out = {}
    for p in paths:
        rows = []
        for e in reqs[:n]:
            res = await cc.chat(session, url, ctx["body"](e, p), ctx["timeout"])
            rows.append(row_from(res, e, p, "correctness"))
        log(rows)
        good = [r for r in rows if r["error"] is None]
        tok_eq = [r for r in good if r["prompt_tokens"] == r["ref_input_tokens"]]
        out[p] = {
            "n": len(rows), "errors": len(rows) - len(good),
            "error_examples": [r["error"] for r in rows if r["error"]][:3],
            "anls_t4": cc.mean([r["anls"] for r in good]), "em_t4": cc.mean([r["em"] for r in good]),
            "anls_4090_same_queries": cc.mean([r["ref_anls"] for r in rows]),
            "answer_identical_to_4090": cc.mean([float(r["same_as_4090"]) for r in good]),
            "anls_vs_4090_answer": cc.mean([r["anls_vs_4090_answer"] for r in good]),
            "prompt_tokens_equal_4090": f"{len(tok_eq)}/{len(good)}",
            "prompt_token_mismatch_examples": [{"qid": r["qid"], "t4": r["prompt_tokens"], "ref": r["ref_input_tokens"]}
                                               for r in good if r["prompt_tokens"] != r["ref_input_tokens"]][:5],
            "completion_tokens_mean": cc.mean([r["completion_tokens"] for r in good]),
            "finish_reasons": {k: sum(r["finish_reason"] == k for r in good) for k in {r["finish_reason"] for r in good}},
            "latency_s": cc.lat_stats([r["latency_s"] for r in good]),
            "ttft_s": cc.lat_stats([r["ttft_s"] for r in good]),
            "ref_generate_s_mean_4090": cc.mean([r["ref_generate_s"] for r in rows]),
        }
        s = out[p]
        print(f"[correctness] {p:6s} n={s['n']} err={s['errors']} ANLS t4={s['anls_t4'] or 0:.3f} "
              f"4090={s['anls_4090_same_queries']:.3f} same-answer={s['answer_identical_to_4090'] or 0:.2f} "
              f"prompt-tokens-equal={s['prompt_tokens_equal_4090']} lat p50={s['latency_s']['p50'] or 0:.3f}s", flush=True)
    return out


async def closed_loop(session, url, bodies, c, timeout):
    results = [None] * len(bodies)
    it = iter(range(len(bodies)))

    async def worker():
        for j in it:   # shared iterator: a worker takes the next request as soon as its previous one finished
            results[j] = await cc.chat(session, url, bodies[j], timeout)

    await asyncio.gather(*[worker() for _ in range(c)])
    return results


async def stage_sweep(session, url, reqs, paths, levels, n_min, n_mult, ctx, log):
    order = np.random.default_rng(cc.SEED).permutation(len(reqs))
    out = {}
    for p in paths:
        out[p] = {}
        for c in levels:
            n_meas = max(n_min, n_mult * c)
            sel = [reqs[order[j % len(order)]] for j in range(c + n_meas)]
            res = await closed_loop(session, url, [ctx["body"](e, p) for e in sel], c, ctx["timeout"])
            rows = [row_from(r, e, p, "sweep", concurrency=c, k=k, warmup=k < c) for k, (r, e) in enumerate(zip(res, sel))]
            log(rows)
            meas = [r for r in rows if not r["warmup"]]
            good = [r for r in meas if r["error"] is None]
            sends = sorted(r["t_send"] for r in meas)
            window = max(r["t_end"] for r in meas) - sends[0]
            out[p][str(c)] = {
                "concurrency": c, "n_measured": len(meas), "errors": len(meas) - len(good),
                "steady_throughput_rps": (len(sends) - 1) / (sends[-1] - sends[0]) if sends[-1] > sends[0] else None,
                "window_throughput_rps": len(good) / window if window > 0 else None,
                "latency_s": cc.lat_stats([r["latency_s"] for r in good]),
                "ttft_s": cc.lat_stats([r["ttft_s"] for r in good]),
                "prompt_tokens_mean": cc.mean([r["prompt_tokens"] for r in good]),
                "completion_tokens_mean": cc.mean([r["completion_tokens"] for r in good]),
            }
            s = out[p][str(c)]
            print(f"[sweep] {p:6s} c={c:2d} thr={s['steady_throughput_rps'] or 0:.3f} req/s "
                  f"lat mean={s['latency_s']['mean'] or 0:.3f}s p95={s['latency_s']['p95'] or 0:.3f}s err={s['errors']}",
                  flush=True)
        base = out[p][str(levels[0])]["latency_s"]["mean"]
        for c in levels:   # T(c)/T(first level): the simulator's batch_curve analogue
            m = out[p][str(c)]["latency_s"]["mean"]
            out[p][str(c)]["latency_ratio_vs_first_level"] = (m / base) if (m and base) else None
        out[p]["max_steady_throughput_rps"] = max((out[p][str(c)]["steady_throughput_rps"] or 0) for c in levels)
    return out


async def stage_service(session, urls, reqs, paths, ctx, out_path):
    done = set()
    if out_path.exists():
        done = {(r["i"], r["path"]) for r in map(json.loads, open(out_path, encoding="utf-8")) if r["error"] is None}
    todo = [(e, p) for e in reqs for p in paths if (e["i"], p) not in done]
    print(f"[service] {len(todo)} requests to run ({len(done)} already done) on {len(urls)} replica(s)", flush=True)
    it = iter(todo)
    t0, n_done = time.perf_counter(), [0]
    with open(out_path, "a", encoding="utf-8") as f:
        async def worker(r, url):
            for e, p in it:
                res = await cc.chat(session, url, ctx["body"](e, p), ctx["timeout"])
                f.write(json.dumps(row_from(res, e, p, "service", replica=r), ensure_ascii=False) + "\n")
                n_done[0] += 1
                if n_done[0] % 100 == 0:
                    f.flush()
                    el = time.perf_counter() - t0
                    print(f"[service] {n_done[0]}/{len(todo)} ({el / n_done[0] * (len(todo) - n_done[0]) / 60:.0f} min left)",
                          flush=True)
        await asyncio.gather(*[worker(r, u) for r, u in enumerate(urls)])
    all_rows = [json.loads(line) for line in open(out_path, encoding="utf-8")]
    out = {"file": out_path.name, "errors_in_file": sum(r["error"] is not None for r in all_rows)}
    for p in paths:
        rows = [r for r in all_rows if r["error"] is None and r["path"] == p]
        out[p] = {"n": len(rows), "latency_s": cc.lat_stats([r["latency_s"] for r in rows]),
                  "ttft_s": cc.lat_stats([r["ttft_s"] for r in rows]),
                  "ref_generate_s_mean_4090": cc.mean([r["ref_generate_s"] for r in rows]),
                  "anls_t4": cc.mean([r["anls"] for r in rows]), "anls_4090": cc.mean([r["ref_anls"] for r in rows]),
                  "answer_identical_to_4090": cc.mean([float(r["same_as_4090"]) for r in rows])}
    return out


async def main_async(args):
    reqs, meta = cc.load_requests(args.requests)
    paths = args.paths.split(",")
    urls = cc.replica_urls(args)
    images = cc.Images(Path(args.requests).parent, args.image_transport, args.server_image_dir)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")
    gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else list(range(len(urls)))
    meter = cc.EnergyMeter(gpu_ids)

    async with cc.make_session() as session:
        servers = await cc.server_info(session, urls)
        model = args.model or cc.served_model(servers)
        ctx = {"timeout": args.timeout,
               "body": lambda e, p: cc.build_body(e["paths"][p]["messages"], images, model, meta["sampling"])}
        result = json.load(open(args.out, encoding="utf-8")) if Path(args.out).exists() else {"runs": []}
        result["runs"].append(cc.run_metadata(args, servers, meter.mode))
        result["requests_meta"] = {k: meta[k] for k in ("n_requests", "sampling", "min_pixels", "max_pixels",
                                                         "model_4090", "model_t4", "split")}

        def save():
            cc.write_json(args.out, result)

        log_f = open(out_dir / "pilot_requests.jsonl", "a", encoding="utf-8")

        def log(rows):
            for r in rows:
                log_f.write(json.dumps({"run": len(result["runs"]) - 1, **r}, ensure_ascii=False) + "\n")
            log_f.flush()

        print(f"servers: {[s['url'] for s in servers]}  model: {model}  vLLM: {servers[0]['version']}  "
              f"energy: {meter.mode}", flush=True)
        if any(s in stages for s in ("correctness", "sweep")):
            await warm_up(session, urls[:1], reqs, paths, ctx)
        for st in stages:
            e0, t0 = meter.read(), time.perf_counter()
            if st == "correctness":
                res = await stage_correctness(session, urls[0], reqs, paths, args.n_correct, ctx, log)
            elif st == "sweep":
                levels = [int(c) for c in args.concurrency.split(",")]
                res = await stage_sweep(session, urls[0], reqs, paths, levels, args.sweep_min, args.sweep_mult, ctx, log)
            elif st == "service":
                await warm_up(session, urls, reqs, paths, ctx, n=1)
                res = await stage_service(session, urls, reqs[:args.service_limit] if args.service_limit else reqs,
                                          paths, ctx, out_dir / "service_times.jsonl")
            else:
                raise SystemExit(f"unknown stage {st}")
            de = cc.energy_window(e0, meter.read())
            res["_stage"] = {"run": len(result["runs"]) - 1, "wall_s": time.perf_counter() - t0,
                             "energy_j_per_gpu": de, "energy_mode": meter.mode, "replicas": urls if st == "service" else urls[:1]}
            result[st] = res
            save()
        log_f.close()
    meter.close()
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cc.add_server_args(ap)
    ap.add_argument("--stages", default="correctness,sweep")
    ap.add_argument("--paths", default="text,visual,hybrid")
    ap.add_argument("--n-correct", type=int, default=50)
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--sweep-min", type=int, default=24, help="measured requests per level: max(sweep-min, sweep-mult*c)")
    ap.add_argument("--sweep-mult", type=int, default=4)
    ap.add_argument("--service-limit", type=int, default=0, help="service stage: first N queries only (0 = all)")
    ap.add_argument("--model", default=None, help="served model name (default: first id from /v1/models)")
    ap.add_argument("--out", default=str(cc.HERE / "results" / "pilot.json"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
