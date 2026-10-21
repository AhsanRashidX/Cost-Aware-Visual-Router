"""Mock OpenAI-compatible chat server for testing pilot.py / loadgen.py without a GPU (stdlib + aiohttp).

Endpoints as in vLLM: GET /health, /v1/models, /version, /metrics; POST /v1/chat/completions (streaming with
stream_options.include_usage, or not). Every request is validated like the real setup expects it (model name,
system + user messages, at most one image part given as a data:image/png;base64 URL holding a valid PNG or a
file:// URL, temperature 0, max_tokens, stop_token_ids); invalid requests get HTTP 400.
The path is inferred from the content (image part -> visual/hybrid, "Document text:" -> text/hybrid). A toy
continuous-batching engine serves it: at most --max-num-seqs requests run at once (the rest wait), and a request's
service time = base time of its path x (1 + --batch-slowdown x (running - 1)) x --scale, split into prefill (TTFT)
and per-token decode. prompt_tokens = text characters / 4 + the image's 28x28 grid (from the PNG header).
The answer is a short fixed string; --fail-rate returns HTTP 500 for that share of requests.
Unknown (vLLM) arguments are ignored, so serve.sh can start it in native mode for a dry run:
  NATIVE_CMD="python3 mock_server.py" MODE=native bash serve.sh 2
Usage: python mock_server.py --port 8000 [--scale 1.0] [--max-num-seqs 16]
"""
import argparse
import asyncio
import base64
import json
import random
import struct
import time

from aiohttp import web

BASE_S = {"text": 0.43, "visual": 0.57, "hybrid": 0.96}   # 4090 mean generate_s per path
PREFILL_SHARE = 0.7
ANSWER = ["mock", " answer", " 42"]


class Engine:
    def __init__(self, args):
        self.args, self.running, self.served = args, 0, 0
        self.sem = asyncio.Semaphore(args.max_num_seqs)
        self.rng = random.Random(1234 + args.port)


def check_request(body, model):
    if body.get("model") != model:
        return f"model {body.get('model')!r} not served"
    if body.get("temperature") != 0.0 or not isinstance(body.get("max_tokens"), int):
        return "expected temperature 0 and integer max_tokens"
    if not isinstance(body.get("stop_token_ids"), list):
        return "expected stop_token_ids"
    msgs = body.get("messages")
    if not (isinstance(msgs, list) and len(msgs) == 2 and msgs[0]["role"] == "system" and msgs[1]["role"] == "user"):
        return "expected [system, user] messages"
    images, texts = [], []
    for m in msgs:
        for c in m["content"]:
            if c["type"] == "text":
                texts.append(c["text"])
            elif c["type"] == "image_url":
                images.append(c["image_url"]["url"])
            else:
                return f"unexpected content part {c['type']}"
    if len(images) > 1:
        return "more than one image (server limit is 1)"
    grid = 0
    for url in images:
        if url.startswith("data:image/png;base64,"):
            raw = base64.b64decode(url.split(",", 1)[1])
            if raw[:8] != b"\x89PNG\r\n\x1a\n":
                return "image is not a PNG"
            w, h = struct.unpack(">II", raw[16:24])
            if w % 28 or h % 28:
                return f"image {w}x{h} not pre-resized to multiples of 28"
            grid = (w // 28) * (h // 28)
        elif not url.startswith("file://"):
            return "image URL must be data:image/png;base64 or file://"
    has_text_ctx = any("Document text:" in t for t in texts)
    path = ("hybrid" if has_text_ctx else "visual") if images else "text"
    return {"path": path, "prompt_tokens": sum(len(t) for t in texts) // 4 + grid + 20}


def chunk(cid, model, delta=None, finish=None, usage=None):
    obj = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
           "choices": [] if usage else [{"index": 0, "delta": delta or {}, "logprobs": None, "finish_reason": finish}]}
    if usage:
        obj["usage"] = usage
    return f"data: {json.dumps(obj)}\n\n".encode()


async def completions(request):
    eng, args = request.app["engine"], request.app["args"]
    body = await request.json()
    info = check_request(body, args.model)
    if isinstance(info, str):
        return web.json_response({"object": "error", "message": info, "code": 400}, status=400)
    if eng.rng.random() < args.fail_rate:
        return web.json_response({"object": "error", "message": "injected failure", "code": 500}, status=500)
    async with eng.sem:
        eng.running += 1
        service = BASE_S[info["path"]] * (1 + args.batch_slowdown * (eng.running - 1)) * args.scale
        service *= eng.rng.uniform(0.8, 1.2)
        n_tok = min(len(ANSWER), body["max_tokens"])
        usage = {"prompt_tokens": info["prompt_tokens"], "completion_tokens": n_tok + 1,
                 "total_tokens": info["prompt_tokens"] + n_tok + 1}
        cid = f"chatcmpl-mock-{args.port}-{eng.served}"
        eng.served += 1
        try:
            if not body.get("stream"):
                await asyncio.sleep(service)
                return web.json_response({"id": cid, "object": "chat.completion", "model": args.model,
                                          "choices": [{"index": 0, "finish_reason": "stop",
                                                       "message": {"role": "assistant", "content": "".join(ANSWER[:n_tok])}}],
                                          "usage": usage})
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await asyncio.sleep(service * PREFILL_SHARE)
            await resp.write(chunk(cid, args.model, {"role": "assistant", "content": ""}))
            for piece in ANSWER[:n_tok]:
                await resp.write(chunk(cid, args.model, {"content": piece}))
                await asyncio.sleep(service * (1 - PREFILL_SHARE) / n_tok)
            await resp.write(chunk(cid, args.model, {"content": ""}, finish="stop"))
            if (body.get("stream_options") or {}).get("include_usage"):
                await resp.write(chunk(cid, args.model, usage=usage))
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        finally:
            eng.running -= 1


async def models(request):
    a = request.app["args"]
    return web.json_response({"object": "list", "data": [{"id": a.model, "object": "model", "owned_by": "mock",
                                                          "root": a.model, "max_model_len": 8192}]})


async def health(request):
    return web.Response(text="")


async def server_version(request):
    return web.json_response({"version": "mock-0.0"})


async def metrics(request):
    eng = request.app["engine"]
    return web.Response(text=f"# HELP vllm:num_requests_running mock\nvllm:num_requests_running {eng.running}\n"
                             f"vllm:request_success_total {eng.served}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="qwen2.5-vl-7b-awq")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies every service time")
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--batch-slowdown", type=float, default=0.15, help="service-time growth per extra running request")
    ap.add_argument("--fail-rate", type=float, default=0.0)
    ap.add_argument("--served-model-name", default=None, help="as in vLLM; overrides --model")
    args, ignored = ap.parse_known_args()   # vLLM flags are accepted and ignored, so serve.sh can launch the mock
    args.model = args.served_model_name or args.model
    if ignored:
        print(f"mock: ignoring {len(ignored)} vLLM arguments", flush=True)
    app = web.Application(client_max_size=64 * 2**20)
    app["args"] = args

    async def start(app_):
        app_["engine"] = Engine(args)
    app.on_startup.append(start)
    app.add_routes([web.get("/health", health), web.get("/version", server_version),
                    web.get("/v1/models", models), web.get("/metrics", metrics),
                    web.post("/v1/chat/completions", completions)])
    web.run_app(app, host=args.host, port=args.port, print=lambda *_: print(f"mock server on :{args.port}", flush=True))


if __name__ == "__main__":
    main()
