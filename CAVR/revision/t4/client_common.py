"""Shared client code for pilot.py and loadgen.py.

Runs inside the vLLM container or as plain Python next to a native vLLM install: only the stdlib, aiohttp and numpy
are required; NVML (pynvml / nvidia-ml-py, or vLLM's vendored copy) is optional, with nvidia-smi power sampling and
then "no energy" as fallbacks.
Provides: request-set loading, image references -> data:/file:// URLs, the OpenAI chat request body, one streaming
chat call (TTFT + completion time + usage), replica URLs and server info, an energy meter, run metadata (hardware,
software, server, package provenance, CLI args) and small statistics helpers.
"""
import base64
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import aiohttp
import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 1234
PATHS = ("text", "visual", "hybrid")


# ---------------------------------------------------------------- request set
def load_requests(path):
    path = Path(path)
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    meta = json.load(open(path.with_name("requests_meta.json"), encoding="utf-8"))
    return rows, meta


class Images:
    """Turns the package's relative image references into URLs the server can open.

    data: base64 data URL built from the local file (works with every vLLM version; cached in memory).
    file: file://<server_dir>/<ref>; the server must see the package images at that path and be started with
          --allowed-local-media-path (serve.sh does this when IMAGE_TRANSPORT=file).
    """

    def __init__(self, base_dir, mode="data", server_dir=None):
        self.base, self.mode, self.server_dir, self.cache = Path(base_dir), mode, server_dir, {}

    def url(self, ref):
        if self.mode == "file":
            return f"file://{self.server_dir.rstrip('/')}/{ref}"
        if ref not in self.cache:
            data = (self.base / ref).read_bytes()
            mime = "image/png" if ref.endswith(".png") else "image/jpeg"
            self.cache[ref] = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
        return self.cache[ref]


def build_body(messages, images, model, sampling, stream=True):
    msgs = []
    for m in messages:
        content = [{"type": "image_url", "image_url": {"url": images.url(c["image_url"]["url"])}}
                   if c["type"] == "image_url" else c for c in m["content"]]
        msgs.append({"role": m["role"], "content": content})
    body = {"model": model, "messages": msgs, "stream": stream, **sampling}
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


# ---------------------------------------------------------------- HTTP
def replica_urls(args):
    if getattr(args, "urls", None):
        return [u.rstrip("/") for u in args.urls.split(",")]
    return [f"http://{args.host}:{args.base_port + i}" for i in range(args.replicas)]


def add_server_args(ap):
    ap.add_argument("--replicas", type=int, default=1, help="replica i at http://HOST:BASE_PORT+i")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--base-port", type=int, default=8000)
    ap.add_argument("--urls", default=None, help="comma-separated base URLs (overrides --replicas/--host/--base-port)")
    ap.add_argument("--requests", default=str(HERE / "requests.jsonl"))
    ap.add_argument("--image-transport", choices=["data", "file"], default=os.environ.get("IMAGE_TRANSPORT", "data"))
    ap.add_argument("--server-image-dir", default=os.environ.get("SERVER_MEDIA_DIR", "/data/t4"),
                    help="where the server sees this package folder (file transport only)")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request timeout (s)")
    ap.add_argument("--gpu-ids", default=None, help="GPUs to meter (default 0..replicas-1)")


async def get_json(session, url, timeout=10):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            text = await r.text()
            try:
                return json.loads(text) if r.status == 200 else {"status": r.status, "body": text[:500]}
            except ValueError:
                return {"status": r.status, "body": text[:500]}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


async def get_text(session, url, timeout=10):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            return await r.text() if r.status == 200 else None
    except Exception:  # noqa: BLE001
        return None


async def server_info(session, urls):
    info = []
    for u in urls:
        models = await get_json(session, f"{u}/v1/models")
        info.append({"url": u, "version": await get_json(session, f"{u}/version"), "models": models})
    return info


def served_model(info):
    try:
        return info[0]["models"]["data"][0]["id"]
    except (KeyError, IndexError, TypeError):
        raise SystemExit(f"cannot read the served model from {info[0]['url']}/v1/models: {info[0]['models']}")


async def chat(session, url, body, timeout):
    """One streaming chat completion. Times are time.perf_counter() values.

    Returns t_send, t_first_chunk, t_first_token (first non-empty content delta), t_end, answer, finish_reason,
    prompt_tokens, completion_tokens, error (None on success)."""
    res = {"t_send": time.perf_counter(), "t_first_chunk": None, "t_first_token": None, "t_end": None,
           "answer": "", "finish_reason": None, "prompt_tokens": None, "completion_tokens": None, "error": None}
    parts = []
    try:
        async with session.post(f"{url}/v1/chat/completions", json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                res["error"] = f"HTTP {r.status}: {(await r.text())[:300]}"
            elif not body.get("stream"):
                obj = await r.json()
                res["t_first_chunk"] = res["t_first_token"] = time.perf_counter()
                parts.append(obj["choices"][0]["message"].get("content") or "")
                res["finish_reason"] = obj["choices"][0].get("finish_reason")
                usage = obj.get("usage") or {}
                res["prompt_tokens"], res["completion_tokens"] = usage.get("prompt_tokens"), usage.get("completion_tokens")
            else:
                async for raw in r.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    now = time.perf_counter()
                    obj = json.loads(data)
                    if "error" in obj and not obj.get("choices"):
                        res["error"] = f"stream error: {str(obj['error'])[:300]}"
                        break
                    if res["t_first_chunk"] is None:
                        res["t_first_chunk"] = now
                    for ch in obj.get("choices") or []:
                        piece = (ch.get("delta") or {}).get("content")
                        if piece:
                            if res["t_first_token"] is None:
                                res["t_first_token"] = now
                            parts.append(piece)
                        if ch.get("finish_reason"):
                            res["finish_reason"] = ch["finish_reason"]
                    usage = obj.get("usage")
                    if usage:
                        res["prompt_tokens"], res["completion_tokens"] = usage.get("prompt_tokens"), usage.get("completion_tokens")
    except Exception as e:  # noqa: BLE001  (timeouts, resets, cancellation by the load generator)
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    res["t_end"] = time.perf_counter()
    res["answer"] = "".join(parts).strip()   # generate.py strips the decoded answer
    return res


def make_session(limit=0):
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=limit, force_close=False),
                                 timeout=aiohttp.ClientTimeout(total=None))


# ---------------------------------------------------------------- energy
def _import_nvml():
    try:
        import pynvml
        return pynvml
    except ImportError:
        pass
    try:
        from vllm.third_party import pynvml  # vendored in newer vLLM releases
        return pynvml
    except Exception:  # noqa: BLE001
        return None


class EnergyMeter:
    """Cumulative energy (J) per metered GPU since start().

    nvml_counter : nvmlDeviceGetTotalEnergyConsumption (Volta and newer, incl. T4)
    nvml_power   : integrated nvmlDeviceGetPowerUsage, sampled every 100 ms
    nvidia_smi   : integrated `nvidia-smi --query-gpu=index,power.draw -lms 100`
    none         : nothing available (energy fields are null)
    """

    def __init__(self, gpu_ids, prefer=None):
        self.ids, self.mode, self.nv, self.handles = list(gpu_ids), "none", None, []
        self.acc = [0.0] * len(self.ids)
        self.lock, self.stop_evt, self.thread, self.proc = threading.Lock(), threading.Event(), None, None
        nv = _import_nvml()
        if nv is not None and prefer in (None, "nvml_counter", "nvml_power"):
            try:
                nv.nvmlInit()
                self.handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in self.ids]
                self.nv = nv
                try:
                    [nv.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles]
                    self.mode = "nvml_counter" if prefer != "nvml_power" else "nvml_power"
                except Exception:  # noqa: BLE001
                    self.mode = "nvml_power"
            except Exception:  # noqa: BLE001
                self.nv = None
        if self.mode == "none" and shutil.which("nvidia-smi") and prefer in (None, "nvidia_smi"):
            self.mode = "nvidia_smi"
        if self.mode == "nvml_counter":
            self.e0 = [self.nv.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles]
        elif self.mode == "nvml_power":
            self.thread = threading.Thread(target=self._nvml_loop, daemon=True)
            self.thread.start()
        elif self.mode == "nvidia_smi":
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=index,power.draw", "--format=csv,noheader,nounits", "-lms", "100"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.thread = threading.Thread(target=self._smi_loop, daemon=True)
            self.thread.start()

    def _nvml_loop(self):
        last = time.perf_counter()
        while not self.stop_evt.wait(0.1):
            now = time.perf_counter()
            watts = [self.nv.nvmlDeviceGetPowerUsage(h) / 1000 for h in self.handles]
            with self.lock:
                for k, w in enumerate(watts):
                    self.acc[k] += w * (now - last)
            last = now

    def _smi_loop(self):
        last = {}
        pos = {g: k for k, g in enumerate(self.ids)}
        for line in self.proc.stdout:
            try:
                idx, w = (x.strip() for x in line.split(","))
                idx, w = int(idx), float(w)
            except ValueError:
                continue
            now = time.perf_counter()
            if idx in pos:
                if idx in last:
                    with self.lock:
                        self.acc[pos[idx]] += w * (now - last[idx])
                last[idx] = now
            if self.stop_evt.is_set():
                break

    def read(self):
        """Cumulative joules per metered GPU (None if no energy source)."""
        if self.mode == "nvml_counter":
            return [(self.nv.nvmlDeviceGetTotalEnergyConsumption(h) - e) / 1000 for h, e in zip(self.handles, self.e0)]
        if self.mode in ("nvml_power", "nvidia_smi"):
            with self.lock:
                return list(self.acc)
        return None

    def power_w(self):
        """Instantaneous power per GPU (W), or None."""
        if self.nv is not None:
            try:
                return [self.nv.nvmlDeviceGetPowerUsage(h) / 1000 for h in self.handles]
            except Exception:  # noqa: BLE001
                return None
        return None

    def close(self):
        self.stop_evt.set()
        if self.proc is not None:
            self.proc.terminate()


def energy_window(e0, e1):
    if e0 is None or e1 is None:
        return None
    return [b - a for a, b in zip(e0, e1)]


# ---------------------------------------------------------------- metadata
def _run(cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:  # noqa: BLE001
        return f"failed: {e}"


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _ram_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    return None


def _pkg(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _read(path, limit=200000):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return None


def run_metadata(args, servers, energy_mode=None):
    """Hardware/software provenance written with every run."""
    smi = shutil.which("nvidia-smi")
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    git = _run(["git", "-C", str(HERE), "rev-parse", "HEAD"]).strip()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "argv": sys.argv, "args": vars(args),
        "python": sys.version, "platform": platform.platform(), "kernel": platform.release(),
        "os_release": _read("/etc/os-release"),
        "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "cpu_affinity": affinity, "ram_gb": _ram_gb(),
        "packages": {p: _pkg(p) for p in ("aiohttp", "numpy", "vllm", "torch", "transformers", "nvidia-ml-py",
                                          "pynvml", "flashinfer-python", "xformers")},
        "nvidia_smi": _run([smi]) if smi else None,
        "nvidia_smi_query": _run([smi, "--query-gpu=index,name,uuid,memory.total,driver_version,power.limit,"
                                       "clocks.max.sm,clocks.max.mem,pcie.link.gen.current,pcie.link.width.current",
                                  "--format=csv"]) if smi else None,
        "nvidia_smi_topo": _run([smi, "topo", "-m"]) if smi else None,
        "energy_mode": energy_mode,
        "env": {k: os.environ.get(k) for k in ("CAVR_MODE", "VLLM_IMAGE", "MODEL", "MODEL_REVISION", "SERVED_NAME",
                                               "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "IMAGE_TRANSPORT")},
        "servers": servers,
        "serve_info": _read(HERE / "results" / "serve_info.txt"),
        "package_info": json.loads(_read(HERE / "package_info.json") or "null"),
        "git_commit": git if len(git) == 40 else None,
    }


# ---------------------------------------------------------------- stats
def pct(xs, q):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, q)) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def lat_stats(xs):
    return {"n": len([x for x in xs if x is not None]), "mean": mean(xs), "p50": pct(xs, 50), "p90": pct(xs, 90),
            "p95": pct(xs, 95), "p99": pct(xs, 99), "max": pct(xs, 100)}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
