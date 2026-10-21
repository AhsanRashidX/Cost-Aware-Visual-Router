# Real multi-GPU serving load test (2 or 3 Tesla T4)

The paper's "GPUs needed at p95 <= 2 s" result comes from a discrete-event simulation (`simulate.py`). This package
measures the real thing: one vLLM OpenAI-compatible server per GPU (Qwen2.5-VL-7B-Instruct-AWQ, fp16, 1-3
replicas), driven by open-loop Poisson traffic per routing policy, recording p50/p95/p99 latency, throughput and
energy against the arrival rate. The pilot also measures T4 service times so the simulator can be re-run with them
and compared with the measurement.

The requests are the 1,606 test queries of split 1 (seed 1234) with the exact prompts of the 4090 study (same system
prompt, OCR text, retrieved pages, page image pre-resized to the same size, so the same number of visual tokens).
Only generation is served; retrieval and the router ran on the workstation and their choices are in the requests.

## Contents

| File | Purpose |
|---|---|
| `config.sh` | all settings: launch mode, image, model, CPU pinning, vLLM flags (version-sensitive ones marked) |
| `serve.sh N` / `stop.sh` | start N replicas (replica i on GPU i, port 8000+i) and wait until healthy / stop them |
| `client.sh` | run a client script inside the vLLM image (docker modes) or with `$PYTHON` (native) |
| `download_model.sh` | download the model into `$HF_CACHE`, print the commit hash to pin and the size |
| `pilot.py` | correctness check, service time vs concurrency, optional per-query service times |
| `loadgen.py` | open-loop Poisson load test of one policy on N replicas |
| `run_matrix.sh` | all replica counts x policies, unattended |
| `requests.jsonl`, `images/` | request set (messages per path, gold answers, 4090 answers, policy choices) and page images |
| `requests_meta.json`, `policies_summary.json` | sampling settings, verification results; path mix per policy |
| `mock_server.py`, `client_common.py`, `vqa_metrics.py` | test server without GPU; shared client code; ANLS/EM |

## Launch modes (`MODE` in config.sh)

- `docker`: Docker >= 19.03 with nvidia-container-toolkit (`--gpus '"device=i"'`). Default.
- `docker-legacy`: Docker 18.06 with nvidia-container-runtime 2.0 (`--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=i`,
  plus `--security-opt seccomp=unconfined`, because old Docker seccomp profiles break images with glibc >= 2.34).
- `native`: no Docker; vLLM pip-installed in `$PYTHON` (for example a Kaggle notebook with 2x T4). Replicas run as
  background processes with `CUDA_VISIBLE_DEVICES=i`, logs in `logs/replica_<i>.log`, PIDs in `run/`.

In the docker modes the host needs no Python: the client scripts run inside the vLLM image (`client.sh`). In native
mode, `bash client.sh pilot.py ...` and `python3 pilot.py ...` are equivalent.

CPU pinning (`CPU_PIN`) is on by default in the docker modes: servers on NUMA node 1 (CPUs 16-31,48-63, where all
three T4s attach), client on node 0. It is off by default in native mode. Check your topology with
`nvidia-smi topo -m` (column "CPU Affinity") and adjust `SERVER_CPUS` / `CLIENT_CPUS`, or set `CPU_PIN=0`.

## 1. Copy and check the package

On the workstation: `scp -r t4_package user@server:~/` (about 0.26 GB). On the server:
```
cd ~/t4_package
sha1sum -c SHA1SUMS --quiet && echo "package OK"
```

## 2. Prerequisites and configuration

Edit `config.sh`: set `MODE`, pin `VLLM_IMAGE` (docker modes), and set `HF_CACHE` to a disk with at least 10 GB free.
On a 2-GPU machine everything below works with 1-2 replicas (`REPLICA_LIST="1 2"` for the matrix).

Check that the GPUs are visible from a container (or from Python in native mode):
```
source config.sh
docker pull "$VLLM_IMAGE"                       # docker modes; several GB
# docker:
docker run --rm --gpus all --entrypoint nvidia-smi "$VLLM_IMAGE"
# docker-legacy:
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --security-opt seccomp=unconfined --entrypoint nvidia-smi "$VLLM_IMAGE"
# native (pin the version you choose):
#   pip install "vllm==<version>"  &&  python3 -c "import vllm, torch; print(vllm.__version__, torch.cuda.get_device_name(0))"
```
The driver must support the image's CUDA version (`nvidia-smi` shows the highest CUDA version the driver supports).

## 3. Download the model

```
bash download_model.sh
```
It prints `MODEL_REVISION=<commit hash>`: put that hash into `config.sh` so every run uses the same weights. The
final line shows the size on disk (roughly 6-7 GB for the AWQ checkpoint; please note the exact number).

## 4. Start one replica and run the pilot

```
bash serve.sh 1                                 # first start compiles/captures CUDA graphs: a few minutes
bash client.sh pilot.py --replicas 1            # about 20-40 min
```
Output: `results/pilot.json`, `results/pilot_requests.jsonl`. Check the printed lines:
- `prompt-tokens-equal=50/50` for every path. This compares vLLM's prompt token count with the 4090 run's; equality
  means the server rendered the same prompt and the same image grid. If not, stop and send `results/` and `logs/`
  (usual causes: chat-template content format, image resized differently, a different chat template in the AWQ repo).
- `err=0`; ANLS t4 close to the 4090 value (int4 AWQ vs bf16, so a small difference is expected).
- the sweep lines: throughput should rise with concurrency up to 16.

## 5. Per-query service times (recommended)

Every test query on every path at concurrency 1, spread over all replicas (one request in flight per replica). This
is the T4 counterpart of the 4090 per-query generation times that drive the simulator.
```
bash stop.sh
bash serve.sh 3                                 # 2 on a 2-GPU machine
bash client.sh pilot.py --replicas 3 --stages service      # 4,818 requests; about 30-60 min with 3 replicas
```
Output: `results/service_times.jsonl` (resumable: rerun the same command after an interruption).

## 6. Load tests

All replica counts x policies {always_hybrid, always_visual, cavr_ce_valsel}, unattended (it survives a closed SSH
session):
```
bash stop.sh
nohup bash run_matrix.sh > logs/matrix.log 2>&1 &
tail -f logs/matrix.log
```
On 2 GPUs: `REPLICA_LIST="1 2" nohup bash run_matrix.sh > logs/matrix.log 2>&1 &`.

Rates are fractions of each policy's capacity estimated from the pilot sweep (`FRACTIONS`, default
`0.15,0.3,0.5,0.7,0.8,0.9,1.0,1.1` x replicas x capacity per replica). Per rate, at most 400 arrivals (fewer at low
rates: about 15 minutes per rate, at least 150 arrivals); the first 10% are warm-up and not counted. A run stops
after the first rate whose p95 exceeds 30 s, that has an error, or that is aborted because a request waited more than
120 s. Expect roughly 4-8 hours for the whole matrix. One run by hand:
```
bash serve.sh 2
bash client.sh loadgen.py --policy cavr_ce_valsel --replicas 2 --rate-fractions 0.15,0.3,0.5,0.7,0.8,0.9,1.0,1.1
bash client.sh loadgen.py --policy always_hybrid --replicas 2 --rates 0.2,0.4,0.8 --tag fixed   # absolute rates
```
Output per run: `results/load/<policy>_r<N>/` with `summary.json`, `requests.jsonl` (every request, including the
arrival trace) and `server_metrics.jsonl` (vLLM `/metrics` before and after each rate).

If the pilot shows a concurrency-1 p95 above 2 s for a path, the 2 s target cannot be met for that path at any rate
on a T4; the curves still give the comparison with the simulator at other targets.

## 7. Send back

```
tar czf t4_results.tar.gz results logs
```
That is: `results/` (pilot.json, pilot_requests.jsonl, service_times.jsonl, load/*, serve_info.txt) and `logs/`
(server logs, matrix.log). Every result file records the hardware, driver/CUDA, CPU, RAM, vLLM version, model,
revision, CLI arguments and launch commands.

## Kaggle (native mode, 2x T4)

Upload `t4_package` as a dataset, then in the notebook copy it to a writable place and run the same steps:
```
!cp -r /kaggle/input/<dataset>/t4_package /kaggle/working/ && cd /kaggle/working/t4_package && pip install -q "vllm==<version>"
!cd /kaggle/working/t4_package && MODE=native bash download_model.sh
!cd /kaggle/working/t4_package && MODE=native bash serve.sh 1 && MODE=native bash client.sh pilot.py --replicas 1
```
Kaggle machines have few CPU cores, which vLLM's image preprocessing and the client share; note this when comparing
with the server results. Sessions are time-limited: run the matrix in parts (`REPLICA_LIST`, `POLICIES`).

## Troubleshooting

- **Threads fail to start** (`can't start new thread`, `Operation not permitted`, `clone3`) in docker mode on an old
  kernel: set `DOCKER_EXTRA=(--security-opt seccomp=unconfined)` in `config.sh` (docker-legacy always adds it).
- **docker-legacy refuses the image** (`unsatisfied condition: cuda>=12...`): `DOCKER_EXTRA=(-e NVIDIA_DISABLE_REQUIRE=1)`;
  this only helps if the driver really supports the image's CUDA version (Tesla GPUs can also use the CUDA forward-
  compatibility libraries shipped in CUDA images).
- **`unrecognized arguments` or a flag error at start-up**: the vLLM version differs from the flag syntax in
  `config.sh`. Adjust the arrays in the "version-sensitive" block (`LIMIT_MM_ARGS`, `CONTENT_FORMAT_ARGS`,
  `GEN_CONFIG_ARGS`, `PREFIX_CACHE_ARGS`, `QUANT_ARGS`, `LOG_ARGS`) and note what you changed.
- **Attention-backend errors on compute capability 7.5**: try `ATTN_BACKEND=FLASHINFER` or `TRITON_ATTN` (or
  `XFORMERS` on older releases).
- **CUDA-graph capture fails or out of memory at start-up**: `EXTRA_ARGS=(--enforce-eager)`. Note it: it changes
  latency.
- **A replica dies**: `serve.sh` prints the last log lines; full logs are in `logs/` (docker logs are saved by
  `stop.sh`).
- **Ports busy**: change `BASE_PORT` (the clients take `--base-port`).
- **Dry run without GPU or vLLM (native mode)**: `NATIVE_CMD="python3 mock_server.py" MODE=native bash serve.sh 2`,
  then the pilot/loadgen commands with `MODE=native`; `bash stop.sh` afterwards.
