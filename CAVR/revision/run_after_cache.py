"""Run steps 4-9 in order once results/path_outputs.jsonl is complete. Stops at the first failure.

GPU benchmarks run one at a time so their timings aren't disturbed by each other.
"""
import subprocess
import sys
import time

from common import ROOT

STEPS = [
    ("labels", ["labels.py"]),
    ("router", ["router.py"]),
    ("colpali_token_feats (GPU)", ["colpali_token_feats.py"]),
    ("router_ce (GPU)", ["router_ce.py"]),
    ("bench_router_ce (GPU)", ["bench_router_ce.py"]),
    ("bench_router (GPU)", ["bench_router.py"]),
    ("evaluate", ["evaluate.py"]),
    ("escalate", ["escalate.py"]),
    ("bench_batching (GPU)", ["bench_batching.py"]),
    ("simulate", ["simulate.py"]),
    ("bench_index (GPU)", ["bench_index.py"]),
    ("multi_gpu_scaling (results/multi_gpu from the multi-GPU node)", ["multi_gpu_scaling.py"]),
    ("multi_gpu_scaling_a10 (results/a10 from the 2x A10 node)",
     ["multi_gpu_scaling.py", "--env", "results/a10/a10_env.json", "--scaling", "results/a10/a10_shard_scaling.json",
      "--out", "results/a10/multi_gpu_scaling.json"]),
    ("make_outputs", ["make_outputs.py"]),
    ("make_method_figure", ["make_method_figure.py"]),
    ("make_training_figure", ["make_training_figure.py"]),
    ("make_summary", ["make_summary.py"]),
    ("make_paper_text", ["make_paper_text.py"]),
]


def main():
    for name, args in STEPS:
        t0 = time.time()
        print(f"=== {name} ...", flush=True)
        with open(ROOT / "logs" / f"{name.split()[0]}.log", "w", encoding="utf-8") as log:
            rc = subprocess.call([sys.executable, "-W", "ignore", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"=== {name}: exit {rc} in {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            sys.exit(rc)
    print("=== ALL DONE", flush=True)


if __name__ == "__main__":
    main()
