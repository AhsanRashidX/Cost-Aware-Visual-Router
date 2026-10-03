"""Fetch the datasets and model weights used by the revision pipeline into HF_HOME.

Usage: python CAVR/revision/download.py [--data-only | --models-only]
"""
import argparse
import os

os.environ.setdefault("HF_HOME", r"D:\hf_cache")

from datasets import load_dataset
from huggingface_hub import snapshot_download

DATASET = "lmms-lab-encoder/DocVQA"
CONFIGS = ["DocVQA", "InfographicVQA"]
SPLIT = "validation"

MODELS = [
    "vidore/colpali-v1.3",
    "vidore/colpaligemma-3b-pt-448-base",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "sentence-transformers/all-mpnet-base-v2",
]
# Skip duplicate weight formats; safetensors is all we load.
IGNORE = ["*.bin", "*.pt", "*.h5", "*.msgpack", "*.onnx", "openvino/*", "onnx/*"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-only", action="store_true")
    ap.add_argument("--models-only", action="store_true")
    args = ap.parse_args()

    if not args.models_only:
        for cfg in CONFIGS:
            ds = load_dataset(DATASET, cfg, split=SPLIT)
            print(f"[data] {DATASET}/{cfg}/{SPLIT}: {len(ds)} rows", flush=True)

    if not args.data_only:
        for repo in MODELS:
            path = snapshot_download(repo, ignore_patterns=IGNORE)
            print(f"[model] {repo} -> {path}", flush=True)


if __name__ == "__main__":
    main()
