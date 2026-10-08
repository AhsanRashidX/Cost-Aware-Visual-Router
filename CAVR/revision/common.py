"""Shared helpers for the revision pipeline: paths, seeding, hardware/software provenance."""
import json
import os
import platform
import random
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\hf_cache")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 1234

PACKAGES = ["torch", "transformers", "sentence-transformers", "colpali-engine", "python-doctr",
            "datasets", "scikit-learn", "numpy", "rank-bm25", "nvidia-ml-py", "qwen-vl-utils", "accelerate"]


def cpu_name():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor()


def set_seed(seed=SEED):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_commit():
    try:
        top = subprocess.check_output(["git", "-c", f"safe.directory={ROOT.parent.parent.as_posix()}", "rev-parse",
                                       "--show-toplevel"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return subprocess.check_output(["git", "-c", f"safe.directory={top}", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def hardware_info():
    import psutil
    info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "cpu_threads": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "git_commit": _git_commit(),
        "packages": {},
        "gpus": [],
    }
    for p in PACKAGES:
        try:
            info["packages"][p] = version(p)
        except PackageNotFoundError:
            pass
    try:
        import torch
        info["cuda_runtime"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    except ImportError:
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        info["driver"] = pynvml.nvmlSystemGetDriverVersion()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info["gpus"].append({
                "index": i,
                "name": pynvml.nvmlDeviceGetName(h),
                "memory_gb": round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**30, 1),
                "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000,
            })
    except Exception as e:
        info["nvml_error"] = str(e)
    return info


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
