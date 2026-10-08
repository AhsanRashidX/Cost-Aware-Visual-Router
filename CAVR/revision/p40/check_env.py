"""Environment check for the multi-GPU server (Tesla P40 node). Standalone: needs torch, numpy, psutil, nvidia-ml-py.

Reports GPUs, compute capability, whether this PyTorch build supports them, driver model (TCC/WDDM), PCIe link,
GPU-CPU topology, a quick FP32/FP16 matmul throughput test per GPU, and whether NVML's energy counter is available.
Usage:  python check_env.py   ->  prints a summary and writes p40_env.json next to this script
"""
import json
import platform
import subprocess
import time
from pathlib import Path

import torch


def smi(args):
    try:
        return subprocess.check_output(["nvidia-smi"] + args, text=True, stderr=subprocess.STDOUT)
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def tflops(dev, dtype, n=4096, iters=10):
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    a @ b
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize(dev)
    return 2 * n ** 3 * iters / (time.perf_counter() - t0) / 1e12


def main():
    info = {"python": platform.python_version(), "os": platform.platform(), "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [], "gpus": []}
    try:
        import psutil
        info["cpu_threads"], info["ram_gb"] = psutil.cpu_count(), round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        info["cpu"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        info["cpu"] = platform.processor()
    info["smi_gpus"] = smi(["--query-gpu=index,name,driver_version,driver_model.current,pcie.link.gen.current,"
                            "pcie.link.width.current,power.limit,memory.total", "--format=csv"])
    info["smi_topology"] = smi(["topo", "-m"])
    energy_ok = None
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(pynvml.nvmlDeviceGetHandleByIndex(0))
            energy_ok = True
        except Exception:
            energy_ok = False
    except ImportError:
        pass
    info["nvml_energy_counter"] = energy_ok
    for i in range(torch.cuda.device_count()):
        dev = torch.device(f"cuda:{i}")
        p = torch.cuda.get_device_properties(i)
        g = {"index": i, "name": p.name, "capability": f"{p.major}.{p.minor}", "memory_gb": round(p.total_memory / 2**30, 1)}
        try:
            g["fp32_tflops"] = round(tflops(dev, torch.float32), 2)
            g["fp16_tflops"] = round(tflops(dev, torch.float16, iters=3), 3)
        except Exception as e:
            g["error"] = repr(e)
        info["gpus"].append(g)
    print(json.dumps({k: v for k, v in info.items() if not k.startswith("smi_")}, indent=2))
    print(info["smi_gpus"])
    print(info["smi_topology"])
    def supported(cap, archs):
        """Binary kernels run on the same major version with an equal or lower minor; PTX JIT-compiles upward."""
        major, minor = map(int, cap.split("."))
        for a in archs:
            kind, num = a.split("_")
            v = int(num)
            if kind == "sm" and v // 10 == major and v % 10 <= minor:
                return True
            if kind == "compute" and v <= major * 10 + minor:
                return True
        return False
    bad = sorted({g["capability"] for g in info["gpus"] if not supported(g["capability"], info["arch_list"])})
    info["pytorch_supports_gpus"] = not bad
    if bad:
        print(f"WARNING: this PyTorch build has no kernels for compute capability {bad}; install a build that does "
              f"(see README).")
    else:
        print("PyTorch build supports all GPUs.")
    if "WDDM" in info["smi_gpus"]:
        print("NOTE: GPUs are in WDDM mode; TCC mode is preferred for compute (nvidia-smi -g <i> -dm 1, as admin, then reboot).")
    Path(__file__).with_name("p40_env.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
