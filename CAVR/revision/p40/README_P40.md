# Multi-GPU index sharding on the 3x Tesla P40 server (Dell R740, Windows)

This measures what the paper currently only emulates: a ColPali index of up to 200k pages split across 1, 2 and 3
GPUs, queried on all shards in parallel, with the top results merged on the host. No model is needed. The package
carries the page index and the query embeddings exported on the main workstation.

Expected run time: about 30-60 minutes. Nothing else should run on the GPUs meanwhile.

## 1. One-time setup

1. **Driver.** Install the NVIDIA data-center driver for the Tesla P40 on Windows Server. Check that `nvidia-smi`
   lists all three GPUs.
2. **TCC mode (recommended).** The R740's display runs on its onboard VGA, so the P40s can run in TCC compute mode,
   which avoids Windows display-driver (WDDM) overheads. In an administrator prompt run `nvidia-smi -dm 1`, then
   reboot. `nvidia-smi` should now show `TCC` for each GPU.
3. **Python.** Install Miniconda, then:
   ```
   conda create -n p40 python=3.11 -y
   conda activate p40
   pip install numpy psutil nvidia-ml-py
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```
4. **Check PyTorch supports the P40** (compute capability 6.1):
   ```
   python check_env.py
   ```
   If it prints `WARNING: this PyTorch build has no kernels for compute capability ['6.1']`, the newest builds have
   dropped Pascal. Uninstall torch and install an older build from the cu118 index, then re-run `check_env.py` until it
   prints `PyTorch build supports all GPUs.`:
   ```
   pip uninstall -y torch
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
   If the newest cu118 build also lacks Pascal kernels, pin an older version (for example `torch==2.4.1`) from the
   same index.

## 2. Run

From this folder (it contains `colpali_index.pt` and `colpali_query_emb.pt`):
```
python check_env.py
python shard_bench.py --data .
```
`shard_bench.py` first checks that sharded search returns the same top page as a single GPU (it prints
`top-1 agreement`; `exact_or_tie` should be 1.0). It then times every corpus size (1k, 10k, 50k, 100k, 200k pages)
on 1, 2 and 3 GPUs and skips any combination that does not fit in GPU memory.

## 3. Send back

Two files from this folder: `p40_env.json` and `p40_shard_scaling.json`.

## Notes

- 64 GB of RAM is enough: large corpora are built directly on the GPUs, not in host memory.
- The index is stored in fp16 and scored in fp32, because the P40 has no bf16 and runs fp16 arithmetic very slowly.
  P40 timings are therefore not directly comparable with the RTX 4090 timings. The point of this run is how latency
  and throughput change with the number of GPUs.
- The R740 attaches GPUs to two CPU sockets. `check_env.py` records the topology (`nvidia-smi topo -m`) so that
  cross-socket effects can be identified.
