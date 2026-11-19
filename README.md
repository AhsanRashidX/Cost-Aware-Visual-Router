# Cost-Aware Visual Router (CAVR)

**Dynamic query routing for cost-efficient multimodal retrieval-augmented generation (RAG)**

Multimodal RAG can answer a question about a document from three kinds of context, each with a different GPU cost:

- the page's OCR text;
- a retrieved page image;
- both.

This repository measures that trade-off end to end and tests whether a router can exploit it. For each query, a router
chooses one context path, and a single shared vision-language generator answers from it.

> **Status:** the accompanying paper is under review. The code on this branch (`revision`) is the revised pipeline;
> `main` holds the original submission code and is kept only for reference.

---

## What the pipeline does

| Path | Context given to the generator | Retrieval |
|---|---|---|
| **text** | OCR text of the top-3 pages (docTR) | BM25 over OCR text |
| **visual** | top-1 page image (at most 1,280 visual tokens) | ColPali v1.3, exhaustive MaxSim |
| **hybrid** | the page image and the OCR text | both |

- **Generator:** one model for all three paths, `Qwen/Qwen2.5-VL-7B-Instruct` with greedy decoding.
- **Measured once per query and path:** ANLS, exact match, wall-clock time per stage, GPU-seconds, peak GPU memory
  and NVML energy. All of it is cached in `results/path_outputs.jsonl`.
- **Routers:** each predicts the answer quality of every path and picks argmax_p ( q̂_p − λ · c̄_p ). Here c̄_p is the
  measured mean cost of the path, and the cost weight λ is set per deployment without retraining.
  - **CAVR:** reads only the question, through all-mpnet-base-v2 embeddings.
  - **CAVR-CE:** a cross-encoder over the question and the passages that retrieval returned, plus retrieval scores.
    It predicts how well each path will answer before any generation runs.
- **Baselines:** fixed policies, a keyword rule, logistic regression, kNN, gradient boosting, random selection at
  equal budget, and a per-query oracle.
- **Serving experiments:**
  - live vLLM serving on 1–2 NVIDIA A10 GPUs, generation only, plus one run with retrieval and routing in the loop;
  - a discrete-event simulator for larger GPU pools;
  - ColPali index scaling up to 100k pages (GPU-resident, host-memory and pooled placement);
  - index sharding across GPUs.

## Main findings

These are the numbers in the revised paper. Every number in the paper and in `results/SUMMARY.md` is generated from
`results/*.json`; the ones below are copied from those files.

| Finding | Value | Source |
|---|---|---|
| Fixed paths over 8,150 DocVQA + InfographicVQA questions (ANLS / GPU-s per query) | text 0.367 / 0.43, visual 0.546 / 0.62, hybrid 0.570 / 1.01 | `results/SUMMARY.md` |
| CAVR-CE vs always-hybrid, 5 page-grouped splits | 29% less GPU compute at 0.008 lower ANLS | `results/eval_main.json` |
| Held-out set (1,499 questions from the training splits) | CAVR-CE keeps a 35% saving vs hybrid; the page image alone is about as accurate (0.600 vs 0.605) at 16% less compute | `results/confirmation.json` |
| Live serving, 2× A10, 5 arrival traces per rate | at p95 ≤ 5 s, CAVR-CE sustains 1.70× [1.57, 1.84] the request rate of always-hybrid; always-visual stays ahead of CAVR-CE | `results/vllm_repeats_r3.json` |
| Simulator vs live serving | simulated capacity at p95 ≤ 5 s is 1.3–1.9× the measured capacity, so simulated GPU counts for larger pools are projections | `results/vllm_repeats_r3.json` |

In short:
- The page image alone is the strongest fixed policy.
- Content-aware routing reliably saves compute relative to always sending both contexts.
- A gain over the page image alone is **not established**.
- A per-query oracle shows large headroom: predicting whether a path will succeed is the open problem.

The original submission reported 86.2% routing accuracy and a cost of $0.00019 per query. **Both are withdrawn.** The
labels behind the accuracy did not agree with measured outcomes (Cohen's κ ≈ 0.01), and the cost was never measured.
See `results/SUMMARY.md`.

## Repository layout

```
CAVR/revision/            the revised pipeline (use this)
  common.py, download.py, prepare_data.py, ocr_pages.py      data and OCR
  retrieve_text.py, retrieve_visual.py, generate.py          retrieval and the cached run of every path
  labels.py, routing_data.py, router.py, router_ce.py        outcome labels and routers
  evaluate.py, escalate.py, simulate.py, sim_experiments.py  evaluation, escalation, load simulation
  bench_*.py, multi_gpu_scaling.py, host_stream_scaling.py   router, batching and index benchmarks
  controls.py, confirm_eval.py, token_sweep.py               controls, held-out set, image-token sweep
  live_pipeline_r3.py, vllm_analysis.py, vllm_repeats_r3.py  live serving analyses
  t4/                       vLLM serving package (server scripts, load generator; see t4/README_T4.md)
  make_outputs.py           every table and figure, from results/*.json
  make_paper_text.py, paper_sections.py, make_bib.py        paper text (LaTeX) with numbers filled from results
  make_manifest.py          ARTIFACT_MANIFEST.md: table/figure -> generator -> result files -> producing scripts
  check_paper_r3.py         build checks (references, labels, unfilled numbers, required statements)
  results/                  all result files (*.json, cached per-query outputs, run logs)
  paper_out/                generated tables, figures and paper text
CAVR/scripts, CAVR/checkpoints, ...   original submission code (superseded; kept for reference)
```

## Installation

Tested with Python 3.11 on Windows 11 and CUDA 12.8 (PyTorch 2.11).

```bash
git clone https://github.com/AhsanRashidX/Cost-Aware-Visual-Router.git
cd Cost-Aware-Visual-Router/CAVR/revision
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements-lock.txt
```

Live serving uses vLLM 0.31 in a separate Linux environment; on Windows we used WSL2 with Ubuntu 24.04. See
`t4/README_T4.md`.

## Reproducing the results

Each script's docstring lists its options. Runs use fixed seeds, and each run records its hardware and library
versions. Model and dataset revisions are pinned, and `record_system_info.py` writes them to
`results/system_info.json`.

```bash
python download.py          # datasets and model weights into HF_HOME
python prepare_data.py      # query table, page store, samples, data/manifest.json
python ocr_pages.py         # docTR OCR of every page (GPU)
python retrieve_text.py     # BM25
python retrieve_visual.py   # ColPali v1.3
python embed.py             # question embeddings for the router
python generate.py          # every path on every query -> results/path_outputs.jsonl (long, resumable)
python run_after_cache.py   # labels, routers, evaluation, escalation, simulation, benchmarks, tables, figures, text
```

The second-node experiments ran on a node with 2× NVIDIA A10 GPUs:
- retrieval-depth and gold-page controls (`controls.py`);
- the held-out confirmation set (`run_confirmation.sh`, `confirm_eval.py`);
- the image-token sweep (`token_sweep.py`);
- live serving (`t4/`, `live_pipeline_r3.py`).

Their result files are in `results/a10/`, and the analysis scripts read them from there:
- `review_round2.py` and `review_round3.py` compute analyses from cached data only;
- `vllm_repeats_r3.py` analyses the repeated live runs.

To regenerate every table, figure and the paper text from the committed results, without re-running any model:

```bash
python make_outputs.py && python make_paper_text.py && python make_manifest.py && python check_paper_r3.py
```

## Hardware used

| Machine | Used for |
|---|---|
| 1× NVIDIA GeForce RTX 4090 (24 GB), Ryzen 9 5950X, 64 GB RAM | cached run of every path, router training and evaluation, benchmarks |
| 2× NVIDIA A10 (24 GB each), Xeon Gold 6330, 504 GB RAM | controls, held-out set, image-token sweep, live vLLM serving, host-memory index |
| a separate multi-GPU node | ColPali index sharding |

## Data and models

- **Datasets:** DocVQA and InfographicVQA, validation splits (8,150 questions over 1,786 pages). The held-out set is
  1,499 questions from the training splits. Datasets are downloaded from the Hugging Face Hub; they are not included
  in this repository.
- **Models:** `Qwen/Qwen2.5-VL-7B-Instruct`, `vidore/colpali-v1.3`, `sentence-transformers/all-mpnet-base-v2` and
  docTR. Their exact revisions are recorded in `results/system_info.json`.

## Citation

The paper is under review; citation details will be added on publication.

## License

No license has been specified yet.
