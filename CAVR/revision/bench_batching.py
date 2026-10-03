"""Step 8c: batched generation throughput per path (input to the load-test simulation).

For each path and batch size B, run the same 32 queries in batches of B (greedy, left padding) and
record time per batch, energy and peak memory. Stops increasing B for a path once reserved memory
gets close to the point where WDDM would start spilling to system RAM.
Output: results/batching.json
"""
import json
import time

import numpy as np
import pynvml
import torch
from PIL import Image

import generate as G
from common import DATA, RESULTS, SEED, hardware_info, set_seed, write_json

N_QUERIES = 32
BATCHES = [1, 2, 4, 8, 16]
RESERVED_LIMIT_GB = 21.0


def main():
    set_seed()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import random
    queries = G.load_jsonl(DATA / "queries.jsonl", "qid")
    qids = random.Random(SEED).sample(json.load(open(DATA / "samples.json"))["stratified"], N_QUERIES)
    rt, rv = G.load_jsonl(RESULTS / "retrieval_text.jsonl", "qid"), G.load_jsonl(RESULTS / "retrieval_visual.jsonl", "qid")
    pages = {(p["dataset"], p["page_id"]): p for p in map(json.loads, open(DATA / "pages.jsonl"))}
    ocr = {(o["dataset"], o["page_id"]): o["text"] for o in map(json.loads, open(DATA / "ocr.jsonl", encoding="utf-8"))}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        G.MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(G.MODEL)
    processor.tokenizer.padding_side = "left"
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

    def build(path, qid):
        q = queries[qid]
        texts = [ocr[(q["dataset"], p)] for p in rt[qid]["top"][:G.K_TEXT]] if path != "visual" else []
        image = None
        if path != "text":
            image = G.resize(Image.open(DATA / pages[(q["dataset"], rv[qid]["top"][0])]["path"]).convert("RGB"))
        chat = processor.apply_chat_template(G.build_messages(path, q["question"], texts, image), add_generation_prompt=True)
        return chat, image

    out = {"n_queries": N_QUERIES, "hardware": hardware_info(), "paths": {}}
    with torch.inference_mode():
        for path in G.PATHS:
            items = [build(path, q) for q in qids]
            res = {}
            for b in BATCHES:
                times, energy, peak = [], 0.0, 0.0
                for i in range(0, N_QUERIES, b):
                    chats = [c for c, _ in items[i:i + b]]
                    imgs = [im for _, im in items[i:i + b]] if path != "text" else None
                    inputs = processor(text=chats, images=imgs, return_tensors="pt", padding=True).to(model.device)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
                    t0 = time.perf_counter()
                    model.generate(**inputs, max_new_tokens=G.MAX_NEW_TOKENS, do_sample=False)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                    energy += (pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu) - e0) / 1000
                    peak = max(peak, torch.cuda.max_memory_reserved() / 2**30)
                res[str(b)] = {"mean_batch_s": float(np.mean(times)), "per_query_s": float(np.sum(times) / N_QUERIES),
                               "queries_per_s": N_QUERIES / float(np.sum(times)), "energy_j_per_query": energy / N_QUERIES,
                               "peak_reserved_gb": peak}
                print(path, b, {k: round(v, 3) for k, v in res[str(b)].items()}, flush=True)
                if peak > RESERVED_LIMIT_GB:
                    res[str(b)]["note"] = "near VRAM limit; larger batches skipped"
                    break
            out["paths"][path] = res
    write_json(RESULTS / "batching.json", out)


if __name__ == "__main__":
    main()
