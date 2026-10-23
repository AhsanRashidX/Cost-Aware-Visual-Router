"""Retrieval-depth and gold-page controls requested in the ARS panel review (run on the 2x A10 node).

The main paths give the generator the top-3 BM25 pages as text (text), the top-1 ColPali page image (visual), or
both (hybrid). These controls separate modality from retrieval depth and from retrieval errors, on a seeded sample
of N_PER_DATASET questions per dataset, with the same generator, prompt and caps as generate.py:
  text_k1          top-1 BM25 page as text
  visual_k3        top-3 ColPali page images
  gold_text        the gold page as text (OCR)
  gold_visual      the gold page image
  gold_hybrid      the gold page image and its OCR text
  check_text, check_visual   the main text and visual paths re-run on the first N_CHECK questions, to confirm the
                   A10 node reproduces the cached workstation answers
Usage:  CUDA_VISIBLE_DEVICES=<i> python controls.py --shard <i>/<n>      (resumable; one process per GPU)
Output: results/controls/controls_<i>.jsonl (+ _meta.json)
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image

import generate as G
from common import DATA, RESULTS, SEED, hardware_info, set_seed, write_json
from metrics import anls, exact_match

N_PER_DATASET = 750
N_CHECK = 200
CONDITIONS = ["text_k1", "visual_k3", "gold_text", "gold_visual", "gold_hybrid", "check_text", "check_visual"]


def sample_qids(queries):
    rng = random.Random(SEED)
    out = []
    for ds in ("docvqa", "infovqa"):
        q = sorted(k for k, v in queries.items() if v["dataset"] == ds)
        out += rng.sample(q, N_PER_DATASET)
    rng.shuffle(out)
    return out


def contexts(cond, q, rt, rv):
    """(text page ids, image page ids) for a condition."""
    gold = q["page_id"]
    return {"text_k1": ([rt["top"][0]], []), "visual_k3": ([], rv["top"][:3]),
            "gold_text": ([gold], []), "gold_visual": ([], [gold]), "gold_hybrid": ([gold], [gold]),
            "check_text": (rt["top"][:G.K_TEXT], []), "check_visual": ([], [rv["top"][0]])}[cond]


def messages(question, texts, n_images):
    """generate.build_messages generalised to any number of page images (same system prompt and text format)."""
    content = [{"type": "image"} for _ in range(n_images)]
    prompt = ""
    if texts:
        ctx = "\n\n".join(f"[Page {i + 1} OCR]\n{t[:G.MAX_CHARS_PER_PAGE]}" for i, t in enumerate(texts))
        prompt += f"Document text:\n{ctx}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    content.append({"type": "text", "text": prompt})
    return [{"role": "system", "content": [{"type": "text", "text": G.SYSTEM}]}, {"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    si, sn = map(int, args.shard.split("/"))
    set_seed()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    queries = G.load_jsonl(DATA / "queries.jsonl", "qid")
    ret_text = G.load_jsonl(RESULTS / "retrieval_text.jsonl", "qid")
    ret_vis = G.load_jsonl(RESULTS / "retrieval_visual.jsonl", "qid")
    pages = {(p["dataset"], p["page_id"]): p for p in map(json.loads, open(DATA / "pages.jsonl"))}
    ocr = {(o["dataset"], o["page_id"]): o["text"] for o in map(json.loads, open(DATA / "ocr.jsonl", encoding="utf-8"))}
    qids = sample_qids(queries)
    jobs = [(qid, c) for n, qid in enumerate(qids) for c in CONDITIONS if not c.startswith("check") or n < N_CHECK]
    jobs = jobs[si::sn]

    out_dir = RESULTS / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"controls_{si}.jsonl"
    done = set()
    if out_path.exists():
        done = {(r["qid"], r["condition"]) for r in map(json.loads, open(out_path, encoding="utf-8"))}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        G.MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(G.MODEL)
    with torch.inference_mode():                         # warm-up, as in generate.py
        for imgs in ([], [Image.new("RGB", (448, 448), "white")]):
            chat = processor.apply_chat_template(messages("warm up", ["warm up"], len(imgs)), add_generation_prompt=True)
            model.generate(**processor(text=[chat], images=imgs or None, return_tensors="pt").to(model.device),
                           max_new_tokens=4, do_sample=False)
    write_json(out_dir / f"controls_{si}_meta.json", {
        "model": G.MODEL, "conditions": CONDITIONS, "n_per_dataset": N_PER_DATASET, "n_check": N_CHECK,
        "shard": args.shard, "max_new_tokens": G.MAX_NEW_TOKENS, "max_chars_per_page": G.MAX_CHARS_PER_PAGE,
        "max_pixels": G.MAX_PIXELS, "script_sha1": hashlib.sha1(open(__file__, "rb").read()).hexdigest(),
        "hardware": hardware_info()})

    n_new = 0
    with open(out_path, "a", encoding="utf-8") as out, torch.inference_mode():
        for n, (qid, cond) in enumerate(jobs):
            if (qid, cond) in done:
                continue
            q = queries[qid]
            tp, ip = contexts(cond, q, ret_text[qid], ret_vis[qid])
            texts = [ocr[(q["dataset"], p)] for p in tp]
            images = [G.resize(Image.open(DATA / pages[(q["dataset"], p)]["path"]).convert("RGB")) for p in ip]
            chat = processor.apply_chat_template(messages(q["question"], texts, len(images)), add_generation_prompt=True)
            inputs = processor(text=[chat], images=images or None, return_tensors="pt").to(model.device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gen = model.generate(**inputs, max_new_tokens=G.MAX_NEW_TOKENS, do_sample=False)
            torch.cuda.synchronize()
            n_in = inputs["input_ids"].shape[1]
            ans = processor.batch_decode(gen[:, n_in:], skip_special_tokens=True)[0].strip()
            out.write(json.dumps({"qid": qid, "dataset": q["dataset"], "condition": cond, "answer": ans,
                                  "anls": anls(ans, q["answers"]), "em": exact_match(ans, q["answers"]),
                                  "input_tokens": n_in, "generate_s": time.perf_counter() - t0,
                                  "text_pages": tp, "image_pages": ip}, ensure_ascii=False) + "\n")
            n_new += 1
            if n_new % 50 == 0:
                out.flush()
                print(f"shard {args.shard}: {n + 1}/{len(jobs)}", flush=True)
    print(f"done: {n_new} new rows -> {out_path}")


if __name__ == "__main__":
    main()
