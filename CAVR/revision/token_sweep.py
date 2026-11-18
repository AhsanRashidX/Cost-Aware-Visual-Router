"""Image-token budget sweep requested in the third review round (R6 concern 4), run on the 2x A10 node.

On the seeded control sample of controls.py (N_PER_DATASET questions per dataset), the visual and hybrid paths are
re-run with the page image capped at 256, 512, 768 and 1,280 visual tokens (1,280 = the cached run's cap, re-run in
the same session as the anchor), and the text path once (it has no image), with the same generator, prompt, OCR
caps and greedy decoding as generate.py. Together with the text path's ~1,550 input tokens this gives an
equal-token-budget comparison of text and image contexts. The image is resized with generate.py's rule
(smart_resize to a multiple of 28, between 256 and BUDGET visual tokens); the processor does not resize it again.
Usage:  CUDA_VISIBLE_DEVICES=<i> python token_sweep.py --shard <i>/<n>      (resumable; one process per GPU)
Output: results/a10/token_sweep_r3/sweep_<i>.jsonl (+ _meta.json)
"""
import argparse
import hashlib
import json
import time

import torch
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize

import generate as G
from common import DATA, RESULTS, hardware_info, set_seed, write_json
from controls import messages, sample_qids
from metrics import anls, exact_match

BUDGETS = [256, 512, 768, 1280]
CONDITIONS = ["text"] + [f"{p}@{b}" for b in BUDGETS for p in ("visual", "hybrid")]
OUT = RESULTS / "a10" / "token_sweep_r3"


def resize(im, budget):
    h, w = smart_resize(im.height, im.width, factor=28, min_pixels=G.MIN_PIXELS, max_pixels=budget * 28 * 28)
    return im.resize((w, h), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="pilot: only the first N questions")
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
    if args.limit:
        qids = qids[:args.limit]
    jobs = [(qid, c) for qid in qids for c in CONDITIONS][si::sn]

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"pilot_{si}" if args.limit else f"{si}"
    out_path = OUT / f"sweep_{tag}.jsonl"
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
    write_json(OUT / f"sweep_{tag}_meta.json", {
        "model": G.MODEL, "conditions": CONDITIONS, "budgets_visual_tokens": BUDGETS, "n_questions": len(qids),
        "shard": args.shard, "max_new_tokens": G.MAX_NEW_TOKENS, "max_chars_per_page": G.MAX_CHARS_PER_PAGE,
        "k_text": G.K_TEXT, "min_pixels": G.MIN_PIXELS,
        "script_sha1": hashlib.sha1(open(__file__, "rb").read()).hexdigest(), "hardware": hardware_info()})

    n_new = 0
    with open(out_path, "a", encoding="utf-8") as out, torch.inference_mode():
        for n, (qid, cond) in enumerate(jobs):
            if (qid, cond) in done:
                continue
            q, rt, rv = queries[qid], ret_text[qid], ret_vis[qid]
            path, _, budget = cond.partition("@")
            tp = rt["top"][:G.K_TEXT] if path in ("text", "hybrid") else []
            ip = [rv["top"][0]] if path in ("visual", "hybrid") else []
            texts = [ocr[(q["dataset"], p)] for p in tp]
            images = [resize(Image.open(DATA / pages[(q["dataset"], p)]["path"]).convert("RGB"), int(budget)) for p in ip]
            chat = processor.apply_chat_template(messages(q["question"], texts, len(images)), add_generation_prompt=True)
            inputs = processor(text=[chat], images=images or None, return_tensors="pt").to(model.device)
            vis_tok = int(inputs["image_grid_thw"].prod(-1).sum()) // 4 if images else 0
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gen = model.generate(**inputs, max_new_tokens=G.MAX_NEW_TOKENS, do_sample=False)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            n_in = inputs["input_ids"].shape[1]
            ans = processor.batch_decode(gen[:, n_in:], skip_special_tokens=True)[0].strip()
            out.write(json.dumps({"qid": qid, "dataset": q["dataset"], "condition": cond, "path": path,
                                  "budget": int(budget) if budget else None, "answer": ans,
                                  "anls": anls(ans, q["answers"]), "em": exact_match(ans, q["answers"]),
                                  "input_tokens": n_in, "visual_tokens": vis_tok, "generate_s": dt,
                                  "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                                  "gold_in_context": q["page_id"] in tp or q["page_id"] in ip},
                                 ensure_ascii=False) + "\n")
            n_new += 1
            if n_new % 50 == 0:
                out.flush()
                print(f"shard {args.shard}: {n + 1}/{len(jobs)}", flush=True)
    print(f"done: {n_new} new rows -> {out_path}")


if __name__ == "__main__":
    main()
