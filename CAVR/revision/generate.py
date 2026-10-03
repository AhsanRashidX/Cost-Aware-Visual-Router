"""Step 3c: run every path on every query with one generator and cache the outputs.

Paths (same generator, Qwen2.5-VL-7B-Instruct, greedy):
  text    BM25 top-K_TEXT OCR pages as text, no image
  visual  ColPali top-1 page image
  hybrid  ColPali top-1 page image + BM25 top-K_TEXT OCR text
Each output row records the answer, ANLS/EM, token counts, per-stage wall time, GPU-seconds,
peak GPU memory and NVML energy. Retrieval-stage costs come from retrieve_text.py /
retrieve_visual.py and are copied into the row so it is self-contained.

Resumable: rows already in --out are skipped.
"""
import argparse
import hashlib
import json
import random
import time

import pynvml
import torch
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize

from common import DATA, RESULTS, SEED, hardware_info, set_seed, write_json
from metrics import anls, exact_match

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
PATHS = ("text", "visual", "hybrid")
K_TEXT = 3
MAX_CHARS_PER_PAGE = 2000  # 3000 pushed hybrid prompts past VRAM into WDDM sysmem fallback
MAX_PIXELS = 1280 * 28 * 28  # <= 1280 visual tokens per page
MIN_PIXELS = 256 * 28 * 28
MAX_NEW_TOKENS = 32
EOS_IDS = []
SYSTEM = ("You answer questions about documents. Reply with only the answer: a short phrase, "
          "number or name, copied exactly from the document when possible. No explanation.")


def load_jsonl(path, key):
    return {r[key]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def resize(im):
    h, w = smart_resize(im.height, im.width, factor=28, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return im.resize((w, h), Image.BICUBIC)


def build_messages(path, question, texts, image):
    content = []
    if path in ("visual", "hybrid"):
        content.append({"type": "image"})
    prompt = ""
    if path in ("text", "hybrid"):
        ctx = "\n\n".join(f"[Page {i + 1} OCR]\n{t[:MAX_CHARS_PER_PAGE]}" for i, t in enumerate(texts))
        prompt += f"Document text:\n{ctx}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    content.append({"type": "text", "text": prompt})
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "path_outputs.jsonl"))
    ap.add_argument("--sample", choices=["natural", "stratified"], default="natural")
    ap.add_argument("--limit", type=int, default=None, help="random subset of N queries (seeded)")
    ap.add_argument("--paths", default=",".join(PATHS))
    args = ap.parse_args()
    paths = args.paths.split(",")
    set_seed()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    queries = load_jsonl(DATA / "queries.jsonl", "qid")
    qids = json.load(open(DATA / "samples.json"))[args.sample]
    if args.limit:
        qids = random.Random(SEED).sample(qids, args.limit)
    ret_text = load_jsonl(RESULTS / "retrieval_text.jsonl", "qid")
    ret_vis = load_jsonl(RESULTS / "retrieval_visual.jsonl", "qid")
    pages = {(p["dataset"], p["page_id"]): p for p in map(json.loads, open(DATA / "pages.jsonl"))}
    ocr = {(o["dataset"], o["page_id"]): o["text"] for o in map(json.loads, open(DATA / "ocr.jsonl", encoding="utf-8"))}

    done = set()
    try:
        done = {(r["qid"], r["path"]) for r in map(json.loads, open(args.out, encoding="utf-8"))}
    except FileNotFoundError:
        pass

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    global EOS_IDS
    eos = model.generation_config.eos_token_id
    EOS_IDS = eos if isinstance(eos, list) else [eos]
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

    # warm-up both text-only and image inputs
    with torch.inference_mode():
        for p in ("text", "visual"):
            msgs = build_messages(p, "warm up", ["warm up"], None)
            chat = processor.apply_chat_template(msgs, add_generation_prompt=True)
            img = [Image.new("RGB", (448, 448), "white")] if p == "visual" else None
            inputs = processor(text=[chat], images=img, return_tensors="pt").to(model.device)
            model.generate(**inputs, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()

    meta = {"model": MODEL, "k_text": K_TEXT, "max_chars_per_page": MAX_CHARS_PER_PAGE,
            "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS, "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": "greedy (do_sample=False); model generation_config otherwise applies",
            "generation_config": model.generation_config.to_dict(), "system_prompt": SYSTEM, "sample": args.sample, "limit": args.limit,
            "script_sha1": hashlib.sha1(open(__file__, "rb").read()).hexdigest(),
            "hardware": hardware_info()}
    write_json(args.out.replace(".jsonl", "_meta.json"), meta)

    n_new = 0
    with open(args.out, "a", encoding="utf-8") as out, torch.inference_mode():
        for qn, qid in enumerate(qids):
            q = queries[qid]
            rt, rv = ret_text[qid], ret_vis[qid]
            text_pages = rt["top"][:K_TEXT]
            vis_page = rv["top"][0]
            for path in paths:
                if (qid, path) in done:
                    continue
                t0 = time.perf_counter()
                texts = [ocr[(q["dataset"], pid)] for pid in text_pages] if path != "visual" else []
                image = None
                if path != "text":
                    image = resize(Image.open(DATA / pages[(q["dataset"], vis_page)]["path"]).convert("RGB"))
                chat = processor.apply_chat_template(build_messages(path, q["question"], texts, image),
                                                     add_generation_prompt=True)
                inputs = processor(text=[chat], images=[image] if image else None,
                                   return_tensors="pt").to(model.device)
                torch.cuda.empty_cache()  # keep reserved memory low so WDDM never spills to system RAM
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
                out_g = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                       return_dict_in_generate=True, output_logits=True)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                energy = (pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu) - e0) / 1000
                gen = out_g.sequences
                n_in = inputs["input_ids"].shape[1]
                answer = processor.batch_decode(gen[:, n_in:], skip_special_tokens=True)[0].strip()
                # generator's own confidence: log-prob of each emitted token under the raw logits
                tok_lp = torch.stack([torch.log_softmax(l[0].float(), -1)[t] for l, t in
                                      zip(out_g.logits, gen[0, n_in:])]).cpu()
                special = torch.isin(gen[0, n_in:].cpu(), torch.tensor(EOS_IDS))
                ans_lp = tok_lp[~special] if (~special).any() else tok_lp
                n_vis = int((inputs["input_ids"] == model.config.image_token_id).sum())

                used_text = path != "visual"
                used_vis = path != "text"
                row = {
                    "qid": qid, "dataset": q["dataset"], "path": path, "answer": answer,
                    "anls": anls(answer, q["answers"]), "em": exact_match(answer, q["answers"]),
                    "input_tokens": n_in, "visual_tokens": n_vis,
                    "output_tokens": int(gen.shape[1] - n_in),
                    "answer_logprob_mean": float(ans_lp.mean()), "answer_logprob_min": float(ans_lp.min()),
                    "text_pages": text_pages if used_text else [], "visual_page": vis_page if used_vis else None,
                    "gold_in_context": (used_text and q["page_id"] in text_pages) or (used_vis and q["page_id"] == vis_page),
                    "preprocess_s": t1 - t0, "generate_s": t2 - t1,
                    "gen_energy_j": energy, "gen_peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "gen_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                    "text_retrieval_cpu_s": rt["cpu_seconds"] if used_text else 0.0,
                    "visual_retrieval_gpu_s": (rv["encode_s"] + rv["score_s"]) if used_vis else 0.0,
                    "visual_retrieval_energy_j": rv["energy_j"] if used_vis else 0.0,
                }
                row["gpu_seconds"] = row["generate_s"] + row["visual_retrieval_gpu_s"]
                row["energy_j"] = row["gen_energy_j"] + row["visual_retrieval_energy_j"]
                row["wall_s"] = (row["preprocess_s"] + row["generate_s"] + row["text_retrieval_cpu_s"]
                                 + row["visual_retrieval_gpu_s"])
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_new += 1
            if qn % 50 == 0:
                out.flush()
                print(f"{qn}/{len(qids)} queries", flush=True)
    print(f"done: {n_new} new rows -> {args.out}")


if __name__ == "__main__":
    main()
