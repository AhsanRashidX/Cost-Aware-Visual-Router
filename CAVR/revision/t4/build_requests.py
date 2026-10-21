"""Build the request set for the real multi-GPU serving test (runs on the workstation, called by make_t4_package.py).

Split 1 = results/router_runs/seed0.npz, i.e. group_split(Q, SEED=1234), the split evaluate.py reports as
eval_main.json per_seed[0]. For its test queries, in evaluation order (row i = simulate.py's request index i), it writes:
  requests.jsonl         qid, dataset, question, gold answers; per path (text, visual, hybrid) the OpenAI chat
                         `messages` built by generate.build_messages itself (same system prompt, OCR formatting,
                         truncation, retrieved pages and order), with the page image referenced as images/<file>.png,
                         plus the 4090 reference (answer, ANLS, EM, token counts, generate_s, GPU-s); and the path
                         chosen by each policy (always_text/visual/hybrid, cavr_ce_valsel, oracle)
  images/                the ColPali top-1 pages of these queries, pre-resized with generate.resize (smart_resize,
                         generate.py's min/max pixels), lossless PNG (grayscale pages stay 1-channel; the server
                         converts to RGB, which restores exactly the 4090 input)
  requests_meta.json     sampling parameters, model/pixel settings, sources, verification results
  policies_summary.json  per policy: path mix, 4090 test ANLS/EM and mean GPU-s
Verification (any failure aborts): split == group_split; cavr_ce_valsel rebuilt from router_ce/seed0.npz exactly as
evaluate.py does and its test ANLS/EM/GPU-s/energy/path mix matched against eval_main.json per_seed[0] (fixed
policies and oracle too); cached pages == retrieval files; image grid == 4090 visual-token count for every row;
PNG round trip lossless; vqa_metrics reproduces every stored ANLS/EM; for a sample (default: all) of prompts the
chat template renders the request messages exactly like generate.build_messages and the processor gives the 4090
input_tokens / visual_tokens (processor loaded offline from the local HF cache; no model weights are loaded).

Usage:  python t4/build_requests.py --out t4_package [--check-n 0|N|-1]
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import generate as G  # noqa: E402  (imports torch/pynvml but loads no model)
from common import DATA, RESULTS, SEED, hardware_info, write_json  # noqa: E402
from evaluate import VALSEL_DELTA, VALSEL_LAMBDAS, oracle  # noqa: E402
from routing_data import LAMBDA_DEFAULT, PATHS, group_split, load  # noqa: E402
import vqa_metrics  # noqa: E402

SPLIT_RUN = "seed0.npz"            # split 1 (seed 1234)
POLICIES = ["always_text", "always_visual", "always_hybrid", "cavr_ce_valsel", "oracle"]
TOL = 1e-9


def sha1(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


def image_name(dataset, page_id):
    return f"{dataset}__{page_id}.png"


def to_openai(hf_messages, image_rel):
    """generate.build_messages output -> OpenAI chat messages: {"type": "image"} becomes an image_url part."""
    out = []
    for m in hf_messages:
        content = [{"type": "image_url", "image_url": {"url": image_rel}} if c["type"] == "image" else dict(c)
                   for c in m["content"]]
        out.append({"role": m["role"], "content": content})
    return out


def to_hf(openai_messages):
    """What vLLM hands the chat template with --chat-template-content-format openai: image_url -> {"type": "image"}."""
    return [{"role": m["role"], "content": [{"type": "image"} if c["type"] == "image_url" else dict(c)
                                            for c in m["content"]]} for m in openai_messages]


def rebuild_valsel(A, C, VR, R, qce):
    """evaluate.py's cavr_ce_valsel: cheapest validation-GPU-s lambda whose val ANLS >= val(always_hybrid) - delta."""
    va = R["split"] == "val"
    cost = R["cost"]
    target = A[va][:, 2].mean() - VALSEL_DELTA
    pts = []
    for lam in VALSEL_LAMBDAS:
        ch = np.argmax(qce[va] - lam * cost, 1)
        g = C[va][np.arange(va.sum()), ch] + (ch == 0) * VR[va]
        pts.append((lam, float(A[va][np.arange(va.sum()), ch].mean()), float(g.mean())))
    ok = [p for p in pts if p[1] >= target]
    lam = min(ok, key=lambda p: p[2])[0] if ok else max(pts, key=lambda p: p[1])[0]
    return lam, not ok, np.argmax(qce - lam * cost, 1)


def policy_metrics(choice, te, M, overhead=None, pre_visual=False):
    """Same formulas as evaluate.metrics (no bootstrap): ANLS, EM, GPU-s, energy, path mix on the test split."""
    idx = np.where(te)[0]
    ch = choice[idx]
    miss = (ch == 0) if pre_visual else np.zeros(len(ch), bool)
    ov = overhead or {"gpu_s": 0.0, "energy_j": 0.0}
    return {"anls": float(M["A"][idx, ch].mean()), "em": float(M["EM"][idx, ch].mean()),
            "gpu_s": float((M["C"][idx, ch] + miss * M["VR"][idx]).mean() + ov["gpu_s"]),
            "energy_j": float((M["E"][idx, ch] + miss * M["VRE"][idx]).mean() + ov["energy_j"]),
            "path_mix": {p: float((ch == k).mean()) for k, p in enumerate(PATHS)}}


def compare(mine, stored):
    diffs = {k: abs(mine[k] - stored[k]) for k in ("anls", "em", "gpu_s", "energy_j")}
    diffs.update({f"path_mix.{p}": abs(mine["path_mix"][p] - stored["path_mix"][p]) for p in PATHS})
    return {"max_abs_diff": max(diffs.values()), "match": max(diffs.values()) <= TOL, "diffs": diffs}


def main(out=None, check_n=-1):
    if out is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", default=str(HERE.parent / "t4_package"))
        ap.add_argument("--check-n", type=int, default=-1,
                        help="queries for the chat-template/token-count check (-1 = all test queries, 0 = skip)")
        a = ap.parse_args()
        out, check_n = a.out, a.check_n
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    ver = {}

    qids, Q, A, C, rows = load()
    R = np.load(RESULTS / "router_runs" / SPLIT_RUN, allow_pickle=True)
    RC = np.load(RESULTS / "router_ce" / SPLIT_RUN, allow_pickle=True)
    assert list(R["qids"]) == qids and list(RC["qids"]) == qids, "router runs built from another outputs file"
    split = R["split"]
    ver["split_equals_group_split_seed1234"] = bool((group_split(Q, SEED) == split).all()
                                                    and (RC["split"] == split).all())
    assert ver["split_equals_group_split_seed1234"], "router_runs split != group_split(Q, SEED)"
    te = split == "test"

    M = {"A": A, "C": C,
         "EM": np.array([[rows[q][p]["em"] for p in PATHS] for q in qids]),
         "E": np.array([[rows[q][p]["energy_j"] for p in PATHS] for q in qids]),
         "VR": np.array([rows[q]["hybrid"]["visual_retrieval_gpu_s"] for q in qids]),
         "VRE": np.array([rows[q]["hybrid"]["visual_retrieval_energy_j"] for q in qids])}

    # ---- policies, rebuilt as evaluate.py does, and checked against eval_main.json per_seed[0] ----
    ev = json.load(open(RESULTS / "eval_main.json", encoding="utf-8"))
    ev0 = ev["per_seed"][0]
    assert ev0["run"] == SPLIT_RUN
    ce_cost = json.load(open(RESULTS / "router_ce_cost.json"))["per_query_batch1"]
    assert ce_cost == ev["router_ce_overhead"], "router_ce_cost.json changed since eval_main.json"
    lam, fallback, ce_choice = rebuild_valsel(A, C, M["VR"], R, RC["qhat_ce_full"])
    choices = {"always_text": np.zeros(len(qids), int), "always_visual": np.ones(len(qids), int),
               "always_hybrid": np.full(len(qids), 2), "cavr_ce_valsel": ce_choice}
    orc = np.zeros(len(qids), int)
    orc[te] = oracle(A[te], C[te], LAMBDA_DEFAULT)
    choices["oracle"] = orc
    checks = {}
    for name in POLICIES:
        ce = name == "cavr_ce_valsel"
        mine = policy_metrics(choices[name], te, M, ce_cost if ce else None, pre_visual=ce)
        checks[name] = {"recomputed": mine, **compare(mine, ev0["policies"][name])}
    checks["cavr_ce_valsel"].update({"lambda_recomputed": lam, "lambda_eval_main": ev0["policies"]["cavr_ce_valsel"]["lambda"],
                                     "lambda_fallback": bool(fallback),
                                     "stored": {k: ev0["policies"]["cavr_ce_valsel"][k]
                                                for k in ("anls", "em", "gpu_s", "energy_j", "path_mix")}})
    ver["eval_main_per_seed0"] = checks
    ok = all(c["match"] for c in checks.values()) and lam == checks["cavr_ce_valsel"]["lambda_eval_main"]
    assert ok, json.dumps({k: (v["max_abs_diff"], v["match"]) for k, v in checks.items()})
    print(f"eval_main per_seed[0] reconstruction: all {len(POLICIES)} policies match (cavr_ce_valsel lambda={lam}, "
          f"test ANLS {checks['cavr_ce_valsel']['recomputed']['anls']:.6f}, GPU-s {checks['cavr_ce_valsel']['recomputed']['gpu_s']:.6f})")

    # ---- requests ----
    queries = {q["qid"]: q for q in Q}
    ret_text = G.load_jsonl(RESULTS / "retrieval_text.jsonl", "qid")
    ret_vis = G.load_jsonl(RESULTS / "retrieval_visual.jsonl", "qid")
    pages = {(p["dataset"], p["page_id"]): p for p in map(json.loads, open(DATA / "pages.jsonl"))}
    ocr = {(o["dataset"], o["page_id"]): o["text"] for o in map(json.loads, open(DATA / "ocr.jsonl", encoding="utf-8"))}
    te_idx = np.where(te)[0]
    n_ret_mismatch = n_tok_mismatch = n_anls_mismatch = n_img = 0
    lossless = []
    img_dims = {}
    req_rows = []
    for k, i in enumerate(te_idx):
        qid = qids[i]
        q = queries[qid]
        text_pages = ret_text[qid]["top"][:G.K_TEXT]
        vis_page = ret_vis[qid]["top"][0]
        texts = [ocr[(q["dataset"], pid)] for pid in text_pages]
        key = (q["dataset"], vis_page)
        fname = image_name(*key)
        if fname not in img_dims:
            im = G.resize(Image.open(DATA / pages[key]["path"]).convert("RGB"))
            arr = np.asarray(im)
            gray = bool((arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all())
            dst = out / "images" / fname
            (im.convert("L") if gray else im).save(dst, compress_level=9)
            if len(lossless) < 40:   # round trip as the server will read it
                lossless.append(bool((np.asarray(Image.open(dst).convert("RGB")) == arr).all()))
            img_dims[fname] = (im.width, im.height, "L" if gray else "RGB")
            n_img += 1
        w, h, _ = img_dims[fname]
        entry = {"i": k, "qid": qid, "dataset": q["dataset"], "question": q["question"], "answers": q["answers"],
                 "page_id": q["page_id"], "paths": {}, "policy": {}}
        for path in PATHS:
            r = rows[qid][path]
            used_text, used_vis = path != "visual", path != "text"
            if (r["text_pages"] != (text_pages if used_text else [])
                    or r["visual_page"] != (vis_page if used_vis else None)):
                n_ret_mismatch += 1
            if used_vis and (w // 28) * (h // 28) != r["visual_tokens"]:
                n_tok_mismatch += 1
            if (abs(vqa_metrics.anls(r["answer"], q["answers"]) - r["anls"]) > 1e-12
                    or vqa_metrics.exact_match(r["answer"], q["answers"]) != r["em"]):
                n_anls_mismatch += 1
            hf = G.build_messages(path, q["question"], texts if used_text else [], None)
            entry["paths"][path] = {
                "messages": to_openai(hf, f"images/{fname}"),
                "image": f"images/{fname}" if used_vis else None,
                "text_pages": text_pages if used_text else [], "visual_page": vis_page if used_vis else None,
                "ref_4090": {k2: r[k2] for k2 in ("answer", "anls", "em", "input_tokens", "visual_tokens",
                                                   "output_tokens", "generate_s", "gpu_seconds", "energy_j")}}
        for name in POLICIES:
            entry["policy"][name] = PATHS[choices[name][i]]
        req_rows.append(entry)
    ver["cache_matches_retrieval_files"] = {"mismatched_rows": n_ret_mismatch}
    ver["image_grid_equals_4090_visual_tokens"] = {"mismatched_rows": n_tok_mismatch,
                                                   "rows_checked": int(2 * len(te_idx))}
    ver["png_roundtrip_lossless"] = {"checked": len(lossless), "all_equal": all(lossless)}
    ver["vqa_metrics_reproduces_stored_anls_em"] = {"mismatched_rows": n_anls_mismatch,
                                                    "rows_checked": int(3 * len(te_idx))}
    assert n_ret_mismatch == 0 and n_tok_mismatch == 0 and n_anls_mismatch == 0 and all(lossless), ver

    with open(out / "requests.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for e in req_rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ---- prompt equivalence: chat template + processor token counts vs generate.py / path_outputs ----
    if check_n:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(G.MODEL, local_files_only=True)
        sample = req_rows if check_n < 0 else random.Random(SEED).sample(req_rows, min(check_n, len(req_rows)))
        n = n_msg = n_chat = n_in = n_vis = 0
        bad = []
        for e in sample:
            q = queries[e["qid"]]
            texts = [ocr[(q["dataset"], pid)] for pid in e["paths"]["text"]["text_pages"]]
            for path in PATHS:
                pe = e["paths"][path]
                ref = G.build_messages(path, q["question"], texts if path != "visual" else [], None)
                mine = to_hf(pe["messages"])
                n += 1
                n_msg += mine == ref
                chat = processor.apply_chat_template(mine, add_generation_prompt=True)
                n_chat += chat == processor.apply_chat_template(ref, add_generation_prompt=True)
                img = [Image.open(out / pe["image"]).convert("RGB")] if pe["image"] else None
                ids = processor(text=[chat], images=img, return_tensors="np")["input_ids"]
                n_tok = int(ids.shape[1])
                n_v = int((ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum())
                n_in += n_tok == pe["ref_4090"]["input_tokens"]
                n_vis += n_v == pe["ref_4090"]["visual_tokens"]
                if n_tok != pe["ref_4090"]["input_tokens"] and len(bad) < 5:
                    bad.append({"qid": e["qid"], "path": path, "tokens": n_tok, "ref": pe["ref_4090"]["input_tokens"]})
        ver["prompt_equivalence"] = {"prompts_checked": n, "queries_checked": len(sample),
                                     "messages_equal_generate_build_messages": n_msg,
                                     "rendered_chat_identical": n_chat, "input_tokens_equal_4090": n_in,
                                     "visual_tokens_equal_4090": n_vis, "examples_mismatch": bad,
                                     "processor": f"AutoProcessor({G.MODEL}) offline, transformers "
                                                  f"{__import__('transformers').__version__}"}
        print("prompt equivalence:", {k: v for k, v in ver["prompt_equivalence"].items() if k != "examples_mismatch"})
        assert n_msg == n_chat == n_in == n_vis == n, ver["prompt_equivalence"]

    # ---- metadata and policy summary ----
    gen_meta = json.load(open(RESULTS / "path_outputs_meta.json", encoding="utf-8"))
    gc = gen_meta["generation_config"]
    sampling = {"temperature": 0.0, "max_tokens": G.MAX_NEW_TOKENS,
                "repetition_penalty": gc.get("repetition_penalty") or 1.0,
                "stop_token_ids": gc["eos_token_id"] if isinstance(gc["eos_token_id"], list) else [gc["eos_token_id"]],
                "seed": SEED}
    summary = {}
    for name in POLICIES:
        ch = choices[name][te_idx]
        summary[name] = {"n_test": int(len(te_idx)),
                         "path_share": {p: float((ch == k).mean()) for k, p in enumerate(PATHS)},
                         "path_count": {p: int((ch == k).sum()) for k, p in enumerate(PATHS)},
                         "anls_4090": checks[name]["recomputed"]["anls"], "em_4090": checks[name]["recomputed"]["em"],
                         "gpu_s_4090_eval_main": checks[name]["recomputed"]["gpu_s"],
                         "generate_s_4090_mean": float(np.mean([rows[qids[i]][PATHS[c]]["generate_s"]
                                                                for i, c in zip(te_idx, ch)]))}
    summary["cavr_ce_valsel"]["lambda"] = lam
    summary["oracle"]["lambda"] = LAMBDA_DEFAULT
    write_json(out / "policies_summary.json", {
        "split": "split 1 = seed 1234 (router_runs/seed0.npz, eval_main.json per_seed[0])",
        "notes": {"cavr_ce_valsel": "content-aware router, lambda chosen on validation (evaluate.py)",
                  "oracle": "per-query argmax ANLS - lambda * measured GPU-s, lambda = 0.1 (evaluate.py default)",
                  "gpu_s_4090_eval_main": "as in eval_main.json: generation + ColPali (+ router overhead for cavr_ce)"},
        "policies": summary})

    sizes = sorted(img_dims.values())
    meta = {
        "description": "Requests for the T4 serving test: split-1 test queries, all three paths, 4090 references",
        "n_requests": len(req_rows), "n_images": n_img, "paths": list(PATHS), "policies": POLICIES,
        "split": {"run": SPLIT_RUN, "seed": SEED, "n_test": int(te.sum()),
                  "order": "evaluation order (routing_data.load qids, test rows) = simulate.py request index"},
        "model_4090": G.MODEL, "model_t4": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "sampling": sampling,
        "sampling_note": "generate.py: greedy (do_sample=False) with the model generation_config otherwise applied, "
                         "i.e. repetition_penalty and the two EOS ids above; temperature 0 = greedy in vLLM",
        "system_prompt": G.SYSTEM, "k_text": G.K_TEXT, "max_chars_per_page": G.MAX_CHARS_PER_PAGE,
        "min_pixels": G.MIN_PIXELS, "max_pixels": G.MAX_PIXELS,
        "image_size_px": {"min_area": int(min(w * h for w, h, _ in sizes)), "max_area": int(max(w * h for w, h, _ in sizes)),
                          "grayscale": int(sum(m == "L" for *_, m in sizes))},
        "message_format": "OpenAI chat; image part {'type': 'image_url', 'image_url': {'url': 'images/<file>.png'}} "
                          "(the client swaps in a data: or file:// URL). Rendered with the model chat template and "
                          "--chat-template-content-format openai this equals generate.py's prompt.",
        "sources": {f: sha1(p) for f, p in (("path_outputs.jsonl", RESULTS / "path_outputs.jsonl"),
                                            ("eval_main.json", RESULTS / "eval_main.json"),
                                            ("router_runs/seed0.npz", RESULTS / "router_runs" / SPLIT_RUN),
                                            ("router_ce/seed0.npz", RESULTS / "router_ce" / SPLIT_RUN),
                                            ("generate.py", HERE.parent / "generate.py"),
                                            ("requests.jsonl", out / "requests.jsonl"))},
        "verification": ver, "build_seconds": round(time.time() - t_start, 1), "hardware": hardware_info()}
    write_json(out / "requests_meta.json", meta)
    print(f"{len(req_rows)} requests, {n_img} images -> {out}")
    return meta


if __name__ == "__main__":
    main()
