"""Step 2b: OCR every page with docTR on the GPU; record per-page time and energy.

The OCR text feeds the cheap (text) path; its cost is that path's offline indexing cost.
Outputs: data/ocr.jsonl (page text), results/ocr_cost.json (timing, energy, hardware).
"""
import json
import time

import numpy as np
import pynvml
import torch
from PIL import Image

from common import DATA, RESULTS, hardware_info, set_seed, write_json

DET_ARCH, RECO_ARCH = "db_resnet50", "crnn_vgg16_bn"
MAX_ASPECT = 2.0  # pages taller than 2x their width (long infographics) are split into vertical tiles


def tiles(im):
    """Split tall images into stacked tiles so docTR's resize doesn't shrink the text away."""
    w, h = im.size
    if h <= MAX_ASPECT * w:
        return [im]
    step = int(MAX_ASPECT * w * 0.75)
    return [im.crop((0, top, w, min(top + step, h))) for top in range(0, h, step)]


def page_text(doc):
    lines = []
    for page in doc.pages:
        for block in page.blocks:
            for line in block.lines:
                lines.append(" ".join(w.value for w in line.words))
    return "\n".join(lines)


def main():
    from doctr.models import ocr_predictor

    set_seed()
    model = ocr_predictor(det_arch=DET_ARCH, reco_arch=RECO_ARCH, pretrained=True).cuda().eval()
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    pages = [json.loads(l) for l in open(DATA / "pages.jsonl", encoding="utf-8")]

    # warm-up so CUDA init / cudnn autotune isn't charged to the first page
    with torch.inference_mode():
        model([np.zeros((1024, 768, 3), dtype=np.uint8)])
    torch.cuda.synchronize()

    per_page = []
    with open(DATA / "ocr.jsonl", "w", encoding="utf-8") as out, torch.inference_mode():
        for i, p in enumerate(pages):
            im = Image.open(DATA / p["path"]).convert("RGB")
            parts = tiles(im)
            torch.cuda.reset_peak_memory_stats()
            e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
            t0 = time.perf_counter()
            text = "\n".join(page_text(model([np.asarray(t)])) for t in parts)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            energy_j = (pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu) - e0) / 1000
            rec = {"dataset": p["dataset"], "page_id": p["page_id"], "tiles": len(parts),
                   "seconds": dt, "energy_j": energy_j,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "chars": len(text)}
            per_page.append(rec)
            out.write(json.dumps({"dataset": p["dataset"], "page_id": p["page_id"], "text": text},
                                 ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(pages)} {dt:.2f}s {len(text)} chars", flush=True)

    secs = np.array([r["seconds"] for r in per_page])
    write_json(RESULTS / "ocr_cost.json", {
        "engine": f"docTR {DET_ARCH}+{RECO_ARCH}", "max_aspect_before_tiling": MAX_ASPECT,
        "pages": len(per_page),
        "total_seconds": float(secs.sum()),
        "p50_seconds": float(np.percentile(secs, 50)), "p95_seconds": float(np.percentile(secs, 95)),
        "total_energy_j": float(sum(r["energy_j"] for r in per_page)),
        "max_peak_mem_gb": float(max(r["peak_mem_gb"] for r in per_page)),
        "empty_pages": sum(r["chars"] == 0 for r in per_page),
        "per_page": per_page,
        "hardware": hardware_info(),
    })
    print(f"done: {len(per_page)} pages, {secs.sum():.0f}s total")


if __name__ == "__main__":
    main()
