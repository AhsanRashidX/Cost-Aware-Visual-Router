"""Step 2: build the query table, page store, samples and manifest from DocVQA + InfoVQA validation.

Outputs (under CAVR/revision/data/):
  pages/<dataset>/<page_id>.<ext>   original image bytes, one file per unique page
  queries.jsonl                     one row per question, both datasets in one schema
  samples.json                      qids of the natural and stratified samples
  manifest.json                     counts, sources, dataset revision, hardware
"""
import hashlib
import io
import json
import random
from collections import Counter, defaultdict

from PIL import Image

from common import DATA, SEED, hardware_info, write_json

DATASET = "lmms-lab-encoder/DocVQA"
SPLIT = "validation"
STRATUM_CAP = 150  # max queries per (dataset, type) stratum in the stratified sample


def docvqa_rows(ds):
    for r in ds:
        page_id = f"{r['ucsf_document_id']}_{r['ucsf_document_page_no']}"
        yield {
            "qid": f"docvqa-{r['questionId']}",
            "dataset": "docvqa",
            "question": r["question"],
            "answers": r["answers"],
            "page_id": page_id,
            "qtype": r["question_types"][0] if r["question_types"] else "unknown",
            "qtypes_all": r["question_types"],
        }, r["image"]


def infovqa_rows(ds):
    for r in ds:
        page_id = hashlib.sha1(r["image_url"].encode()).hexdigest()[:16]
        reasoning = r["operation/reasoning"]
        yield {
            "qid": f"infovqa-{r['questionId']}",
            "dataset": "infovqa",
            "question": r["question"],
            "answers": r["answers"],
            "page_id": page_id,
            # reasoning type when present (comparison/arithmetic/counting), else answer type
            "qtype": reasoning[0] if reasoning else (r["answer_type"][0] if r["answer_type"] else "unknown"),
            "qtypes_all": r["answer_type"] + reasoning,
            "image_url": r["image_url"],
        }, r["image"]


def main():
    from datasets import Image as HFImage, load_dataset

    pages_meta, queries = {}, []
    sources = {}
    for cfg, name, rows_fn in [("DocVQA", "docvqa", docvqa_rows), ("InfographicVQA", "infovqa", infovqa_rows)]:
        ds = load_dataset(DATASET, cfg, split=SPLIT)
        sources[name] = {"hf_dataset": DATASET, "config": cfg, "split": SPLIT,
                         "fingerprint": ds._fingerprint, "rows": len(ds)}
        if "ocr" in ds.column_names:
            ds = ds.remove_columns("ocr")  # we run our own OCR (docTR) for all pages
        ds = ds.cast_column("image", HFImage(decode=False))
        out_dir = DATA / "pages" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for q, img in rows_fn(ds):
            queries.append(q)
            key = (name, q["page_id"])
            digest = hashlib.sha1(img["bytes"]).hexdigest()
            if key in pages_meta:
                assert pages_meta[key]["sha1"] == digest, f"page id collision: {key}"
                continue
            im = Image.open(io.BytesIO(img["bytes"]))
            ext = {"PNG": "png", "JPEG": "jpg"}.get(im.format, im.format.lower())
            path = out_dir / f"{q['page_id']}.{ext}"
            if not path.exists():
                path.write_bytes(img["bytes"])
            pages_meta[key] = {"dataset": name, "page_id": q["page_id"], "path": str(path.relative_to(DATA)),
                               "width": im.width, "height": im.height, "mode": im.mode, "sha1": digest,
                               "bytes": len(img["bytes"])}

    qids = [q["qid"] for q in queries]
    assert len(qids) == len(set(qids)), "duplicate qids"

    with open(DATA / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(DATA / "pages.jsonl", "w", encoding="utf-8") as f:
        for p in pages_meta.values():
            f.write(json.dumps(p) + "\n")

    # Natural sample: every validation question, in dataset order.
    natural = qids
    # Stratified sample: up to STRATUM_CAP per (dataset, qtype), seeded.
    rng = random.Random(SEED)
    strata = defaultdict(list)
    for q in queries:
        strata[(q["dataset"], q["qtype"])].append(q["qid"])
    stratified = []
    for key in sorted(strata):
        members = strata[key]
        stratified += rng.sample(members, min(STRATUM_CAP, len(members)))
    write_json(DATA / "samples.json", {
        "seed": SEED,
        "natural": natural,
        "stratified": stratified,
        "stratified_note": f"up to {STRATUM_CAP} per (dataset, qtype); a route-label-stratified set is built after step 4",
    })

    def counts(qs):
        return {
            "queries": len(qs),
            "by_dataset": dict(Counter(q["dataset"] for q in qs)),
            "by_dataset_qtype": {f"{d}/{t}": n for (d, t), n in sorted(Counter((q["dataset"], q["qtype"]) for q in qs).items())},
        }

    by_qid = {q["qid"]: q for q in queries}
    page_list = list(pages_meta.values())
    write_json(DATA / "manifest.json", {
        "sources": sources,
        "pages": {
            "total": len(page_list),
            "by_dataset": dict(Counter(p["dataset"] for p in page_list)),
            "total_bytes": sum(p["bytes"] for p in page_list),
            "median_megapixels": sorted(p["width"] * p["height"] / 1e6 for p in page_list)[len(page_list) // 2],
        },
        "natural": counts(queries),
        "stratified": counts([by_qid[i] for i in stratified]),
        "notes": [
            "DocVQA page_id = ucsf_document_id + '_' + page_no; InfoVQA page_id = sha1(image_url)[:16].",
            "Bundled InfoVQA Textract OCR is dropped; all pages are OCR'd with docTR in ocr_pages.py.",
            "qtype: DocVQA first question_type; InfoVQA reasoning type if any, else first answer_type.",
        ],
        "hardware": hardware_info(),
    })
    print(json.dumps({"queries": len(queries), "pages": len(page_list), "stratified": len(stratified)}))


if __name__ == "__main__":
    main()
