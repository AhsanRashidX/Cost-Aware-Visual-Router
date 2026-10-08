"""Record the system, library versions and pinned model/dataset revisions used by the pipeline.

Revisions are read from the local Hugging Face cache (refs/main of each downloaded snapshot), so they are the exact
weights the experiments loaded. Output: results/system_info.json
"""
import os
from pathlib import Path

from common import RESULTS, hardware_info, write_json

MODELS = ["Qwen/Qwen2.5-VL-7B-Instruct", "vidore/colpali-v1.3", "vidore/colpaligemma-3b-pt-448-base",
          "sentence-transformers/all-mpnet-base-v2"]
DATASET = ("lmms-lab-encoder/DocVQA", "lmms-lab-encoder___doc_vqa")


def main():
    hub = Path(os.environ["HF_HOME"]) / "hub"
    revs = {}
    for m in MODELS:
        ref = hub / ("models--" + m.replace("/", "--")) / "refs" / "main"
        revs[m] = ref.read_text().strip() if ref.exists() else None
    ds_dir = Path(os.environ["HF_HOME"]) / "datasets" / DATASET[1] / "DocVQA" / "0.0.0"
    ds_rev = sorted(p.name for p in ds_dir.iterdir())[0] if ds_dir.exists() else None
    write_json(RESULTS / "system_info.json", {"hardware": hardware_info(), "model_revisions": revs,
                                               "dataset": {"name": DATASET[0], "revision": ds_rev}})
    print(revs, ds_rev)


if __name__ == "__main__":
    main()
