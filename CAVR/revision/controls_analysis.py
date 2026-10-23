"""Retrieval-depth and gold-page controls (controls.py) against the cached main paths, on the same questions.

  matched depth    visual_k3 vs the main text path (both 3 pages) and the main visual path vs text_k1 (both 1 page):
                   separates the input modality from the number of retrieved pages
  gold page        gold_text / gold_visual / gold_hybrid vs the main paths: how much of each path's loss is retrieval
  reproduction     check_text / check_visual / check_hybrid on the A10 node vs the cached workstation answers
Differences are paired per question; 95% intervals from a page-cluster bootstrap (1000 resamples, fixed seed).
Inputs: results/controls/controls_*.jsonl, results/path_outputs.jsonl, results/retrieval_*.jsonl
Output: results/controls_analysis.json
"""
import json
from collections import defaultdict

import numpy as np

from common import RESULTS, SEED, hardware_info, write_json

MAIN = ["text", "visual", "hybrid"]
CONTROLS = ["text_k1", "visual_k3", "gold_text", "gold_visual", "gold_hybrid"]
DIFFS = {  # name: (a, b) -> mean ANLS(a) - ANLS(b)
    "visual_k3_minus_text": ("visual_k3", "text"),
    "visual_minus_text_k1": ("visual", "text_k1"),
    "visual_k3_minus_visual": ("visual_k3", "visual"),
    "text_minus_text_k1": ("text", "text_k1"),
    "gold_visual_minus_gold_text": ("gold_visual", "gold_text"),
    "gold_text_minus_text": ("gold_text", "text"),
    "gold_visual_minus_visual": ("gold_visual", "visual"),
    "gold_hybrid_minus_hybrid": ("gold_hybrid", "hybrid"),
    "gold_hybrid_minus_gold_visual": ("gold_hybrid", "gold_visual"),
}


def rows(path):
    return list(map(json.loads, open(path, encoding="utf-8")))


def main():
    ctrl = [r for f in sorted((RESULTS / "controls").glob("controls_*.jsonl")) for r in rows(f)]
    by = defaultdict(dict)                                  # qid -> condition -> row
    for r in ctrl:
        by[r["qid"]][r["condition"]] = r
    cached = defaultdict(dict)
    for r in rows(RESULTS / "path_outputs.jsonl"):
        if r["qid"] in by:
            cached[r["qid"]][r["path"]] = r
    rt = {r["qid"]: r for r in rows(RESULTS / "retrieval_text.jsonl") if r["qid"] in by}
    rv = {r["qid"]: r for r in rows(RESULTS / "retrieval_visual.jsonl") if r["qid"] in by}

    qids = sorted(q for q in by if all(c in by[q] for c in CONTROLS) and all(p in cached[q] for p in MAIN))
    ds = np.array([by[q][CONTROLS[0]]["dataset"] for q in qids])
    gold = {q: by[q]["gold_text"]["text_pages"][0] for q in qids}
    S = {c: np.array([by[q][c]["anls"] for q in qids]) for c in CONTROLS}
    S.update({p: np.array([cached[q][p]["anls"] for q in qids]) for p in MAIN})
    tok = {c: float(np.mean([by[q][c]["input_tokens"] for q in qids])) for c in CONTROLS}
    tok.update({p: float(np.mean([cached[q][p]["input_tokens"] for q in qids])) for p in MAIN})

    pages = np.array([f"{d}/{gold[q]}" for d, q in zip(ds, qids)])
    rng = np.random.default_rng(SEED)

    def boot(d, mask):
        up, pidx = np.unique(pages[mask], return_inverse=True)
        b = rng.integers(0, len(up), (1000, len(up)))
        sums, cnts = np.bincount(pidx, d[mask], len(up)), np.bincount(pidx, None, len(up))
        m = sums[b].sum(1) / cnts[b].sum(1)
        return [float(x) for x in np.percentile(m, [2.5, 97.5])]

    out = {"n_questions": len(qids), "conditions": MAIN + CONTROLS, "by_dataset": {}}
    for name, mask in (("all", np.ones(len(qids), bool)), ("docvqa", ds == "docvqa"), ("infovqa", ds == "infovqa")):
        block = {"n": int(mask.sum()),
                 "anls": {c: float(S[c][mask].mean()) for c in S},
                 "recall@1_bm25": float(np.mean([gold[q] == rt[q]["top"][0] for q, m in zip(qids, mask) if m])),
                 "recall@3_bm25": float(np.mean([gold[q] in rt[q]["top"][:3] for q, m in zip(qids, mask) if m])),
                 "recall@1_colpali": float(np.mean([gold[q] == rv[q]["top"][0] for q, m in zip(qids, mask) if m])),
                 "recall@3_colpali": float(np.mean([gold[q] in rv[q]["top"][:3] for q, m in zip(qids, mask) if m])),
                 "diffs": {}}
        for k, (a, b) in DIFFS.items():
            d = S[a] - S[b]
            block["diffs"][k] = {"mean": float(d[mask].mean()), "ci95": boot(d, mask)}
        out["by_dataset"][name] = block
    out["input_tokens_mean"] = tok

    # A10 vs workstation reproduction on the check questions
    rep = {}
    for c, p in (("check_text", "text"), ("check_visual", "visual"), ("check_hybrid", "hybrid")):
        qs = [q for q in by if c in by[q] and p in cached[q]]
        same = [by[q][c]["answer"].strip() == cached[q][p]["answer"].strip() for q in qs]
        a10 = np.array([by[q][c]["anls"] for q in qs])
        ws = np.array([cached[q][p]["anls"] for q in qs])
        rep[p] = {"n": len(qs), "identical_answer_share": float(np.mean(same)), "anls_a10": float(a10.mean()),
                  "anls_workstation": float(ws.mean()), "max_abs_anls_diff": float(np.abs(a10 - ws).max()),
                  "generate_s_a10_mean": float(np.mean([by[q][c]["generate_s"] for q in qs])),
                  "generate_s_workstation_mean": float(np.mean([cached[q][p]["generate_s"] for q in qs]))}
    out["reproduction"] = rep
    out["generate_s_a10_mean"] = {c: float(np.mean([by[q][c]["generate_s"] for q in qids])) for c in CONTROLS}
    out["hardware"] = hardware_info()
    write_json(RESULTS / "controls_analysis.json", out)
    a = out["by_dataset"]["all"]
    print(json.dumps({"anls": a["anls"], "diffs": a["diffs"], "reproduction": rep}, indent=1))


if __name__ == "__main__":
    main()
