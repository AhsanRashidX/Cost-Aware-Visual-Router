"""Shared routing data: cached path outputs, page-grouped splits, measured costs, outcome labels."""
import json
import random
from collections import defaultdict

import numpy as np

from common import DATA, RESULTS

PATHS = ("text", "visual", "hybrid")
COST_KEY = "gpu_seconds"           # primary cost: measured GPU-seconds per query on this GPU
LAMBDA_DEFAULT = 0.1               # ANLS per GPU-second; a runtime knob, swept in evaluation
LAMBDA_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
ANLS_OK = 0.5                      # "good answer" threshold for the threshold label and confidence
SPLIT_FRACS = (0.6, 0.2, 0.2)


def load(outputs=RESULTS / "path_outputs.jsonl"):
    """Return (qids, Q, A, C, rows): per-query arrays over PATHS for queries with all paths cached.

    A[i, p] = ANLS, C[i, p] = measured cost (COST_KEY), rows[qid][path] = full cached row.
    """
    rows = defaultdict(dict)
    for r in map(json.loads, open(outputs, encoding="utf-8")):
        rows[r["qid"]][r["path"]] = r
    queries = {q["qid"]: q for q in map(json.loads, open(DATA / "queries.jsonl", encoding="utf-8"))}
    qids = [q for q in queries if q in rows and all(p in rows[q] for p in PATHS)]
    A = np.array([[rows[q][p]["anls"] for p in PATHS] for q in qids])
    C = np.array([[rows[q][p][COST_KEY] for p in PATHS] for q in qids])
    return qids, [queries[q] for q in qids], A, C, rows


def group_split(Q, seed):
    """60/20/20 split by page (all questions on a page share a split), per dataset."""
    rng = random.Random(seed)
    split = {}
    for ds in ("docvqa", "infovqa"):
        pages = sorted({q["page_id"] for q in Q if q["dataset"] == ds})
        rng.shuffle(pages)
        n_tr, n_va = int(SPLIT_FRACS[0] * len(pages)), int(SPLIT_FRACS[1] * len(pages))
        for i, p in enumerate(pages):
            split[(ds, p)] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return np.array([split[(q["dataset"], q["page_id"])] for q in Q])


def path_costs(C, mask):
    """Expected per-path cost, estimated on the training split only."""
    return C[mask].mean(axis=0)


def utility_labels(A, cost_vec, lam):
    """argmax_p ANLS - lam * cost; ties go to the cheaper path."""
    u = A - lam * cost_vec[None, :]
    order = np.argsort(cost_vec)  # cheapest first, so argmax on the reordered matrix breaks ties cheaply
    return order[np.argmax(u[:, order] >= u.max(axis=1, keepdims=True) - 1e-12, axis=1)]


def threshold_labels(A, cost_vec, tau=ANLS_OK):
    """Cheapest path with ANLS >= tau; if none reaches tau, the cheapest path (nothing helps)."""
    order = np.argsort(cost_vec)
    ok = A[:, order] >= tau
    first = np.argmax(ok, axis=1)
    return np.where(ok.any(axis=1), order[first], order[0])


def keyword_submitted(question):
    """The submitted paper's labelling rule, verbatim: patterns tested as plain substrings
    (so regex-looking patterns never match). Mapped to paths text/visual/hybrid."""
    q = question.lower()
    visual = ['where is', 'located', 'position', 'placement', 'layout', 'color', 'shape', 'size',
              'appearance', 'look like', 'what does the (diagram|chart|graph|figure|image|infographic) show',
              'what is (shown|depicted|illustrated|displayed)', 'arrangement of', 'placed', 'situated']
    hybrid = ['why', 'how (does|do|is|are|can)', 'explain', 'compare', 'contrast', 'difference between',
              'relationship between', 'trend', 'pattern', 'what can (you|we) (infer|conclude|learn|determine)',
              'what information', 'what insight', 'what conclusion', 'what does this (mean|indicate|suggest)',
              'interpret']
    if any(p in q for p in hybrid):
        return "hybrid"
    if any(p in q for p in visual):
        return "visual"
    return "text"


def keyword_regex(question):
    """Same rule with the patterns applied as regular expressions (the evident intent)."""
    import re
    q = question.lower()
    visual = [r'where is', r'located', r'position', r'placement', r'layout', r'colou?r', r'shape', r'size',
              r'appearance', r'look like', r'what does the (diagram|chart|graph|figure|image|infographic) show',
              r'what is (shown|depicted|illustrated|displayed)', r'arrangement of', r'placed', r'situated']
    hybrid = [r'why', r'how (does|do|is|are|can)', r'explain', r'compare', r'contrast', r'difference between',
              r'relationship between', r'trend', r'pattern', r'what can (you|we) (infer|conclude|learn|determine)',
              r'what information', r'what insight', r'what conclusion', r'what does this (mean|indicate|suggest)',
              r'interpret']
    if any(re.search(p, q) for p in hybrid):
        return "hybrid"
    if any(re.search(p, q) for p in visual):
        return "visual"
    return "text"
