"""Analyses requested by the third review round, computed from cached results only (no model is re-run).

1. Union page recall (R6 minor): the gold page in BM25's top k OR ColPali's top k, per dataset, on the evaluation
   set and the confirmation set (results*/retrieval_{text,visual}.jsonl).
2. Router-versus-oracle confusion (R6 minor): CAVR-CE at its validation-chosen lambda against the per-query oracle at
   the same lambda (evaluate.oracle, true per-query cost), on each split's test pages, pooled and per dataset.
3. Memory held per query (R6 minor): peak reserved GPU memory during generation (model weights included) times the
   query's GPU-seconds, in GiB-seconds, per path and per policy; and the evaluation index's residency.
4. Path set and input budgets (R6 concern 4): the retrieval-depth controls (1,500 questions, generation re-run on the
   A10 node) put text top-1 and visual top-3 on the same cost-quality plane as the three paths. Generation time of
   the three paths on the A10 is the cached workstation time of each question scaled by the A10/workstation ratio
   measured on the same 200 questions in the same session (controls_analysis.json, reproduction). Reported: input
   tokens, ANLS per 1,000 input tokens, membership of the upper convex hull of fixed policies, and the per-query
   oracle with three and with five paths.
5. Where routing helps (R2.2, R3): CAVR-CE minus always-visual and always-hybrid, per dataset and question type,
   on the main splits (t interval over splits) and on the confirmation set (page-cluster bootstrap); and how often
   a path with OCR text answers better than the page image alone.
6. Escalation against the lambda sweep (R2.1, R6 minor): for CAVR-CE, the ANLS of every escalation operating point
   minus the ANLS of the lambda frontier (upper hull of the lambda sweep) at the same GPU-seconds.
Output: results/review_round3.json
"""
import json

import numpy as np

from common import DATA, RESULTS, ROOT, SEED, write_json
from evaluate import oracle
from review_round2 import N_BOOT, TXT, VIS, HYB, confirm_data, t_interval
from routing_data import PATHS, load

FIVE = ("text_k1", "text", "visual", "hybrid", "visual_k3")


def jl(path):
    return [json.loads(x) for x in open(path, encoding="utf-8")]


def ci(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def page_boot(groups, seed=SEED, n=N_BOOT):
    up, pidx = np.unique(groups, return_inverse=True)
    pick = np.random.default_rng(seed).integers(0, len(up), (n, len(up)))
    cnt = np.bincount(pidx, minlength=len(up)).astype(float)

    def boot(x):
        s = np.bincount(pidx, np.asarray(x, float), len(up))
        return s[pick].sum(1) / cnt[pick].sum(1)
    return boot


# 1 ------------------------------------------------------------------------------------------------------------
def union_recall():
    out = {}
    for name, res, dat in (("evaluation", RESULTS, DATA), ("confirmation", ROOT / "results_confirm", ROOT / "data_confirm")):
        Q = {q["qid"]: q for q in jl(dat / "queries.jsonl")}
        bt = {r["qid"]: r for r in jl(res / "retrieval_text.jsonl")}
        bv = {r["qid"]: r for r in jl(res / "retrieval_visual.jsonl")}
        qids = [q for q in Q if q in bt and q in bv]
        out[name] = {}
        for ds in ("all", "docvqa", "infovqa"):
            sel = [q for q in qids if ds == "all" or Q[q]["dataset"] == ds]
            r = {"n": len(sel)}
            for k in (1, 3):
                inb = np.array([Q[q]["page_id"] in bt[q]["top"][:k] for q in sel])
                inv = np.array([Q[q]["page_id"] in bv[q]["top"][:k] for q in sel])
                r[f"bm25@{k}"], r[f"colpali@{k}"] = float(inb.mean()), float(inv.mean())
                r[f"union@{k}"], r[f"both@{k}"] = float((inb | inv).mean()), float((inb & inv).mean())
            # what the generator is actually given: text = BM25 top 3, visual = ColPali top 1, hybrid = both
            inb3 = np.array([Q[q]["page_id"] in bt[q]["top"][:3] for q in sel])
            inv1 = np.array([Q[q]["page_id"] in bv[q]["top"][:1] for q in sel])
            r["hybrid_context_recall"] = float((inb3 | inv1).mean())
            out[name][ds] = r
    return out


# 2, 3, 5 (main splits) ---------------------------------------------------------------------------------------
def main_splits():
    qids, Q, A, C, rows = load()
    ev = json.load(open(RESULTS / "eval_main.json"))
    ce_over = json.load(open(RESULTS / "router_ce_cost.json"))["per_query_batch1"]["gpu_s"]
    VR = np.array([rows[q]["hybrid"]["visual_retrieval_gpu_s"] for q in qids])
    MEM = np.array([[rows[q][p]["gen_peak_reserved_gb"] for p in PATHS] for q in qids])
    DS = np.array([q["dataset"] for q in Q])
    QT = np.array([q["qtype"] for q in Q])
    conf = {"pooled": np.zeros((3, 3), int), "docvqa": np.zeros((3, 3), int), "infovqa": np.zeros((3, 3), int)}
    regret, per_seed_where, mem_rows = [], [], []
    for k in range(5):
        R = np.load(RESULTS / "router_runs" / f"seed{k}.npz", allow_pickle=True)
        G = np.load(RESULTS / "router_ce" / f"seed{k}.npz", allow_pickle=True)
        assert list(R["qids"]) == qids and list(G["qids"]) == qids
        te, cost = R["split"] == "test", R["cost"]
        lam = ev["per_seed"][k]["policies"]["cavr_ce_valsel"]["lambda"]
        ch = np.argmax(G["qhat_ce_full"][te] - lam * cost, 1)
        orc = oracle(A[te], C[te], lam)
        i = np.arange(te.sum())
        for key, m in (("pooled", np.ones(te.sum(), bool)), ("docvqa", DS[te] == "docvqa"), ("infovqa", DS[te] == "infovqa")):
            np.add.at(conf[key], (ch[m], orc[m]), 1)
        a_ce, a_or = A[te][i, ch], A[te][i, orc]
        g_ce = C[te][i, ch] + (ch == TXT) * VR[te] + ce_over
        best = A[te] >= A[te].max(1, keepdims=True) - 1e-12        # paths that reach the query's best ANLS
        regret.append({"agreement": float((ch == orc).mean()),
                       "router_path_reaches_best_anls": float(best[i, ch].mean()),
                       "oracle_tie_share": float((best.sum(1) > 1).mean()),
                       "all_paths_fail_share": float((A[te].max(1) == 0).mean()),
                       "anls_router": float(a_ce.mean()),
                       "anls_oracle": float(a_or.mean()), "gpu_s_router": float(g_ce.mean()),
                       "gpu_s_oracle": float(C[te][i, orc].mean())})
        # memory held: peak reserved GiB x GPU-seconds of the query (ColPali charged at the visual path's footprint)
        mt = MEM[te] * C[te]
        mem_rows.append({"cavr_ce_valsel": float((mt[i, ch] + (ch == TXT) * VR[te] * MEM[te][:, VIS]).mean()),
                         "always_visual": float(mt[:, VIS].mean()), "always_hybrid": float(mt[:, HYB].mean()),
                         "always_text": float(mt[:, TXT].mean())})
        w = {}
        for ds in ("docvqa", "infovqa"):
            for t in ["all"] + sorted(set(QT[te][DS[te] == ds])):
                m = (DS[te] == ds) & ((QT[te] == t) if t != "all" else True)
                if m.sum() < 20:
                    continue
                w[f"{ds}/{t}"] = {"n": int(m.sum()),
                                  "ce_minus_visual": float((a_ce - A[te][:, VIS])[m].mean()),
                                  "ce_minus_hybrid": float((a_ce - A[te][:, HYB])[m].mean()),
                                  "ocr_helps_rate": float((A[te][m][:, [TXT, HYB]].max(1) > A[te][m][:, VIS]).mean()),
                                  "ocr_hurts_rate": float((A[te][m][:, HYB] < A[te][m][:, VIS]).mean()),
                                  "ce_path_mix": [float((ch[m] == j).mean()) for j in range(3)]}
        per_seed_where.append(w)
    keys = set.intersection(*(set(w) for w in per_seed_where))
    where = {}
    for key in sorted(keys):
        where[key] = {"n_mean": float(np.mean([w[key]["n"] for w in per_seed_where]))}
        for f in ("ce_minus_visual", "ce_minus_hybrid", "ocr_helps_rate", "ocr_hurts_rate"):
            where[key][f] = t_interval([w[key][f] for w in per_seed_where])
        where[key]["ce_path_mix"] = [float(np.mean([w[key]["ce_path_mix"][j] for w in per_seed_where])) for j in range(3)]
    norm = {k: (v / v.sum(1, keepdims=True).clip(1)).round(4).tolist() for k, v in conf.items()}
    mem = {p: t_interval([r[p] for r in mem_rows]) for p in mem_rows[0]}
    mem["saving_vs_hybrid"] = t_interval([1 - r["cavr_ce_valsel"] / r["always_hybrid"] for r in mem_rows])
    mem["extra_vs_visual"] = t_interval([r["cavr_ce_valsel"] / r["always_visual"] - 1 for r in mem_rows])
    mem["peak_reserved_gib_mean"] = {p: float(MEM[:, j].mean()) for j, p in enumerate(PATHS)}
    ix = json.load(open(RESULTS / "index_cost_visual.json"))["index"]
    mem["eval_index_gib"] = ix["index_bytes_bf16"] / 2**30
    return {"confusion": {"rows": "router choice (text, visual, hybrid)", "cols": "oracle choice",
                          "counts": {k: v.tolist() for k, v in conf.items()}, "row_normalised": norm,
                          "per_split": regret,
                          "agreement": t_interval([r["agreement"] for r in regret]),
                          "router_path_reaches_best_anls": t_interval([r["router_path_reaches_best_anls"] for r in regret]),
                          "oracle_tie_share": t_interval([r["oracle_tie_share"] for r in regret]),
                          "all_paths_fail_share": t_interval([r["all_paths_fail_share"] for r in regret]),
                          "anls_gap_to_oracle": t_interval([r["anls_oracle"] - r["anls_router"] for r in regret])},
            "memory_time_gib_s": mem, "where_routing_helps_main": where}


# 5 (confirmation set) ------------------------------------------------------------------------------------------
def where_confirmation():
    rows, Qd = confirm_data()
    qh = np.load(ROOT / "results_confirm" / "confirm_ce_qhat.npz", allow_pickle=True)
    meta = json.load(open(ROOT / "results_confirm" / "confirm_ce_meta.json"))
    qids = list(qh["qids"])
    A = np.array([[rows[q][p]["anls"] for p in PATHS] for q in qids])
    i = np.arange(len(qids))
    ch = [np.argmax(qh[f"qhat_seed{k}"] - s["lambda_valsel"] * qh[f"cost_seed{k}"], 1) for k, s in enumerate(meta["seeds"])]
    a = np.mean([A[i, c] for c in ch], 0)
    DS = np.array([Qd[q]["dataset"] for q in qids])
    pages = np.array([f"{Qd[q]['dataset']}/{Qd[q]['page_id']}" for q in qids])
    out = {}
    for ds in ("docvqa", "infovqa"):
        m = DS == ds
        boot = page_boot(pages[m])
        out[ds] = {"n": int(m.sum()),
                   "ce_minus_visual": {"mean": float((a - A[:, VIS])[m].mean()), "ci95": ci(boot((a - A[:, VIS])[m]))},
                   "ce_minus_hybrid": {"mean": float((a - A[:, HYB])[m].mean()), "ci95": ci(boot((a - A[:, HYB])[m]))},
                   "ocr_helps_rate": float((A[m][:, [TXT, HYB]].max(1) > A[m][:, VIS]).mean())}
    return out


# 4 -------------------------------------------------------------------------------------------------------------
def upper_hull(pts):
    """Indices of the points on the upper-left convex hull of (cost, quality), cheapest first."""
    order = sorted(range(len(pts)), key=lambda j: (pts[j][0], -pts[j][1]))
    hull = []
    for j in order:
        if hull and pts[j][1] <= pts[hull[-1]][1]:
            continue                                  # dearer and no better: dominated
        while len(hull) >= 2:
            (x1, y1), (x2, y2), (x3, y3) = pts[hull[-2]], pts[hull[-1]], pts[j]
            if (y2 - y1) * (x3 - x1) <= (y3 - y1) * (x2 - x1):
                hull.pop()                            # middle point lies on or below the chord
            else:
                break
        hull.append(j)
    return hull


def path_set():
    ca = json.load(open(RESULTS / "controls_analysis.json"))
    ctl = {}
    for f in sorted((RESULTS / "controls").glob("controls_*.jsonl")):
        for r in jl(f):
            ctl.setdefault(r["qid"], {})[r["condition"]] = r
    qids_all, Q, A, C, rows = load()
    qset = [q for q in ctl if all(c in ctl[q] for c in ("text_k1", "visual_k3")) and q in rows]
    rep = ca["reproduction"]
    scale = {p: rep[p]["generate_s_a10_mean"] / rep[p]["generate_s_workstation_mean"] for p in PATHS}
    Qd = {q["qid"]: q for q in Q}

    def get(q, c):
        if c in PATHS:
            r = rows[q][c]
            return r["anls"], r["generate_s"] * scale[c], r["input_tokens"]
        r = ctl[q][c]
        return r["anls"], r["generate_s"], r["input_tokens"]
    out = {"n_questions": len(qset), "cost": "A10 generation seconds (ColPali retrieval, ~0.1 s, not included)",
           "a10_over_workstation_scale": scale}
    for ds in ("all", "docvqa", "infovqa"):
        sel = [q for q in qset if ds == "all" or Qd[q]["dataset"] == ds]
        X = np.array([[get(q, c) for c in FIVE] for q in sel])          # (n, 5, 3): anls, seconds, tokens
        pages = np.array([f"{Qd[q]['dataset']}/{Qd[q]['page_id']}" for q in sel])
        boot = page_boot(pages)
        pts = [(float(X[:, j, 1].mean()), float(X[:, j, 0].mean())) for j in range(5)]
        hull = upper_hull(pts)
        r = {"n": len(sel), "paths": {}}
        for j, c in enumerate(FIVE):
            r["paths"][c] = {"anls": pts[j][1], "anls_ci95": ci(boot(X[:, j, 0])), "gen_s": pts[j][0],
                             "input_tokens": float(X[:, j, 2].mean()),
                             "anls_per_1k_tokens": float(1000 * pts[j][1] / X[:, j, 2].mean()),
                             "on_upper_hull": j in hull}
        # quality the hull reaches at each path's cost (how far below the frontier a path sits)
        hx = [pts[j] for j in hull]
        for j, c in enumerate(FIVE):
            x = pts[j][0]
            y = np.interp(x, [p[0] for p in hx], [p[1] for p in hx]) if hx[0][0] <= x <= hx[-1][0] else pts[j][1]
            r["paths"][c]["hull_anls_at_same_cost"] = float(y)
        # paired differences that decide the comparisons R6 asked about
        d = lambda a, b: {"mean": float((X[:, FIVE.index(a), 0] - X[:, FIVE.index(b), 0]).mean()),
                          "ci95": ci(boot(X[:, FIVE.index(a), 0] - X[:, FIVE.index(b), 0]))}
        r["diff"] = {"visual_k3_minus_hybrid": d("visual_k3", "hybrid"), "text_k1_minus_text": d("text_k1", "text"),
                     "visual_minus_text_equal_tokens": d("visual", "text")}
        # per-query oracle with 3 and 5 paths at a few cost weights (per generation second)
        orc = {}
        for lam in (0.0, 0.05, 0.1, 0.2):
            for name, cols in (("three_paths", [1, 2, 3]), ("five_paths", list(range(5)))):
                Aq, Cq = X[:, cols, 0], X[:, cols, 1]
                o = oracle(Aq, Cq, lam)
                ii = np.arange(len(o))
                orc[f"{name}@{lam}"] = {"anls": float(Aq[ii, o].mean()), "gen_s": float(Cq[ii, o].mean()),
                                        "mix": {FIVE[cols[m]]: float((o == m).mean()) for m in range(len(cols))}}
        r["oracle"] = orc
        out[ds] = r
    return out


# 6 -------------------------------------------------------------------------------------------------------------
def escalation_vs_lambda():
    """Frontier = upper hull of CAVR-CE over a fine lambda grid plus the fixed policies, with the same accounting as
    escalation.json (ColPali charged to text-routed queries, CE overhead on every query)."""
    es = json.load(open(RESULTS / "escalation.json"))
    qids, Q, A, C, rows = load()
    VR = np.array([rows[q]["hybrid"]["visual_retrieval_gpu_s"] for q in qids])
    ce_over = json.load(open(RESULTS / "router_ce_cost.json"))["per_query_batch1"]["gpu_s"]
    lams = np.unique(np.concatenate([np.linspace(0, 0.3, 61), np.linspace(0.3, 2.0, 35)]))
    per = []
    for k, s in enumerate(es["per_seed"]):
        R = np.load(RESULTS / "router_runs" / f"seed{k}.npz", allow_pickle=True)
        G = np.load(RESULTS / "router_ce" / f"seed{k}.npz", allow_pickle=True)
        te, cost = R["split"] == "test", R["cost"]
        At, Ct, i = A[te], C[te], np.arange(te.sum())
        pts = []
        for lam in lams:
            ch = np.argmax(G["qhat_ce_full"][te] - lam * cost, 1)
            pts.append((float((Ct[i, ch] + (ch == TXT) * VR[te] + ce_over).mean()), float(At[i, ch].mean())))
        pts += [(float(Ct[:, VIS].mean()), float(At[:, VIS].mean())), (float(Ct[:, HYB].mean()), float(At[:, HYB].mean()))]
        chk = np.argmax(G["qhat_ce_full"][te] - s["ce_lambda"] * cost, 1)
        check = abs(float((Ct[i, chk] + (chk == TXT) * VR[te] + ce_over).mean()) - s["results_ce"]["none"]["gpu_s"])
        hull = [pts[j] for j in upper_hull(pts)]
        hx, hy = [p[0] for p in hull], [p[1] for p in hull]
        gaps = []
        for sig, modes in s["results_ce"].items():
            if sig == "none":
                continue
            for mode, taus in modes.items():
                for tau, v in taus.items():
                    if hx[0] <= v["gpu_s"]:        # beyond the last vertex the frontier is flat (free disposal)
                        gaps.append((v["anls"] - float(np.interp(v["gpu_s"], hx, hy)), sig, mode, tau))
        best = max(gaps)
        per.append({"accounting_check_abs_gpu_s": check, "n_points_in_range": len(gaps),
                    "max_gap": best[0], "max_gap_at": list(best[1:]),
                    "mean_gap": float(np.mean([g[0] for g in gaps])),
                    "share_above_frontier": float(np.mean([g[0] > 0 for g in gaps]))})
    return {"per_split": per, "max_gap": t_interval([p["max_gap"] for p in per]),
            "mean_gap": t_interval([p["mean_gap"] for p in per]),
            "share_above_frontier": t_interval([p["share_above_frontier"] for p in per]),
            "note": "escalation point ANLS minus the CAVR-CE lambda-sweep frontier ANLS at the same mean GPU-s; "
                    "fine lambda grid, fixed policies included, linear interpolation between hull vertices, flat beyond the last"}


# 7 -------------------------------------------------------------------------------------------------------------
def token_sweep():
    """Image-token budget sweep (token_sweep.py, A10 node): ANLS, input tokens and generation time per path and budget,
    per dataset, with page-cluster bootstrap intervals of the differences to the 1,280-token anchor and to the text
    path; and the equal-token-budget comparison (the image budget whose mean input tokens are closest to the text
    path's)."""
    d = RESULTS / "a10" / "token_sweep_r3"
    files = sorted(f for f in d.glob("sweep_[0-9]*.jsonl")) if d.exists() else []
    if not files:
        return None
    rows = {}
    for f in files:
        for r in jl(f):
            rows.setdefault(r["qid"], {})[r["condition"]] = r
    conds = sorted({c for v in rows.values() for c in v})
    qids = [q for q, v in rows.items() if all(c in v for c in conds)]
    Q = {q["qid"]: q for q in jl(DATA / "queries.jsonl")}
    out = {"n_questions": len(qids), "conditions": conds}
    for ds in ("all", "docvqa", "infovqa"):
        sel = [q for q in qids if ds == "all" or Q[q]["dataset"] == ds]
        boot = page_boot(np.array([f"{Q[q]['dataset']}/{Q[q]['page_id']}" for q in sel]))
        X = {c: np.array([[rows[q][c]["anls"], rows[q][c]["generate_s"], rows[q][c]["input_tokens"],
                           rows[q][c]["visual_tokens"], rows[q][c]["peak_reserved_gb"]] for q in sel]) for c in conds}
        r = {"n": len(sel), "conditions": {}}
        for c in conds:
            r["conditions"][c] = {"anls": float(X[c][:, 0].mean()), "anls_ci95": ci(boot(X[c][:, 0])),
                                  "gen_s": float(X[c][:, 1].mean()), "input_tokens": float(X[c][:, 2].mean()),
                                  "visual_tokens": float(X[c][:, 3].mean()), "peak_reserved_gb": float(X[c][:, 4].max())}
            if "@" in c and not c.endswith("@1280"):
                ref = c.split("@")[0] + "@1280"
                dd = X[c][:, 0] - X[ref][:, 0]
                r["conditions"][c]["minus_1280"] = {"mean": float(dd.mean()), "ci95": ci(boot(dd))}
        tt = r["conditions"]["text"]["input_tokens"]
        eq = min((c for c in conds if c.startswith("visual@")), key=lambda c: abs(r["conditions"][c]["input_tokens"] - tt))
        dd = X[eq][:, 0] - X["text"][:, 0]
        r["equal_budget"] = {"visual_condition": eq, "visual_tokens_in": r["conditions"][eq]["input_tokens"],
                             "text_tokens_in": tt, "visual_minus_text": {"mean": float(dd.mean()), "ci95": ci(boot(dd))}}
        # every image budget against the text path, and hybrid against visual at the same image budget
        for c in conds:
            if c.startswith("visual@"):
                dd = X[c][:, 0] - X["text"][:, 0]
                r["conditions"][c]["minus_text"] = {"mean": float(dd.mean()), "ci95": ci(boot(dd))}
            if c.startswith("hybrid@"):
                dd = X[c][:, 0] - X["visual@" + c.split("@")[1]][:, 0]
                r["conditions"][c]["minus_visual"] = {"mean": float(dd.mean()), "ci95": ci(boot(dd))}
        out[ds] = r
    return out


def main():
    ms = main_splits()
    out = {"union_recall": union_recall(), "router_vs_oracle": ms["confusion"],
           "memory_time": ms["memory_time_gib_s"],
           "where_routing_helps": {"main_splits": ms["where_routing_helps_main"], "confirmation": where_confirmation()},
           "path_set": path_set(), "escalation_vs_lambda": escalation_vs_lambda(), "token_sweep": token_sweep()}
    write_json(RESULTS / "review_round3.json", out)
    u = out["union_recall"]
    for s in u:
        print(s, {d: {k: round(v, 3) for k, v in u[s][d].items() if "@" in k or "hybrid" in k} for d in u[s]})
    rv = out["router_vs_oracle"]
    print("confusion pooled (rows router, cols oracle):", rv["counts"]["pooled"], "agreement",
          round(rv["agreement"]["mean"], 3), "gap", round(rv["anls_gap_to_oracle"]["mean"], 3))
    print("memory GiB*s:", {k: round(v["mean"], 2) for k, v in out["memory_time"].items() if isinstance(v, dict) and "mean" in v})
    for d in ("all", "docvqa", "infovqa"):
        ps = out["path_set"][d]
        print(d, {c: (round(v["anls"], 3), round(v["gen_s"], 2), round(v["input_tokens"]), v["on_upper_hull"])
                  for c, v in ps["paths"].items()})
        print("   diffs", {k: round(v["mean"], 3) for k, v in ps["diff"].items()},
              "oracle3/5@0.1", round(ps["oracle"]["three_paths@0.1"]["anls"], 3), round(ps["oracle"]["five_paths@0.1"]["anls"], 3))
    for k, v in out["where_routing_helps"]["main_splits"].items():
        print(f"  {k:28s} n={v['n_mean']:.0f} ce-vis {v['ce_minus_visual']['mean']:+.3f} "
              f"[{v['ce_minus_visual']['ci95_t_over_splits'][0]:+.3f},{v['ce_minus_visual']['ci95_t_over_splits'][1]:+.3f}] "
              f"ocr_helps {v['ocr_helps_rate']['mean']:.2f}")
    print("confirmation:", json.dumps(out["where_routing_helps"]["confirmation"]))
    print("escalation vs lambda:", {k: round(out["escalation_vs_lambda"][k]["mean"], 4)
                                    for k in ("max_gap", "mean_gap", "share_above_frontier")})


if __name__ == "__main__":
    main()
