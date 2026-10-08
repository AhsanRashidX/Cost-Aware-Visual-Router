"""Phase 0 feasibility probes, computed from the cache only (no new generation, no downloads).

Probe 1 (option C)  cascades: answer with a cheaper path first, escalate when the answer looks unreliable.
Probe 2 (option A)  sensitivity: what if the text path were s x cheaper and kept a fraction r of its correct answers?
Probe 3 (option B)  content-aware router: does retrieved-content evidence predict per-path success better than
                    the question alone?
lambda / tau are chosen on the val split; numbers are reported on test; 5 page-grouped seeds.
Outputs: results/phase0/probes.json, results/phase0/REPORT.md
"""
import json
import re
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import DATA, RESULTS, SEED, hardware_info, set_seed, write_json
from generate import K_TEXT, MAX_CHARS_PER_PAGE
from metrics import normalize
from routing_data import ANLS_OK, PATHS, group_split, load, path_costs

OUT = RESULTS / "phase0"
N_SEEDS = 5
LAMBDAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
TAUS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
DELTA = 0.005           # "matches hybrid" = val ANLS >= val(always_hybrid) - DELTA
TXT, VIS, HYB = 0, 1, 2
STOP = set("the a an of in on at to for by with from and or is are was were be what which who whom whose when where "
           "why how does do did this that these those as it its into than then there their per".split())
WIN, STRIDE = 50, 25    # OCR window size / stride in words for semantic-match features


def content_tokens(s):
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP]


def overlap(qtok, text):
    if not qtok:
        return 0.0
    tt = set(re.findall(r"[a-z0-9]+", text.lower()))
    return sum(t in tt for t in qtok) / len(qtok)


def window_index(ocr):
    """mpnet embeddings of overlapping OCR windows for every page (offline / index-time cost)."""
    path = DATA / "ocr_windows.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["emb"], dict(z["spans"].item()), float(z["seconds"])
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    texts, spans = [], {}
    for key, text in ocr.items():
        words = text.split()
        start = len(texts)
        for i in range(0, max(1, len(words) - WIN // 2), STRIDE):
            texts.append(" ".join(words[i:i + WIN]) or "<empty>")
        spans[key] = (start, len(texts))
    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False).astype(np.float32)
    secs = time.perf_counter() - t0
    np.savez(path, emb=emb, spans=np.array(spans, dtype=object), seconds=secs)
    return emb, spans, secs


def build_features(qids, Q, rows):
    ocr = {(o["dataset"], o["page_id"]): o["text"] for o in map(json.loads, open(DATA / "ocr.jsonl", encoding="utf-8"))}
    rt = {r["qid"]: r for r in map(json.loads, open(RESULTS / "retrieval_text.jsonl", encoding="utf-8"))}
    rv = {r["qid"]: r for r in map(json.loads, open(RESULTS / "retrieval_visual.jsonl", encoding="utf-8"))}
    E = np.load(DATA / "embeddings.npz")
    qemb = dict(zip(E["qids"], E["emb"]))
    wemb, spans, win_secs = window_index(ocr)

    def sem(qv, key):
        a, b = spans[key]
        return float((wemb[a:b] @ qv).max()) if b > a else 0.0

    retr, cont, post = [], [], []
    for qid, q in zip(qids, Q):
        ds, t, v = q["dataset"], rt[qid], rv[qid]
        bm, cp = t["top"], v["top"]
        qv, qt = qemb[qid], content_tokens(q["question"])
        k_b1, k_c1 = (ds, bm[0]), (ds, cp[0])
        retr.append([t["scores"][0], t["scores"][0] - t["scores"][1], t["scores"][0] - t["scores"][2],
                     v["scores"][0], v["scores"][0] - v["scores"][1],
                     float(bm[0] == cp[0]),
                     (bm.index(cp[0]) + 1) if cp[0] in bm else 11, (cp.index(bm[0]) + 1) if bm[0] in cp else 11])
        ctx_text = "\n".join(ocr[(ds, p)][:MAX_CHARS_PER_PAGE] for p in bm[:K_TEXT])
        cont.append([overlap(qt, ocr[k_b1]), overlap(qt, ctx_text), overlap(qt, ocr[k_c1]),
                     sem(qv, k_b1), max(sem(qv, (ds, p)) for p in bm[:K_TEXT]), sem(qv, k_c1),
                     np.log1p(len(ocr[k_b1])), np.log1p(len(ocr[k_c1]))])
        # Post-generation evidence about each path's own answer (used only by cascades).
        f = []
        for p, ctx in (("text", ctx_text), ("visual", ocr[k_c1]), ("hybrid", ctx_text + "\n" + ocr[k_c1])):
            r = rows[qid][p]
            ans = normalize(r["answer"])
            f += [r["answer_logprob_mean"], r["answer_logprob_min"],
                  float(bool(ans) and ans in normalize(ctx)), np.log1p(len(ans))]
        post.append(f)
    names = {"retr": ["bm25_top", "bm25_m12", "bm25_m13", "col_top", "col_m12", "agree", "bm25_rank_of_col1",
                      "col_rank_of_bm1"],
             "cont": ["ovl_bm1", "ovl_ctx", "ovl_col1", "sem_bm1", "sem_bm3", "sem_col1", "len_bm1", "len_col1"],
             "post": [f"{p}_{k}" for p in PATHS for k in ("lp_mean", "lp_min", "ans_in_ctx", "ans_len")]}
    return np.array(retr, np.float32), np.array(cont, np.float32), np.array(post, np.float32), names, win_secs


def stage_costs(qids, rows):
    G = np.array([[rows[q][p]["generate_s"] for p in PATHS] for q in qids])
    VR = np.array([rows[q]["hybrid"]["visual_retrieval_gpu_s"] for q in qids])
    return G, VR


def select_on_val(points, val_target):
    """points: list of (param, val_anls, val_gpu, test_anls, test_gpu). Cheapest param on VALIDATION cost whose val
    ANLS reaches the target; test values are only reported."""
    ok = [p for p in points if p[1] >= val_target]
    if not ok:
        return None
    best = min(ok, key=lambda p: p[2])
    return {"param": best[0], "val_anls": best[1], "test_anls": best[3], "test_gpu_s": best[4]}


def main():
    set_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    qids, Q, A, C, rows = load()
    G, VR = stage_costs(qids, rows)
    F_retr, F_cont, F_post, names, win_secs = build_features(qids, Q, rows)
    E = np.load(DATA / "embeddings.npz")
    emb = dict(zip(E["qids"], E["emb"]))
    ds1h = np.array([[q["dataset"] == "docvqa", q["dataset"] == "infovqa"] for q in Q], np.float32)
    X_q = np.hstack([np.stack([emb[q] for q in qids]), ds1h])
    feature_sets = {   # name -> (matrix, needs ColPali scores before routing)
        "Q (question only)": (X_q, False),
        "Q+BM25-content": (np.hstack([X_q, F_retr[:, :3], F_cont[:, [0, 1, 3, 4, 6]]]), False),
        "Q+retrieval+content": (np.hstack([X_q, F_retr, F_cont]), True),
    }
    idx = np.arange(len(qids))
    out = {"window_index_seconds": win_secs, "feature_names": names, "seeds": []}

    for s in range(N_SEEDS):
        seed = SEED + s
        split = group_split(Q, seed)
        tr, va, te = (split == k for k in ("train", "val", "test"))
        cost = path_costs(C, tr)
        res = {"seed": seed}

        def evaluate(choice_fn, mask, pre_col=False):
            ch = choice_fn(mask)
            g = C[mask][np.arange(mask.sum()), ch] + (pre_col * (ch == TXT) * VR[mask])
            return float(A[mask][np.arange(mask.sum()), ch].mean()), float(g.mean())

        fixed = {p: {"val_anls": float(A[va, k].mean()), "test_anls": float(A[te, k].mean()),
                     "test_gpu_s": float(C[te, k].mean())} for k, p in enumerate(PATHS)}
        res["fixed"] = fixed
        target_h, target_v = fixed["hybrid"]["val_anls"] - DELTA, fixed["visual"]["val_anls"] - DELTA

        # ---- Probe 3: per-path quality regressors on different feature sets -------------------------------
        probe3 = {}
        for fs, (X, pre_col) in feature_sets.items():
            qhat = np.zeros_like(A)
            for k in range(3):
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                  l2_regularization=1.0, random_state=seed)
                qhat[:, k] = np.clip(m.fit(X[tr], A[tr, k]).predict(X), 0, 1)
            auc = {p: float(roc_auc_score(A[te, k] >= ANLS_OK, qhat[te, k])) for k, p in enumerate(PATHS)}
            pts = []
            for lam in LAMBDAS:
                fn = lambda m, lam=lam: np.argmax(qhat[m] - lam * cost, 1)
                va_a, va_g = evaluate(fn, va, pre_col)
                te_a, te_g = evaluate(fn, te, pre_col)
                pts.append((lam, va_a, va_g, te_a, te_g))
            probe3[fs] = {"auc_success": auc, "sweep": pts,
                          "match_hybrid": select_on_val(pts, target_h), "match_visual": select_on_val(pts, target_v)}
        res["probe3_router"] = probe3

        # ---- Probe 1: cascades with a calibrated "answer is good" probability -----------------------------
        def p_good(first, cols):
            Xc = np.hstack([F_post[:, first * 4:(first + 1) * 4], F_retr, F_cont][:cols])
            y = (A[:, first] >= ANLS_OK).astype(int)
            mu, sd = Xc[va].mean(0), Xc[va].std(0) + 1e-6   # calibrated on val (cascade never sees train labels)
            return LogisticRegression(max_iter=3000).fit((Xc[va] - mu) / sd, y[va]).predict_proba((Xc - mu) / sd)[:, 1]

        cascades = {"text->visual": (TXT, VIS), "text->hybrid": (TXT, HYB), "visual->hybrid": (VIS, HYB)}
        probe1 = {}
        for name, (first, second) in cascades.items():
            for sig, cols in (("answer-only", 1), ("answer+retrieval+content", 3)):
                pg = p_good(first, cols)
                pts = []
                for tau in TAUS:
                    def run(m):
                        esc = pg[m] < tau
                        final = np.where(esc, second, first)
                        # cost: first generation + its retrievals; escalation adds second generation and any
                        # retrieval not yet done (ColPali when going text->visual/hybrid). sig with retrieval
                        # features needs ColPali up front.
                        need_col_first = (first != TXT) or cols == 3
                        gpu = G[m, first] + need_col_first * VR[m] + esc * (G[m, second] + (not need_col_first) * VR[m])
                        return float(A[m, :][np.arange(m.sum()), final].mean()), float(gpu.mean()), float(esc.mean())
                    va_a, va_g, _ = run(va)
                    te_a, te_g, te_e = run(te)
                    pts.append((tau, va_a, va_g, te_a, te_g, te_e))
                probe1[f"{name} [{sig}]"] = {"sweep": pts,
                                             "match_hybrid": select_on_val([p[:5] for p in pts], target_h),
                                             "match_visual": select_on_val([p[:5] for p in pts], target_v)}
        res["probe1_cascade"] = probe1

        # ---- Probe 2: hypothetical cheap text path (cost factor s, keeps fraction r of correct answers) --------
        rng = np.random.default_rng(seed)
        X_best, pre_col = feature_sets["Q+retrieval+content"]
        qhat_vh = {}
        for k in (VIS, HYB):
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=seed)
            qhat_vh[k] = np.clip(m.fit(X_best[tr], A[tr, k]).predict(X_best), 0, 1)
        probe2 = {}
        for s_cost in (1.0, 0.5, 0.33, 0.2):
            for r in (1.0, 0.9, 0.8, 0.7):
                A2, C2 = A.copy(), C.copy()
                C2[:, TXT] = G[:, TXT] * s_cost                 # text path has no GPU retrieval
                good = A2[:, TXT] > 0
                drop = good & (rng.random(len(A2)) > r)         # lose a fraction (1-r) of the text path's hits
                A2[drop, TXT] = 0.0
                cost2 = path_costs(C2, tr)
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                  l2_regularization=1.0, random_state=seed)
                qt = np.clip(m.fit(X_best[tr], A2[tr, TXT]).predict(X_best), 0, 1)
                qh = np.column_stack([qt, qhat_vh[VIS], qhat_vh[HYB]])
                pts = []
                for lam in LAMBDAS:
                    def run(msk, lam=lam):
                        ch = np.argmax(qh[msk] - lam * cost2, 1)
                        g = C2[msk][np.arange(msk.sum()), ch] + pre_col * (ch == TXT) * VR[msk]
                        return float(A2[msk][np.arange(msk.sum()), ch].mean()), float(g.mean())
                    va_a, va_g = run(va)
                    te_a, te_g = run(te)
                    pts.append((lam, va_a, va_g, te_a, te_g))
                u = A2[te] - 1e-6 * C2[te]
                orc = np.argmax(u, 1)
                probe2[f"s={s_cost},r={r}"] = {
                    "router_match_hybrid": select_on_val(pts, target_h),
                    "oracle": {"test_anls": float(A2[te][np.arange(te.sum()), orc].mean()),
                               "test_gpu_s": float(C2[te][np.arange(te.sum()), orc].mean())},
                    "always_text": {"test_anls": float(A2[te, TXT].mean()), "test_gpu_s": float(C2[te, TXT].mean())}}
        res["probe2_cheap_text"] = probe2
        out["seeds"].append(res)
        print(f"seed {seed} done", flush=True)

    out["hardware"] = hardware_info()
    write_json(OUT / "probes.json", out)
    report(out)


def report(out):
    S = out["seeds"]
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")

    def agg(get):
        vals = [get(s) for s in S]
        vals = [v for v in vals if v is not None]
        return (mean([v["test_anls"] for v in vals]), mean([v["test_gpu_s"] for v in vals]), len(vals))

    hy = agg(lambda s: s["fixed"]["hybrid"])
    vi = agg(lambda s: s["fixed"]["visual"])
    lines = ["# Phase 0 probes (from cache; lambda/tau picked on val, reported on test, mean over seeds)", "",
             f"Reference: always-hybrid ANLS {hy[0]:.3f} at {hy[1]:.3f} GPU-s; always-visual ANLS {vi[0]:.3f} at {vi[1]:.3f} GPU-s.",
             f"'Match hybrid' = cheapest setting whose val ANLS >= val(hybrid) - {DELTA}. n = seeds where such a setting exists.",
             "", "## Probe 3 - router features (option B)", "",
             "| Features | AUC text ok | AUC visual ok | AUC hybrid ok | Match hybrid: ANLS / GPU-s (saving) | n | Match visual: ANLS / GPU-s | n |",
             "|---|---|---|---|---|---|---|---|"]
    for fs in S[0]["probe3_router"]:
        au = {p: mean([s["probe3_router"][fs]["auc_success"][p] for s in S]) for p in PATHS}
        mh = agg(lambda s: s["probe3_router"][fs]["match_hybrid"])
        mv = agg(lambda s: s["probe3_router"][fs]["match_visual"])
        lines.append(f"| {fs} | {au['text']:.3f} | {au['visual']:.3f} | {au['hybrid']:.3f} | "
                     f"{mh[0]:.3f} / {mh[1]:.3f} ({100 * (1 - mh[1] / hy[1]):.0f}%) | {mh[2]} | {mv[0]:.3f} / {mv[1]:.3f} | {mv[2]} |")
    lines += ["", "## Probe 1 - cascades with the current shared 7B generator (option C)", "",
              "| Cascade [signal] | Match hybrid: ANLS / GPU-s (saving) | n | Match visual: ANLS / GPU-s | n |", "|---|---|---|---|---|"]
    for c in S[0]["probe1_cascade"]:
        mh = agg(lambda s: s["probe1_cascade"][c]["match_hybrid"])
        mv = agg(lambda s: s["probe1_cascade"][c]["match_visual"])
        lines.append(f"| {c} | {mh[0]:.3f} / {mh[1]:.3f} ({100 * (1 - mh[1] / hy[1]):.0f}%) | {mh[2]} | {mv[0]:.3f} / {mv[1]:.3f} | {mv[2]} |")
    lines += ["", "## Probe 2 - hypothetical cheap text path (option A): text GPU cost x s, keeps fraction r of its correct answers", "",
              "Router = Q+retrieval+content features. HYPOTHETICAL: no smaller model was run.", "",
              "| s (cost factor) | r (answers kept) | Always-text ANLS / GPU-s | Router match hybrid: ANLS / GPU-s (saving) | n | Oracle ANLS / GPU-s |",
              "|---|---|---|---|---|---|"]
    for key in S[0]["probe2_cheap_text"]:
        sc, rr = key.split(",")
        at = agg(lambda s: s["probe2_cheap_text"][key]["always_text"])
        mh = agg(lambda s: s["probe2_cheap_text"][key]["router_match_hybrid"])
        orc = agg(lambda s: s["probe2_cheap_text"][key]["oracle"])
        lines.append(f"| {sc[2:]} | {rr[2:]} | {at[0]:.3f} / {at[1]:.3f} | {mh[0]:.3f} / {mh[1]:.3f} ({100 * (1 - mh[1] / hy[1]):.0f}%) | {mh[2]} | {orc[0]:.3f} / {orc[1]:.3f} |")
    lines += ["", f"OCR window index for content features: {out['window_index_seconds']:.1f} s of GPU time, once, at index time."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
