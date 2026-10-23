"""Phase 1b: content-aware router - mpnet cross-encoder over (question, retrieved OCR passages) + tabular
retrieval / content / ColPali-token features, predicting each path's answer quality.

Passages (all available before generation): best-matching OCR windows of the BM25 top-1 page (2), BM25 pages
2-3 (1 each) and the ColPali top-1 page (2). ColPali runs for every query, so its GPU cost is charged to
text-routed queries; the cross-encoder's own latency is measured and charged to every query.

Systems compared (same splits, lambda chosen on val, reported on test):
  ce_full        cross-encoder + all tabular features (incl. ColPali token stats)
  ce_no_tok      cross-encoder + tabular features without ColPali token stats
  gbm_tab_tok    gradient boosting on question embedding + all tabular features (no cross-encoder)
  gbm_tab        same without ColPali token stats (= Phase 0 probe 3 best)
Go criteria (fixed before running): mean test success AUC >= 0.87 AND hybrid quality (val ANLS >= val hybrid
- 0.005) reached at >= 30% less GPU on test.
Outputs: results/router_ce/{seed*.npz, summary.json, REPORT.md}
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from common import DATA, RESULTS, SEED, hardware_info, set_seed, write_json
from phase0 import STRIDE, WIN, build_features, stage_costs
from routing_data import ANLS_OK, PATHS, group_split, load, path_costs

OUT = RESULTS / "router_ce"
BACKBONE = "sentence-transformers/all-mpnet-base-v2"
N_SEEDS = 5
MAX_LEN = 512   # 384 truncated 91% of inputs in the smoke test
HP = {"epochs": 5, "batch": 16, "lr_backbone": 2e-5, "lr_head": 1e-3, "weight_decay": 0.01, "warmup": 0.1,
      "dropout": 0.1, "head_hidden": 256}
LAMBDAS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
BUDGETS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
DELTA = 0.005
TXT, VIS, HYB = 0, 1, 2


# ---------------------------------------------------------------- passages ------------------------------
def build_passages(qids, Q):
    ocr = {(o["dataset"], o["page_id"]): o["text"] for o in map(json.loads, open(DATA / "ocr.jsonl", encoding="utf-8"))}
    z = np.load(DATA / "ocr_windows.npz", allow_pickle=True)
    wemb, spans = z["emb"], dict(z["spans"].item())
    rt = {r["qid"]: r for r in map(json.loads, open(RESULTS / "retrieval_text.jsonl", encoding="utf-8"))}
    rv = {r["qid"]: r for r in map(json.loads, open(RESULTS / "retrieval_visual.jsonl", encoding="utf-8"))}
    E = np.load(DATA / "embeddings.npz")
    qemb = dict(zip(E["qids"], E["emb"]))

    def windows(key):  # must mirror phase0.window_index
        words = ocr[key].split()
        return [" ".join(words[i:i + WIN]) or "<empty>" for i in range(0, max(1, len(words) - WIN // 2), STRIDE)]

    def best(key, qv, k, skip=()):
        a, b = spans[key]
        if b <= a:
            return []
        order = np.argsort(-(wemb[a:b] @ qv))
        ws = windows(key)
        return [ws[i] for i in order if ws[i] not in skip][:k]

    out = []
    for qid, q in zip(qids, Q):
        ds, qv = q["dataset"], qemb[qid]
        bm, cp = rt[qid]["top"], rv[qid]["top"]
        t1 = best((ds, bm[0]), qv, 2)
        v1 = best((ds, cp[0]), qv, 2, skip=set(t1))
        t23 = [w for p in bm[1:3] for w in best((ds, p), qv, 1)]
        # interleave by importance so truncation never removes one source entirely
        parts = []
        for i in range(2):
            if i < len(t1):
                parts.append("[bm25 p1] " + t1[i])
            if i < len(v1):
                parts.append("[colpali p1] " + v1[i])
        parts += ["[bm25 p2-3] " + w for w in t23]
        out.append(" | ".join(parts))
    return out


# ---------------------------------------------------------------- model ---------------------------------
class CrossRouter(nn.Module):
    def __init__(self, n_tab):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(BACKBONE)
        d = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Linear(d + n_tab, HP["head_hidden"]), nn.ReLU(), nn.Dropout(HP["dropout"]),
                                  nn.Linear(HP["head_hidden"], len(PATHS)))

    def forward(self, input_ids, attention_mask, tab):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(torch.cat([pooled.float(), tab], 1))


def train_ce(enc, tab, A, tr, va, seed, log=None, extra=None):
    """log: optional dict; if given, per-epoch train/val BCE and val AUC are appended to it.
    extra: optional dict with input_ids, attention_mask, tab of other queries (e.g. the confirmation set); the kept
    model's predictions for them are stored in extra["qhat"]."""
    set_seed(seed)
    dev = "cuda"
    model = CrossRouter(tab.shape[1]).to(dev)
    params = [{"params": model.enc.parameters(), "lr": HP["lr_backbone"]},
              {"params": model.head.parameters(), "lr": HP["lr_head"]}]
    opt = torch.optim.AdamW(params, weight_decay=HP["weight_decay"])
    tr_idx = np.where(tr)[0]
    steps = HP["epochs"] * int(np.ceil(len(tr_idx) / HP["batch"]))
    warm = int(HP["warmup"] * steps)
    def lr_factor(step):   # linear warm-up, then linear decay to 0
        if step < warm:
            return (step + 1) / max(1, warm)
        return max(0.0, (steps - step) / max(1, steps - warm))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    loss_fn = nn.BCEWithLogitsLoss()     # ANLS in [0,1] used as soft targets
    ids, mask = enc["input_ids"], enc["attention_mask"]
    tab_t, A_t = torch.tensor(tab), torch.tensor(A, dtype=torch.float32)
    rng = np.random.default_rng(seed)

    def predict(idx):
        model.eval()
        out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(idx), 128):
                b = idx[i:i + 128]
                out.append(torch.sigmoid(model(ids[b].to(dev), mask[b].to(dev), tab_t[b].to(dev)).float()).cpu())
        return torch.cat(out).numpy()

    best_auc, best_state, history = -1, None, []
    va_idx = np.where(va)[0]
    for ep in range(HP["epochs"]):
        model.train()
        perm = rng.permutation(tr_idx)
        ep_sum, ep_n = 0.0, 0
        for i in range(0, len(perm), HP["batch"]):
            b = perm[i:i + HP["batch"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids[b].to(dev), mask[b].to(dev), tab_t[b].to(dev))
            loss = loss_fn(logits.float(), A_t[b].to(dev))
            if log is not None:
                ep_sum += len(b) * loss.item()
                ep_n += len(b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        pv = predict(va_idx)
        auc = float(np.mean([roc_auc_score(A[va_idx, p] >= ANLS_OK, pv[:, p]) for p in range(3)]))
        history.append(auc)
        if log is not None:
            p = np.clip(pv, 1e-6, 1 - 1e-6)
            t = A[va_idx]
            log.setdefault("train_bce", []).append(ep_sum / max(ep_n, 1))
            log.setdefault("val_bce", []).append(float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()))
            log.setdefault("val_auc", []).append(auc)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    if log is not None:
        log["best_epoch"] = int(np.argmax(history))   # 0-based; the epoch whose weights are kept
    qhat = predict(np.arange(len(A)))
    if extra is not None:
        model.eval()
        e_ids, e_mask, e_tab = extra["input_ids"], extra["attention_mask"], torch.tensor(extra["tab"])
        out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(e_ids), 128):
                out.append(torch.sigmoid(model(e_ids[i:i + 128].to(dev), e_mask[i:i + 128].to(dev),
                                               e_tab[i:i + 128].to(dev)).float()).cpu())
        extra["qhat"] = torch.cat(out).numpy()

    # router latency at batch 1 (tokenized input of a real query; tokenization excluded, measured separately)
    model.eval()
    lat = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in va_idx[:200]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(ids[i:i + 1].to(dev), mask[i:i + 1].to(dev), tab_t[i:i + 1].to(dev))
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
    del model
    torch.cuda.empty_cache()
    return qhat, history, float(np.median(lat))


# ---------------------------------------------------------------- evaluation ----------------------------
def evaluate(qhat, A, EM, C, VR, cost, va, te, overhead_s, pre_col=True):
    def run(mask, lam):
        ch = np.argmax(qhat[mask] - lam * cost, 1)
        i = np.arange(mask.sum())
        g = C[mask][i, ch] + pre_col * (ch == TXT) * VR[mask] + overhead_s
        return ch, float(A[mask][i, ch].mean()), float(g.mean())

    pts = [(lam, *run(va, lam)[1:], *run(te, lam)[1:]) for lam in LAMBDAS]   # (lam, val_anls, val_gpu, test_anls, test_gpu)
    srt = sorted((p[4], p[3]) for p in pts)
    xs, ys = np.array([s[0] for s in srt]), np.maximum.accumulate([s[1] for s in srt])
    curve = [float(np.interp(b, xs, ys)) if b >= xs[0] else None for b in BUDGETS]
    auc = {p: float(roc_auc_score(A[te, k] >= ANLS_OK, qhat[te, k])) for k, p in enumerate(PATHS)}

    def select(target):
        ok = [p for p in pts if p[1] >= target]
        if not ok:
            return None
        lam = min(ok, key=lambda p: p[2])[0]   # cheapest on VALIDATION cost
        ch, a, g = run(te, lam)
        i = np.arange(te.sum())
        em_r, em_h = EM[te][i, ch], EM[te][:, HYB]
        b, c = int(((em_r == 1) & (em_h == 0)).sum()), int(((em_r == 0) & (em_h == 1)).sum())
        d = A[te][i, ch] - A[te][:, HYB]
        boot = np.random.default_rng(SEED).integers(0, len(d), (1000, len(d)))
        return {"lambda": lam, "test_anls": a, "test_gpu_s": g,
                "path_mix": {p: float((ch == k).mean()) for k, p in enumerate(PATHS)},
                "vs_hybrid_anls_diff": float(d.mean()),
                "vs_hybrid_anls_ci95": [float(x) for x in np.percentile(d[boot].mean(1), [2.5, 97.5])],
                "vs_hybrid_mcnemar_p": float(binomtest(b, b + c).pvalue) if b + c else 1.0}

    return {"auc_success": auc, "auc_mean": float(np.mean(list(auc.values()))), "curve": curve, "sweep": pts,
            "match_hybrid": select(float(A[va, HYB].mean()) - DELTA),
            "match_visual": select(float(A[va, VIS].mean()) - DELTA)}


def main():
    import argparse
    import sys
    global N_SEEDS, OUT
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--epochs", type=int, default=HP["epochs"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    N_SEEDS, HP["epochs"] = args.seeds, args.epochs
    OUT = type(OUT)(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    qids, Q, A, C, rows = load()
    EM = np.array([[rows[q][p]["em"] for p in PATHS] for q in qids])
    G, VR = stage_costs(qids, rows)
    F_retr, F_cont, _, _, _ = build_features(qids, Q, rows)
    T = np.load(DATA / "colpali_token_feats.npz")
    tpos = {q: i for i, q in enumerate(T["qids"])}
    F_tok = T["feats"][[tpos[q] for q in qids]]
    ds1h = np.array([[q["dataset"] == "docvqa", q["dataset"] == "infovqa"] for q in Q], np.float32)
    E = np.load(DATA / "embeddings.npz")
    emb = dict(zip(E["qids"], E["emb"]))
    Xq = np.stack([emb[q] for q in qids])

    passages = build_passages(qids, Q)
    tokzr = AutoTokenizer.from_pretrained(BACKBONE)
    t0 = time.perf_counter()
    enc = tokzr([q["question"] for q in Q], passages, truncation="only_second", max_length=MAX_LEN,
                padding="max_length", return_tensors="pt")
    tok_s = (time.perf_counter() - t0) / len(Q)
    trunc = float((enc["attention_mask"].sum(1) >= MAX_LEN).float().mean())

    summary = {"hyperparameters": HP, "max_len": MAX_LEN, "backbone": BACKBONE, "truncated_share": trunc,
               "tokenize_s_per_query": tok_s, "colpali_token_feat_ms_median": float(1000 * np.median(T["seconds"])),
               "seeds": []}
    for k in range(N_SEEDS):
        seed = SEED + k
        split = group_split(Q, seed)
        tr, va, te = (split == s for s in ("train", "val", "test"))
        cost = path_costs(C, tr)

        def std(F):
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
            return ((F - mu) / sd).astype(np.float32)

        tab_full = np.hstack([std(np.hstack([F_retr, F_cont, F_tok])), ds1h])
        tab_no_tok = np.hstack([std(np.hstack([F_retr, F_cont])), ds1h])
        res = {"seed": seed, "fixed": {p: {"val_anls": float(A[va, i].mean()), "test_anls": float(A[te, i].mean()),
                                            "test_gpu_s": float(C[te, i].mean())} for i, p in enumerate(PATHS)}}
        preds = {}
        for name, tab in (("ce_full", tab_full), ("ce_no_tok", tab_no_tok)):
            qhat, hist, lat = train_ce(enc, tab, A, tr, va, seed)
            preds[name] = qhat
            res[name] = evaluate(qhat, A, EM, C, VR, cost, va, te, overhead_s=lat + tok_s)
            res[name].update({"val_auc_by_epoch": hist, "router_latency_s": lat})
            print(f"seed {seed} {name}: test AUC {res[name]['auc_mean']:.3f}, match-hybrid {res[name]['match_hybrid']}", flush=True)
        for name, X in (("gbm_tab_tok", np.hstack([Xq, ds1h, F_retr, F_cont, F_tok])),
                        ("gbm_tab", np.hstack([Xq, ds1h, F_retr, F_cont]))):
            qhat = np.column_stack([np.clip(HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                random_state=seed).fit(X[tr], A[tr, p]).predict(X), 0, 1) for p in range(3)])
            preds[name] = qhat
            res[name] = evaluate(qhat, A, EM, C, VR, cost, va, te, overhead_s=0.0)
        np.savez(OUT / f"seed{k}.npz", qids=np.array(qids), split=split, **{f"qhat_{n}": v for n, v in preds.items()})
        summary["seeds"].append(res)
        write_json(OUT / "summary.json", summary)
    summary["hardware"] = hardware_info()
    write_json(OUT / "summary.json", summary)
    report(summary)


def report(summary):
    S = summary["seeds"]
    m = lambda xs: float(np.mean(xs))
    hy = (m([s["fixed"]["hybrid"]["test_anls"] for s in S]), m([s["fixed"]["hybrid"]["test_gpu_s"] for s in S]))
    vi = (m([s["fixed"]["visual"]["test_anls"] for s in S]), m([s["fixed"]["visual"]["test_gpu_s"] for s in S]))
    L = ["# Phase 1: content-aware router (generated from results/router_ce/summary.json)", "",
         f"Always-hybrid ANLS {hy[0]:.3f} at {hy[1]:.3f} GPU-s; always-visual {vi[0]:.3f} at {vi[1]:.3f}. "
         f"Router overhead charged: cross-encoder latency + tokenization; ColPali charged to text-routed queries.", "",
         "Go criteria (fixed before running): mean test success AUC >= 0.87 AND hybrid-matching operating point "
         "(lambda chosen on val) at >= 30% less GPU than always-hybrid.", "",
         "| System | AUC text / visual / hybrid (mean) | " + " | ".join(f"ANLS@{b}" for b in BUDGETS) +
         " | Match hybrid (val-chosen λ): test ANLS / GPU-s / saving / n | ΔANLS vs hybrid [95% CI], McNemar p (seed 1) |",
         "|---|---|" + "---|" * len(BUDGETS) + "---|---|"]
    for name in ("ce_full", "ce_no_tok", "gbm_tab_tok", "gbm_tab"):
        au = {p: m([s[name]["auc_success"][p] for s in S]) for p in PATHS}
        curve = [m([s[name]["curve"][i] for s in S if s[name]["curve"][i] is not None]) if any(
            s[name]["curve"][i] is not None for s in S) else None for i in range(len(BUDGETS))]
        mh = [s[name]["match_hybrid"] for s in S if s[name]["match_hybrid"]]
        mh_txt = (f"{m([x['test_anls'] for x in mh]):.3f} / {m([x['test_gpu_s'] for x in mh]):.3f} / "
                  f"{100 * (1 - m([x['test_gpu_s'] for x in mh]) / hy[1]):.0f}% / {len(mh)}") if mh else "none / 0"
        s0 = S[0][name]["match_hybrid"]
        t0 = (f"{s0['vs_hybrid_anls_diff']:+.3f} [{s0['vs_hybrid_anls_ci95'][0]:+.3f}, {s0['vs_hybrid_anls_ci95'][1]:+.3f}], "
              f"p={s0['vs_hybrid_mcnemar_p']:.2g}") if s0 else "-"
        L.append(f"| {name} | {au['text']:.3f} / {au['visual']:.3f} / {au['hybrid']:.3f} ({m(list(au.values())):.3f}) | "
                 + " | ".join(f"{c:.3f}" if c is not None else "-" for c in curve) + f" | {mh_txt} | {t0} |")
    lat = m([s["ce_full"]["router_latency_s"] for s in S])
    L += ["", f"Cross-encoder latency (batch 1, GPU): {1000 * lat:.1f} ms; tokenization {1000 * summary['tokenize_s_per_query']:.2f} ms; "
          f"inputs truncated at {summary['max_len']} tokens: {100 * summary['truncated_share']:.0f}%."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
