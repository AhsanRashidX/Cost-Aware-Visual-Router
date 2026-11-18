"""Live serving with retrieval and routing in the loop (third review round, R6 concern 3), on the 2x A10 node.

The vLLM load test (t4/loadgen.py) serves generation only: retrieval and the router ran beforehand. Here every
request goes through the whole pipeline at its arrival time:
  front end (one worker, batch 1, on GPU 1 of this Windows process; CUDA_VISIBLE_DEVICES selects the card):
    BM25 over the dataset's OCR text (CPU) -> ColPali query encoding and exhaustive MaxSim over the dataset's pages
    -> CAVR-CE features (retrieval scores, content overlap, ColPali per-token matches) and cross-encoder -> path
  generation: the chosen path's prompt, built from the live retrieval, is sent to the vLLM replica (GPU 0, in WSL)
Policies: cavr_ce (live router at the split's validation-chosen lambda) and always_visual (live ColPali, no router).
Arrivals and query picks replay loadgen.py exactly (same seed, rates, counts and draw order), so latencies compare
request by request with the generation-only run of the same policy and rate.

The weights of the main run's CAVR-CE were not stored, so `train` re-trains the split-1 router with router_ce.train_ce
(same code, data, features, hyperparameters and seed) and saves it with its feature standardisation; its offline
choices on split 1 are saved too, to check that the live features reproduce the cached ones.
Usage:  CUDA_VISIBLE_DEVICES=1 python live_pipeline_r3.py train
        CUDA_VISIBLE_DEVICES=1 python live_pipeline_r3.py run --policy cavr_ce --rates 0.303,0.605 [--limit-n 20]
Output: results/a10/live_pipeline_r3/ (ce_split1.pt, ce_split1_offline.npz, <policy>_r1_s<seed>/{requests.jsonl,summary.json})
"""
import argparse
import asyncio
import base64
import io
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

import generate as G
import router_ce as R
from colpali_token_feats import N_AUG, stats
from common import DATA, RESULTS, ROOT, SEED, hardware_info, set_seed, write_json
from metrics import anls, exact_match
from phase0 import STRIDE, WIN, build_features, content_tokens, overlap
from retrieve_text import tokenize
from retrieve_visual import load_model, maxsim
from routing_data import PATHS, group_split, load, path_costs

sys.path.insert(0, str(ROOT / "t4"))
import client_common as cc  # noqa: E402  (t4/client_common.py: streaming chat client)

OUT = RESULTS / "a10" / "live_pipeline_r3"
REQ = ROOT / "t4_package" / "requests.jsonl"


# ---------------------------------------------------------------- train ----------------------------------------
def train():
    from transformers import AutoTokenizer
    set_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    qids, Q, A, C, rows = load()
    F_retr, F_cont, _, _, _ = build_features(qids, Q, rows)
    T = np.load(DATA / "colpali_token_feats.npz")
    tpos = {q: i for i, q in enumerate(T["qids"])}
    F = np.hstack([F_retr, F_cont, T["feats"][[tpos[q] for q in qids]]])
    ds1h = np.array([[q["dataset"] == "docvqa", q["dataset"] == "infovqa"] for q in Q], np.float32)
    split = group_split(Q, SEED)
    tr, va = split == "train", split == "val"
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    tab = np.hstack([((F - mu) / sd).astype(np.float32), ds1h])
    tokzr = AutoTokenizer.from_pretrained(R.BACKBONE)
    enc = tokzr([q["question"] for q in Q], R.build_passages(qids, Q), truncation="only_second",
                max_length=R.MAX_LEN, padding="max_length", return_tensors="pt")
    captured = []

    class Capture(R.CrossRouter):          # train_ce builds R.CrossRouter and keeps the best epoch's weights in it
        def __init__(self, n_tab):
            super().__init__(n_tab)
            captured.append(self)
    R.CrossRouter = Capture
    qhat, hist, lat = R.train_ce(enc, tab, A, tr, va, SEED)
    ev = json.load(open(RESULTS / "eval_main.json"))
    lam = ev["per_seed"][0]["policies"]["cavr_ce_valsel"]["lambda"]
    cost = path_costs(C, tr)
    torch.save({"state": {k: v.cpu() for k, v in captured[-1].state_dict().items()}, "mu": mu, "sd": sd,
                "n_tab": tab.shape[1], "lambda": lam, "cost": cost, "val_auc_by_epoch": hist}, OUT / "ce_split1.pt")
    G0 = np.load(RESULTS / "router_ce" / "seed0.npz", allow_pickle=True)
    te = split == "test"
    new = np.argmax(qhat - lam * cost, 1)
    old = np.argmax(G0["qhat_ce_full"] - lam * cost, 1)
    np.savez(OUT / "ce_split1_offline.npz", qids=np.array(qids), split=split, qhat=qhat, choice=new)
    info = {"lambda": lam, "cost": cost.tolist(), "val_auc_by_epoch": hist,
            "test_choice_agreement_with_original": float((new[te] == old[te]).mean()),
            "test_anls_new": float(A[te][np.arange(te.sum()), new[te]].mean()),
            "test_anls_original": float(A[te][np.arange(te.sum()), old[te]].mean()),
            "hardware": hardware_info()}
    write_json(OUT / "ce_split1_train.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "hardware"}, indent=1))


# ---------------------------------------------------------------- front end --------------------------------------
class FrontEnd:
    """Retrieval and routing for one query at a time on this process's GPU."""

    def __init__(self, policy):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer
        self.policy = policy
        self.ocr = {(o["dataset"], o["page_id"]): o["text"] for o in map(json.loads, open(DATA / "ocr.jsonl", encoding="utf-8"))}
        pages = {(p["dataset"], p["page_id"]): p for p in map(json.loads, open(DATA / "pages.jsonl", encoding="utf-8"))}
        self.bm25 = {}
        for ds in ("docvqa", "infovqa"):
            docs = [(k[1], t) for k, t in self.ocr.items() if k[0] == ds]
            self.bm25[ds] = (BM25Okapi([tokenize(t) or ["<empty>"] for _, t in docs]), [p for p, _ in docs])
        self.model, self.proc = load_model()
        saved = torch.load(DATA / "colpali_index.pt")
        keys = saved["keys"]
        self.corpora = {}
        for ds in ("docvqa", "infovqa"):
            rows = [i for i, k in enumerate(keys) if k[0] == ds]
            ids = [keys[i][1] for i in rows]
            self.corpora[ds] = (saved["index"][rows].cuda(), ids, {p: j for j, p in enumerate(ids)})
        tok = self.proc.tokenizer
        self.n_prefix = len(tok(tok.bos_token + self.proc.query_prefix, add_special_tokens=False)["input_ids"])
        # page images resized with generate.py's rule and PNG-encoded once (an index-time artefact, as in loadgen)
        self.img = {}
        for k, p in pages.items():
            buf = io.BytesIO()
            G.resize(Image.open(DATA / p["path"]).convert("RGB")).save(buf, format="PNG")
            self.img[k] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        if policy == "cavr_ce":
            ck = torch.load(OUT / "ce_split1.pt", weights_only=False)
            self.ce = R.CrossRouter(ck["n_tab"]).cuda().eval()
            self.ce.load_state_dict(ck["state"])
            self.mu, self.sd, self.lam, self.cost = ck["mu"], ck["sd"], ck["lambda"], np.asarray(ck["cost"])
            self.tokzr = AutoTokenizer.from_pretrained(R.BACKBONE)
            self.mpnet = SentenceTransformer(R.BACKBONE, device="cuda").half()
            z = np.load(DATA / "ocr_windows.npz", allow_pickle=True)
            self.wemb, self.spans = z["emb"], dict(z["spans"].item())
        self.warm()

    def warm(self):
        q = {"dataset": "docvqa", "question": "what is the date of the letter?"}
        for _ in range(3):
            self.process(q)

    def windows(self, key):          # mirrors phase0.window_index / router_ce.build_passages
        words = self.ocr[key].split()
        return [" ".join(words[i:i + WIN]) or "<empty>" for i in range(0, max(1, len(words) - WIN // 2), STRIDE)]

    def sem(self, qv, key):
        a, b = self.spans[key]
        return float((self.wemb[a:b] @ qv).max()) if b > a else 0.0

    def best(self, key, qv, k, skip=()):
        a, b = self.spans[key]
        if b <= a:
            return []
        order = np.argsort(-(self.wemb[a:b] @ qv))
        ws = self.windows(key)
        return [ws[i] for i in order if ws[i] not in skip][:k]

    @torch.inference_mode()
    def process(self, q):
        t = {"start": time.perf_counter()}
        ds, question = q["dataset"], q["question"]
        bm25, ids = self.bm25[ds]
        s = bm25.get_scores(tokenize(question))
        order = np.argsort(-s)
        bm, bs = [ids[i] for i in order[:10]], [float(s[i]) for i in order[:10]]
        t["bm25"] = time.perf_counter()
        corpus, cids, cpos = self.corpora[ds]
        inp = self.proc.process_queries([question]).to(self.model.device)
        qemb = self.model(**inp)[0]
        sc = maxsim(qemb, corpus).float()
        top = torch.topk(sc, k=len(cids))
        cp, cs = [cids[i] for i in top.indices[:10].tolist()], top.values[:10].tolist()
        torch.cuda.synchronize()
        t["colpali"] = time.perf_counter()
        out = {"bm25_top": bm[:3], "colpali_top": cp[:3]}
        if self.policy == "always_visual":
            path = "visual"
        else:
            emb = qemb[inp["attention_mask"][0].bool()]
            real = emb[self.n_prefix:len(emb) - N_AUG] if len(emb) - N_AUG > self.n_prefix else emb
            per_tok = torch.stack([(real @ corpus[cpos[p]].T).amax(1) for p in (cp[0], cp[1], bm[0])]).float().cpu().numpy()
            c1, c2, b1 = per_tok
            d = c1 - c2
            f_tok = stats(c1) + stats(c2) + stats(b1) + [float(d.mean()), float(d.min()), float((d > 0).mean()),
                                                         float((b1 - c1).mean()), float(len(real))]
            qv = self.mpnet.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
            qt = content_tokens(question)
            k_b1, k_c1 = (ds, bm[0]), (ds, cp[0])
            f_retr = [bs[0], bs[0] - bs[1], bs[0] - bs[2], cs[0], cs[0] - cs[1], float(bm[0] == cp[0]),
                      (bm.index(cp[0]) + 1) if cp[0] in bm else 11, (cp.index(bm[0]) + 1) if bm[0] in cp else 11]
            ctx_text = "\n".join(self.ocr[(ds, p)][:G.MAX_CHARS_PER_PAGE] for p in bm[:G.K_TEXT])
            f_cont = [overlap(qt, self.ocr[k_b1]), overlap(qt, ctx_text), overlap(qt, self.ocr[k_c1]),
                      self.sem(qv, k_b1), max(self.sem(qv, (ds, p)) for p in bm[:G.K_TEXT]), self.sem(qv, k_c1),
                      np.log1p(len(self.ocr[k_b1])), np.log1p(len(self.ocr[k_c1]))]
            F = np.array(f_retr + f_cont + f_tok, np.float32)
            tab = np.hstack([(F - self.mu) / self.sd, [ds == "docvqa", ds == "infovqa"]]).astype(np.float32)
            t1 = self.best((ds, bm[0]), qv, 2)
            v1 = self.best((ds, cp[0]), qv, 2, skip=set(t1))
            t23 = [w for p in bm[1:3] for w in self.best((ds, p), qv, 1)]
            parts = []
            for i in range(2):
                if i < len(t1):
                    parts.append("[bm25 p1] " + t1[i])
                if i < len(v1):
                    parts.append("[colpali p1] " + v1[i])
            parts += ["[bm25 p2-3] " + w for w in t23]
            enc = self.tokzr([question], [" | ".join(parts)], truncation="only_second", max_length=R.MAX_LEN,
                             return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qhat = torch.sigmoid(self.ce(enc["input_ids"], enc["attention_mask"],
                                             torch.tensor(tab).unsqueeze(0).cuda()).float())[0].cpu().numpy()
            torch.cuda.synchronize()
            path = PATHS[int(np.argmax(qhat - self.lam * self.cost))]
            out["qhat"] = qhat.tolist()
        t["router"] = time.perf_counter()
        texts = [self.ocr[(ds, p)] for p in bm[:G.K_TEXT]] if path in ("text", "hybrid") else []
        msgs = G.build_messages(path, question, texts, None)
        url = self.img[(ds, cp[0])]
        out["messages"] = [{"role": m["role"], "content": [{"type": "image_url", "image_url": {"url": url}}
                                                           if c["type"] == "image" else c for c in m["content"]]}
                           for m in msgs]
        t["prompt"] = time.perf_counter()
        out["path"], out["t_start"] = path, t["start"]
        out["times"] = {"bm25_s": t["bm25"] - t["start"], "colpali_s": t["colpali"] - t["bm25"],
                        "router_s": t["router"] - t["colpali"], "prompt_s": t["prompt"] - t["router"],
                        "frontend_s": t["prompt"] - t["start"]}
        return out


# ---------------------------------------------------------------- run --------------------------------------------
async def run_rate(session, url, fe, pool, reqs, rate, n, warmup, args, sampling, model, ref):
    rng = np.random.default_rng(args.seed)
    arrivals = np.cumsum(rng.exponential(1 / rate, n))           # loadgen.run_rate's draw order
    picks = rng.integers(0, len(reqs), n)
    loop = asyncio.get_running_loop()
    rows = []
    t0 = time.perf_counter() + 0.5

    async def one(k, t_sched):
        e = reqs[picks[k]]
        f = await loop.run_in_executor(pool, fe.process, e)          # single front-end worker: FIFO queue
        body = {"model": model, "messages": f["messages"], "stream": True,
                "stream_options": {"include_usage": True}, **sampling}
        res = await cc.chat(session, url, body, args.timeout)
        ok = res["error"] is None
        rows.append({"rate": rate, "k": k, "warmup": k < warmup, "i": int(picks[k]), "qid": e["qid"],
                     "path": f["path"], "path_cached_router": e["policy"]["cavr_ce_valsel"],
                     "path_offline_retrained": ref.get(e["qid"]),
                     "bm25_top": f["bm25_top"], "colpali_top": f["colpali_top"],
                     "frontend_wait_s": f["t_start"] - t_sched,   # queue for the single front-end worker
                     **f["times"], "scheduled_s": t_sched - t0,
                     "gen_latency_s": res["t_end"] - res["t_send"] if res["t_send"] else None,
                     "latency_s": res["t_end"] - t_sched,
                     "ttft_s": (res["t_first_token"] - t_sched) if res["t_first_token"] else None,
                     "prompt_tokens": res["prompt_tokens"], "answer": res["answer"], "error": res["error"],
                     "anls": anls(res["answer"], e["answers"]) if ok else None,
                     "em": exact_match(res["answer"], e["answers"]) if ok else None})

    tasks = []
    for k in range(n):
        t_sched = t0 + arrivals[k]
        delay = t_sched - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(k, t_sched)))
    await asyncio.gather(*tasks)
    rows.sort(key=lambda x: x["k"])
    return rows


async def run_async(args):
    reqs, meta = cc.load_requests(REQ)
    OUT.mkdir(parents=True, exist_ok=True)
    out_dir = OUT / f"{args.policy}_r1_s{args.seed}{'_' + args.tag if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = {}
    if (OUT / "ce_split1_offline.npz").exists():
        z = np.load(OUT / "ce_split1_offline.npz", allow_pickle=True)
        ref = {q: PATHS[int(c)] for q, c in zip(z["qids"], z["choice"])}
    print("loading front end ...", flush=True)
    fe = FrontEnd(args.policy)
    pool = ThreadPoolExecutor(max_workers=1)
    rates = [float(r) for r in args.rates.split(",")]
    summary = {"policy": args.policy, "url": args.url, "seed": args.seed, "rates_planned": rates,
               "hardware_frontend": hardware_info(), "rates": {}, "stop_reason": None}
    async with cc.make_session() as session:
        info = await cc.server_info(session, [args.url])
        model = cc.served_model(info)
        summary["server"] = info
        for e in reqs[:4]:                                          # warm the replica, not recorded
            f = fe.process(e)
            await cc.chat(session, args.url, {"model": model, "messages": f["messages"], "stream": True,
                                              "stream_options": {"include_usage": True}, **meta["sampling"]}, args.timeout)
        with open(out_dir / "requests.jsonl", "w", encoding="utf-8") as fh:
            for rate in rates:
                n = args.limit_n or int(min(400, max(150, math.ceil(rate * 900))))
                warm = max(10, int(0.1 * n))
                rows = await run_rate(session, args.url, fe, pool, reqs, rate, n, warm, args, meta["sampling"], model, ref)
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                rec = [r for r in rows if not r["warmup"] and r["error"] is None]
                lat = [r["latency_s"] for r in rec]
                s = {"n": len(rows), "n_recorded": len(rec), "errors": sum(r["error"] is not None for r in rows),
                     "latency_s": cc.lat_stats(lat),
                     "frontend_s": cc.lat_stats([r["frontend_s"] for r in rec]),
                     "frontend_wait_s": cc.lat_stats([r["frontend_wait_s"] for r in rec]),
                     "gen_latency_s": cc.lat_stats([r["gen_latency_s"] for r in rec]),
                     "anls": cc.mean([r["anls"] for r in rec]),
                     "path_share": {p: float(np.mean([r["path"] == p for r in rec])) for p in PATHS},
                     "agree_cached_router": float(np.mean([r["path"] == r["path_cached_router"] for r in rec])),
                     "agree_offline_retrained": float(np.mean([r["path"] == r["path_offline_retrained"] for r in rec]))
                     if ref else None}
                summary["rates"][str(rate)] = s
                write_json(out_dir / "summary.json", summary)
                print(f"rate {rate}: p95 {s['latency_s']['p95']:.2f} s, front end p50 {1000 * s['frontend_s']['p50']:.0f} ms, "
                      f"ANLS {s['anls']:.3f}, agree offline {s['agree_offline_retrained']}", flush=True)
                if s["latency_s"]["p95"] > args.p95_cap:
                    summary["stop_reason"] = f"p95 above {args.p95_cap} s at rate {rate}"
                    break
                await asyncio.sleep(5)
    summary["stop_reason"] = summary["stop_reason"] or "all rates done"
    write_json(out_dir / "summary.json", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "run"])
    ap.add_argument("--policy", default="cavr_ce", choices=["cavr_ce", "always_visual"])
    ap.add_argument("--rates", default="")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--p95-cap", type=float, default=30.0)
    ap.add_argument("--limit-n", type=int, default=0, help="pilot: arrivals per rate")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.stage == "train":
        train()
    else:
        asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
