"""Frozen CAVR-CE on the confirmation set (fresh questions from the DocVQA / InfographicVQA train splits).

The router design was developed on the five validation splits; the confirmation set was never seen. Two stages:
  features  (run with CAVR_DATA=data_confirm CAVR_RESULTS=results_confirm, after OCR, retrieval, embeddings,
            generation and ColPali token features exist there) -> results_confirm/confirm_ce_inputs.npz:
            the same tabular features and passages router_ce.py builds
  train     (default data folders) -> results_confirm/confirm_ce_qhat.npz: CAVR-CE (ce_full) is re-trained per seed
            exactly as in router_ce.py (same split, features, standardisation and seed), the confirmation queries are
            standardised with the training split's statistics and scored by the kept model; the re-trained model is
            compared with the saved one on the original queries (prediction difference, and decisions, ANLS and GPU
            cost on each seed's test split at its validation lambda). --seeds runs a subset (one process per GPU);
            each seed is saved under results_confirm/confirm_ce/ and the last process merges them
"""
import argparse
import json
import time

import numpy as np

from common import DATA, RESULTS, ROOT, SEED, hardware_info, write_json

CONF_RESULTS = ROOT / "results_confirm"


def stage_features():
    from phase0 import build_features
    from router_ce import build_passages
    from routing_data import load
    qids, Q, A, C, rows = load()
    F_retr, F_cont, _, _, _ = build_features(qids, Q, rows)
    T = np.load(DATA / "colpali_token_feats.npz")
    tpos = {q: i for i, q in enumerate(T["qids"])}
    F_tok = T["feats"][[tpos[q] for q in qids]]
    ds1h = np.array([[q["dataset"] == "docvqa", q["dataset"] == "infovqa"] for q in Q], np.float32)
    np.savez(RESULTS / "confirm_ce_inputs.npz", qids=np.array(qids), questions=np.array([q["question"] for q in Q]),
             passages=np.array(build_passages(qids, Q)), F_retr=F_retr, F_cont=F_cont, F_tok=F_tok, ds1h=ds1h)
    print(f"wrote {RESULTS / 'confirm_ce_inputs.npz'} ({len(qids)} queries)")


def stage_train(seeds):
    from transformers import AutoTokenizer
    import router_ce as R
    from phase0 import build_features
    from routing_data import group_split, load, path_costs

    qids, Q, A, C, rows = load()
    VR = np.array([rows[q]["hybrid"]["visual_retrieval_gpu_s"] for q in qids])
    F_retr, F_cont, _, _, _ = build_features(qids, Q, rows)
    T = np.load(DATA / "colpali_token_feats.npz")
    tpos = {q: i for i, q in enumerate(T["qids"])}
    F_tok = T["feats"][[tpos[q] for q in qids]]
    ds1h = np.array([[q["dataset"] == "docvqa", q["dataset"] == "infovqa"] for q in Q], np.float32)
    tokzr = AutoTokenizer.from_pretrained(R.BACKBONE)
    enc = tokzr([q["question"] for q in Q], R.build_passages(qids, Q), truncation="only_second",
                max_length=R.MAX_LEN, padding="max_length", return_tensors="pt")

    ci = np.load(CONF_RESULTS / "confirm_ce_inputs.npz", allow_pickle=True)
    enc_c = tokzr(list(ci["questions"]), list(ci["passages"]), truncation="only_second", max_length=R.MAX_LEN,
                  padding="max_length", return_tensors="pt")
    F_all = np.hstack([F_retr, F_cont, F_tok])
    Fc_all = np.hstack([ci["F_retr"], ci["F_cont"], ci["F_tok"]])
    ev = json.load(open(RESULTS / "eval_main.json"))["per_seed"]
    part = CONF_RESULTS / "confirm_ce"
    part.mkdir(exist_ok=True)
    for k in seeds:
        seed = SEED + k
        split = group_split(Q, seed)
        tr, va, te = split == "train", split == "val", split == "test"
        mu, sd = F_all[tr].mean(0), F_all[tr].std(0) + 1e-6
        tab = np.hstack([((F_all - mu) / sd).astype(np.float32), ds1h])
        tab_c = np.hstack([((Fc_all - mu) / sd).astype(np.float32), ci["ds1h"]])
        extra = {"input_ids": enc_c["input_ids"], "attention_mask": enc_c["attention_mask"], "tab": tab_c}
        t0 = time.perf_counter()
        qhat, hist, lat = R.train_ce(enc, tab, A, tr, va, seed, extra=extra)
        saved = np.load(RESULTS / "router_ce" / f"seed{k}.npz", allow_pickle=True)["qhat_ce_full"]
        lam, cost = ev[k]["policies"]["cavr_ce_valsel"]["lambda"], path_costs(C, tr)
        # the re-trained router against the saved one, on this seed's original test split, same decision rule
        rep = {}
        for name, qh in (("saved", saved), ("retrained", qhat)):
            ch = np.argmax(qh[te] - lam * cost, 1)
            j = np.arange(te.sum())
            rep[name] = {"choice": ch, "anls": float(A[te][j, ch].mean()),
                         "gpu_s": float((C[te][j, ch] + (ch == 0) * VR[te]).mean())}
        np.savez(part / f"seed{k}.npz", qhat=extra["qhat"], cost=cost)
        meta = {"seed": seed, "lambda_valsel": lam, "router_latency_s": lat, "train_s": time.perf_counter() - t0,
                "val_auc_by_epoch": hist,
                "max_abs_diff_vs_saved_qhat": float(np.abs(qhat - saved).max()),
                "mean_abs_diff_vs_saved_qhat": float(np.abs(qhat - saved).mean()),
                "test_decision_agreement": float((rep["saved"]["choice"] == rep["retrained"]["choice"]).mean()),
                "test_anls_saved": rep["saved"]["anls"], "test_anls_retrained": rep["retrained"]["anls"],
                "test_gpu_s_saved": rep["saved"]["gpu_s"], "test_gpu_s_retrained": rep["retrained"]["gpu_s"],
                "hardware": hardware_info()}
        write_json(part / f"seed{k}.json", meta)
        print(meta, flush=True)

    # merge once every seed is done (whichever process finishes last)
    if all((part / f"seed{k}.json").exists() for k in range(R.N_SEEDS)):
        out, metas = {"qids": ci["qids"]}, []
        for k in range(R.N_SEEDS):
            z = np.load(part / f"seed{k}.npz")
            out[f"qhat_seed{k}"], out[f"cost_seed{k}"] = z["qhat"], z["cost"]
            metas.append(json.load(open(part / f"seed{k}.json")))
        np.savez(CONF_RESULTS / "confirm_ce_qhat.npz", **out)
        write_json(CONF_RESULTS / "confirm_ce_meta.json", {"seeds": metas, "hardware": hardware_info()})
        print(f"wrote {CONF_RESULTS / 'confirm_ce_qhat.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "train"], required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4", help="train stage: seed indices to run in this process")
    a = ap.parse_args()
    stage_features() if a.stage == "features" else stage_train([int(x) for x in a.seeds.split(",")])
