"""Draft the revised Results section (LaTeX) and the response to reviewers (Markdown) from results/*.json.

Every number is filled in from the results files; only the prose is fixed. Templates use <<key>> tokens
(not str.format) so LaTeX braces need no escaping. Items the pipeline did not do are marked [AUTHOR ACTION].
Conventions: "all queries" = all validation questions; policy comparisons = mean over seeds on held-out test
pages; figures marked "seed 1" show the first split only.
Outputs: paper_out/text/results_section.tex, paper_out/text/response_to_reviewers.md
"""
import json
import re

import numpy as np

from common import DATA, RESULTS, ROOT
from paper_sections import write_sections

OUT = ROOT / "paper_out" / "text"


def J(name):
    return json.load(open(RESULTS / name, encoding="utf-8"))


def values():
    man = json.load(open(DATA / "manifest.json"))
    meta, ocr = J("path_outputs_meta.json"), J("ocr_cost.json")
    tx, vi, lb, ev, es = (J("index_cost_text.json"), J("index_cost_visual.json"), J("labels.json"),
                          J("eval_main.json"), J("escalation.json"))
    lt, ix, rc, bj = J("load_test.json"), J("index_scaling.json"), J("router_cost.json"), J("batching.json")
    ce_sum, ce_lc, ce_cost = J("router_ce/summary.json"), J("router_ce/learning_curve.json"), J("router_ce_cost.json")
    hw, a = meta["hardware"], ev["aggregate"]
    V = {}
    V["max_new_tokens"] = meta["max_new_tokens"]

    # data, hardware
    V["n_q"], V["n_pages"] = f"{man['natural']['queries']:,}", f"{man['pages']['total']:,}"
    g = hw["gpus"][0]
    V["gpu"], V["gpu_mem"] = g["name"].replace("NVIDIA ", ""), f"{g['memory_gb']:.0f}"
    V["generator"], V["k_text"], V["max_vis_tok"] = meta["model"].split("/")[-1], meta["k_text"], meta["max_pixels"] // 784
    V["ocr_s"], V["ocr_kj"] = f"{ocr['total_seconds']:.0f}", f"{ocr['total_energy_j'] / 1000:.1f}"
    V["n_seeds"], V["lam"] = ev["n_seeds"], f"{ev['lambda_default']:g}"
    nt = [s["n_test"] for s in ev["per_seed"]]
    V["n_test_min"], V["n_test_max"] = f"{min(nt):,}", f"{max(nt):,}"

    # per path, over all queries
    rows = [json.loads(l) for l in open(RESULTS / "path_outputs.jsonl", encoding="utf-8")]
    gpu = {}
    for p in ("text", "visual", "hybrid"):
        r = [x for x in rows if x["path"] == p]
        gpu[p] = np.mean([x["gpu_seconds"] for x in r])
        V[f"{p}_anls"] = f"{np.mean([x['anls'] for x in r]):.3f}"
        V[f"{p}_gpu"] = f"{gpu[p]:.2f}"
        V[f"{p}_j"] = f"{np.mean([x['energy_j'] for x in r]):.0f}"
    V["hyb_x_vis"], V["hyb_x_txt"] = f"{gpu['hybrid'] / gpu['visual']:.1f}", f"{gpu['hybrid'] / gpu['text']:.1f}"
    by = {}
    for x in rows:
        by.setdefault(x["qid"], {})[x["path"]] = x
    fail = [d for d in by.values() if max(x["anls"] for x in d.values()) == 0]
    V["all_fail"] = f"{100 * len(fail) / len(by):.0f}"
    V["fail_nogold"] = f"{100 * np.mean([not any(x['gold_in_context'] for x in d.values()) for d in fail]):.0f}"
    V["r1_bm25"], V["r1_col"] = f"{tx['recall']['@1']:.2f}", f"{vi['recall']['@1']:.2f}"

    # labels (seed-1 split)
    kw = lb["keyword_rule_vs_outcome"]["keyword_submitted"]
    V["kappa"], V["kw_agree"] = f"{kw['kappa_with_utility_label']:.2f}", f"{100 * kw['agreement_with_utility_label']:.1f}"
    V["kw_solv_n"] = f"{kw['solvable_queries']:,}"
    V["kw_agree_solv"] = f"{100 * kw['agreement_with_utility_label_solvable']:.1f}"
    V["kw_kappa_solv"] = f"{kw['kappa_with_utility_label_solvable']:.3f}"
    ra = J("review_analyses.json")
    bm, ev0 = ra["blind_mix"], ra["escalation_vs_lambda0"]
    V.update({
        "bm_mix_anls": f"{bm['anls_at_ce_cost']['mean']:.3f}", "bm_diff": f"{bm['ce_minus_mix_anls']['mean']:.3f}",
        "bm_mix_gpu": f"{bm['gpu_at_ce_anls']['mean']:.3f}", "bm_save": f"{100 * bm['saving_vs_mix']['mean']:.0f}",
        "bm_save_min": f"{100 * bm['saving_vs_mix']['min']:.0f}", "bm_save_max": f"{100 * bm['saving_vs_mix']['max']:.0f}",
        "esc_l0_anls": f"{ev0['lambda0_anls']['mean']:.3f}", "esc_l0_gpu": f"{ev0['lambda0_gpu_s']['mean']:.3f}",
        "disc_min": f"{ra['mcnemar_power']['discordant_pairs']['min']:.0f}",
        "disc_max": f"{ra['mcnemar_power']['discordant_pairs']['max']:.0f}",
        "cov_tested": f"{ra['split_coverage']['tested_at_least_once']:,}",
        "cov_multi": f"{ra['split_coverage']['tested_more_than_once']:,}",
        "cov_never": f"{ra['split_coverage']['never_tested']:,}",
        "cap_hyb_max": f"{ra['capacity_1gpu']['always_hybrid_max']:g}",
        "cap_hyb_zero": ra["capacity_1gpu"]["seeds_below_first_step"],
    })
    ca = J("controls_analysis.json")
    cA, cd = ca["by_dataset"]["all"], ca["by_dataset"]
    dd = lambda b, k: b["diffs"][k]
    ci = lambda x: f"{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}"
    V.update({
        "ct_n": f"{ca['n_questions']:,}", "ct_n_ds": f"{cd['docvqa']['n']:,}",
        "ct_v3_t": f"{dd(cA, 'visual_k3_minus_text')['mean']:.2f}", "ct_v3_t_ci": ci(dd(cA, "visual_k3_minus_text")),
        "ct_v_t1": f"{dd(cA, 'visual_minus_text_k1')['mean']:.2f}", "ct_v_t1_ci": ci(dd(cA, "visual_minus_text_k1")),
        "ct_v3_v": f"{dd(cA, 'visual_k3_minus_visual')['mean']:.3f}", "ct_t_t1": f"{dd(cA, 'text_minus_text_k1')['mean']:.3f}",
        "ct_gv_gt": f"{dd(cA, 'gold_visual_minus_gold_text')['mean']:.2f}",
        "ct_gv_gt_doc": f"{dd(cd['docvqa'], 'gold_visual_minus_gold_text')['mean']:.2f}",
        "ct_gv_gt_info": f"{dd(cd['infovqa'], 'gold_visual_minus_gold_text')['mean']:.2f}",
        "ct_vis": f"{cA['anls']['visual']:.3f}", "ct_gvis": f"{cA['anls']['gold_visual']:.3f}",
        "ct_txt": f"{cA['anls']['text']:.3f}", "ct_gtxt": f"{cA['anls']['gold_text']:.3f}",
        "ct_hyb": f"{cA['anls']['hybrid']:.3f}", "ct_ghyb": f"{cA['anls']['gold_hybrid']:.3f}",
        "ct_v3": f"{cA['anls']['visual_k3']:.3f}",
        "ct_gv_v_doc": f"{dd(cd['docvqa'], 'gold_visual_minus_visual')['mean']:.2f}",
        "ct_gv_v_info": f"{dd(cd['infovqa'], 'gold_visual_minus_visual')['mean']:.2f}",
        "ct_gh_gv": f"{dd(cA, 'gold_hybrid_minus_gold_visual')['mean']:.3f}",
        "ct_gh_gv_ci": ci(dd(cA, "gold_hybrid_minus_gold_visual")),
        "ct_v3_x": f"{ca['generate_s_a10_mean']['visual_k3'] / ca['generate_s_a10_mean']['gold_visual']:.1f}",
        "ct_rep_t": f"{100 * ca['reproduction']['text']['identical_answer_share']:.1f}",
        "ct_rep_v": f"{100 * ca['reproduction']['visual']['identical_answer_share']:.1f}",
        "ct_rep_n": ca["reproduction"]["text"]["n"],
        "ct_rep_vis_a10": f"{ca['reproduction']['visual']['anls_a10']:.3f}",
        "ct_rep_vis_ws": f"{ca['reproduction']['visual']['anls_workstation']:.3f}",
    })
    rep_h = ca["reproduction"]["hybrid"]
    V.update({"ct_rep_h": f"{100 * rep_h['identical_answer_share']:.1f}",
              "ct_rep_hyb_a10": f"{rep_h['anls_a10']:.3f}", "ct_rep_hyb_ws": f"{rep_h['anls_workstation']:.3f}",
              "ct_slow": f"{rep_h['generate_s_a10_mean'] / rep_h['generate_s_workstation_mean']:.1f}"})
    cf = J("confirmation.json")
    cs, cfx = cf["seeds"], cf["fixed"]
    man_c = json.load(open(ROOT / "data_confirm" / "manifest.json"))
    V.update({
        "cf_n": f"{cf['n_queries']:,}", "cf_pages": f"{cf['pages']:,}",
        "cf_n_ds": f"{man_c['natural']['by_dataset']['docvqa']:,}",
        "cf_pool_doc": f"{man_c['pages']['by_dataset']['docvqa']:,}",
        "cf_pool_info": f"{man_c['pages']['by_dataset']['infovqa']:,}",
        "cf_rev": man_c["sources"]["docvqa"]["revision"][:12],
        "cf_n_shards": len(man_c["sources"]["docvqa"]["shards"]) + len(man_c["sources"]["infovqa"]["shards"]),
        "cf_oom_sent": "One question was" if len(cf["excluded_oom"]) == 1 else f"{len(cf['excluded_oom'])} questions were",
        "cf_oom_tok": f"{cf['excluded_oom'][0]['input_tokens']:,}",
        "cf_agree_min": f"{100 * min(r['test_decision_agreement'] for r in cf['reproducibility']):.0f}",
        "cf_agree_max": f"{100 * max(r['test_decision_agreement'] for r in cf['reproducibility']):.0f}",
        "cf_rep_anls": f"{max(abs(r['test_anls_retrained'] - r['test_anls_saved']) for r in cf['reproducibility']):.3f}",
        "cf_anls": f"{cf['summary']['anls']['mean']:.3f}", "cf_anls_min": f"{cf['summary']['anls']['min']:.3f}",
        "cf_anls_max": f"{cf['summary']['anls']['max']:.3f}", "cf_gpu": f"{cf['summary']['gpu_s']['mean']:.2f}",
        "cf_hyb_anls": f"{cfx['hybrid']['anls']:.3f}", "cf_hyb_gpu": f"{cfx['hybrid']['gpu_s']:.2f}",
        "cf_vis_anls": f"{cfx['visual']['anls']:.3f}", "cf_vis_gpu": f"{cfx['visual']['gpu_s']:.2f}",
        "cf_save": f"{100 * cf['summary']['saving_vs_hybrid']['mean']:.0f}",
        "cf_save_min": f"{100 * cf['summary']['saving_vs_hybrid']['min']:.0f}",
        "cf_save_max": f"{100 * cf['summary']['saving_vs_hybrid']['max']:.0f}",
        "cf_dh": f"{cf['summary']['anls_diff_vs_hybrid']['mean']:.3f}",
        "cf_dh_pos": sum(s["anls_diff_vs_hybrid_ci95"][0] > 0 for s in cs),
        "cf_mc_sig": sum(s["mcnemar_vs_hybrid"]["p"] < 0.05 and s["mcnemar_vs_hybrid"]["ce_only"] > s["mcnemar_vs_hybrid"]["hybrid_only"] for s in cs),
        "cf_dv": f"{cf['summary']['anls_diff_vs_visual']['mean']:.3f}",
        "cf_dv_incl": sum(s["anls_diff_vs_visual_ci95"][0] <= 0 <= s["anls_diff_vs_visual_ci95"][1] for s in cs),
        "cf_vs_vis": f"{100 * (cf['summary']['gpu_s']['mean'] / cfx['visual']['gpu_s'] - 1):.0f}",
        "cf_vis_less": f"{100 * (1 - cfx['visual']['gpu_s'] / cf['summary']['gpu_s']['mean']):.0f}",
        "cf_oracle": f"{cf['oracle']['anls']:.3f}",
    })
    tp = J("token_pooling.json")["methods"]
    best_f = max((k for k in tp if k.startswith("hier")), key=lambda k: int(k[4:]))
    a10_budget_gib = J("a10/a10_shard_scaling.json")["budget_gib_per_gpu"]
    V.update({
        "tp_f": best_f[4:], "tp_vec": tp[best_f]["vectors_per_page"], "tp_r1": f"{tp[best_f]['recall@1']:.3f}",
        "tp_r5": f"{tp[best_f]['recall@5']:.3f}", "tp_full_r1": f"{tp['full']['recall@1']:.3f}",
        "tp_full_r5": f"{tp['full']['recall@5']:.3f}", "tp_2x2_r1": f"{tp['pooled2x2']['recall@1']:.3f}",
        "tp_2x2_r5": f"{tp['pooled2x2']['recall@5']:.3f}", "tp_gib_1m": f"{tp[best_f]['index_gib_1m_pages']:.0f}",
        "tp_a10": int(-(-tp[best_f]["index_gib_1m_pages"] // a10_budget_gib)),
        "tp_x": f"{tp['full']['vectors_per_page'] / tp[best_f]['vectors_per_page']:.1f}",
    })
    se = J("sim_experiments.json")
    sw, sav = se["slo_sweep"], se["slo_sweep_saving_vs_hybrid"]
    vis_more = {T: sw["cavr_ce_valsel"][T]["mean"] / sw["always_visual"][T]["mean"] - 1 for T in sav}
    bmod = se["batch_model_saving_vs_hybrid"]
    jn = se["joint"]["by_pages"][str(max(int(n) for n in se["joint"]["by_pages"]))]
    ad = se["adaptive"]["policies"]
    V.update({
        "sw_min_t": f"{min(float(t) for t in sav):g}", "sw_max_t": f"{max(float(t) for t in sav):g}",
        "sw_save_min": f"{100 * min(sav.values()):.0f}", "sw_save_max": f"{100 * max(sav.values()):.0f}",
        "sw_vis_min": f"{100 * min(vis_more.values()):.0f}", "sw_vis_max": f"{100 * max(vis_more.values()):.0f}",
        "bm_long_2": f"{100 * bmod['longest_member']['2.0']:.0f}", "bm_long_5": f"{100 * bmod['longest_member']['5.0']:.0f}",
        "bm_mean_5": f"{100 * bmod['mean_member']['5.0']:.0f}",
        "jn_pages": "a million" if int(max(se["joint"]["by_pages"], key=int)) == 10**6 else max(se["joint"]["by_pages"], key=int),
        "jn_ret": jn["policies"]["cavr_ce_valsel"]["retrieval_gpus"], "jn_hold": jn["gpus_to_hold_index"],
        "jn_ce": f"{jn['policies']['cavr_ce_valsel']['total']:.0f}", "jn_hyb": f"{jn['policies']['always_hybrid']['total']:.0f}",
        "jn_vis": f"{jn['policies']['always_visual']['total']:.0f}", "jn_cavr": f"{jn['policies']['cavr']['total']:.0f}",
        "jn_cavr_text": f"{100 * (1 - se['joint']['colpali_share']['cavr']):.0f}",
        "ad_pool": se["adaptive"]["pool_gpus"], "ad_base": f"{se['adaptive']['burst']['base_qps']:g}",
        "ad_burst": f"{se['adaptive']['burst']['burst_qps']:g}", "ad_base_s": f"{se['adaptive']['burst']['base_s']:g}",
        "ad_burst_s": f"{se['adaptive']['burst']['burst_s']:g}",
        "ad_fix_slo": f"{100 * ad['cavr_ce_valsel']['slo_2s']:.1f}", "ad_fix_anls": f"{ad['cavr_ce_valsel']['anls']:.3f}",
        "ad_adp_slo": f"{100 * ad['cavr_ce_adaptive']['slo_2s']:.1f}", "ad_adp_anls": f"{ad['cavr_ce_adaptive']['anls']:.3f}",
        "ad_02_slo": f"{100 * ad['cavr_ce@0.2']['slo_2s']:.1f}", "ad_02_anls": f"{ad['cavr_ce@0.2']['anls']:.3f}",
        "ad_vis_slo": f"{100 * ad['always_visual']['slo_2s']:.1f}", "ad_hyb_slo": f"{100 * ad['always_hybrid']['slo_2s']:.1f}",
        "ad_up": f"{100 * (1 - ad['cavr_ce_adaptive']['level_share'][0]):.0f}",
    })
    V["kw_text_share"] = f"{100 * kw['distribution'].get('text', 0) / lb['n_queries']:.0f}"

    # policies: mean (+- std) over seeds on test
    for n in ("always_visual", "always_hybrid", "cavr", "cavr_ce_valsel", "oracle"):
        V[f"{n}_anls"] = f"{a[n]['anls']['mean']:.3f}"
        V[f"{n}_gpu"] = f"{a[n]['gpu_s']['mean']:.3f}"
        V[f"{n}_anls_pm"] = f"{a[n]['anls']['mean']:.3f} $\\pm$ {a[n]['anls']['std']:.3f}"
    cv = [s["tests_vs_cavr"]["always_visual"]["paired_boot_anls"]["ci95"] for s in ev["per_seed"]]
    V["cavr_vis_ci_all0"] = all(c[0] <= 0 <= c[1] for c in cv)
    V["cavr_vis_bound"] = f"{max(abs(x) for c in cv for x in c):.3f}"
    V["cavr_vs_vis_cost"] = f"{100 * (a['cavr']['gpu_s']['mean'] / a['always_visual']['gpu_s']['mean'] - 1):.0f}"
    S = [s for s in ev["per_seed"] if "cavr_ce_valsel" in s["policies"]]
    T = [s["tests_cavr_ce_valsel"] for s in S]
    ph = [t["always_hybrid"]["mcnemar_em"]["p_value"] for t in T]
    pv = [t["always_visual"]["mcnemar_em"]["p_value"] for t in T]
    bh = [t["always_hybrid"]["paired_boot_anls"] for t in T]
    bv = [t["always_visual"]["paired_boot_anls"] for t in T]
    sv = [100 * (1 - s["policies"]["cavr_ce_valsel"]["gpu_s"] / s["policies"]["always_hybrid"]["gpu_s"]) for s in S]
    V["ce_n"] = len(T)
    V["ce_nonsig_h"] = sum(p > 0.05 for p in ph)
    V["ce_ph_min"], V["ce_ph_max"] = f"{min(ph):.2g}", f"{max(ph):.2g}"
    V["ce_dh_mean"] = f"{np.mean([b['mean_diff'] for b in bh]):+.3f}"
    V["ce_dh_min"], V["ce_dh_max"] = f"{min(b['mean_diff'] for b in bh):+.3f}", f"{max(b['mean_diff'] for b in bh):+.3f}"
    V["ce_lower_n"] = sum(b["mean_diff"] < 0 for b in bh)
    V["ce_ci_excl"] = sum(b["ci95"][1] < 0 for b in bh)
    V["ce_sig_v"], V["ce_pv_max"] = sum(p < 0.05 for p in pv), f"{max(pv):.2g}"
    V["ce_v_ci_excl"] = sum(b["ci95"][0] > 0 for b in bv)
    V["ce_dv_min"], V["ce_dv_max"] = f"{min(b['mean_diff'] for b in bv):+.3f}", f"{max(b['mean_diff'] for b in bv):+.3f}"
    V["ce_save"], V["ce_save_min"], V["ce_save_max"] = f"{np.mean(sv):.0f}", f"{min(sv):.0f}", f"{max(sv):.0f}"
    V["ce_vs_vis_cost"] = f"{100 * (a['cavr_ce_valsel']['gpu_s']['mean'] / a['always_visual']['gpu_s']['mean'] - 1):.0f}"
    V["ece_head"] = f"{np.mean([s['confidence_ece'] for s in ev['per_seed']]):.3f}"

    # lambda sweep of CAVR-CE (mean over seeds)
    def sw(lam):
        return (np.mean([s["lambda_sweep"][lam]["cavr_ce"]["anls"] for s in ev["per_seed"]]),
                np.mean([s["lambda_sweep"][lam]["cavr_ce"]["gpu_s"] for s in ev["per_seed"]]))
    lo, hi = sw("1.0"), sw("0.0")
    V["sw_lo_anls"], V["sw_lo_gpu"], V["sw_hi_anls"], V["sw_hi_gpu"] = f"{lo[0]:.3f}", f"{lo[1]:.2f}", f"{hi[0]:.3f}", f"{hi[1]:.2f}"

    # router
    V["ce_auc"] = f"{np.mean([s['ce_full']['auc_mean'] for s in ce_sum['seeds']]):.3f}"
    V["gbm_auc"] = f"{np.mean([s['gbm_tab']['auc_mean'] for s in ce_sum['seeds']]):.3f}"
    lc = {f: np.mean([r["ce_auc"] for r in ce_lc["runs"] if r["frac"] == f]) for f in ce_lc["fracs"]}
    V["lc_25"], V["lc_100"], V["lc_seeds"] = f"{lc[0.25]:.3f}", f"{lc[1.0]:.3f}", len(ce_lc["seeds"])
    V["ce_ms"], V["ce_j"] = f"{1000 * ce_cost['per_query_batch1']['latency_s']:.1f}", f"{ce_cost['per_query_batch1']['energy_j']:.2f}"
    V["ce_qps"] = f"{max(v['queries_per_s'] for v in ce_cost['cross_encoder_throughput'].values()):,.0f}"
    V["cavr_router_ms"], V["cavr_router_cpu_ms"] = f"{1000 * rc['cuda']['batch1_p50_s']:.1f}", f"{1000 * rc['cpu']['batch1_p50_s']:.1f}"
    V["cavr_router_qps"] = f"{max(v['queries_per_s'] for v in rc['cuda']['throughput'].values()):,.0f}"

    # escalation on CAVR-CE, mean over seeds
    def esc(key, sig, mode, tau):
        vs = [s[key]["none"] if sig == "none" else s[key][sig][mode][tau] for s in es["per_seed"] if key in s]
        return {k: float(np.mean([v[k] for v in vs])) for k in vs[0]}
    e0, e1 = esc("results_ce", "none", None, None), esc("results_ce", "colpali_margin", "reroute", "0.5")
    ser, spe = esc("results_ce", "answer_logprob_min", "serial", "0.5"), esc("results_ce", "answer_logprob_min", "speculative", "0.5")
    hyb_gpu = a["always_hybrid"]["gpu_s"]["mean"]
    V["esc_none_anls"], V["esc_none_gpu"], V["esc_none_p95"] = f"{e0['anls']:.3f}", f"{e0['gpu_s']:.3f}", f"{e0['latency_p95_s']:.2f}"
    V["esc_anls"], V["esc_gpu"], V["esc_p95"] = f"{e1['anls']:.3f}", f"{e1['gpu_s']:.3f}", f"{e1['latency_p95_s']:.2f}"
    V["esc_rate"], V["esc_save"] = f"{100 * e1['escalation_rate']:.0f}", f"{100 * (1 - e1['gpu_s'] / hyb_gpu):.0f}"
    V["ser_p95"], V["ser_gpu"] = f"{ser['latency_p95_s']:.2f}", f"{ser['gpu_s']:.2f}"
    V["spe_p95"], V["spe_gpu"] = f"{spe['latency_p95_s']:.2f}", f"{spe['gpu_s']:.2f}"
    V["spe_vs_hyb"] = f"{100 * (spe['gpu_s'] / hyb_gpu - 1):.0f}"
    V["hyb_p95"] = f"{a['always_hybrid']['latency_p95_s']['mean']:.2f}"
    ece_ce = {k: np.mean([s["ece_ce"][k] for s in es["per_seed"] if "ece_ce" in s]) for k in es["per_seed"][0]["ece_ce"]}
    V["ece_esc_min"], V["ece_esc_max"] = f"{min(ece_ce.values()):.3f}", f"{max(ece_ce.values()):.3f}"

    # serving, mean +- std over seeds
    ss = lt["seeds_summary"]["policies"]
    for n in ("always_hybrid", "always_visual", "cavr", "cavr_ce_valsel", "oracle"):
        gd = ss[n]["gpus_needed"]["100"]
        V[f"g100_{n}"] = f"{gd['mean']:.0f}"
        V[f"g100_{n}_pm"] = f"{gd['mean']:.1f} $\\pm$ {gd['std']:.1f}"
    V["g100_save"] = f"{100 * (1 - ss['cavr_ce_valsel']['gpus_needed']['100']['mean'] / ss['always_hybrid']['gpus_needed']['100']['mean']):.0f}"
    V["g100_ratio"] = f"{ss['cavr_ce_valsel']['gpus_needed']['100']['mean'] / ss['always_hybrid']['gpus_needed']['100']['mean']:.2f}"
    V["gs_ratio"] = f"{a['cavr_ce_valsel']['gpu_s']['mean'] / a['always_hybrid']['gpu_s']['mean']:.2f}"
    P1 = lt["policies"]
    V["g100_5s_hyb"], V["g100_5s_ce"] = P1["always_hybrid"]["gpus_needed"]["5.0"]["100"], P1["cavr_ce_valsel"]["gpus_needed"]["5.0"]["100"]
    V["g100_5s_save"] = f"{100 * (1 - V['g100_5s_ce'] / V['g100_5s_hyb']):.0f}"
    V["ix1m_gb"] = f"{10 * ix['scaling']['100000']['index_gb_bf16']:.0f}"
    for p, d in bj["paths"].items():
        V[f"bq_{p}"] = f"{max(x['queries_per_s'] for x in d.values() if 'note' not in x):.2f}"
    s100, s1k, s50k = ix["scaling"]["100000"], ix["scaling"]["1000"], ix["scaling"]["50000"]
    V["ix100_gb"], V["ix100_stream_ms"] = f"{s100['index_gb_bf16']:.1f}", f"{1000 * s100['cpu_stream']['p50_s']:,.0f}"
    V["ix100_pool_gb"], V["ix100_pool_ms"] = f"{s100['index_gb_pooled2x2']:.1f}", f"{1000 * s100['gpu_pooled2x2']['p50_s']:.1f}"
    V["ix100_shards"], V["ix100_shard_ms"] = s100["sharded"]["gpus_needed"], f"{1000 * s100['sharded']['emulated_parallel_p50_s']:.1f}"
    V["ix1k_ms"], V["ix50k_ms"] = f"{1000 * s1k['gpu_bf16']['p50_s']:.2f}", f"{1000 * s50k['gpu_bf16']['p50_s']:.1f}"
    V["pool_r1_full"], V["pool_r1"] = f"{ix['pooling_recall_at1']['full']:.3f}", f"{ix['pooling_recall_at1']['pooled2x2']:.3f}"
    mg = J("multi_gpu_scaling.json")
    st, sm, ms = mg["strong"], mg["summary"], mg["source"]
    weak_per, weak = next(iter(mg["weak"].items()))
    wmax = weak[max(weak, key=int)]
    V.update({
        "mg_gpu": ms["devices"][0], "mg_n": mg["n_gpus"], "mg_n_word": {2: "two", 3: "three", 4: "four"}[mg["n_gpus"]],
        "mg_mem": f"{ms['node']['gpus'][0]['memory_gb']:.1f}", "mg_topk": ms["topk"], "mg_nlat": ms["n_lat"],
        "mg_ncheck": ms["n_check"], "mg_budget": f"{ms['budget_gib_per_gpu']:.1f}",
        "mg_batches": ", ".join(ms["batches"][:-1]) + " and " + ms["batches"][-1], "mg_tb": mg["throughput_batch"],
        "mg_exact": f"{100 * sm['check_exact_min']:.1f}", "mg_tie": f"{sm['check_exact_or_tie_min']:.3f}",
        "mg_ovh_ms": f"{1000 * sm['max_overhead_s_ge50k']:.1f}",
        "mg_lat_eff_min": f"{100 * sm['min_latency_efficiency_ge50k']:.1f}",
        "mg_thr_eff_min": f"{100 * sm['min_throughput_efficiency_ge10k']:.1f}",
        "mg_idle_w": f"{sm['idle_w_per_gpu']:.0f}", "mg_idle_lo": f"{sm['idle_w_range'][0]:.0f}",
        "mg_idle_hi": f"{sm['idle_w_range'][1]:.0f}", "mg_min_pages": f"{mg['min_pages_fit'] // 1000}k",
        "mg_weak_k": f"{int(weak_per) // 1000}k", "mg_weak_eff": f"{100 * wmax['efficiency']:.1f}",
        "mg_weak_pages": f"{wmax['pages'] // 1000}k", "mg_weak_gib": f"{wmax['index_gib_total']:.1f}",
        "mg_weak_ms": f"{1000 * wmax['p50_s']:.0f}", "mg_maxpages": f"{max(int(n) for n in st) // 1000}k",
    })
    for n in ("1000", "10000", "50000", "100000"):
        k = f"mg{int(n) // 1000}k"
        for g, v in st[n]["by_gpus"].items():
            V[f"{k}_ms{g}"] = f"{1000 * v['p50_s']:.0f}" if v["p50_s"] >= 0.1 else f"{1000 * v['p50_s']:.1f}"
            V[f"{k}_sp{g}"] = f"{v['latency_speedup']:.2f}"
            V[f"{k}_eff{g}"] = f"{100 * v['latency_efficiency']:.1f}"
            V[f"{k}_qps{g}"] = f"{v['throughput_qps']:.1f}"
            V[f"{k}_teff{g}"] = f"{100 * v['throughput_efficiency']:.0f}"
            V[f"{k}_e{g}"] = f"{v['energy_j_per_query_node']:.1f}"
            V[f"{k}_ea{g}"] = f"{v['energy_j_per_query_active']:.1f}"
    V["gpu_short"] = V["gpu"].replace("GeForce ", "")
    V["mg_vs_ws"] = f"{st['50000']['by_gpus']['1']['p50_s'] / s50k['gpu_bf16']['p50_s']:.0f}"
    # second multi-GPU node (2x A10), measured with the same benchmark
    ma = J("a10/multi_gpu_scaling.json")
    sa, sma = ma["strong"], ma["summary"]
    wa_per, wa = "50000", ma["weak"]["50000"]
    wa_top = wa[max(wa, key=int)]
    n_max = max(sa, key=int)
    r_max = sa[n_max]["by_gpus"][max(sa[n_max]["by_gpus"], key=int)]
    a1, p1 = sa["50000"]["by_gpus"]["1"], st["50000"]["by_gpus"]["1"]
    V.update({
        "a10_gpu": ma["source"]["devices"][0], "a10_n_word": {2: "two", 3: "three", 4: "four"}[ma["n_gpus"]],
        "a10_exact": f"{100 * sma['check_exact_min']:.1f}", "a10_budget": f"{ma['source']['budget_gib_per_gpu']:.1f}",
        "a10_ovh_ms": f"{1000 * sma['max_overhead_s_ge50k']:.1f}",
        "a10_lat_eff_min": f"{100 * sma['min_latency_efficiency_ge50k']:.1f}",
        "a10_thr_eff_min": f"{100 * sma['min_throughput_efficiency_ge10k']:.1f}",
        "a10_weak_k": f"{int(wa_per) // 1000}k", "a10_weak_eff": f"{100 * wa_top['efficiency']:.1f}",
        "a10_weak_pages": f"{wa_top['pages'] // 1000}k", "a10_weak_gib": f"{wa_top['index_gib_total']:.1f}",
        "a10_weak_ms": f"{1000 * wa_top['p50_s']:.1f}", "a10_maxpages": f"{int(n_max) // 1000}k",
        "a10_max_ms": f"{1000 * r_max['p50_s']:.1f}",
        "a10_vs_p40": f"{p1['p50_s'] / a1['p50_s']:.0f}",
        "a10_e_ratio": f"{p1['energy_j_per_query_active'] / a1['energy_j_per_query_active']:.1f}",
        "a10_vs_ws": f"{a1['p50_s'] / s50k['gpu_bf16']['p50_s']:.1f}",
    })
    for n in sa:
        for g, v in sa[n]["by_gpus"].items():
            k = f"a10_{int(n) // 1000}k"
            V[f"{k}_ms{g}"] = f"{1000 * v['p50_s']:.0f}" if v["p50_s"] >= 0.1 else f"{1000 * v['p50_s']:.1f}"
            V[f"{k}_eff{g}"] = f"{100 * v['latency_efficiency']:.1f}"
            V[f"{k}_ea{g}"] = f"{v['energy_j_per_query_active']:.1f}"
    # index in host memory, streamed to the A10s
    hs = J("a10/host_stream_scaling.json")
    hr = {(r["pages"], r["gpus"]): r for r in hs["runs"]}
    hmax, hg, hb = hs["summary"]["max_pages"], max(r["gpus"] for r in hs["runs"]), str(hs["source"]["big_batch"])
    h1, h2, vr, r1m = hr[(hmax, 1)], hr[(hmax, hg)], hs["vs_resident"], hs["resident_1m"]
    gib_page = hs["summary"]["max_gib"] / hmax

    def kpages(n):
        return f"{n // 10**6}M" if n % 10**6 == 0 else f"{n // 1000}k"
    sizes = [kpages(n) for n in sorted({r["pages"] for r in hs["runs"]})]
    cross = [r["crossover_batch"] for r in hs["runs"]]
    src = hs["source"]
    V.update({
        "hs_max_words": {10**6: "a million"}.get(hmax, f"{hmax:,}"), "hs_max_k": kpages(hmax),
        "hs_gib": f"{hs['summary']['max_gib']:.0f}", "hs_sizes": ", ".join(sizes[:-1]) + " and " + sizes[-1],
        "hs_ncheck": src["checks"]["1"]["queries"], "hs_check_k": kpages(src["checks"]["1"]["pages"]),
        "hs_copy1": f"{hs['copy_only']['1']['gb_per_s_total']:.1f}",
        "hs_copy2": f"{hs['copy_only'][str(hg)]['gb_per_s_total']:.1f}",
        "hs_share": f"{100 * hs['pcie_two_gpu_share']:.0f}",
        "hs_lat_sp": f"{hs['speedup'][str(hmax)]['latency']:.1f}",
        "hs_thr_sp": f"{hs['speedup'][str(hmax)]['throughput_big']:.1f}",
        "hs_p50_1": f"{h1['p50_s']:.1f}", "hs_p50_2": f"{h2['p50_s']:.1f}",
        "hs_over_copy": f"{100 * (hs['summary']['max_p50_over_copy'] - 1):.0f}",
        "hs_cross": f"{min(cross):.0f}--{max(cross):.0f}", "hs_b": hb,
        "hs_qps1": f"{h1['qps'][hb]:.2f}", "hs_qps2": f"{h2['qps'][hb]:.2f}",
        "hs_b_lat2": f"{int(hb) / h2['qps'][hb]:.0f}",
        "hs_gpus_q": f"{hs['summary']['gpu_s_per_query_1m']:.2f}",
        "hs_res_gpus_q": f"{r1m['gpu_s_per_query_big']:.2f}",
        "hs_res_pages": kpages(vr["pages"]), "hs_vs_res": f"{100 * vr['throughput_ratio']:.0f}",
        "hs_e_stream": f"{vr['streamed_j_per_query_big']:.1f}", "hs_e_res": f"{vr['resident_j_per_query_big']:.1f}",
        "hs_res_gpus": r1m["gpus_needed"], "hs_res_ms": f"{1000 * r1m['projected_p50_s']:.0f}",
        "hs_fewer": f"{r1m['gpus_needed'] / hg:.0f}",
        "hs_reg_gib": f"{src['host_index']['register_block_pages'] * gib_page:.0f}",
        "hs_chunk": f"{src['chunk_pages']:,}", "hs_chunk_gib": f"{src['chunk_pages'] * gib_page:.1f}",
        "hs_nlat": src["n_lat"], "hs_reps": src["reps"],
        "hs_batches": ", ".join(map(str, src["batches"][:-1])) + " and " + str(src["batches"][-1]),
    })
    V.update({
        "hs_budget_pct": f"{100 * r1m['budget_share_of_memory']:.0f}", "hs_res_nodes": r1m["nodes_needed"],
        "hs_100k_p50_1": f"{hr[(vr['pages'], 1)]['p50_s']:.2f}",
        "hs_b1_gpus_s": f"{vr['streamed_gpu_s_b1']:.2f}", "hs_b1_gpus_r": f"{vr['resident_gpu_s_b1']:.3f}",
        "hs_b1_e_s": f"{vr['streamed_j_per_query_b1']:.0f}", "hs_b1_e_r": f"{vr['resident_j_per_query_b1']:.0f}",
        "hs_cv": f"{100 * hs['summary']['max_latency_cv']:.1f}",
    })
    b = ix["build"]
    V["build_h"], V["build_pps"], V["build_b"] = (f"{b['extrapolated_build_hours']['100000']:.1f}",
                                                  f"{b['by_batch'][str(b['best_batch'])]['pages_per_s']:.1f}", b["best_batch"])
    vidx = vi["index"]
    V["full_build_pps"], V["full_build_b"] = f"{vidx['pages'] / vidx['gpu_s']:.1f}", vidx["page_batch"]
    return V


def fill(template, V):
    out = re.sub(r"<<(\w+)>>", lambda m: str(V[m.group(1)]), template)
    assert "<<" not in out, "unfilled token"
    return out


RESULTS_TEX = r"""% Revised Results section. Generated by make_paper_text.py from results/*.json -- do not edit numbers by hand.
% Tables and figures are in paper_out/tables and paper_out/figures (make_outputs.py).
\section{Results}
\label{sec:results}

All numbers in this section come from one cached run of every path on every query
(<<n_q>> validation questions from DocVQA and InfographicVQA over <<n_pages>> page images) on a single
<<gpu>> (Table~\ref{tab:hardware}). Costs are measured, not priced: GPU-seconds, energy from NVML, and wall-clock
latency per query. Routers are trained and evaluated on <<n_seeds>> page-grouped 60/20/20 splits, so no page
appears in more than one split; each test split holds <<n_test_min>>--<<n_test_max>> questions. Unless marked
otherwise, policy results are means ($\pm$ standard deviation) over the <<n_seeds>> seeds; figures marked
``seed 1'' show the first split.


\subsection{Cost and answer quality of each path}
\label{sec:res-paths}

All three paths use the same generator (<<generator>>, greedy decoding). The text path reads the top-<<k_text>>
BM25 pages of OCR text; the visual path reads the top-1 ColPali page image (at most <<max_vis_tok>> visual
tokens); the hybrid path reads both. Over all <<n_q>> questions, always-text reaches ANLS <<text_anls>> at
<<text_gpu>> GPU-s per query, always-visual <<visual_anls>> at <<visual_gpu>> GPU-s, and always-hybrid
<<hybrid_anls>> at <<hybrid_gpu>> GPU-s (<<hybrid_j>>\,J).
With a shared local generator the text path is therefore \emph{not} a cheap path:
OCR text costs about as many input tokens as a page image, and the text path is much less accurate.
The real cost differences are between one context and two (hybrid costs <<hyb_x_vis>>$\times$ the visual path and
<<hyb_x_txt>>$\times$ the text path) and the size of the visual index at scale (Section~\ref{sec:res-scale}).
In <<all_fail>>\% of queries every path fails. For <<fail_nogold>>\% of these, no path had the gold page in its
context (page recall@1 is <<r1_bm25>> for BM25 and <<r1_col>> for ColPali; Table~\ref{tab:retrieval}). No router
can recover these queries; they bound every policy.

Two controls on <<ct_n>> of these questions (<<ct_n_ds>> per dataset; Table~\ref{tab:controls}) show where the
differences between paths come from. First, the visual advantage is a matter of modality, not of how many pages
the generator reads: at equal depth, one page image beats one page of OCR text by <<ct_v_t1>> ANLS (95\% interval
<<ct_v_t1_ci>>), and three page images beat the three-page text path by <<ct_v3_t>> (<<ct_v3_t_ci>>), whereas going
from one page to three adds only <<ct_t_t1>> for text and <<ct_v3_v>> for images. Second, retrieval errors cost more
than any choice of path. Given the gold page, the visual path rises from <<ct_vis>> to <<ct_gvis>>, the text path
from <<ct_txt>> to <<ct_gtxt>> and the hybrid path from <<ct_hyb>> to <<ct_ghyb>>; the gain is largest on DocVQA
(+<<ct_gv_v_doc>> for the visual path, against +<<ct_gv_v_info>> on InfographicVQA). Even with the gold page, the
image beats its OCR text by <<ct_gv_gt>> ANLS (<<ct_gv_gt_doc>> on DocVQA, <<ct_gv_gt_info>> on InfographicVQA), and
adding the OCR text to the gold image adds only <<ct_gh_gv>> (<<ct_gh_gv_ci>>). Three page images reach ANLS
<<ct_v3>>, above the hybrid path, but take <<ct_v3_x>>$\times$ the generation time of one image; such a path is
outside the evaluated path set.

\input{tables/controls}

\subsection{Outcome-based route labels}
\label{sec:res-labels}

Route labels are derived from measured outcomes: the label of a query is the path that maximises
$\mathrm{ANLS} - \lambda \cdot \mathrm{cost}$, with ties broken towards the cheaper path (Table~\ref{tab:labels}).
Labels from a keyword rule over the question text agree with these outcome labels on <<kw_agree>>\% of queries,
with Cohen's $\kappa = <<kappa>>$, i.e.\ no agreement beyond chance; the keyword rule sends <<kw_text_share>>\% of
queries to the text path. On the <<kw_solv_n>> queries that at least one path answers, agreement falls to
<<kw_agree_solv>>\% ($\kappa = <<kw_kappa_solv>>$): most of the overall agreement comes from queries that every path
fails, which tie and default to the cheaper text path. Routing accuracy measured against keyword labels therefore says nothing about cost or answer
quality.

\input{tables/labels}

\subsection{Routing policies}
\label{sec:res-policies}

Table~\ref{tab:main_results} compares fixed policies, a keyword rule, simple learned baselines,
CAVR, the content-aware CAVR-CE and the per-query oracle. All router costs include the router's own measured
overhead and, where the router reads ColPali scores, ColPali retrieval for every query.

CAVR on question embeddings alone (ANLS <<cavr_anls_pm>>) shows no significant difference from always-visual:
on every seed the 95\% bootstrap interval of the ANLS difference includes zero and lies within
$\pm$<<cavr_vis_bound>>, while CAVR uses <<cavr_vs_vis_cost>>\% more GPU. The question text alone does not predict
which path will succeed.
CAVR-CE adds what is available before generation: an mpnet cross-encoder reads the question with the
best-matching OCR passages of the retrieved pages, plus retrieval scores and ColPali per-token match statistics.
With $\lambda$ chosen on each seed's validation split (the cheapest $\lambda$ within 0.005 ANLS of always-hybrid),
CAVR-CE reaches ANLS <<cavr_ce_valsel_anls_pm>> at <<cavr_ce_valsel_gpu>> GPU-s per query, against
<<always_hybrid_anls_pm>> at <<always_hybrid_gpu>> GPU-s for always-hybrid: <<ce_save>>\% less GPU compute
(<<ce_save_min>>--<<ce_save_max>>\% across seeds; Table~\ref{tab:router_ce_tests}).
The price is a small ANLS loss: CAVR-CE is lower than always-hybrid on <<ce_lower_n>> of <<ce_n>> seeds
(mean difference $<<ce_dh_mean>>$, range $<<ce_dh_min>>$ to $<<ce_dh_max>>$), and the 95\% bootstrap interval
excludes zero on <<ce_ci_excl>> of <<ce_n>> seeds. On exact match, McNemar's test finds no significant difference
on <<ce_nonsig_h>> of <<ce_n>> seeds ($p = <<ce_ph_min>>$--$<<ce_ph_max>>$), but with only <<disc_min>>--<<disc_max>>
discordant pairs per seed it is underpowered, so this does not show that the two policies are equivalent.
Against always-visual, CAVR-CE gains $<<ce_dv_min>>$ to $<<ce_dv_max>>$ ANLS (bootstrap interval above zero on
<<ce_v_ci_excl>> of <<ce_n>> seeds; McNemar on exact match $p \le <<ce_pv_max>>$ on <<ce_sig_v>> of <<ce_n>> seeds)
for <<ce_vs_vis_cost>>\% more GPU.
Part of the saving over always-hybrid comes from using the hybrid path less often, whichever queries are chosen.
A quality-blind random mix of always-visual and always-hybrid at CAVR-CE's GPU cost reaches ANLS <<bm_mix_anls>>
(in expectation, computed per seed), and at CAVR-CE's ANLS it costs <<bm_mix_gpu>> GPU-s. The saving attributable
to choosing queries is therefore <<bm_save>>\% (<<bm_save_min>>--<<bm_save_max>>\% across seeds), or $+<<bm_diff>>$
ANLS at equal cost.

The per-query oracle reaches ANLS <<oracle_anls_pm>> at only <<oracle_gpu>> GPU-s, so substantial headroom
remains. It is limited by prediction: CAVR-CE predicts per-path success with a mean AUC of <<ce_auc>>
(<<gbm_auc>> for gradient boosting on the same tabular features), and a learning curve over 25--100\% of the
training pages (<<lc_seeds>> seeds) rises only from <<lc_25>> to <<lc_100>>.

\input{tables/main_results}
\input{tables/router_ce_tests}

\subsection{Held-out confirmation}
\label{sec:res-confirm}

The router was designed on the same five splits that we report on. To test it on questions that played no part in
its design, each seed's CAVR-CE was re-trained with the same code, data and seed on the <<a10_gpu>> node and frozen,
with its validation-chosen $\lambda$ and training-split cost estimates, and applied to <<cf_n>> new questions from the
DocVQA and InfographicVQA training splits (<<cf_pages>> pages, none shared with the validation data;
Section~\ref{sec:setup-confirm}). The re-trained routers make the same choice as the saved ones on
<<cf_agree_min>>--<<cf_agree_max>>\% of their original test queries, with test ANLS within $\pm$<<cf_rep_anls>>.

On the confirmation set (Table~\ref{tab:confirmation}), the saving over always-hybrid holds. CAVR-CE reaches ANLS
<<cf_anls>> (<<cf_anls_min>>--<<cf_anls_max>> across seeds) at <<cf_gpu>> GPU-s per query, against <<cf_hyb_anls>> at
<<cf_hyb_gpu>> GPU-s for always-hybrid: <<cf_save>>\% less GPU compute (<<cf_save_min>>--<<cf_save_max>>\%), with ANLS
higher by <<cf_dh>> rather than lower (bootstrap interval above zero on <<cf_dh_pos>> of <<ce_n>> seeds; McNemar
$p < 0.05$ in CAVR-CE's favour on <<cf_mc_sig>> of <<ce_n>>). The gain over the best fixed path does not carry over.
On these questions the hybrid path is less accurate than the visual path (<<cf_hyb_anls>> against <<cf_vis_anls>>),
so always-visual is the best fixed policy: CAVR-CE exceeds it by only <<cf_dv>> ANLS (interval includes zero on
<<cf_dv_incl>> of <<ce_n>> seeds) and uses <<cf_vs_vis>>\% more GPU compute. A quality-blind mix of the visual and
hybrid paths is then dominated by always-visual, so the query-selection saving of Section~\ref{sec:res-policies} is
not confirmed on this set. GPU-seconds here are measured on the <<a10_gpu>>, about <<ct_slow>>$\times$ slower per
query than the workstation, so they are comparable within the set but not with Table~\ref{tab:main_results}.

\input{tables/confirmation}

\subsection{Runtime budget control}
\label{sec:res-lambda}

The decision rule $\arg\max_p \hat{q}_p - \lambda c_p$ takes the cost weight $\lambda$ at serving time, so one
trained router covers a range of budgets without retraining. Sweeping $\lambda$ from 1 to 0 moves CAVR-CE from
ANLS <<sw_lo_anls>> at <<sw_lo_gpu>> GPU-s to <<sw_hi_anls>> at <<sw_hi_gpu>> GPU-s (mean over seeds;
Figure~\ref{fig:pareto} shows seed 1). Table~\ref{tab:load_test_ops} gives the serving capacity of each operating
point on seed 1.

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/pareto.png}
  \caption{Cost--quality frontier of each router as $\lambda$ is swept (seed 1), with fixed policies and the
  per-query oracle. Costs are measured GPU-seconds per query including router overhead.}
  \label{fig:pareto}
\end{figure}

\subsection{Confidence-based escalation}
\label{sec:res-escalation}

Escalation sends a query to the hybrid path when a calibrated probability that its routed answer is good falls
below $\tau$ (Table~\ref{tab:escalation}, Figure~\ref{fig:escalation}). Probabilities are calibrated on the
validation split by logistic regression; their expected calibration error on test is <<ece_esc_min>>--<<ece_esc_max>>
depending on the signal. On top of CAVR-CE, re-routing on the ColPali score margin at $\tau = 0.5$ raises ANLS
from <<esc_none_anls>> to <<esc_anls>> and GPU cost from <<esc_none_gpu>> to <<esc_gpu>> GPU-s
(<<esc_rate>>\% of queries escalated), still <<esc_save>>\% below always-hybrid.
Setting $\lambda = 0$ reaches the same point without escalation (ANLS <<esc_l0_anls>> at <<esc_l0_gpu>> GPU-s), and
the signal and $\tau$ shown here were highlighted after inspecting test results, so this comparison is exploratory.
Because the re-routing decision is made before generation, it adds no serial step, and its latency cost is small:
p95 latency rises from <<esc_none_p95>>\,s to <<esc_p95>>\,s, against <<hyb_p95>>\,s for always-hybrid. Signals
that judge the generated answer (token log-probabilities) can be computed only after the routed answer exists. Computed from the measured
per-stage times, a serial cascade has p95 <<ser_p95>>\,s; launching the hybrid path alongside the routed path cuts
p95 to <<spe_p95>>\,s but costs <<spe_gpu>> GPU-s per query, <<spe_vs_hyb>>\% more than always-hybrid (assuming two
concurrent workers).

\input{tables/escalation}
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/escalation.png}
  \caption{Escalation on CAVR-CE: answer quality against measured GPU cost as $\tau$ is swept (seed 1).}
  \label{fig:escalation}
\end{figure}

\subsection{Scalability, throughput and energy}
\label{sec:res-scale}

\paragraph{Serving under load.} A Poisson load test with dynamic batching, simulated from the measured per-query
service times and measured batching curves, gives the highest arrival rate one GPU sustains at p95 $\le$ 2\,s and
the number of GPUs needed for a given load (Table~\ref{tab:load_test}, mean over seeds; Figure~\ref{fig:gpus},
seed 1). Serving 100 queries/s at p95 $\le$ 2\,s needs <<g100_always_hybrid_pm>> GPUs for always-hybrid,
<<g100_cavr_ce_valsel_pm>> for CAVR-CE (<<g100_save>>\% fewer), <<g100_always_visual_pm>> for always-visual and
<<g100_oracle_pm>> for the oracle. At this target batches rarely form: the ratio of GPUs needed by CAVR-CE and always-hybrid (<<g100_ratio>>)
tracks their ratio of batch-1 GPU-seconds (<<gs_ratio>>). The saving depends on the target, as the sweep below shows. With HF Transformers
generation (no serving engine such as vLLM~\cite{vllm}), one GPU serves at most <<bq_text>> (text), <<bq_visual>>
(visual) and <<bq_hybrid>> (hybrid) queries/s at the best batch size within memory. The simulation keeps the
evaluation index (<<n_pages>> pages) on every GPU; routing saves generation compute, not index memory, because
CAVR-CE reads ColPali scores for every query.
Across p95 targets from <<sw_min_t>> to <<sw_max_t>>\,s, on all five splits, CAVR-CE needs <<sw_save_min>>--<<sw_save_max>>\%
fewer GPUs than always-hybrid and <<sw_vis_min>>--<<sw_vis_max>>\% more than always-visual (Table~\ref{tab:slo_sweep}).
The simulator charges a batch the mean generation time of its members, scaled by the measured batching curve;
charging the longest member instead, a conservative bound, the saving over always-hybrid is <<bm_long_2>>\% at 2\,s and
<<bm_long_5>>\% at 5\,s (against <<bm_mean_5>>\% with the mean member).

\paragraph{Generation and retrieval together.} The load test keeps the small evaluation index on every GPU. For a
large corpus, a retrieval tier is needed as well. Projected from the workstation's measured MaxSim throughput at
batch 64, scaled linearly in pages and at full utilisation, 100 queries/s over <<jn_pages>> pages needs <<jn_ret>> GPUs
for retrieval (<<jn_hold>> just to hold the index) for every policy that reads ColPali on every query. CAVR-CE then
needs <<jn_ce>> GPUs in total, against <<jn_hyb>> for always-hybrid and <<jn_vis>> for always-visual. The question-only
router could skip ColPali for the queries it sends to the text path, but it sends only <<jn_cavr_text>>\% there and
needs <<jn_cavr>> GPUs. Routing thus saves the same number of generation GPUs at any corpus size, while the index
adds a cost that every visual policy pays.

\paragraph{Load-adaptive $\lambda$.} On a fixed pool of <<ad_pool>> GPUs, the number CAVR-CE needs for 100 queries/s
at p95 $\le$ 2\,s, arrivals alternate between <<ad_base>> queries/s for <<ad_base_s>>\,s and <<ad_burst>> queries/s for
<<ad_burst_s>>\,s (split 1). CAVR-CE at its validation-chosen $\lambda$ answers <<ad_fix_slo>>\% of requests within 2\,s
(ANLS <<ad_fix_anls>>); always-hybrid manages <<ad_hyb_slo>>\%. A controller that raises $\lambda$ when the oldest
queued request has waited longer than a threshold (chosen on validation queries) moves <<ad_up>>\% of requests to a
higher $\lambda$ and reaches <<ad_adp_slo>>\% (ANLS <<ad_adp_anls>>), whereas a fixed $\lambda = 0.2$ reaches
<<ad_02_slo>>\% (ANLS <<ad_02_anls>>) and always-visual <<ad_vis_slo>>\%. In this simulator overload appears mostly as
larger and slower batches rather than long queues, so a queue-driven controller reacts little; adapting $\lambda$
to load needs a better overload signal, such as GPU busy time, which we leave open.

\paragraph{Visual index scaling.} ColPali stores 1,031 vectors per page. At 100k pages the index needs
<<ix100_gb>>\,GiB in bf16, more than our <<gpu_mem>>\,GiB GPU holds before any model weights or activations, and a
million pages (about <<ix1m_gb>>\,GiB) would exceed the memory of most current accelerators
(Table~\ref{tab:index_scaling}, Figure~\ref{fig:index}). Streaming it from pinned host memory costs
<<ix100_stream_ms>>\,ms per query; 2$\times$2 pooling shrinks it to <<ix100_pool_gb>>\,GiB and <<ix100_pool_ms>>\,ms
but lowers recall@1 on the real corpus from <<pool_r1_full>> to <<pool_r1>> (both measured with the same batched
query encoding). Hierarchical token pooling~\cite{tokenpooling}, which clusters each page's patch vectors and keeps
one mean vector per cluster, does much better at the same size: with <<tp_vec>> vectors per page it keeps recall@1 at <<tp_r1>> and recall@5 at
<<tp_r5>>, against <<tp_full_r1>> and <<tp_full_r5>> for the full index and <<tp_2x2_r1>> and <<tp_2x2_r5>> for 2$\times$2
pooling (Table~\ref{tab:token_pooling}). A <<tp_x>>$\times$ smaller index makes a million pages <<tp_gib_1m>>\,GiB,
which <<tp_a10>> A10s could hold, and scoring cost falls in proportion to the vectors per page. Emulated sharding over
<<ix100_shards>> GPUs (shards timed one after another on this GPU) gives
<<ix100_shard_ms>>\,ms. A GPU-resident index answers in <<ix1k_ms>>\,ms at 1k pages and <<ix50k_ms>>\,ms at
50k pages.

\paragraph{Offline cost.} Building a 100k-page index would take about <<build_h>> GPU-hours, extrapolated from a 64-page
benchmark at batch <<build_b>> (<<build_pps>> pages/s); the 1,786-page evaluation index was built at batch
<<full_build_b>>, which exceeded GPU memory, and ran at <<full_build_pps>> pages per GPU-second. docTR OCR of the
<<n_pages>> evaluation pages took <<ocr_s>>\,s and <<ocr_kj>>\,kJ.

\paragraph{Multi-GPU index sharding.} On the <<mg_n_word>>-GPU node (Table~\ref{tab:multi_gpu},
Figure~\ref{fig:multi_gpu}), sharded search returned the same top page as one GPU for all <<mg_ncheck>> check questions
(<<mg_exact>>\% exactly, the rest exact score ties). Latency is set by the number of pages per GPU. A 50k-page index is
searched in <<mg50k_ms1>>\,ms on one GPU, <<mg50k_ms2>>\,ms on two and <<mg50k_ms3>>\,ms on three (speedup
<<mg50k_sp3>>$\times$); from 50k pages up, parallel efficiency is at least <<mg_lat_eff_min>>\% and dispatch and merge
add at most <<mg_ovh_ms>>\,ms over an ideal split. Throughput at batch <<mg_tb>> scales by the same factor, with
efficiency of at least <<mg_thr_eff_min>>\% from <<mg_min_pages>> pages up. With <<mg_weak_k>> pages per GPU, a
<<mg_weak_pages>>-page index of <<mg_weak_gib>>\,GiB, larger than the memory of any one GPU on the node, is searched in
<<mg_weak_ms>>\,ms, as fast as <<mg_weak_k>> pages on one GPU (weak-scaling efficiency <<mg_weak_eff>>\%). Smaller
corpora gain less at batch 1 (three GPUs give <<mg1k_sp3>>$\times$ at 1k pages and <<mg10k_sp3>>$\times$ at 10k pages),
where per-query overheads are a larger share of a short search; their batch-<<mg_tb>> throughput still reaches
<<mg1k_teff3>>\% and <<mg10k_teff3>>\% of linear scaling.

Energy per query for the whole node falls from
<<mg50k_e1>>\,J on one GPU to <<mg50k_e3>>\,J on three at 50k pages, but not because the GPUs work more efficiently: an
unused GPU, which holds a CUDA context in the benchmark process, draws about <<mg_idle_w>>\,W (estimated from the runs; <<mg_idle_lo>>--<<mg_idle_hi>>\,W across
configurations), and without the idle draw the active energy is <<mg50k_ea1>>\,J per query whatever the number of GPUs.
On a node whose GPUs are powered anyway, spreading the shards over all of them therefore lowers the energy per query.
These GPUs are much slower than the workstation's: one <<mg_gpu>> takes <<mg50k_ms1>>\,ms for the 50k-page search that
the <<gpu_short>> completes in <<ix50k_ms>>\,ms in bf16, <<mg_vs_ws>>$\times$ faster. Absolute times therefore do not
transfer between the machines; the scaling with the number of GPUs does. It also shows that the emulated sharding above
omitted little: merging the shard results costs well under a millisecond.

\paragraph{Sharding on data-centre GPUs.} The same benchmark, unchanged, ran on a node with <<a10_n_word>>
<<a10_gpu>> GPUs (Ampere, scored in fp16; Table~\ref{tab:hardware_multigpu}). Sharded search again returned the same
top page as one GPU for all <<mg_ncheck>> check questions (<<a10_exact>>\% exactly, the rest exact score ties). A
50k-page index is searched in <<a10_50k_ms1>>\,ms on one A10 and <<a10_50k_ms2>>\,ms on two (parallel efficiency
<<a10_50k_eff2>>\%); from 50k pages up, dispatch and merge add at most <<a10_ovh_ms>>\,ms, and batch-<<mg_tb>>
throughput efficiency is at least <<a10_thr_eff_min>>\% from <<mg_min_pages>> pages up. A <<a10_weak_pages>>-page index
(<<a10_weak_gib>>\,GiB, more than one A10 holds) is searched in <<a10_weak_ms>>\,ms on two GPUs, as fast as
<<a10_weak_k>> pages on one (weak-scaling efficiency <<a10_weak_eff>>\%), and the largest corpus run on two A10s,
<<a10_maxpages>> pages, took <<a10_max_ms>>\,ms. One A10 is <<a10_vs_p40>>$\times$ faster than one P40 on the 50k-page search and uses
<<a10_e_ratio>>$\times$ less active energy per query (<<a10_50k_ea1>> against <<mg50k_ea1>>\,J), largely because it
scores in fp16 on tensor cores, which the P40 lacks; it is still <<a10_vs_ws>>$\times$ slower than the <<gpu_short>>.
As on the P40 node, active energy per query does not depend on the number of GPUs (<<a10_50k_ea1>> and
<<a10_50k_ea2>>\,J at 50k pages). Near-linear sharding therefore holds across two GPU generations and two scoring
precisions; the parallel efficiency transfers between machines, the absolute speed does not.

\paragraph{Host-memory placement at a million pages.} With the index held in host memory, two <<a10_gpu>> GPUs
searched <<hs_max_words>> pages (<<hs_gib>>\,GiB). Held in GPU memory within the same <<hs_budget_pct>>\% per-GPU budget
(<<a10_budget>>\,GiB), this index would need <<hs_res_gpus>> A10s, or <<hs_res_nodes>> nodes like this one
(Table~\ref{tab:host_stream}, Figure~\ref{fig:host_stream}). Streamed search returned the same top page, up to the
repetition of real pages in the corpus, and the same score as GPU-resident search for all <<hs_ncheck>> check questions
on <<hs_check_k>> pages. Single queries were bound by the PCIe link: their latency stayed within <<hs_over_copy>>\% of
the time needed to copy each GPU's shard, and at <<hs_max_words>> pages it was <<hs_p50_1>>\,s on one GPU and
<<hs_p50_2>>\,s on two. The workstation shows the same bound, streaming <<hs_res_pages>> pages from host memory in
<<ix100_stream_ms>>\,ms against <<hs_100k_p50_1>>\,s on one A10. One A10 read host memory at <<hs_copy1>>\,GB/s and two
together at <<hs_copy2>>\,GB/s, <<hs_share>>\% of twice the single-GPU rate, so a second GPU cut single-query latency by
only <<hs_lat_sp>>$\times$. The cause of this shortfall was not identified: nvidia-smi reports a path through several PCIe
bridges between the two GPUs, and the placement of host memory across the two CPU sockets was not controlled.

Batching removes the PCIe bound, because one pass over the index serves every query in the batch. Scoring rather than
copying sets the pace from about <<hs_cross>> queries per batch, a crossover derived from the batch-1 and
batch-<<hs_b>> times on the assumption that scoring time grows linearly with batch size. At batch <<hs_b>> and
<<hs_max_words>> pages, throughput was <<hs_qps1>> queries/s on one GPU and <<hs_qps2>> on two (<<hs_thr_sp>>$\times$),
or <<hs_gpus_q>>\,GPU-s per query; GPU-resident search would cost about <<hs_res_gpus_q>>\,GPU-s, scaled linearly from
its <<hs_res_pages>>-page run. Where both placements were measured on the same corpus (<<hs_res_pages>> pages, two GPUs),
streamed search reached <<hs_vs_res>>\% of the GPU-resident throughput at batch <<hs_b>> with a GPU energy of
<<hs_e_stream>> against <<hs_e_res>>\,J per query; the host CPU and memory energy that streaming adds was not measured.
At batch 1 the comparison reverses: streamed search used <<hs_b1_gpus_s>> against <<hs_b1_gpus_r>>\,GPU-s and
<<hs_b1_e_s>> against <<hs_b1_e_r>>\,J per query. Host-memory placement therefore trades latency for GPU count at about
the same throughput per GPU, but only for workloads that fill large batches. At <<hs_p50_2>>\,s per single query and
about <<hs_b_lat2>>\,s per batch of <<hs_b>>, it misses the 2\,s p95 target of the load test and suits batch or offline
search. Interactive search over <<hs_max_words>> pages needs GPU-resident sharding, projected at about
<<hs_res_ms>>\,ms on <<hs_res_gpus>> A10s (linear fit of the resident runs against pages per GPU, without the cost of
merging results across nodes), or a compressed index.

\paragraph{Router overhead.} The CAVR router (mpnet embedding and MLP) takes <<cavr_router_ms>>\,ms per query at
batch 1 on the GPU (<<cavr_router_cpu_ms>>\,ms on the CPU) and reaches <<cavr_router_qps>> queries/s at large
batch. CAVR-CE takes <<ce_ms>>\,ms and <<ce_j>>\,J per query at batch 1, and its cross-encoder sustains
<<ce_qps>> queries/s. Neither is a bottleneck next to generation (Table~\ref{tab:router_throughput}).

\input{tables/load_test}
\input{tables/load_test_operating_points}
\input{tables/slo_sweep}
\input{tables/index_scaling}
\input{tables/token_pooling}
\input{tables/multi_gpu_scaling}
\input{tables/host_stream}
\input{tables/router_throughput}
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/gpus_vs_quality.png}
  \caption{GPUs needed to serve 100 queries/s at p95 $\le$ 2\,s against answer quality (seed 1).}
  \label{fig:gpus}
\end{figure}
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/index_scaling.png}
  \caption{ColPali MaxSim query latency against index size for a GPU-resident, host-streamed and pooled index.}
  \label{fig:index}
\end{figure}
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/multi_gpu_scaling.pdf}
  \caption{Measured multi-GPU sharding of the ColPali index on <<mg_n_word>> <<mg_gpu>> GPUs (a, b) and <<a10_n_word>>
  <<a10_gpu>> GPUs (c, d): p50 latency of single queries (a, c) and throughput at batch <<mg_tb>> (b, d), against the number of GPUs, for each corpus size. Dashed lines show
  linear scaling from the fewest GPUs that hold the corpus; the dotted line joins runs with <<mg_weak_k>> pages per GPU
  (weak scaling).}
  \label{fig:multi_gpu}
\end{figure*}
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/host_stream.pdf}
  \caption{ColPali index held in host memory and streamed to one and <<a10_n_word>> <<a10_gpu>> GPUs: (a) p50 latency of
  single queries against index size, with the time to copy each GPU's shard over PCIe (black dashes, within
  <<hs_over_copy>>\% of the latency); (b) throughput at batch
  <<hs_b>>, streamed (filled markers) and GPU-resident where the index fits in GPU memory (open markers).}
  \label{fig:host_stream}
\end{figure*}

\subsection{Limitations}
\label{sec:res-limits}

All measurements except index sharding and the host-memory index use one consumer GPU (<<gpu>>) under Windows; caps on visual tokens and OCR length keep memory
below the point where the driver spills to system RAM. Generation uses HF Transformers rather than a serving
engine such as vLLM~\cite{vllm}, so absolute capacity is pessimistic; the comparison between policies is the main
result. The 2\,s latency target is our choice; the saving is smaller at looser targets.

The load test is a simulation driven by measured service times, not live traffic, and escalation latencies are
computed from measured per-stage times rather than measured on concurrent workers. The simulator assumes an ideal
global queue and unlimited CPU, charges a batch the mean generation time of its members (the longest-member bound is reported), and keeps the
<<n_pages>>-page evaluation index on every GPU, so the retrieval tier for a large corpus is projected rather than simulated. Costs
are measured without prefix or encoder caching, which would favour paths whose page image repeats across
questions. Indexes of 10k--<<hs_max_k>> pages
repeat the real page embeddings, which is valid for latency and memory but not for retrieval quality at that
scale; the recall of pooled indexes is measured on the <<n_pages>> real pages only. Multi-GPU sharding was measured for retrieval only, on two server nodes with at most <<mg_n_word>> GPUs
each (<<mg_gpu>>, <<a10_gpu>>), whose absolute times do not transfer to the workstation, and the load test does not include sharded retrieval. The host-memory index was measured in one process on one node, with <<hs_nlat>> latency queries and <<hs_reps>>
repetitions per batch (latency varied by at most <<hs_cv>>\% within the run) and with host memory placed across CPU
sockets without control, and GPU-resident search at <<hs_max_words>> pages is
projected from smaller resident runs, not measured.

Energy covers the GPU only (NVML); host CPU and memory energy is not counted. Answers are short (at most
<<max_new_tokens>> tokens), so cost is dominated by processing the input; with longer answers the relative cost of an
extra context shrinks. The savings are relative to this path set and one shared generator, and they depend on the
retrieval depth and input caps of each path (the depth controls cover <<ct_n>> questions and one more depth per
modality); routing to a smaller generator was not tested. The router handles text
queries over visual documents; queries that contain an image are out of scope. The router design was developed on the same five splits that we
report on, and the feasibility analyses that motivated it looked at their test portions; no parameter of the final
pipeline is selected on test data. On a held-out confirmation set, the saving over always-hybrid held but the
gain over always-visual did not (Section~\ref{sec:res-confirm}): which fixed path is best, and so what routing
gains over it, depends on the question distribution. The confirmation questions come from the training splits of
the same two datasets and were run on a different GPU, so they test new questions and pages, not new datasets. Splits group
questions by page, so pages of one DocVQA document can fall into different splits. Finally, the <<all_fail>>\% of
queries that every path fails bound what routing can achieve on this benchmark.
"""

RESPONSE_MD = r"""# Response to the editor and reviewers

**Manuscript:** Cost-Aware Visual Router (CAVR): Dynamic Query Routing for Cost-Efficient Multimodal RAG
**Manuscript ID:** [AUTHOR ACTION: add]

*Draft generated by `make_paper_text.py` from `results/*.json`; every number below comes from the results files.
Reviewer comments are paraphrased; replace each with the reviewer's verbatim text before submission. Items marked
[AUTHOR ACTION] are not yet done and must be completed or removed before this letter is sent. Table and figure
names refer to the labels in the revised manuscript (for example "Table main_results" = \label{tab:main_results}).*

Dear Editor,

We thank you and the three reviewers for a careful and demanding review. The reviewers found inconsistent
numbers, a cost model that contradicted our own measurements, and released code that did not match the paper.
They were right on each point. We rebuilt the experimental pipeline from scratch, re-ran every experiment, and
rewrote the paper around what the new measurements support. Several claims of the submitted version are
withdrawn; we list them first so that nothing is hidden behind the new results.

## Summary of changes

1. **One reproducible pipeline.** Every path (text, visual, hybrid) is run on every query once, with real BM25
   and ColPali retrieval and a local generator (<<generator>>). Outputs are cached, and every table and figure is
   generated from the results files by released code with fixed seeds. [AUTHOR ACTION: tag the release, archive
   it with a DOI and cite it.]
2. **Outcome-based labels.** Route labels now come from measured answer quality and cost, not from a keyword rule.
3. **End-to-end answer quality.** Every path is run on all <<n_q>> DocVQA and InfographicVQA validation questions.
   Routing policies are compared on ANLS and exact match of the answers they produce, on the held-out test pages of
   <<n_seeds>> page-grouped splits (<<n_test_min>>--<<n_test_max>> questions each).
4. **Measured cost.** GPU-seconds, NVML energy and wall-clock latency are measured per query on one <<gpu>>
   (Table hardware). Dollar estimates from API prices and A100 rates are removed.
5. **A content-aware router (CAVR-CE).** It reads retrieved content before generation and uses <<ce_save>>% less
   GPU compute than always-hybrid, at a mean ANLS cost of <<ce_dh_mean>>.
6. **Held-out confirmation and controls.** The frozen routers were applied to <<cf_n>> new questions from the training
   splits (Section "Held-out confirmation"): the saving over always-hybrid holds (<<cf_save>>%, at <<cf_dh>> higher
   ANLS), but always-visual comes within <<cf_dv>> ANLS of the router at <<cf_vis_less>>% less GPU, and the paper says
   so. Retrieval-depth and gold-page controls (Table controls) separate input modality from retrieval depth and errors.
7. **Supercomputing case.** New experiments on serving under load, GPUs needed at a latency target, visual-index
   scaling to 100k pages, router throughput and energy.
8. **Related work and verified references.** A new related-work section; every reference was checked against its
   DOI, arXiv, OpenReview or publisher record.
9. **Narrowed claims and a full limitations section.**

## Withdrawn or corrected claims

| Submitted claim | Status | What the revised paper reports |
|---|---|---|
| 86.2% routing accuracy (231/268) | Withdrawn | The keyword labels behind it agree with outcome labels at Cohen's κ = <<kappa>>, so routing accuracy is no longer reported. The submitted confusion matrix gave 229/268 = 85.4%, which was also misreported. |
| $0.00019 per query; 21.5% cost saving | Withdrawn | They came from word count × GPT-3.5 prices and router latency × an A100 rate, with no retrieval or generation run. Costs are now measured. |
| Text path is ~100× cheaper than visual | Withdrawn | With a shared generator, text costs <<text_gpu>> GPU-s and visual <<visual_gpu>> GPU-s per query. |
| 92.6% saving on the natural distribution | Withdrawn | Not measured in the submitted version. All results now use the full validation sets at their natural question mix. |
| Out-of-distribution generalisation on InfoVQA | Withdrawn | InfoVQA was also used for training; both datasets are now in-distribution and split by page. [AUTHOR ACTION: confirm; or add a held-out benchmark.] |
| "Production-ready" system | Reworded | "Deployable system design", supported by the serving experiments. |
| Training details (AdamW, loss weights α/β/γ, temperature 0.7) | Corrected | The released code trained a plain classifier. The revised routers and their hyperparameters are documented and released. |
| Results without released code (temperature and α sweeps, bootstrap intervals, imbalanced baseline) | Removed | Replaced by experiments generated by released code. |

## Editor

**E.1 — Justify the need for supercomputing, real-time and parallel or distributed computing throughout.**
The revised paper frames CAVR as a scheduler for GPU resources in multimodal RAG serving. Late-interaction visual
retrieval stores 1,031 vectors per page, so index memory and scoring time grow with the corpus: at 100k pages the
ColPali index needs <<ix100_gb>> GiB, more than our <<gpu_mem>> GiB GPU holds, and a million pages (about <<ix1m_gb>>
GiB) would exceed the memory of most current accelerators. Such an index must be streamed from host memory (<<ix100_stream_ms>> ms per
query on our hardware), pooled (recall@1 drops from <<pool_r1_full>> to <<pool_r1>>) or sharded. We measured sharding on a node
with <<mg_n_word>> <<mg_gpu>> GPUs: sharded search returns the same top page as one GPU, a 50k-page index is searched in
<<mg50k_ms1>>, <<mg50k_ms2>> and <<mg50k_ms3>> ms on one, two and three GPUs (parallel efficiency <<mg50k_eff3>>%), and a
<<mg_weak_pages>>-page index (<<mg_weak_gib>> GiB, more than one GPU holds) is searched in <<mg_weak_ms>> ms, as fast as
<<mg_weak_k>> pages on one GPU. On a second node with <<a10_n_word>> <<a10_gpu>> GPUs, the same benchmark searches 50k pages in <<a10_50k_ms1>> and <<a10_50k_ms2>> ms on one and two GPUs (efficiency <<a10_50k_eff2>>%) and a <<a10_weak_pages>>-page index (<<a10_weak_gib>> GiB) in <<a10_weak_ms>> ms. We also held an index of <<hs_max_words>> pages (<<hs_gib>> GiB) in host memory on that node and streamed it to the GPUs for every query batch: two A10s search it at <<hs_qps2>> queries/s at batch <<hs_b>>, close to the GPU-seconds per query of GPU-resident search (<<hs_vs_res>>% of its throughput where both were measured, at <<hs_res_pages>> pages), whereas a GPU-resident index would need <<hs_res_gpus>> A10s (projected). A single query takes <<hs_p50_2>> s, bound by PCIe, so this placement suits batch rather than interactive search. Routing saves generation compute rather than index memory. In a simulated Poisson load test with dynamic batching, driven by measured service
times, serving 100 queries/s at p95 ≤ 2 s needs <<g100_always_hybrid>> GPUs for always-hybrid and
<<g100_cavr_ce_valsel>> for CAVR-CE (<<g100_save>>% fewer; means over <<n_seeds>> seeds). We added these
experiments, a hardware table for each machine, energy per query, router throughput up to batch 1,024, and serial versus
speculative-parallel escalation.
*Changes:* Abstract; Introduction (new subsection on why routing is a supercomputing problem; research question
and contribution); Related work (new section: visual-document retrieval, adaptive retrieval and model routing,
efficient late-interaction and vector search, inference serving; every reference checked against its arXiv, DOI
or publisher record); Method (Figure method: the pipeline as built and its deployment); Results, subsection
"Scalability, throughput and
energy" (Tables hardware, hardware_multigpu, load_test, load_test_operating_points, index_scaling,
multi_gpu_scaling, host_stream, router_throughput; Figures gpus_vs_quality, index_scaling, multi_gpu_scaling,
host_stream); Setup (multi-GPU sharding and index in host memory); Conclusion.
Routing and generation were measured on one <<gpu>>, capacity projections come from a simulation driven by
measured service times, and multi-GPU sharding and the host-memory index were measured on separate nodes for retrieval only,
with GPU-resident search at <<hs_max_words>> pages projected rather than measured; the paper states all of this.

**E.2 — Report results accurately, rewrite overstated conclusions and explain limitations fully.**
Every number now comes from one cached run, and tables and figures are generated from the results files. The
withdrawn claims are listed above. A new Limitations subsection covers the single GPU, simulation versus live
traffic, computed rather than measured escalation latencies, synthetic index scaling, sharding measured on separate server nodes for retrieval only,
GPU-only energy, text-only queries, and the <<all_fail>>% of queries that no path answers.

## Reviewer 1

**R1.1 — Perplexity-based escalation forces a generate-then-escalate chain, so escalation can be slower than going
visual from the start.**
We agree. The revised paper quantifies this with latencies computed from measured per-stage times; two concurrent
workers were not run. On CAVR-CE, signals available before generation (router confidence, ColPali score margin)
re-route without a serial step: re-routing on the ColPali margin at τ = 0.5 raises p95 latency from
<<esc_none_p95>> s to <<esc_p95>> s, against <<hyb_p95>> s for always-hybrid. Answer-level signals (token
log-probabilities) require the first answer, and a serial cascade has p95 <<ser_p95>> s. A speculative design that
launches the hybrid path alongside the routed path cuts p95 to <<spe_p95>> s but costs <<spe_gpu>> GPU-s per query,
<<spe_vs_hyb>>% more than always-hybrid. The paper gives both designs and their latency accounting, and recommends
pre-generation escalation.
*Changes:* Method (escalation designs and latency accounting); Results, subsection "Confidence-based
escalation" (Table escalation, Figure escalation).

**R1.2 — No runtime budget control without retraining.**
The decision rule is now $\arg\max_p \hat{q}_p - \lambda c_p$, where the router predicts per-path quality and the
cost weight λ is set at serving time. One trained router covers a range of budgets: sweeping λ from 1 to 0 moves
CAVR-CE from ANLS <<sw_lo_anls>> at <<sw_lo_gpu>> GPU-s to <<sw_hi_anls>> at <<sw_hi_gpu>> GPU-s (mean over seeds).
*Changes:* Method (decision rule); Results, subsection "Runtime budget control" (Figure pareto, Table
load_test_operating_points). [AUTHOR ACTION: decide whether to add a load-adaptive λ experiment or leave it as
future work.]

**R1.3 — The query encoder is text-only; queries containing images are not supported.**
We narrowed the scope to text queries over visual documents and say so in the title, abstract and Limitations.
[AUTHOR ACTION: confirm narrowing the scope, or add a proof of concept with an image embedding.]

## Reviewer 2

**R2.1 — Estimated and measured costs are inconsistent; which gives the 21.5% saving?**
Neither is used any more. The 21.5% came from list-price estimates (Table 2 of the submission), which contradicted
the measured costs in Table 8, and neither involved real retrieval or generation. Costs are now measured per query
on one <<gpu>>: over all queries, always-text costs <<text_gpu>>, always-visual <<visual_gpu>> and always-hybrid
<<hybrid_gpu>> GPU-s (<<text_j>>, <<visual_j>> and <<hybrid_j>> J). All savings are computed from these
measurements, router overhead included.
*Changes:* Setup (cost model, Table hardware); Results (Tables main_results, router_ce_tests).

**R2.2 — Constrained cost minimisation is not equivalent to maximising the quality/cost ratio.**
We agree. The ratio objective is replaced by a Lagrangian penalty: the router picks argmax over paths p of
q̂_p − λ·c_p. The Method section now derives this as the Lagrangian relaxation of the budget-constrained problem,
states when it is exact (randomised policies, strong duality), and gives an example where the ratio rule and the
constrained rule disagree. [AUTHOR ACTION: check the derivation.]
*Changes:* Method, problem formulation.

**R2.3 — How are labels produced, and how many samples are in each class?**
In the submitted version, labels came from a keyword rule over InfographicVQA questions; we state this plainly. They
agree with outcome-based labels at Cohen's κ = <<kappa>>. Labels are now produced by running every path on every
query, scoring each answer with ANLS, and taking the path that maximises ANLS − λ·cost (ties to the cheaper path).
Class counts per split (seed-1 split) and a threshold-based variant are in Table labels.
*Changes:* Method (labels); Results, subsection "Outcome-based route labels" (Table labels).

**R2.4 — Ablate escalation; ECE 0.06 vs 0.042; justify τ = 0.6.**
The escalation ablation covers no escalation, each signal alone, all pre-generation signals and all signals, for
both routers, with τ swept from 0.3 to 0.9 (Table escalation, mean over seeds; Figure escalation). We report the
whole τ curve and no single tuned τ. The conflicting ECE values are replaced by measurements on test with 10 bins:
the CAVR confidence head has ECE <<ece_head>>, and the escalation probabilities, recalibrated on the validation
split by logistic regression, have ECE <<ece_esc_min>>–<<ece_esc_max>> depending on the signal (CAVR-CE base;
Table escalation). [AUTHOR ACTION: add a reliability diagram if the reviewer requires one.]

**R2 minor 1 — Table 5 columns reversed.** The table is removed; targets are no longer reported.
**R2 minor 2 — "79" vs "+79.9 pp".** The sentence is removed with the withdrawn results.
**R2 minor 3 — "Two orders of magnitude" is about 50×.** Router overhead is now reported as measured:
<<cavr_router_ms>> ms (CAVR) and <<ce_ms>> ms (CAVR-CE) per query. [AUTHOR ACTION: fix the phrase wherever it
remains in the manuscript.]
**R2 minor 4 — "volto brio" and other garbled text.** [AUTHOR ACTION: full copy-edit pass.]
**R2 minor 5 — "Production-ready" is too strong.** Reworded to "deployable system design".
[AUTHOR ACTION: Reviewer 2's attachment was not available to us; address any further comments it contains.]

## Reviewer 3

**R3.1 — Answer correctness is never checked, and there is no comparison with escalation switched off.**
This was the central gap and is now the core experiment: every path is run on every query, answers are scored with
ANLS and exact match, and every policy is evaluated on the answers it produces (Table main_results). Escalation off
is the "none" row of Table escalation. New controls on <<ct_n>> questions (Table controls) separate the input
modality from retrieval depth and from retrieval errors: at equal depth a page image beats OCR text by <<ct_v_t1>>
ANLS, and giving the generator the gold page raises the visual path from <<ct_vis>> to <<ct_gvis>>.

**R3.2 — 229/268 is not 231/268; the rule-based baseline, path counts, training curve and calibration differ across
the paper.**
All of these came from results that could not be reproduced and have been removed. Every number in the revision
comes from one cached run, and tables and figures are generated from the results files. Policy results are mean ±
standard deviation over <<n_seeds>> seeds with McNemar and paired bootstrap tests; figures that show a single split
are labelled "seed 1".

**R3.3 — Visual is 33× more expensive, not 100×, and Table 8 shows visual cheaper than text.**
The reviewer is right that the cost claims were inconsistent. With measured costs over all queries, text and visual
cost <<text_gpu>> and <<visual_gpu>> GPU-s per query, and the paper no longer claims a large text–visual cost gap.

**R3.4 — The objective equivalence is wrong; labels and utility targets are unclear.**
See R2.2 (Lagrangian formulation) and R2.3 (outcome-based labels).

**R3.5 — The repository does not match the paper: InfoVQA only, keyword labels, a classification-only model, no
real visual retrieval.**
The reviewer's audit was correct on every point. The revision uses a new pipeline: DocVQA and InfographicVQA, real
BM25 and ColPali retrieval, generation for every path, outcome-based labels, multi-task routers, and released
scripts for every table and figure. [AUTHOR ACTION: release tag and DOI.]

**R3.6 — Wrong arXiv IDs for [9] and [29]; incorrect and duplicated references.**
[AUTHOR ACTION: audit every reference against its DOI or arXiv page; fix [9] and [29], remove duplicates, correct
authors and venues, and remove any entry that cannot be found. Rebuild Table 1 afterwards, because RouteLLM and
FrugalGPT do model cost.]

## Headline result of the revised paper

With λ chosen on validation data, CAVR-CE uses <<cavr_ce_valsel_gpu>> instead of <<always_hybrid_gpu>> GPU-s per query
(<<ce_save>>% less than always-hybrid). Its ANLS is <<cavr_ce_valsel_anls>> against <<always_hybrid_anls>>: lower on
<<ce_lower_n>> of <<ce_n>> seeds, with the 95% interval excluding zero on <<ce_ci_excl>> of <<ce_n>> splits. On exact match the
difference is not significant on <<ce_nonsig_h>> of <<ce_n>> seeds. CAVR-CE beats always-visual by
<<ce_dv_min>> to <<ce_dv_max>> ANLS for <<ce_vs_vis_cost>>% more GPU. At 100 queries/s and p95 ≤ 2 s it needs
<<g100_cavr_ce_valsel>> GPUs instead of <<g100_always_hybrid>> (simulated). The per-query oracle (ANLS <<oracle_anls>>
at <<oracle_gpu>> GPU-s) shows that predicting per-query success remains the open problem, and the paper says so.
On <<cf_n>> held-out questions, the frozen router keeps a <<cf_save>>% saving over always-hybrid, but there
always-visual comes within <<cf_dv>> ANLS of it at <<cf_vis_less>>% less GPU; the paper reports this as a limit on
what routing gains over the best single path.

Sincerely,
[AUTHOR ACTION: authors]
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    V = values()
    (OUT / "results_section.tex").write_text(fill(RESULTS_TEX, V), encoding="utf-8")
    md = fill(RESPONSE_MD, V).replace(" $\\pm$ ", " ± ")
    (OUT / "response_to_reviewers.md").write_text(md, encoding="utf-8")
    write_sections(V, OUT, fill)
    print(f"wrote results_section, abstract, introduction, related, method, setup, conclusion (.tex), "
          f"and response_to_reviewers.md to {OUT}")


if __name__ == "__main__":
    main()
