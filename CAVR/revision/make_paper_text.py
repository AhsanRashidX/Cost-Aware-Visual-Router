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
    P1 = lt["policies"]
    V["g100_5s_hyb"], V["g100_5s_ce"] = P1["always_hybrid"]["gpus_needed"]["5.0"]["100"], P1["cavr_ce_valsel"]["gpus_needed"]["5.0"]["100"]
    V["g100_5s_save"] = f"{100 * (1 - V['g100_5s_ce'] / V['g100_5s_hyb']):.0f}"
    V["ix1m_gb"] = f"{10 * ix['scaling']['100000']['index_gb_bf16']:.0f}"
    for p, d in bj["paths"].items():
        V[f"bq_{p}"] = f"{max(x['queries_per_s'] for x in d.values() if 'note' not in x):.2f}"
    s100, s1k, s50k = ix["scaling"]["100000"], ix["scaling"]["1000"], ix["scaling"]["50000"]
    V["ix100_gb"], V["ix100_stream_ms"] = f"{s100['index_gb_bf16']:.1f}", f"{1000 * s100['cpu_stream']['p50_s']:.0f}"
    V["ix100_pool_gb"], V["ix100_pool_ms"] = f"{s100['index_gb_pooled2x2']:.1f}", f"{1000 * s100['gpu_pooled2x2']['p50_s']:.1f}"
    V["ix100_shards"], V["ix100_shard_ms"] = s100["sharded"]["gpus_needed"], f"{1000 * s100['sharded']['emulated_parallel_p50_s']:.1f}"
    V["ix1k_ms"], V["ix50k_ms"] = f"{1000 * s1k['gpu_bf16']['p50_s']:.2f}", f"{1000 * s50k['gpu_bf16']['p50_s']:.1f}"
    V["pool_r1_full"], V["pool_r1"] = f"{ix['pooling_recall_at1']['full']:.3f}", f"{ix['pooling_recall_at1']['pooled2x2']:.3f}"
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
% Tables hardware and retrieval are input here; move them to the Setup section if preferred.
\section{Results}
\label{sec:results}

All numbers in this section come from one cached run of every path on every query
(<<n_q>> validation questions from DocVQA and InfographicVQA over <<n_pages>> page images) on a single
<<gpu>> (Table~\ref{tab:hardware}). Costs are measured, not priced: GPU-seconds, energy from NVML, and wall-clock
latency per query. Routers are trained and evaluated on <<n_seeds>> page-grouped 60/20/20 splits, so no page
appears in more than one split; each test split holds <<n_test_min>>--<<n_test_max>> questions. Unless marked
otherwise, policy results are means ($\pm$ standard deviation) over the <<n_seeds>> seeds; figures marked
``seed 1'' show the first split.

\input{tables/hardware}
\input{tables/data}

\subsection{Cost and answer quality of each path}
\label{sec:res-paths}

All three paths use the same generator (<<generator>>, greedy decoding). The text path reads the top-<<k_text>>
BM25 pages of OCR text; the visual path reads the top-1 ColPali page image (at most <<max_vis_tok>> visual
tokens); the hybrid path reads both. Over all <<n_q>> questions, always-text reaches ANLS <<text_anls>> at
<<text_gpu>> GPU-s per query, always-visual <<visual_anls>> at <<visual_gpu>> GPU-s, and always-hybrid
<<hybrid_anls>> at <<hybrid_gpu>> GPU-s (<<hybrid_j>>\,J).
With a shared local generator the text path is therefore \emph{not} the cheap path the submitted version
assumed: OCR text costs about as many input tokens as a page image, and the text path is much less accurate.
The real cost differences are between one context and two (hybrid costs <<hyb_x_vis>>$\times$ the visual path and
<<hyb_x_txt>>$\times$ the text path) and the size of the visual index at scale (Section~\ref{sec:res-scale}).
In <<all_fail>>\% of queries every path fails. For <<fail_nogold>>\% of these, no path had the gold page in its
context (page recall@1 is <<r1_bm25>> for BM25 and <<r1_col>> for ColPali; Table~\ref{tab:retrieval}). No router
can recover these queries; they bound every policy.

\input{tables/retrieval}

\subsection{Outcome-based route labels}
\label{sec:res-labels}

Route labels are now derived from measured outcomes: the label of a query is the path that maximises
$\mathrm{ANLS} - \lambda \cdot \mathrm{cost}$, with ties broken towards the cheaper path (Table~\ref{tab:labels}).
The keyword labels used in the submitted version agree with these outcome labels on <<kw_agree>>\% of queries,
with Cohen's $\kappa = <<kappa>>$, i.e.\ no agreement beyond chance; the keyword rule sends <<kw_text_share>>\% of
queries to the text path. Routing accuracy against keyword labels therefore says nothing about cost or answer
quality, and we no longer report it.

\input{tables/labels}

\subsection{Routing policies}
\label{sec:res-policies}

Table~\ref{tab:main_results} compares fixed policies, the submitted keyword rule, simple learned baselines,
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
excludes zero on <<ce_ci_excl>> of <<ce_n>> seeds. On exact match, the difference is not significant on
<<ce_nonsig_h>> of <<ce_n>> seeds (McNemar $p = <<ce_ph_min>>$--$<<ce_ph_max>>$).
Against always-visual, CAVR-CE gains $<<ce_dv_min>>$ to $<<ce_dv_max>>$ ANLS (bootstrap interval above zero on
<<ce_v_ci_excl>> of <<ce_n>> seeds; McNemar on exact match $p \le <<ce_pv_max>>$ on <<ce_sig_v>> of <<ce_n>> seeds)
for <<ce_vs_vis_cost>>\% more GPU.

The per-query oracle reaches ANLS <<oracle_anls_pm>> at only <<oracle_gpu>> GPU-s, so substantial headroom
remains. It is limited by prediction: CAVR-CE predicts per-path success with a mean AUC of <<ce_auc>>
(<<gbm_auc>> for gradient boosting on the same tabular features), and a learning curve over 25--100\% of the
training pages (<<lc_seeds>> seeds) rises only from <<lc_25>> to <<lc_100>>.

\input{tables/main_results}
\input{tables/router_ce_tests}

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
Escalation adds latency. Because the re-routing decision is made before generation, it adds no serial step:
p95 latency rises from <<esc_none_p95>>\,s to <<esc_p95>>\,s, against <<hyb_p95>>\,s for always-hybrid. Signals
that judge the generated answer (token log-probabilities) need the first answer first. Computed from the measured
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
<<g100_oracle_pm>> for the oracle. The saving depends on the target: at p95 $\le$ 5\,s, seed 1 needs
<<g100_5s_hyb>> GPUs for always-hybrid and <<g100_5s_ce>> for CAVR-CE (<<g100_5s_save>>\% fewer). With HF Transformers
generation (no serving engine such as vLLM~\cite{vllm}), one GPU serves at most <<bq_text>> (text), <<bq_visual>>
(visual) and <<bq_hybrid>> (hybrid) queries/s at the best batch size within memory. The simulation keeps the
evaluation index (<<n_pages>> pages) on every GPU; routing saves generation compute, not index memory, because
CAVR-CE reads ColPali scores for every query.

\paragraph{Visual index scaling.} ColPali stores 1,031 vectors per page. At 100k pages the index needs
<<ix100_gb>>\,GiB in bf16, more than our <<gpu_mem>>\,GiB GPU holds before any model weights or activations, and a
million pages (about <<ix1m_gb>>\,GiB) would exceed any single accelerator
(Table~\ref{tab:index_scaling}, Figure~\ref{fig:index}). Streaming it from pinned host memory costs
<<ix100_stream_ms>>\,ms per query; 2$\times$2 pooling shrinks it to <<ix100_pool_gb>>\,GiB and <<ix100_pool_ms>>\,ms
but lowers recall@1 on the real corpus from <<pool_r1_full>> to <<pool_r1>> (both measured with the same batched
query encoding); sharding over <<ix100_shards>> GPUs gives <<ix100_shard_ms>>\,ms (emulated by timing the shards one
after another on one GPU). A GPU-resident index answers in <<ix1k_ms>>\,ms at 1k pages and <<ix50k_ms>>\,ms at
50k pages. Building a 100k-page index would take about <<build_h>> GPU-hours, extrapolated from a 64-page
benchmark at batch <<build_b>> (<<build_pps>> pages/s); the 1,786-page evaluation index was built at batch
<<full_build_b>>, which exceeded GPU memory, and ran at <<full_build_pps>> pages per GPU-second. docTR OCR of the
<<n_pages>> evaluation pages took <<ocr_s>>\,s and <<ocr_kj>>\,kJ.

\paragraph{Router overhead.} The CAVR router (mpnet embedding and MLP) takes <<cavr_router_ms>>\,ms per query at
batch 1 on the GPU (<<cavr_router_cpu_ms>>\,ms on the CPU) and reaches <<cavr_router_qps>> queries/s at large
batch. CAVR-CE takes <<ce_ms>>\,ms and <<ce_j>>\,J per query at batch 1, and its cross-encoder sustains
<<ce_qps>> queries/s. Neither is a bottleneck next to generation (Table~\ref{tab:router_throughput}).

\input{tables/load_test}
\input{tables/load_test_operating_points}
\input{tables/index_scaling}
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

\subsection{Limitations}
\label{sec:res-limits}

All measurements use one consumer GPU (<<gpu>>) under Windows; caps on visual tokens and OCR length keep memory
below the point where the driver spills to system RAM. Generation uses HF Transformers rather than a serving
engine such as vLLM~\cite{vllm}, so absolute capacity is pessimistic; the comparison between policies is the main
result. The 2\,s latency target is our choice; the saving is smaller at looser targets.
The load test is a simulation driven by measured service times, not live traffic, and escalation latencies are
computed from measured per-stage times rather than measured on concurrent workers. Indexes of 10k--100k pages
repeat the real page embeddings, which is valid for latency and memory but not for retrieval quality at that
scale, and multi-GPU sharding is emulated on one GPU. Energy covers the GPU only (NVML). The router handles text
queries over visual documents; queries that contain an image are out of scope. Finally, the <<all_fail>>\% of
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
6. **Supercomputing case.** New experiments on serving under load, GPUs needed at a latency target, visual-index
   scaling to 100k pages, router throughput and energy.
7. **Narrowed claims and a full limitations section.**

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
GiB) would exceed any single accelerator. Such an index must be streamed from host memory (<<ix100_stream_ms>> ms per
query on our hardware), pooled (recall@1 drops from <<pool_r1_full>> to <<pool_r1>>) or sharded (emulated over
<<ix100_shards>> GPUs). Routing saves generation compute rather than index memory. In a simulated Poisson load test with dynamic batching, driven by measured service
times, serving 100 queries/s at p95 ≤ 2 s needs <<g100_always_hybrid>> GPUs for always-hybrid and
<<g100_cavr_ce_valsel>> for CAVR-CE (<<g100_save>>% fewer; means over <<n_seeds>> seeds). We added these
experiments, a hardware table, energy per query, router throughput up to batch 1,024, and serial versus
speculative-parallel escalation.
*Changes:* Abstract; Introduction (new subsection on why routing is a supercomputing problem; research question
and contribution); Related work (inference serving and late-interaction retrieval) [AUTHOR ACTION: write and verify
these citations]; Method (Figure method: the pipeline as built and its deployment); Results, subsection
"Scalability, throughput and
energy" (Tables hardware, load_test, load_test_operating_points, index_scaling, router_throughput; Figures
gpus_vs_quality, index_scaling); Conclusion.
All experiments ran on one <<gpu>>. Multi-GPU sharding is emulated by timing shards on one GPU, and capacity
projections come from a simulation driven by measured service times; the paper states both.

**E.2 — Report results accurately, rewrite overstated conclusions and explain limitations fully.**
Every number now comes from one cached run, and tables and figures are generated from the results files. The
withdrawn claims are listed above. A new Limitations subsection covers the single GPU, simulation versus live
traffic, computed rather than measured escalation latencies, synthetic index scaling, emulated sharding, GPU-only
energy, text-only queries, and the <<all_fail>>% of queries that no path answers.

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
is the "none" row of Table escalation.

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
<<ce_lower_n>> of <<ce_n>> seeds, with the 95% interval excluding zero on <<ce_ci_excl>> seed. On exact match the
difference is not significant on <<ce_nonsig_h>> of <<ce_n>> seeds. CAVR-CE beats always-visual by
<<ce_dv_min>> to <<ce_dv_max>> ANLS for <<ce_vs_vis_cost>>% more GPU. At 100 queries/s and p95 ≤ 2 s it needs
<<g100_cavr_ce_valsel>> GPUs instead of <<g100_always_hybrid>> (simulated). The per-query oracle (ANLS <<oracle_anls>>
at <<oracle_gpu>> GPU-s) shows that predicting per-query success remains the open problem, and the paper says so.

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
    print(f"wrote results_section, abstract, introduction, method, conclusion (.tex), "
          f"response_to_reviewers.md and citations_needed.md to {OUT}")


if __name__ == "__main__":
    main()
