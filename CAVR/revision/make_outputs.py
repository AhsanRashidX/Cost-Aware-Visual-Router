"""Step 9: regenerate every table and figure from results/*.json (nothing typed by hand).

Writes LaTeX (booktabs) tables to paper_out/tables/ and PNG figures to paper_out/figures/.
Each piece is skipped, with a note, if its source file doesn't exist yet.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import DATA, RESULTS, ROOT  # noqa: E402

OUT = ROOT / "paper_out"
TAB, FIG = OUT / "tables", OUT / "figures"

# Fixed entity -> color (reference categorical palette, light mode, fixed order).
COLOR = {"cavr": "#2a78d6", "always_visual": "#eb6834", "always_hybrid": "#1baf7a", "always_text": "#eda100",
         "keyword_submitted": "#e87ba4", "knn": "#008300", "cavr_ce": "#4a3aa7", "cavr_ce_valsel": "#4a3aa7",
         "cavr_rf": "#e34948", "oracle": "#52514e"}
MARK = {"cavr": "o", "always_visual": "s", "always_hybrid": "D", "always_text": "^", "keyword_submitted": "v",
        "knn": "P", "logreg": "X", "cavr_rf": "h", "cavr_ce": "p", "cavr_ce_valsel": "p", "oracle": "*"}
LABEL = {"always_text": "Always text", "always_visual": "Always visual", "always_hybrid": "Always hybrid",
         "keyword_submitted": "Keyword rule (submitted)", "keyword_regex": "Keyword rule (regex fixed)",
         "logreg": "Logistic regression", "knn": "kNN quality", "cavr": "CAVR", "cavr_cls_head": "CAVR (class head only)",
         "oracle": "Per-query oracle", "random_equal_budget": "Random (equal budget)",
         "cavr_bm25": "CAVR + BM25 scores", "cavr_rf": "CAVR + BM25 + ColPali scores",
         "cavr_ce": "CAVR-CE (content-aware)", "cavr_ce_valsel": "CAVR-CE, val-chosen $\lambda$"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Per-policy annotation offsets (points) so neighbouring labels don't collide.
OFFSET = {"always_text": (8, -4), "keyword_submitted": (8, 6), "always_visual": (6, -12), "always_hybrid": (-20, -16),
          "oracle": (6, -4)}


def style():
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
                         "lines.markersize": 6, "figure.facecolor": "white", "axes.facecolor": "white"})


def load(name):
    p = RESULTS / name
    if not p.exists():
        print(f"  skip: {name} not found")
        return None
    return json.load(open(p, encoding="utf-8"))


def table(name, caption, label, header, rows, align=None):
    align = align or "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\",
             r"\midrule"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  table {name}.tex")


def pm(d, k, fmt="{:.3f}"):
    return (fmt + r" $\pm$ " + fmt).format(d[k]["mean"], d[k]["std"])


def os_name(platform_str):
    """platform.platform() calls Windows 11 'Windows-10-10.0.<build>'; builds >= 22000 are Windows 11."""
    parts = platform_str.split("-")
    if parts[0] == "Windows" and len(parts) > 2 and int(parts[2].split(".")[-1]) >= 22000:
        return f"Windows 11 (build {parts[2].split('.')[-1]})"
    return " ".join(parts[:2])


def t_hardware():
    meta = load("path_outputs_meta.json")
    if not meta:
        return
    hw = meta["hardware"]
    g = hw["gpus"][0]
    pk = hw.get("packages", {})
    rows = [["GPU", f"1 $\\times$ {g['name']} ({g['memory_gb']:.0f}\\,GB, power limit {g['power_limit_w']:.0f}\\,W)"],
            ["Driver / CUDA / cuDNN", f"{hw['driver']} / {hw['cuda_runtime']} / "
                                      f"{hw['cudnn'] // 10000}.{hw['cudnn'] // 100 % 100}.{hw['cudnn'] % 100}"],
            ["CPU threads / RAM", f"{hw['cpu_threads']} / {hw['ram_gb']:.0f}\\,GB"],
            ["OS / Python", f"{os_name(hw['os'])} / {hw['python']}"],
            ["PyTorch / Transformers", f"{pk.get('torch', '?')} / {pk.get('transformers', '?')}"],
            ["colpali-engine / docTR", f"{pk.get('colpali-engine', '?')} / {pk.get('python-doctr', '?')}"],
            ["Interconnect", "single GPU (multi-GPU sharding emulated)"]]
    table("hardware", "Hardware and software used for every measurement in this paper.", "tab:hardware",
          ["Component", "Value"], rows, "ll")


def t_data():
    m = json.load(open(DATA / "manifest.json"))
    ocr = load("ocr_cost.json")
    rows = [[ds, m["natural"]["by_dataset"][ds], m["pages"]["by_dataset"][ds], m["stratified"]["by_dataset"][ds]]
            for ds in ("docvqa", "infovqa")]
    rows.append(["Total", m["natural"]["queries"], m["pages"]["total"], m["stratified"]["queries"]])
    cap = "Evaluation data (validation splits). Natural = all questions; stratified = up to 150 per (dataset, question type)."
    if ocr:
        cap += f" docTR OCR of all pages: {ocr['total_seconds']:.0f}\\,s, {ocr['total_energy_j']/1000:.1f}\\,kJ on one RTX 4090."
    table("data", cap, "tab:data", ["Dataset", "Questions", "Pages", "Stratified"], rows)


def t_retrieval():
    t, v = load("index_cost_text.json"), load("index_cost_visual.json")
    if not (t and v):
        return
    rows = [["BM25 (docTR text)"] + [f"{t['recall'][f'@{k}']:.3f}" for k in (1, 3, 5, 10)] + [f"{t['mrr']:.3f}"],
            ["ColPali v1.3"] + [f"{v['recall'][f'@{k}']:.3f}" for k in (1, 3, 5, 10)] + [f"{v['mrr']:.3f}"]]
    q = v["query"]
    cap = (f"Page retrieval over each dataset's page pool. ColPali per query: encode {1000*q['encode_p50_s']:.0f}\\,ms + "
           f"score {1000*q['score_p50_s']:.1f}\\,ms (p50); index build {v['index']['wall_s']:.0f}\\,s for "
           f"{v['index']['pages']} pages, {v['index']['index_bytes_bf16']/2**20:.0f}\\,MB.")
    table("retrieval", cap, "tab:retrieval", ["Retriever", "R@1", "R@3", "R@5", "R@10", "MRR"], rows)


def t_main(ev):
    a = ev["aggregate"]
    s0 = ev["per_seed"][0]["tests_vs_cavr"]
    order = ["always_text", "always_visual", "always_hybrid", "keyword_submitted", "keyword_regex",
             "random_equal_budget", "logreg", "knn", "cavr_cls_head", "cavr", "cavr_bm25", "cavr_rf", "cavr_ce",
             "cavr_ce_valsel", "oracle"]
    rows = []
    for n in order:
        if n not in a:
            continue
        p = s0.get(n, {}).get("mcnemar_em", {}).get("p_value")
        rows.append([LABEL[n], pm(a[n], "anls"), pm(a[n], "em"), pm(a[n], "gpu_s"), pm(a[n], "energy_j", "{:.0f}"),
                     pm(a[n], "latency_p95_s", "{:.2f}"), "--" if p is None else f"{p:.2g}"])
    cap = (f"Routing policies on held-out test pages (mean $\\pm$ std over {ev['n_seeds']} seeds, $\\lambda$ = "
           f"{ev['lambda_default']}). Cost is measured per query on one RTX 4090"
           f"{' including router overhead' if ev['router_overhead_included'] else ''}. p: McNemar (EM) vs CAVR, seed 1.")
    table("main_results", cap, "tab:main_results",
          ["Policy", "ANLS", "EM", "GPU-s", "Energy (J)", "p95 lat. (s)", "p"], rows)

    # Per-seed tests of the content-aware router (val-chosen lambda) against always-hybrid and always-visual.
    rows = []
    for s in ev["per_seed"]:
        if "cavr_ce_valsel" not in s["policies"]:
            continue
        p, h, t = s["policies"]["cavr_ce_valsel"], s["policies"]["always_hybrid"], s["tests_cavr_ce_valsel"]
        bh, bv = t["always_hybrid"]["paired_boot_anls"], t["always_visual"]["paired_boot_anls"]
        rows.append([str(int(s["run"].replace(".npz", "").replace("seed", "")) + 1), f"{p['lambda']:g}", f"{p['anls']:.3f}",
                     f"{h['anls']:.3f}", f"{p['gpu_s']:.3f}", f"{h['gpu_s']:.3f}",
                     f"{100 * (1 - p['gpu_s'] / h['gpu_s']):.0f}\\%",
                     f"${bh['mean_diff']:+.3f}$ [${bh['ci95'][0]:+.3f}$, ${bh['ci95'][1]:+.3f}$]",
                     f"{t['always_hybrid']['mcnemar_em']['p_value']:.2g}",
                     f"${bv['mean_diff']:+.3f}$", f"{t['always_visual']['mcnemar_em']['p_value']:.2g}"])
    if rows:
        table("router_ce_tests", "CAVR-CE with $\lambda$ chosen on each seed's validation split (cheapest "
              "$\lambda$ within " + str(ev.get("valsel_delta", 0.005)) + " ANLS of always-hybrid on val), against "
              "always-hybrid and always-visual on test. GPU-s include ColPali for every query and the router's measured "
              "overhead.", "tab:router_ce_tests",
              ["Seed", "$\lambda$", "ANLS", "Hybrid ANLS", "GPU-s", "Hybrid GPU-s", "Saving",
               "$\Delta$ANLS vs hybrid [95\\% CI]", "McNemar p", "$\Delta$ANLS vs visual", "McNemar p"], rows)


def f_pareto(ev):
    sw = ev["per_seed"][0]["lambda_sweep"]
    pol = ev["per_seed"][0]["policies"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for n in ("cavr", "cavr_rf", "cavr_ce", "knn"):
        if n not in next(iter(sw.values())):
            continue
        pts = sorted((v[n]["gpu_s"], v[n]["anls"]) for v in sw.values())
        ax.plot(*zip(*pts), marker=MARK[n], color=COLOR[n], label=LABEL[n])
    pts = sorted((v["oracle"]["gpu_s"], v["oracle"]["anls"]) for v in sw.values())
    ax.plot(*zip(*pts), linestyle="--", marker=MARK["oracle"], color=COLOR["oracle"], label=LABEL["oracle"])
    for n in ("always_text", "always_visual", "always_hybrid", "keyword_submitted"):
        ax.scatter(pol[n]["gpu_s"], pol[n]["anls"], marker=MARK[n], color=COLOR[n], s=45, zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(LABEL[n], (pol[n]["gpu_s"], pol[n]["anls"]), textcoords="offset points",
                    xytext=OFFSET.get(n, (5, -10)), fontsize=7, color=INK2)
    ax.set_xlabel("Measured GPU-seconds per query")
    ax.set_ylabel("ANLS")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Cost-quality frontier (λ sweep, seed 1)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "pareto.png")
    plt.close(fig)
    print("  figure pareto.png")


def t_labels():
    lb = load("labels.json")
    if not lb:
        return
    c = lb["utility_labels_by_lambda"][str(lb["lambda_default"])]["counts"]
    rows = [[s] + [c[s].get(p, 0) for p in ("text", "visual", "hybrid")] for s in ("train", "val", "test")]
    th = lb["threshold_labels"]["counts"]
    rows += [[f"{s} (threshold)"] + [th[s].get(p, 0) for p in ("text", "visual", "hybrid")] for s in ("train", "val", "test")]
    kw = lb["keyword_rule_vs_outcome"]["keyword_submitted"]
    cap = (f"Outcome-based route labels on the seed-1 split (utility label at $\\lambda$={lb['lambda_default']}; threshold label = cheapest path "
           f"with ANLS $\\geq$ 0.5). The submitted keyword labels agree with the utility labels on "
           f"{100*kw['agreement_with_utility_label']:.1f}\\% of queries (Cohen's $\\kappa$ = {kw['kappa_with_utility_label']:.2f}).")
    table("labels", cap, "tab:labels", ["Split", "Text", "Visual", "Hybrid"], rows)


def t_escalation():
    es = load("escalation.json")
    if not es:
        return
    s0 = es["per_seed"][0]
    S = es["per_seed"]
    rows = []

    def mean_of(get):
        return {k: sum(get(x)[k] for x in S) / len(S) for k in ("anls", "gpu_s", "latency_p95_s", "escalation_rate")}
    for key, base, ece_key in (("results", "CAVR", "ece"), ("results_ce", "CAVR-CE", "ece_ce")):
        if key not in s0:
            continue
        v = mean_of(lambda x: x[key]["none"])
        rows.append([base, "none", "--", f"{v['anls']:.3f}", f"{v['gpu_s']:.3f}", f"{v['latency_p95_s']:.2f}", "0.0", "--"])
        for sig, modes in s0[key].items():
            if sig == "none":
                continue
            ece = sum(x[ece_key][sig] for x in S) / len(S) if ece_key in s0 else None
            for mode in modes:
                v = mean_of(lambda x: x[key][sig][mode]["0.5"])
                rows.append([base, sig.replace("_", " "), mode, f"{v['anls']:.3f}", f"{v['gpu_s']:.3f}",
                             f"{v['latency_p95_s']:.2f}", f"{100 * v['escalation_rate']:.1f}",
                             f"{ece:.3f}" if ece is not None else "--"])
    table("escalation", "Escalation to hybrid when the calibrated probability of a good answer is $<\\tau = 0.5$ "
          f"(mean over {len(S)} seeds). Probabilities are calibrated on the validation split by logistic regression; "
          "ECE is measured on test (10 bins). Costs include router overhead; serial and speculative latencies are "
          "computed from measured per-stage times (speculative assumes two concurrent workers).", "tab:escalation",
          ["Base router", "Signal", "Mode", "ANLS", "GPU-s", "p95 (s)", "Escalated \\%", "ECE"], rows, "lllrrrrr")
    r = s0.get("results_ce", s0["results"])
    base_name = "CAVR-CE" if "results_ce" in s0 else "CAVR"
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for (sig, mode), color, mk in zip((("conf_head", "reroute"), ("colpali_margin", "reroute"),
                                       ("answer_logprob_min", "serial"), ("all", "serial")),
                                      ("#2a78d6", "#eb6834", "#008300", "#e87ba4"), ("o", "s", "P", "v")):
        if sig not in r:
            continue
        pts = sorted((v["gpu_s"], v["anls"]) for v in r[sig][mode].values())
        ax.plot(*zip(*pts), marker=mk, color=color, label=f"{sig.replace('_', ' ')} ({mode})")
    ax.scatter(r["none"]["gpu_s"], r["none"]["anls"], color=INK, marker="*", s=80, zorder=3, label="no escalation")
    ev = load("eval_main.json")
    if ev:
        h = ev["per_seed"][0]["policies"]["always_hybrid"]
        ax.scatter(h["gpu_s"], h["anls"], marker=MARK["always_hybrid"], color=COLOR["always_hybrid"], s=45, zorder=3,
                   edgecolor="white", linewidth=1.5, label=LABEL["always_hybrid"])
    ax.set_xlabel("Measured GPU-seconds per query")
    ax.set_ylabel("ANLS")
    ax.legend(fontsize=7)
    ax.set_title(f"Escalation on {base_name}, τ swept 0.3–0.9 (seed 1)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "escalation.png")
    plt.close(fig)
    print("  figure escalation.png")


def index_scaling():
    ix = load("index_scaling.json")
    if not ix:
        return
    sc = ix["scaling"]
    rows = []
    for n, r in sc.items():
        def cell(k):
            return f"{1000*r[k]['p50_s']:.1f}" if k in r else "--"
        sh = r.get("sharded")
        rows.append([f"{int(n):,}", f"{r['index_gb_bf16']:.1f}", cell("gpu_bf16"), cell("cpu_stream"),
                     f"{r['index_gb_pooled2x2']:.1f}", cell("gpu_pooled2x2"),
                     f"{sh['gpus_needed']} / {1000*sh['emulated_parallel_p50_s']:.1f}" if sh else "1 / --"])
    pr = ix["pooling_recall_at1"]
    cap = (f"ColPali index scaling on one RTX 4090: p50 query latency (ms, batch 1). Pooling 2$\\times$2 cuts vectors "
           f"per page {ix['vectors_per_page']}$\\to${pr['vectors_per_page_pooled']}; R@1 on the real corpus "
           f"{pr['full']:.3f}$\\to${pr['pooled2x2']:.3f}. Sharded latency is emulated (shards timed one by one on one GPU).")
    table("index_scaling", cap, "tab:index_scaling",
          ["Pages", "Index GB", "GPU", "CPU stream", "Pooled GB", "Pooled GPU", "GPUs / sharded"], rows)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for k, color, mk, lab in (("gpu_bf16", COLOR["cavr"], "o", "GPU-resident"),
                              ("cpu_stream", COLOR["always_visual"], "s", "Host RAM, streamed"),
                              ("gpu_pooled2x2", COLOR["knn"], "D", "GPU, pooled 2×2")):
        pts = [(int(n), 1000 * r[k]["p50_s"]) for n, r in sc.items() if k in r]
        if pts:
            ax.plot(*zip(*pts), marker=mk, color=color, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pages in index")
    ax.set_ylabel("p50 query latency (ms)")
    ax.legend(fontsize=7)
    ax.set_title("ColPali MaxSim latency vs index size", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "index_scaling.png")
    plt.close(fig)
    print("  figure index_scaling.png")


def load_test():
    lt = load("load_test.json")
    if not lt:
        return
    pol = {n: v for n, v in lt["policies"].items() if "@" not in n}
    ops = {n: v for n, v in lt["policies"].items() if "@" in n}
    ss = lt.get("seeds_summary", {}).get("policies", {})

    def mstd(d, fmt="{:.1f}"):
        return "--" if not d else (fmt + r" $\pm$ " + fmt).format(d["mean"], d["std"])
    rows = [[LABEL[n], mstd(v["max_rate_1gpu"], "{:.2f}")] + [mstd(v["gpus_needed"][str(l)]) for l in (10, 25, 50, 100)]
            for n, v in ss.items()]
    n_seeds = len(next(iter(ss.values()))["anls_test"]["per_seed"]) if ss else 0
    table("load_test", "Serving capacity from a Poisson load test with dynamic batching, simulated from measured "
          "per-query service times and measured batching curves on one RTX 4090 (mean $\\pm$ std over "
          f"{n_seeds} seeds): highest arrival rate one GPU sustains at p95 $\\leq$ 2\\,s, and GPUs needed to hold "
          "p95 $\\leq$ 2\\,s at a given load (queries/s).", "tab:load_test",
          ["Policy", "QPS per GPU", "GPUs@10", "GPUs@25", "GPUs@50", "GPUs@100"], rows)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for n, v in pol.items():
        pts = [(float(r), x["p95_s"]) for r, x in v["latency_vs_rate_1gpu"].items()]
        ax.plot(*zip(*pts), marker=MARK.get(n, "o"), color=COLOR.get(n, INK2), label=LABEL[n],
                linestyle="--" if n == "oracle" else "-")
    ax.axhline(2.0, color=INK2, linewidth=1, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel("Arrival rate (queries/s, Poisson)")
    ax.set_ylabel("p95 latency (s)")
    ax.legend(fontsize=7)
    ax.set_title("p95 latency under load, 1 GPU", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "load_test.png")
    plt.close(fig)
    print("  figure load_test.png")
    if not ops:
        return
    # GPUs needed for 100 QPS at p95 <= 2 s vs answer quality, fixed policies + CAVR operating points.
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for fam, color, mk, lab in (("cavr", COLOR["cavr"], MARK["cavr"], "CAVR (λ sweep)"),
                                ("cavr_rf", COLOR["cavr_rf"], MARK["cavr_rf"], "CAVR + retrieval scores (λ sweep)"),
                                ("cavr_ce", COLOR["cavr_ce"], MARK["cavr_ce"], "CAVR-CE (λ sweep)")):
        pts = sorted((v["gpus_needed"]["2.0"]["100"], v["anls_test"]) for n, v in ops.items()
                     if n.split("@")[0] == fam and v["gpus_needed"]["2.0"]["100"])
        if pts:
            ax.plot(*zip(*pts), marker=mk, color=color, label=lab)
    for n in ("always_text", "always_visual", "always_hybrid", "keyword_submitted", "oracle"):
        v = pol[n]
        if v["gpus_needed"]["2.0"]["100"]:
            ax.scatter(v["gpus_needed"]["2.0"]["100"], v["anls_test"], marker=MARK[n], color=COLOR[n],
                       s=110 if n == "oracle" else 45, zorder=3, edgecolor="white", linewidth=1.5)
            ax.annotate(LABEL[n], (v["gpus_needed"]["2.0"]["100"], v["anls_test"]), textcoords="offset points",
                        xytext=OFFSET.get(n, (5, -10)), fontsize=7, color=INK2)
    ax.set_xlabel("GPUs needed for 100 QPS at p95 ≤ 2 s")
    ax.set_ylabel("ANLS (test)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Serving cost vs answer quality (seed 1)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "gpus_vs_quality.png")
    plt.close(fig)
    print("  figure gpus_vs_quality.png")
    rows = [[n.replace("_rf", " + ret. scores").replace("_ce", "-CE").replace("cavr", "CAVR").replace("@", r", $\lambda$="),
             f"{v['anls_test']:.3f}", f"{v['max_rate_1gpu']['2.0']:g}"]
            + [str(v["gpus_needed"]["2.0"][str(l)] or ">256") for l in (10, 25, 50, 100)] for n, v in ops.items()]
    table("load_test_operating_points", "CAVR operating points ($\lambda$ is set at run time): test ANLS, capacity of "
          "one GPU and GPUs needed at p95 $\leq$ 2\,s (seed 1).", "tab:load_test_ops",
          ["Policy", "ANLS", "QPS/GPU", "GPUs@10", "GPUs@25", "GPUs@50", "GPUs@100"], rows)


def router_cost():
    rc = load("router_cost.json")
    if not rc:
        return
    rows = [[b, f"{1000*v['latency_s']:.2f}", f"{v['queries_per_s']:,.0f}"] for b, v in rc["cuda"]["throughput"].items()]
    cap = (f"Router (mpnet embedding + CAVR MLP) on one RTX 4090: batch-1 p50 {1000*rc['cuda']['batch1_p50_s']:.1f}\\,ms; "
           "throughput by batch size.")
    ce = load("router_ce_cost.json")
    if ce:
        b1 = ce["per_query_batch1"]
        best = max(v["queries_per_s"] for v in ce["cross_encoder_throughput"].values())
        cap += (f" CAVR-CE (embedding + passage selection + cross-encoder): batch-1 p50 {1000*b1['latency_s']:.1f}\\,ms, "
                f"{b1['energy_j']:.2f}\\,J per query; cross-encoder throughput up to {best:,.0f} queries/s.")
    table("router_throughput", cap, "tab:router_throughput", ["Batch", "Latency (ms)", "Queries/s"], rows)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    t_hardware()
    t_data()
    t_retrieval()
    t_labels()
    ev = load("eval_main.json")
    if ev:
        t_main(ev)
        f_pareto(ev)
    t_escalation()
    index_scaling()
    load_test()
    router_cost()


if __name__ == "__main__":
    main()
