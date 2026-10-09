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
    rows = [["GPU", f"1 $\\times$ {g['name']} ({g['memory_gb']:.0f}\\,GiB, power limit {g['power_limit_w']:.0f}\\,W)"],
            ["Driver / CUDA / cuDNN", f"{hw['driver']} / {hw['cuda_runtime']} / "
                                      f"{hw['cudnn'] // 10000}.{hw['cudnn'] // 100 % 100}.{hw['cudnn'] % 100}"],
            ["CPU threads / RAM", f"{hw['cpu_threads']} / {hw['ram_gb']:.0f}\\,GiB"],
            ["OS / Python", f"{os_name(hw['os'])} / {hw['python']}"],
            ["PyTorch / Transformers", f"{pk.get('torch', '?')} / {pk.get('transformers', '?')}"],
            ["colpali-engine / docTR", f"{pk.get('colpali-engine', '?')} / {pk.get('python-doctr', '?')}"],
            ["Interconnect", "single GPU (multi-GPU sharding: Table~\\ref{tab:hardware_multigpu})"]]
    si = load("system_info.json")
    if si:
        sh, sp = si["hardware"], si["hardware"]["packages"]
        rows.insert(2, ["CPU", sh["cpu"]])
        rows.insert(-1, ["sentence-transformers / scikit-learn",
                         f"{sp.get('sentence-transformers', '?')} / {sp.get('scikit-learn', '?')}"])
        rows.insert(-1, ["rank-bm25 / qwen-vl-utils / datasets",
                         f"{sp.get('rank-bm25', '?')} / {sp.get('qwen-vl-utils', '?')} / {sp.get('datasets', '?')}"])
        for m, r in si["model_revisions"].items():
            rows.insert(-1, [f"\\texttt{{{m.split('/')[-1]}}} revision", f"\\texttt{{{(r or '?')[:7]}}}"])
    table("hardware", "Workstation used for every measurement except multi-GPU index sharding "
          "(Table~\\ref{tab:hardware_multigpu}): hardware, software and model revisions.", "tab:hardware",
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
           f"{pr['full']:.3f}$\\to${pr['pooled2x2']:.3f}. Sharded latency is emulated here (shards timed one by one "
           "on one GPU); measured sharding on three GPUs is in Table~\\ref{tab:multi_gpu}.")
    table("index_scaling", cap, "tab:index_scaling",
          ["Pages", "Index GiB", "GPU", "CPU stream", "Pooled GiB", "Pooled GPU", "GPUs / sharded"], rows)

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


SIZE_STYLE = {1000: ("#eda100", "^"), 10000: ("#1baf7a", "s"), 50000: ("#2a78d6", "o"), 100000: ("#4a3aa7", "D"),
              150000: ("#eb6834", "*")}


def multi_gpu():
    mg = load("multi_gpu_scaling.json")
    if not mg:
        return
    src, node, G = mg["source"], mg["source"]["node"], mg["n_gpus"]
    g0 = node["gpus"][0]
    tf = [g["fp32_tflops"] for g in node["gpus"]]
    cpu = node["cpu"].replace("(R)", "").replace("(TM)", "")
    rows = [["GPU", f"{G} $\\times$ {g0['name']} ({g0['memory_gb']:.1f}\\,GiB each, power limit "
                    f"{g0['power_limit_w']:.0f}\\,W, compute capability {g0['capability']})"],
            ["Driver / mode", f"{node['driver']} / {node['driver_model']} (compute mode)"],
            ["fp32 matrix multiply (measured)", f"{min(tf):.1f}--{max(tf):.1f} TFLOPS per GPU"],
            ["PCIe link width", f"$\\times${g0['pcie_width']} per GPU"
                                + ("" if node["topology_recorded"] else " (topology not recorded)")],
            ["CPU", cpu], ["CPU threads / RAM", f"{node['cpu_threads']} / {node['ram_gib']:.0f}\\,GiB"],
            ["OS / Python", f"Windows (build {node['os'].split('-')[2].split('.')[-1]}) / {node['python']}"],
            ["PyTorch / CUDA", f"{node['torch']} / {node['cuda_build']}"],
            ["Energy", f"NVML energy counters, summed over all {G} GPUs"]]
    table("hardware_multigpu", "Multi-GPU node used for the index-sharding measurements "
          "(Table~\\ref{tab:multi_gpu}). No model runs on this node: page and query embeddings are exported from "
          "the workstation of Table~\\ref{tab:hardware}.", "tab:hardware_multigpu", ["Component", "Value"], rows, "ll")

    def num(x):
        return f"{x:,.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}"
    rows = []
    for n, s in mg["strong"].items():
        for g, v in s["by_gpus"].items():
            rows.append([f"{int(n) // 1000}k", g, f"{v['index_gib_per_gpu']:.1f}", f"{1000 * v['p50_s']:.1f}",
                         f"{1000 * v['p95_s']:.1f}", f"{v['latency_speedup']:.2f}",
                         f"{100 * v['latency_efficiency']:.0f}", num(v["throughput_qps"]),
                         num(v["energy_j_per_query_node"])])
    sm, b = mg["summary"], mg["throughput_batch"]
    cap = (f"Measured multi-GPU sharding of the ColPali index on {G} {g0['name']} GPUs "
           f"(Table~\\ref{{tab:hardware_multigpu}}): index stored in {src['store_dtype']} and scored in "
           f"{src['compute_dtype'].replace('float', 'fp')}, one contiguous shard per GPU, all shards queried in "
           f"parallel and their top-{src['topk']} merged on the host. p50 and p95 over {src['n_lat']} single queries; "
           "speedup and parallel efficiency of p50 latency relative to the fewest GPUs that hold the corpus (two for "
           f"100k pages); throughput and energy at batch {b}. Energy is read for the whole node, so runs on fewer "
           f"than {G} GPUs include the idle GPUs (about {sm['idle_w_per_gpu']:.0f}\\,W each). Sharded top-1 results "
           f"matched one GPU for all {src['n_check']} check queries ({100 * sm['check_exact_min']:.1f}\\% exactly, "
           f"the rest exact score ties). Shards above {src['budget_gib_per_gpu']:.1f}\\,GiB per GPU were not run.")
    table("multi_gpu_scaling", cap, "tab:multi_gpu",
          ["Pages", "GPUs", "GiB/GPU", "p50 (ms)", "p95 (ms)", "Speedup", "Eff. (\\%)", "Queries/s", "J/query"],
          rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    handles = {}
    for ax, key, ylab, title in ((axes[0], "p50_s", "p50 query latency (ms)", "(a) Latency, batch 1"),
                                 (axes[1], "throughput_qps", "Queries/s", f"(b) Throughput, batch {b}")):
        for n, s in mg["strong"].items():
            color, mk = SIZE_STYLE.get(int(n), (INK2, "o"))
            by = s["by_gpus"]
            gs = [int(g) for g in by]
            ys = [(1000 if key == "p50_s" else 1) * by[str(g)][key] for g in gs]
            h, = ax.plot(gs, ys, marker=mk, color=color, markersize=9 if mk == "*" else 6,
                         label=f"{int(n) // 1000}k pages")
            handles.setdefault(f"{int(n) // 1000}k pages", h)
            if len(gs) > 1:
                g_b = gs[0]
                ideal = [ys[0] * (g_b / g if key == "p50_s" else g / g_b) for g in gs]
                ax.plot(gs, ideal, color=color, linestyle="--", linewidth=1, alpha=0.7)
        for per, d in mg["weak"].items():
            pts = [(int(g), (1000 if key == "p50_s" else 1) * (v["p50_s"] if key == "p50_s" else
                   mg["strong"][str(v["pages"])]["by_gpus"][g]["throughput_qps"])) for g, v in d.items()]
            ax.plot(*zip(*pts), color=INK2, linestyle=":", linewidth=1.5, zorder=0)
            weak_label = f"{int(per) // 1000}k pages per GPU"
        ax.set_yscale("log")
        ax.set_xticks(range(1, G + 1))
        ax.set_xlabel("GPUs")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    from matplotlib.lines import Line2D
    handles["ideal (linear)"] = Line2D([], [], color=INK2, linestyle="--", linewidth=1)
    if mg["weak"]:
        handles[weak_label] = Line2D([], [], color=INK2, linestyle=":", linewidth=1.5)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles), fontsize=6.5,
               handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "multi_gpu_scaling.png")
    fig.savefig(FIG / "multi_gpu_scaling.pdf")
    plt.close(fig)
    print("  figure multi_gpu_scaling.png")


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
    multi_gpu()
    load_test()
    router_cost()


if __name__ == "__main__":
    main()
