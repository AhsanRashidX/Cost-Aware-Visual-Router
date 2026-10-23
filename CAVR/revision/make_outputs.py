"""Step 9: regenerate every table and figure from results/*.json (nothing typed by hand).

Writes LaTeX (booktabs) tables to paper_out/tables/ and PNG figures to paper_out/figures/.
Each piece is skipped, with a note, if its source file doesn't exist yet.
"""
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

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
         "keyword_submitted": "Keyword rule", "keyword_regex": "Keyword rule (regex)",
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
           f"with ANLS $\\geq$ 0.5). Keyword-rule labels agree with the utility labels on "
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


def confirmation():
    cf = load("confirmation.json")
    if not cf:
        return
    fx, cs = cf["fixed"], cf["seeds"]
    rows = [[f"Always-{p}", f"{fx[p]['anls']:.3f}", f"{fx[p]['em']:.3f}", f"{fx[p]['gpu_s']:.2f}"]
            for p in ("text", "visual", "hybrid")]
    m = lambda k, d: f"{mean(s[k] for s in cs):.{d}f} $\\pm$ {stdev(s[k] for s in cs):.{d}f}"
    rows.append(["CAVR-CE (frozen)", m("anls", 3), m("em", 3), m("gpu_s", 2)])
    rows.append(["Per-query oracle", f"{cf['oracle']['anls']:.3f}", "--", f"{cf['oracle']['gpu_s']:.2f}"])
    cap = (f"Held-out confirmation set: {cf['n_queries']:,} new questions from the DocVQA and InfographicVQA training "
           "splits, run on the A10 node. CAVR-CE is each seed's router re-trained on its original training split and "
           "frozen (validation-chosen $\\lambda$); mean $\\pm$ standard deviation over the five seeds. GPU-seconds per "
           "query on the A10, including retrieval; the router's overhead is the one measured on the workstation.")
    table("confirmation", cap, "tab:confirmation", ["Policy", "ANLS", "EM", "GPU-s"], rows)


def controls():
    ca = load("controls_analysis.json")
    if not ca:
        return
    names = [("text_k1", "Text, top-1 BM25 page"), ("text", "Text, top-3 BM25 pages (main)"),
             ("visual", "Visual, top-1 ColPali image (main)"), ("visual_k3", "Visual, top-3 ColPali images"),
             ("hybrid", "Hybrid (main)"), ("gold_text", "Gold page as text"), ("gold_visual", "Gold page as image"),
             ("gold_hybrid", "Gold page, image and text")]
    b = ca["by_dataset"]
    rows = [[lab, f"{b['all']['anls'][k]:.3f}", f"{b['docvqa']['anls'][k]:.3f}", f"{b['infovqa']['anls'][k]:.3f}",
             f"{ca['input_tokens_mean'][k]:,.0f}"] for k, lab in names]
    cap = (f"Retrieval-depth and gold-page controls on {ca['n_questions']:,} questions ({b['docvqa']['n']} per dataset): "
           "mean ANLS of the same generator given each context, and mean input tokens. Main paths are the cached "
           "workstation runs; the other rows were run on the A10 node.")
    table("controls", cap, "tab:controls", ["Context", "ANLS (all)", "DocVQA", "InfoVQA", "Input tokens"], rows)


def token_pooling():
    tp = load("token_pooling.json")
    if not tp:
        return
    names = {"full": "Full index", "pooled2x2": "2$\\times$2 spatial pooling"}
    rows = []
    for k, v in tp["methods"].items():
        label = names.get(k, f"Hierarchical pooling, $f = {k[4:]}$")
        rows.append([label, v["vectors_per_page"], f"{v['recall@1']:.3f}", f"{v['recall@5']:.3f}",
                     f"{v['index_gib_1m_pages']:.0f}", f"{v['relative_scoring_cost']:.2f}"])
    cap = ("Compressed ColPali indexes on the real corpus: recall of the gold page over each dataset's page pool for all "
           "questions (batched query encoding), index size at a million pages and scoring cost relative to the full index. "
           "Hierarchical pooling clusters each page's patch vectors by Ward linkage and keeps the prompt-token vectors.")
    table("token_pooling", cap, "tab:token_pooling",
          ["Index", "Vectors/page", "Recall@1", "Recall@5", "GiB at 1M pages", "Rel. scoring cost"], rows)


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
           "(batched query encoding, which differs slightly from the per-query encoding of Table~\\ref{tab:retrieval}) "
           f"{pr['full']:.3f}$\\to${pr['pooled2x2']:.3f}. Sharded latency is emulated here (shards timed one by one "
           "on one GPU); measured sharding on two multi-GPU nodes is in Table~\\ref{tab:multi_gpu} and "
           "host-memory streaming up to a million pages in Table~\\ref{tab:host_stream}.")
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


SIZE_STYLE = {1000: ("#eda100", "^"), 10000: ("#1baf7a", "s"), 25000: ("#e87ba4", "v"), 50000: ("#2a78d6", "o"),
              100000: ("#4a3aa7", "D"), 120000: ("#008300", "P"), 150000: ("#eb6834", "*")}
# Measured multi-GPU nodes: (results file, short name), in the order the paper describes them.
NODES = (("multi_gpu_scaling.json", "P40"), ("a10/multi_gpu_scaling.json", "A10"))


def multi_gpu():
    mgs = [(mg, short) for mg, short in ((load(f), short) for f, short in NODES) if mg]
    if not mgs:
        return

    def hw(mg):
        node, G = mg["source"]["node"], mg["n_gpus"]
        g0 = node["gpus"][0]
        tf = [g["fp32_tflops"] for g in node["gpus"]]
        return {"GPU": f"{G} $\\times$ {g0['name']}",
                "Memory / power limit": f"{g0['memory_gb']:.1f}\\,GiB / {g0['power_limit_w']:.0f}\\,W each",
                "Compute capability": g0["capability"],
                "Driver / mode": f"{node['driver']} / {node['driver_model']} (compute mode)",
                "fp32 matrix multiply (measured)": f"{min(tf):.1f}--{max(tf):.1f} TFLOPS per GPU",
                "Index scoring precision": mg["source"]["compute_dtype"].replace("float", "fp"),
                "PCIe link width": f"$\\times${g0['pcie_width']} per GPU"
                                   + ("" if node["topology_recorded"] else " (topology not recorded)"),
                "CPU": node["cpu"].replace("(R)", "").replace("(TM)", "").replace(" CPU", ""),
                "CPU threads / RAM": f"{node['cpu_threads']} / {node['ram_gib']:.0f}\\,GiB",
                "OS / Python": f"Windows (build {node['os'].split('-')[2].split('.')[-1]}) / {node['python']}",
                "PyTorch / CUDA": f"{node['torch']} / {node['cuda_build']}",
                "Energy": f"NVML energy counters, summed over all {G} GPUs"}
    cols = [hw(mg) for mg, _ in mgs]
    rows = [[k] + [c[k] for c in cols] for k in cols[0]]
    table("hardware_multigpu", "Multi-GPU nodes used for the index-sharding and host-memory measurements "
          "(Tables~\\ref{tab:multi_gpu} and~\\ref{tab:host_stream}). No model runs on these nodes: page and query embeddings are exported from "
          "the workstation of Table~\\ref{tab:hardware}.", "tab:hardware_multigpu",
          ["Component"] + [f"{short} node" for _, short in mgs], rows, "l" * (len(mgs) + 1))

    def num(x):
        return f"{x:,.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}"
    rows = []
    for i, (mg, _) in enumerate(mgs):
        title = f"{mg['n_gpus']} $\\times$ {mg['source']['node']['gpus'][0]['name']}"
        rows.append([("\\midrule " if i else "") + f"\\multicolumn{{9}}{{l}}{{\\emph{{{title}}}}}"])
        for n, s in mg["strong"].items():
            for g, v in s["by_gpus"].items():
                rows.append([f"{int(n) // 1000}k", g, f"{v['index_gib_per_gpu']:.1f}", f"{1000 * v['p50_s']:.1f}",
                             f"{1000 * v['p95_s']:.1f}", f"{v['latency_speedup']:.2f}",
                             f"{100 * v['latency_efficiency']:.0f}", num(v["throughput_qps"]),
                             num(v["energy_j_per_query_node"])])
    src0, b = mgs[0][0]["source"], mgs[0][0]["throughput_batch"]

    def per_node(f):
        return " and ".join(f"{f(mg)} ({short})" for mg, short in mgs)
    dtype = per_node(lambda m: m["source"]["compute_dtype"].replace("float", "fp"))
    idle = per_node(lambda m: f"{m['summary']['idle_w_per_gpu']:.0f}\\,W")
    exact = per_node(lambda m: f"{100 * m['summary']['check_exact_min']:.1f}\\%")
    budget = per_node(lambda m: f"{m['source']['budget_gib_per_gpu']:.1f}\\,GiB")
    cap = (f"Measured multi-GPU sharding of the ColPali index on {['one', 'two', 'three'][len(mgs) - 1]} nodes "
           f"(Table~\\ref{{tab:hardware_multigpu}}): index stored in {src0['store_dtype']} and scored in {dtype}; "
           f"one contiguous shard per GPU, all shards queried in parallel and their top-{src0['topk']} merged on the "
           f"host. p50 and p95 over {src0['n_lat']} single queries; speedup and parallel efficiency of p50 latency "
           f"relative to the fewest GPUs that hold the corpus; throughput and energy at batch {b}. Energy is read for "
           "the whole node, so runs on fewer GPUs include the unused GPUs, which hold a CUDA context and draw about "
           f"{idle} each. Sharded top-1 results matched one GPU for all {src0['n_check']} check queries on every node "
           f"({exact} exactly, the rest exact score ties). Shards above 70\\% of GPU memory ({budget}) were not run.")
    table("multi_gpu_scaling", cap, "tab:multi_gpu",
          ["Pages", "GPUs", "GiB/GPU", "p50 (ms)", "p95 (ms)", "Speedup", "Eff. (\\%)", "Queries/s", "J/query"],
          rows)

    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(len(mgs), 2, figsize=(7.0, 2.6 * len(mgs) + 0.5), squeeze=False)
    handles, letters = {}, iter("abcdefgh")
    for (mg, short), row in zip(mgs, axes):
        G = mg["n_gpus"]
        for ax, key, ylab, what in ((row[0], "p50_s", "p50 query latency (ms)", "latency, batch 1"),
                                    (row[1], "throughput_qps", "Queries/s", f"throughput, batch {b}")):
            for n, s in mg["strong"].items():
                color, mk = SIZE_STYLE.get(int(n), (INK2, "o"))
                by = s["by_gpus"]
                gs = [int(g) for g in by]
                ys = [(1000 if key == "p50_s" else 1) * by[str(g)][key] for g in gs]
                h, = ax.plot(gs, ys, marker=mk, color=color, markersize=9 if mk == "*" else 6)
                handles.setdefault(int(n), h)
                if len(gs) > 1:
                    g_b = gs[0]
                    ideal = [ys[0] * (g_b / g if key == "p50_s" else g / g_b) for g in gs]
                    ax.plot(gs, ideal, color=color, linestyle="--", linewidth=1, alpha=0.7)
            if "50000" in mg["weak"]:
                pts = [(int(g), (1000 if key == "p50_s" else 1) * (v["p50_s"] if key == "p50_s" else
                       mg["strong"][str(v["pages"])]["by_gpus"][g]["throughput_qps"]))
                       for g, v in mg["weak"]["50000"].items()]
                ax.plot(*zip(*pts), color=INK2, linestyle=":", linewidth=1.5, zorder=0)
            ax.set_yscale("log")
            ax.set_xticks(range(1, G + 1))
            ax.set_xlim(0.8, G + 0.2)
            ax.set_xlabel("GPUs")
            ax.set_ylabel(ylab)
            ax.set_title(f"({next(letters)}) {G}$\\times${short}: {what}", fontsize=9, color=INK, loc="left")
    leg = {f"{n // 1000}k pages": handles[n] for n in sorted(handles)}
    leg["ideal (linear)"] = Line2D([], [], color=INK2, linestyle="--", linewidth=1)
    leg["50k pages per GPU"] = Line2D([], [], color=INK2, linestyle=":", linewidth=1.5)
    fig.legend(leg.values(), leg.keys(), loc="lower center", ncol=(len(leg) + 1) // 2, fontsize=6.5,
               handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "multi_gpu_scaling.png")
    fig.savefig(FIG / "multi_gpu_scaling.pdf")
    plt.close(fig)
    print("  figure multi_gpu_scaling.png")


def host_stream():
    hs = load("a10/host_stream_scaling.json")
    rs = load("a10/a10_shard_scaling.json")
    if not hs:
        return
    src, b = hs["source"], str(hs["source"]["big_batch"])
    topk = load("a10/a10_host_stream.json")["topk"]
    rows = []
    for r in sorted(hs["runs"], key=lambda r: (r["pages"], r["gpus"])):
        n = r["pages"]
        rows.append([f"{n // 10**6}M" if n % 10**6 == 0 else f"{n // 1000}k", f"{r['index_gib_total']:.1f}", r["gpus"],
                     f"{r['p50_s']:.2f}", f"{r['stream_gb_per_s_per_gpu_b1']:.1f}", f"{r['qps']['16']:.2f}",
                     f"{r['qps'][b]:.2f}", f"{r['gpu_s_per_query_big']:.3f}", f"{r['energy_j_per_query'][b]:.1f}"])
    c, ck = hs["copy_only"], src["checks"]["1"]
    cap = (f"ColPali index held in host memory and streamed to the GPUs of the A10 node "
           f"(Table~\\ref{{tab:hardware_multigpu}}): index stored in {src['compute_dtype'].replace('float', 'fp')}, "
           f"page-locked with \\texttt{{cudaHostRegister}}, split into one contiguous shard per GPU and streamed in "
           f"chunks of {src['chunk_pages']:,} pages with copying and scoring overlapped; exhaustive MaxSim in "
           f"{src['compute_dtype'].replace('float', 'fp')}, per-GPU top-{topk} merged on the host. p50 over {src['n_lat']} "
           f"single queries; PCIe rate per GPU at batch 1; throughput at batch 16 and {b}; GPU-seconds and node energy "
           f"per query at batch {b} ({src['reps']} repetitions per batch). Energy is the NVML sum over both GPUs, so "
           f"one-GPU runs include the unused GPU; host CPU and memory energy is not measured. Copying alone reached "
           f"{c['1']['gb_per_s_total']:.1f}\\,GB/s on one GPU and {c['2']['gb_per_s_total']:.1f}\\,GB/s on two. Streamed "
           f"top-1 results matched GPU-resident search for all {ck['queries']} check queries on "
           f"{ck['pages'] // 1000}k pages.")
    table("host_stream", cap, "tab:host_stream",
          ["Pages", "Index GiB", "GPUs", "p50 (s)", "GB/s/GPU", "Q/s, b16", f"Q/s, b{b}", "GPU-s/query", "GPU J/query"],
          rows)

    color = {1: COLOR["cavr"], 2: COLOR["always_hybrid"]}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for g in sorted({r["gpus"] for r in hs["runs"]}):
        rr = sorted((r for r in hs["runs"] if r["gpus"] == g), key=lambda r: r["pages"])
        x = [r["pages"] for r in rr]
        ax1.plot(x, [r["p50_s"] for r in rr], marker="o", color=color[g], label=f"{g} GPU{'s' if g > 1 else ''}")
        ax1.plot(x, [r["copy_only_s"] for r in rr], color=INK, linestyle=(0, (3, 2)), linewidth=1, zorder=3)
        ax2.plot(x, [r["qps"][b] for r in rr], marker="o", color=color[g], label=f"{g} GPU{'s' if g > 1 else ''}, streamed")
        if rs:
            res = sorted((r for r in rs["runs"] if r["gpus"] == g and r["pages"] >= 10000), key=lambda r: r["pages"])
            ax2.plot([r["pages"] for r in res], [r["throughput"][b]["queries_per_s"] for r in res], marker="o",
                     linestyle="none", markerfacecolor="white", color=color[g],
                     label=f"{g} GPU{'s' if g > 1 else ''}, GPU-resident")
    from matplotlib.lines import Line2D
    h, lab = ax1.get_legend_handles_labels()
    ax1.legend(h + [Line2D([], [], color=INK, linestyle=(0, (3, 2)), linewidth=1)], lab + ["PCIe copy time"], fontsize=7)
    ax2.legend(fontsize=7)
    for ax, ylab, title in ((ax1, "p50 latency, single query (s)", "(a) Latency, batch 1"),
                            (ax2, "Queries/s", f"(b) Throughput, batch {b}")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Pages in index")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "host_stream.png")
    fig.savefig(FIG / "host_stream.pdf")
    plt.close(fig)
    print("  figure host_stream.png")


def slo_sweep():
    se = load("sim_experiments.json")
    if not se:
        return
    sw, sav = se["slo_sweep"], se["slo_sweep_saving_vs_hybrid"]
    rows = []
    for T in sw["always_hybrid"]:
        rows.append([f"{float(T):g}"] + [f"{sw[p][T]['mean']:.1f} $\\pm$ {sw[p][T]['std']:.1f}"
                                         for p in ("always_visual", "cavr_ce_valsel", "always_hybrid")]
                    + [f"{100 * sav[T]:.0f}"])
    bm = se["batch_model_saving_vs_hybrid"]["longest_member"]
    cap = (f"GPUs needed to serve {se['load_qps']} queries/s against the p95 latency target (simulated from measured "
           f"service times; mean $\\pm$ standard deviation over the 5 splits) and the saving of CAVR-CE over "
           f"always-hybrid. Charging each batch its longest member instead of the mean member gives a saving of "
           f"{100 * bm['2.0']:.0f}\\% at 2\\,s and {100 * bm['5.0']:.0f}\\% at 5\\,s.")
    table("slo_sweep", cap, "tab:slo_sweep",
          ["p95 target (s)", "Always visual", "CAVR-CE", "Always hybrid", "Saving vs hybrid (\\%)"], rows)


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
          ["Policy", "1-GPU capacity (q/s)", "GPUs@10", "GPUs@25", "GPUs@50", "GPUs@100"], rows)
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


def live_serving():
    vl = load("vllm_a10.json")
    if not vl:
        return
    runs = vl["runs"]
    order = [(p, n) for n in (1, 2) for p in ("always_visual", "cavr_ce_valsel", "always_hybrid") if f"{p}_r{n}" in runs]
    rows = []
    for p, n in order:
        r = runs[f"{p}_r{n}"]
        c, lb = r["capacity_rps_at_p95_target"], r["measured_capacity_is_lower_bound"]
        cells = []
        for t in ("2.0", "5.0"):
            cells += [("$\\geq$" if lb[t] else "") + f"{c[t]['measured']:.2f}", f"{c[t]['predicted_a10']:.2f}"]
        ok = [v for v in r["rates"].values() if not v["aborted"] and v["n_errors"] == 0]
        anls = mean(v["anls"] for v in ok)
        ref = mean(v["anls_4090_same_requests"] for v in ok)
        rows.append([LABEL[p].split(",")[0], n] + cells + [f"{anls:.3f} ({ref:.3f})"])
    table("live_serving", "Live serving with vLLM on one and two A10 GPUs (one replica per GPU, open-loop Poisson "
          "arrivals): highest arrival rate (queries/s) at which the measured p95 latency stays within the target, "
          "interpolated between tested rates, against the simulator's prediction from the A10's own service times "
          "and batching curve. ANLS is the mean over the tested rates; in brackets, the 4090 reference run on the "
          "same requests.", "tab:live_serving",
          ["Policy", "GPUs", "p95$\\leq$2\\,s live", "sim.", "p95$\\leq$5\\,s live", "sim.", "ANLS (4090)"], rows)
    reps = sorted({n for _, n in order})
    fig, axes = plt.subplots(1, len(reps), figsize=(3.4 * len(reps), 3.0), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], reps):
        for p in ("always_visual", "cavr_ce_valsel", "always_hybrid"):
            r = runs.get(f"{p}_r{n}")
            if not r:
                continue
            ok = sorted((v["offered_rps"], v["p95_s"], v["predicted"]["a10"]["p95_s"]) for v in r["rates"].values()
                        if not v["aborted"] and v["n_errors"] == 0)
            x, live, sim = zip(*ok)
            ax.plot(x, live, marker=MARK[p], color=COLOR[p], label=LABEL[p].split(",")[0])
            ax.plot(x, sim, color=COLOR[p], linestyle="--", linewidth=1.2)
        for t in (2.0, 5.0):
            ax.axhline(t, color=INK2, linewidth=1, linestyle=":")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 5, 10, 20])
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Arrival rate (queries/s, Poisson)")
        ax.set_title(f"{n} A10 GPU" + ("s" if n > 1 else ""), fontsize=9, color=INK, loc="left")
    axes[0][0].set_ylabel("p95 latency (s)")
    h, lab = axes[0][0].get_legend_handles_labels()
    h += [matplotlib.lines.Line2D([], [], color=INK2, linestyle="-"),
          matplotlib.lines.Line2D([], [], color=INK2, linestyle="--", linewidth=1.2)]
    axes[0][0].legend(h, lab + ["live (vLLM)", "simulated"], fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "live_serving.png")
    fig.savefig(FIG / "live_serving.pdf")
    plt.close(fig)
    print("  figure live_serving.png")


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
    controls()
    confirmation()
    ev = load("eval_main.json")
    if ev:
        t_main(ev)
        f_pareto(ev)
    t_escalation()
    index_scaling()
    token_pooling()
    multi_gpu()
    host_stream()
    load_test()
    slo_sweep()
    live_serving()
    router_cost()


if __name__ == "__main__":
    main()
