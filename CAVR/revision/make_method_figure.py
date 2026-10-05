"""Method figure: the CAVR pipeline as built (offline cache and labels, router training, online serving).

Numbers shown in the figure are read from results/*.json. Writes paper_out/figures/method.pdf (vector, for the
paper) and method.png (preview). Colours follow the other figures: text = yellow, visual = orange, hybrid = aqua,
CAVR = blue, CAVR-CE = violet.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from common import DATA, RESULTS, ROOT  # noqa: E402

FIG = ROOT / "paper_out" / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
NEUTRAL = ("#f4f3f0", "#8a8984")
TEXT = ("#fdf1d6", "#c98500")       # yellow
VISUAL = ("#fbe3d8", "#d95926")     # orange
HYBRID = ("#d6f2e7", "#199e70")     # aqua
CAVR = ("#dce9f8", "#2a78d6")       # blue
CE = ("#e4e0f5", "#4a3aa7")         # violet


def values():
    rows = [json.loads(l) for l in open(RESULTS / "path_outputs.jsonl", encoding="utf-8")]
    gpu = {p: sum(r["gpu_seconds"] for r in rows if r["path"] == p) / sum(r["path"] == p for r in rows)
           for p in ("text", "visual", "hybrid")}
    man = json.load(open(DATA / "manifest.json"))
    meta = json.load(open(RESULTS / "path_outputs_meta.json"))
    ix = json.load(open(RESULTS / "index_scaling.json"))
    ce = json.load(open(RESULTS / "router_ce_cost.json"))
    ev = json.load(open(RESULTS / "eval_main.json"))
    return {"gpu": gpu, "n_q": man["natural"]["queries"], "n_pages": man["pages"]["total"],
            "vec": ix["vectors_per_page"], "k_text": meta["k_text"], "gen": meta["model"].split("/")[-1],
            "ce_ms": 1000 * ce["per_query_batch1"]["latency_s"], "delta": ev.get("valsel_delta", 0.005),
            "seeds": ev["n_seeds"]}


def box(ax, x, y, w, h, colors, title=None, lines=(), title_size=7.2, size=6.3, align="center", lw=1.0, dashed=False):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=ec, lw=lw,
                                ls=(0, (3, 2)) if dashed else "-", zorder=2))
    cx = x + w / 2 if align == "center" else x + 1.0
    ty = y + h - 1.6
    if title:
        ax.text(cx, ty, title, ha=align, va="top", fontsize=title_size, fontweight="bold", color=INK, zorder=3)
        ty -= 3.0
    for ln in lines:
        ax.text(cx, ty, ln, ha=align, va="top", fontsize=size, color=INK2 if not ln.startswith("$") else INK, zorder=3)
        ty -= 2.55


def arrow(ax, p, q, color=INK2, style="-|>", lw=1.0, rad=0.0, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, color=color, lw=lw, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}", ls=(0, (3, 2)) if dashed else "-",
                                 shrinkA=0, shrinkB=0))


def band(ax, y0, y1, label):
    ax.add_patch(FancyBboxPatch((0.4, y0), 99.2, y1 - y0, boxstyle="round,pad=0,rounding_size=1.0", fc="white",
                                ec=GRID, lw=0.8, zorder=0))
    ax.text(1.4, y1 - 1.2, label, ha="left", va="top", fontsize=7.6, fontweight="bold", color=INK, zorder=3)


def main():
    V = values()
    g = V["gpu"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(7.16, 5.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 78)
    ax.axis("off")

    # ---------------- band 1: offline ----------------
    band(ax, 52.5, 77.5, "1  Offline: run every path once, cache, label")
    box(ax, 1.5, 63.5, 13.0, 9.0, NEUTRAL, "Queries", [f"{V['n_q']:,} questions", "DocVQA + InfoVQA"])
    box(ax, 1.5, 53.4, 13.0, 9.4, NEUTRAL, "Pages", [f"{V['n_pages']:,} page images"])
    box(ax, 17.0, 63.5, 15.0, 9.0, TEXT, "OCR + BM25", ["docTR OCR (offline)", "BM25 index (CPU)"])
    box(ax, 17.0, 53.4, 15.0, 9.4, VISUAL, "ColPali v1.3", [f"{V['vec']:,} vectors/page", "index on GPU"])
    arrow(ax, (14.5, 57.5), (17.0, 57.5))
    arrow(ax, (14.5, 59.0), (17.0, 66.5))

    ax.text(52.0, 73.9, f"Three paths, one generator ({V['gen']}, greedy)", ha="center", va="center",
            fontsize=6.9, fontweight="bold", color=INK)
    rows = (("Text", "top-%d BM25 pages, OCR text" % V["k_text"], g["text"], TEXT, 66.6),
            ("Hybrid", "both contexts", g["hybrid"], HYBRID, 60.1),
            ("Visual", "top-1 ColPali page image", g["visual"], VISUAL, 53.6))
    for name, desc, cost, col, y in rows:
        box(ax, 35.0, y, 33.5, 5.6, col, None, [])
        ax.text(36.2, y + 2.8, name, ha="left", va="center", fontsize=6.8, fontweight="bold", color=INK, zorder=3)
        ax.text(43.2, y + 2.8, desc, ha="left", va="center", fontsize=6.2, color=INK2, zorder=3)
        ax.text(67.7, y + 2.8, f"{cost:.2f} GPU-s", ha="right", va="center", fontsize=6.2, color=INK, zorder=3)
    arrow(ax, (32.0, 69.4), (35.0, 69.4))      # OCR -> text
    arrow(ax, (32.0, 65.0), (35.0, 63.6))      # OCR -> hybrid
    arrow(ax, (32.0, 59.5), (35.0, 61.6))      # ColPali -> hybrid
    arrow(ax, (32.0, 56.4), (35.0, 56.4))      # ColPali -> visual

    box(ax, 71.0, 54.0, 13.5, 18.5, NEUTRAL, "Cache", ["per query × path:", "ANLS, exact match,", "GPU-s, energy,",
                                                        "latency,", "answer log-probs"])
    for y in (69.4, 62.9, 56.4):
        arrow(ax, (68.5, y), (71.0, y))
    box(ax, 86.5, 54.0, 12.5, 18.5, NEUTRAL, "Outcome labels",
        ["$y^\\ast = \\arg\\max_p$", "$Q(x,p) - \\lambda\\,\\bar{c}_p$", "", "quality targets", "$Q(x,p)$ = ANLS"])
    arrow(ax, (84.5, 63.25), (86.5, 63.25))

    # training targets flow down to both routers through one connector in the gap between bands 1 and 2
    yc = 50.9
    ax.plot([77.75, 77.75], [54.0, yc], color=INK2, lw=1.0, zorder=1)
    ax.plot([92.75, 92.75], [54.0, yc], color=INK2, lw=1.0, zorder=1)
    ax.plot([36.0, 92.75], [yc, yc], color=INK2, lw=1.0, zorder=1)
    arrow(ax, (36.0, yc), (36.0, 46.0))
    arrow(ax, (60.0, yc), (60.0, 46.0))
    ax.text(48.0, yc, "targets: ANLS per path, labels, $\\bar{c}_p$", ha="center", va="center", fontsize=5.9,
            color=INK2, zorder=3, bbox=dict(fc="white", ec="none", pad=0.6))

    # ---------------- band 2: training ----------------
    band(ax, 28.0, 49.6, "2  Router training")
    ax.text(98.6, 48.4, f"{V['seeds']} page-grouped splits · λ chosen on validation", ha="right", va="top",
            fontsize=6.0, color=INK2, zorder=3)
    box(ax, 1.5, 29.5, 40.0, 16.5, CAVR, "CAVR (multi-task)", [], align="left")
    box(ax, 3.0, 31.0, 11.0, 11.0, ("white", CAVR[1]), None, [], lw=0.8)
    ax.text(8.5, 39.4, "question", ha="center", va="center", fontsize=6.3, color=INK, zorder=3)
    ax.text(8.5, 36.0, "mpnet", ha="center", va="center", fontsize=6.3, color=INK2, zorder=3)
    ax.text(8.5, 33.6, "embedding", ha="center", va="center", fontsize=6.3, color=INK2, zorder=3)
    box(ax, 16.5, 31.0, 8.5, 11.0, ("white", CAVR[1]), None, [], lw=0.8)
    ax.text(20.75, 36.5, "MLP\ntrunk", ha="center", va="center", fontsize=6.3, color=INK2, zorder=3, linespacing=1.3)
    arrow(ax, (14.0, 36.5), (16.5, 36.5), color=CAVR[1])
    for i, (lab, yy) in enumerate((("path logits (aux.)", 40.0), ("quality $\\hat{q}_p$", 36.5), ("confidence", 33.0))):
        box(ax, 27.5, yy - 1.5, 12.5, 3.0, ("white", CAVR[1]), None, [], lw=0.8)
        ax.text(33.75, yy, lab, ha="center", va="center", fontsize=6.2, color=INK if i == 1 else INK2, zorder=3,
                fontweight="bold" if i == 1 else "normal")
        arrow(ax, (25.0, 36.5), (27.5, yy), color=CAVR[1])

    box(ax, 43.5, 29.5, 55.5, 16.5, CE, "CAVR-CE (content-aware)", [], align="left")
    box(ax, 45.0, 37.3, 26.5, 4.7, ("white", CE[1]), None, [], lw=0.8)
    ax.text(58.25, 39.65, "question + best OCR passages", ha="center", va="center", fontsize=6.2, color=INK, zorder=3)
    box(ax, 45.0, 31.0, 26.5, 5.6, ("white", CE[1]), None, [], lw=0.8)
    ax.text(58.25, 34.85, "retrieval features: scores, margins,", ha="center", va="center", fontsize=5.8, color=INK2, zorder=3)
    ax.text(58.25, 32.75, "agreement, overlap, ColPali token matches", ha="center", va="center", fontsize=5.8, color=INK2, zorder=3)
    box(ax, 73.5, 37.3, 12.0, 4.7, ("white", CE[1]), None, [], lw=0.8)
    ax.text(79.5, 39.65, "mpnet\ncross-encoder", ha="center", va="center", fontsize=5.9, color=INK2, zorder=3, linespacing=1.1)
    arrow(ax, (71.5, 39.65), (73.5, 39.65), color=CE[1])
    box(ax, 87.0, 33.5, 10.5, 5.5, ("white", CE[1]), None, [], lw=0.8)
    ax.text(92.25, 36.25, "quality $\\hat{q}_p$", ha="center", va="center", fontsize=6.4, color=INK, fontweight="bold", zorder=3)
    arrow(ax, (85.5, 39.65), (87.0, 37.5), color=CE[1])
    arrow(ax, (71.5, 33.8), (87.0, 35.0), color=CE[1])
    ax.text(92.25, 31.6, "passages from BM25", ha="center", va="center", fontsize=5.5, color=INK2, zorder=3)
    ax.text(92.25, 30.4, "and ColPali top pages", ha="center", va="center", fontsize=5.5, color=INK2, zorder=3)

    # ---------------- band 3: online serving ----------------
    band(ax, 0.5, 26.0, "3  Online serving on a GPU pool")
    box(ax, 1.5, 8.0, 9.0, 13.0, NEUTRAL, "Query", ["text", "question"])
    box(ax, 13.0, 8.0, 16.0, 13.0, NEUTRAL, "Retrieval", ["BM25 (CPU)", "ColPali scores (GPU)", "for every query"])
    box(ax, 31.5, 8.0, 20.0, 13.0, CE, "Route", ["$p = \\arg\\max_p\\, \\hat{q}_p - \\lambda\\,\\bar{c}_p$",
                                                  "λ set at serving time", f"CAVR-CE: {V['ce_ms']:.1f} ms / query"])
    box(ax, 54.0, 8.0, 21.5, 13.0, HYBRID, "Escalate (optional)", ["calibrated P(good) < τ → hybrid",
                                                                    "pre-generation re-route,", "serial, or speculative"])
    box(ax, 78.0, 8.0, 20.5, 13.0, NEUTRAL, "Generate", ["per-path batching queues",
                                                         f"{V['gen'].split('-Instruct')[0]} replicas", "→ answer"])
    for a, b in ((10.5, 13.0), (29.0, 31.5), (51.5, 54.0), (75.5, 78.0)):
        arrow(ax, (a, 14.5), (b, 14.5))
    box(ax, 13.0, 1.5, 38.5, 5.0, NEUTRAL, None, [], dashed=True)
    ax.text(32.25, 4.0, "index: GPU-resident → sharded, host-streamed or pooled",
            ha="center", va="center", fontsize=5.8, color=INK2, zorder=3)
    box(ax, 54.0, 1.5, 44.5, 5.0, NEUTRAL, None, [], dashed=True)
    ax.text(76.25, 4.0, "design option, not evaluated: raise λ when queues grow (load shedding)",
            ha="center", va="center", fontsize=5.8, color=INK2, style="italic", zorder=3)
    arrow(ax, (21.0, 8.0), (21.0, 6.5), style="-", dashed=True)
    arrow(ax, (42.5, 29.5), (42.5, 21.0), color=CE[1], dashed=True)
    ax.text(43.4, 24.8, "trained router", ha="left", va="center", fontsize=5.8, color=CE[1], zorder=3)

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "method.pdf")
    fig.savefig(FIG / "method.png", dpi=300)
    plt.close(fig)
    print("figure method.pdf / method.png")


if __name__ == "__main__":
    main()
