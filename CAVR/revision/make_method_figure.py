"""Method figure: the CAVR pipeline as built (offline cache and labels, router training, online serving).

Numbers in the figure are read from results/*.json. Writes paper_out/figures/method.pdf (vector, for the paper)
and method.png (preview). Colours follow the other figures: text = yellow, visual = orange, hybrid = aqua,
CAVR = blue, CAVR-CE = violet. Cost bars are drawn to scale (fraction of the hybrid path's mean GPU-seconds).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

from common import DATA, RESULTS, ROOT  # noqa: E402

FIG = ROOT / "paper_out" / "figures"
INK, INK2, RULE = "#0b0b0b", "#52514e", "#b9b8b2"
NEUTRAL = ("#f4f3f0", "#8a8984", "#dcdbd6")      # body fill, edge, header fill
TEXT = ("#fdf3dc", "#c98500", "#f7d98f")
VISUAL = ("#fce6dc", "#d95926", "#f4b89c")
HYBRID = ("#ddf3ea", "#199e70", "#a6dcc5")
CAVR = ("#e3eefa", "#2a78d6", "#a9c9f0")
CE = ("#e8e4f6", "#4a3aa7", "#c3bbeb")
ENC = ("#f6b83c", "#b07a00")                     # encoder trapezoids
ARROW = "#3f6fd1"
FORMULA = ("#fff6dd", "#c98500")


def values():
    rows = [json.loads(l) for l in open(RESULTS / "path_outputs.jsonl", encoding="utf-8")]
    gpu = {p: sum(r["gpu_seconds"] for r in rows if r["path"] == p) / sum(r["path"] == p for r in rows)
           for p in ("text", "visual", "hybrid")}
    man = json.load(open(DATA / "manifest.json"))
    meta = json.load(open(RESULTS / "path_outputs_meta.json"))
    ix = json.load(open(RESULTS / "index_scaling.json"))
    ce = json.load(open(RESULTS / "router_ce_cost.json"))
    rc = json.load(open(RESULTS / "router_cost.json"))
    ev = json.load(open(RESULTS / "eval_main.json"))
    return {"gpu": gpu, "n_q": man["natural"]["queries"], "n_pages": man["pages"]["total"],
            "vec": ix["vectors_per_page"], "k_text": meta["k_text"], "gen": meta["model"].split("/")[-1],
            "ce_ms": 1000 * ce["per_query_batch1"]["latency_s"], "cavr_ms": 1000 * rc["per_query_batch1"]["latency_s"],
            "seeds": ev["n_seeds"]}


def rbox(ax, x, y, w, h, fc, ec, lw=0.9, dashed=False, z=2, r=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=(0, (3, 2)) if dashed else "-", zorder=z))


def hbox(ax, x, y, w, h, colors, title, hh=2.6, title_size=6.9):
    """Box with a coloured header strip carrying the title."""
    body, edge, head = colors
    rbox(ax, x, y, w, h, body, edge)
    rbox(ax, x + 0.5, y + h - hh - 0.5, w - 1.0, hh, head, edge, lw=0.7, z=3, r=0.5)
    ax.text(x + w / 2, y + h - hh / 2 - 0.5, title, ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=INK, zorder=4)


def txt(ax, x, y, s, size=5.9, color=INK2, weight="normal", ha="center", style="normal", z=4):
    return ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, fontweight=weight, style=style, zorder=z)


def pill(ax, x, y, w, h, s, ec, size=5.6, weight="normal", fc="white"):
    rbox(ax, x, y, w, h, fc, ec, lw=0.7, z=4, r=0.6)
    txt(ax, x + w / 2, y + h / 2, s, size=size, color=INK, weight=weight, z=5)


def arrow(ax, p, q, color=ARROW, lw=1.0, rad=0.0, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, zorder=6,
                                 connectionstyle=f"arc3,rad={rad}", ls=(0, (3, 2)) if dashed else "-",
                                 shrinkA=0, shrinkB=0))


def line(ax, xs, ys, color=ARROW, lw=1.0, dashed=False):
    ax.plot(xs, ys, color=color, lw=lw, zorder=6, ls=(0, (3, 2)) if dashed else "-", solid_capstyle="butt")


def trapezoid(ax, x, y, w, h, label):
    pts = [(x, y), (x + w, y + h * 0.16), (x + w, y + h * 0.84), (x, y + h)]
    ax.add_patch(Polygon(pts, closed=True, fc=ENC[0], ec=ENC[1], lw=0.9, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, rotation=90, ha="center", va="center", fontsize=5.6, color=INK,
            fontweight="bold", zorder=4, linespacing=1.15)


def stage(ax, y, num, lines):
    ax.text(5.7, y, f"Stage {num}", ha="center", va="center", fontsize=8.2, fontweight="bold", color=INK, zorder=4)
    for i, s in enumerate(lines):
        ax.text(5.7, y - 2.7 - 2.2 * i, s, ha="center", va="center", fontsize=5.9, color=INK2, zorder=4)


def main():
    V = values()
    g = V["gpu"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(7.16, 5.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 77)
    ax.axis("off")
    ax.add_patch(Rectangle((0.2, 0.2), 99.6, 76.6, fc="white", ec=INK2, lw=0.9, zorder=0))
    for yy in (51.0, 26.0):
        line(ax, [0.2, 99.8], [yy, yy], color="#8aa6e0", lw=0.8, dashed=True)
    line(ax, [11.4, 11.4], [0.6, 76.4], color=RULE, lw=0.6, dashed=True)
    stage(ax, 66.0, 1, ["Offline: run every", "path once, cache,", "label"])
    stage(ax, 41.0, 2, ["Router", "training"])
    stage(ax, 16.5, 3, ["Online serving", "on a GPU pool"])

    # ================= Stage 1 =================
    for x, s in ((18.4, "Inputs"), (32.3, "Retrievers"), (53.0, "Three paths · one generator"),
                 (83.4, "Cache & outcome labels")):
        txt(ax, x, 75.2, s, size=7.0, color=INK, weight="bold")
    txt(ax, 53.0, 73.3, f"{V['gen']}, greedy decoding", size=5.5)

    hbox(ax, 12.6, 62.0, 11.6, 9.4, NEUTRAL, "Queries")
    txt(ax, 18.4, 66.2, f"{V['n_q']:,} questions", size=5.4)
    txt(ax, 18.4, 64.1, "DocVQA + InfoVQA", size=5.4)
    hbox(ax, 12.6, 52.3, 11.6, 8.8, NEUTRAL, "Pages")
    txt(ax, 18.4, 55.9, f"{V['n_pages']:,} page images", size=5.4)

    rbox(ax, 25.7, 51.8, 13.2, 20.0, "white", RULE, dashed=True, z=1)
    hbox(ax, 26.2, 62.2, 12.2, 9.1, TEXT, "OCR + BM25")
    txt(ax, 32.3, 66.4, "docTR OCR (offline)", size=5.4)
    txt(ax, 32.3, 64.2, "BM25 index (CPU)", size=5.4)
    hbox(ax, 26.2, 52.3, 12.2, 9.1, VISUAL, "ColPali v1.3")
    txt(ax, 32.3, 56.6, f"{V['vec']:,} vectors / page", size=5.4)
    txt(ax, 32.3, 54.4, "index on GPU", size=5.4)
    arrow(ax, (24.2, 66.8), (26.2, 66.8))
    arrow(ax, (24.2, 56.8), (26.2, 56.8))

    paths = (("Text", f"top-{V['k_text']} BM25 pages, OCR text", g["text"], TEXT, 65.6),
             ("Hybrid", "both contexts (OCR text + page image)", g["hybrid"], HYBRID, 59.0),
             ("Visual", "top-1 ColPali page image", g["visual"], VISUAL, 52.4))
    for name, desc, cost, (body, edge, head), y in paths:
        rbox(ax, 40.6, y, 25.0, 6.0, body, edge, lw=1.0)
        txt(ax, 41.6, y + 4.45, name, size=6.9, color=INK, weight="bold", ha="left")
        txt(ax, 41.6, y + 2.6, desc, size=5.4, ha="left")
        pill(ax, 57.7, y + 3.5, 7.2, 2.0, f"{cost:.2f} GPU-s", edge, size=5.4, weight="bold")
        ax.add_patch(Rectangle((41.6, y + 0.8), 16.0, 0.7, fc="white", ec=RULE, lw=0.4, zorder=4))
        ax.add_patch(Rectangle((41.6, y + 0.8), 16.0 * cost / g["hybrid"], 0.7, fc=edge, ec="none", zorder=5))
    arrow(ax, (38.4, 68.7), (40.6, 68.7))       # OCR -> text
    arrow(ax, (38.4, 64.2), (40.6, 63.0))       # OCR -> hybrid
    arrow(ax, (38.4, 59.0), (40.6, 61.0))       # ColPali -> hybrid
    arrow(ax, (38.4, 55.4), (40.6, 55.4))       # ColPali -> visual

    rbox(ax, 67.5, 51.8, 13.0, 20.0, "white", RULE, dashed=True, z=1)
    hbox(ax, 68.0, 52.3, 12.0, 19.0, NEUTRAL, "Cache")
    txt(ax, 74.0, 66.4, "per query × path:", size=5.5)
    for i, s in enumerate(("ANLS", "exact match", "GPU-s · energy", "latency", "answer log-probs")):
        pill(ax, 69.0, 63.4 - 2.25 * i, 10.0, 1.8, s, NEUTRAL[1], size=5.2)
    for y in (68.6, 62.0, 55.4):
        arrow(ax, (65.6, y), (68.0, y))

    rbox(ax, 82.0, 51.8, 17.3, 20.0, "white", RULE, dashed=True, z=1)
    hbox(ax, 82.5, 52.3, 16.3, 19.0, NEUTRAL, "Outcome labels")
    txt(ax, 90.65, 66.6, "best-path label", size=5.5)
    rbox(ax, 83.5, 60.9, 14.3, 4.6, FORMULA[0], FORMULA[1], lw=0.7, z=4)
    txt(ax, 90.65, 64.3, "$y^\\ast(x) = \\arg\\max_p$", size=6.1, color=INK, z=5)
    txt(ax, 90.65, 62.2, "$Q(x,p) - \\lambda\\,\\bar{c}_p$", size=6.1, color=INK, z=5)
    txt(ax, 90.65, 58.8, "quality targets", size=5.5)
    rbox(ax, 83.5, 53.8, 14.3, 3.4, FORMULA[0], FORMULA[1], lw=0.7, z=4)
    txt(ax, 90.65, 55.5, "$Q(x,p)$ = ANLS", size=6.1, color=INK, z=5)
    arrow(ax, (80.0, 61.8), (82.5, 61.8))

    # targets -> both routers
    yc = 49.2
    line(ax, [74.0, 74.0], [51.8, yc])
    line(ax, [90.65, 90.65], [51.8, yc])
    line(ax, [32.0, 90.65], [yc, yc])
    arrow(ax, (32.0, yc), (32.0, 47.4))
    arrow(ax, (77.0, yc), (77.0, 47.4))
    t = txt(ax, 54.5, yc, "targets: ANLS per path · labels $y^\\ast$ · mean costs $\\bar{c}_p$", size=5.5, z=7)
    t.set_bbox(dict(fc="white", ec="none", pad=0.3))

    # ================= Stage 2 =================
    rbox(ax, 12.6, 28.4, 41.0, 19.0, "white", CAVR[1], dashed=True, z=1)
    hbox(ax, 13.1, 28.9, 40.0, 18.0, CAVR, "CAVR (multi-task)")
    pill(ax, 14.0, 33.8, 7.3, 6.8, "", "#333333")
    txt(ax, 17.65, 38.4, "question $x$", size=5.8, color=INK, weight="bold")
    txt(ax, 17.65, 36.1, "+ dataset id", size=5.3)
    trapezoid(ax, 22.4, 30.6, 4.6, 12.8, "mpnet embedding\n(frozen)")
    for i in range(6):                                # e(x) vector glyph
        ax.add_patch(Rectangle((28.2, 32.9 + 1.35 * i), 1.15, 1.15,
                               fc="#ffe066" if i % 2 == 0 else "#f6c544", ec=ENC[1], lw=0.4, zorder=4))
    txt(ax, 28.8, 31.7, "$e(x)$", size=5.5, color=INK)
    rbox(ax, 31.0, 32.0, 6.8, 9.6, "#cfe0f7", CAVR[1], lw=0.9, z=4)
    txt(ax, 34.4, 38.6, "MLP", size=6.5, color=INK, weight="bold", z=5)
    txt(ax, 34.4, 36.8, "trunk", size=6.5, color=INK, weight="bold", z=5)
    txt(ax, 34.4, 34.0, "$h = f_\\theta(e)$", size=5.5, color=INK, z=5)
    heads = (("path logits (aux.)", "trained on labels $y^\\ast$", 40.4, False),
             ("quality $\\hat{q}_p$", "regresses $Q(x,p)$", 36.0, True),
             ("confidence", "escalation signal", 31.6, False))
    for title, sub, yy, main_head in heads:
        rbox(ax, 40.0, yy - 1.7, 12.4, 3.9, FORMULA[0] if main_head else "white",
             FORMULA[1] if main_head else CAVR[1], lw=1.0 if main_head else 0.8, z=4)
        txt(ax, 46.2, yy + 1.05, title, size=5.7, color=INK, weight="bold", z=5)
        txt(ax, 46.2, yy - 0.7, sub, size=4.9, z=5)
        arrow(ax, (37.8, 36.8), (40.0, yy + 0.25))
    arrow(ax, (21.3, 37.2), (22.4, 37.2))
    arrow(ax, (27.0, 37.0), (28.1, 37.0))
    arrow(ax, (29.4, 37.0), (31.0, 37.0))

    rbox(ax, 55.4, 28.4, 43.9, 19.0, "white", CE[1], dashed=True, z=1)
    hbox(ax, 55.9, 28.9, 42.9, 18.0, CE, "CAVR-CE (content-aware)")
    pill(ax, 56.9, 38.6, 17.6, 3.5, "question + OCR passages", "#333333", size=5.5, weight="bold")
    txt(ax, 65.7, 37.1, "from BM25 + ColPali top pages", size=4.7)
    rbox(ax, 56.9, 29.7, 17.6, 6.3, "#efecfa", CE[1], lw=0.8, z=4)
    txt(ax, 65.7, 34.8, "retrieval features", size=5.7, color=INK, weight="bold", z=5)
    txt(ax, 65.7, 33.0, "scores · margins · agreement", size=5.0, z=5)
    txt(ax, 65.7, 31.4, "overlap · ColPali token matches", size=5.0, z=5)
    trapezoid(ax, 76.3, 31.0, 4.4, 13.2, "cross-encoder\n(mpnet, tuned)")
    arrow(ax, (74.5, 40.35), (76.3, 40.35))
    ax.add_patch(Circle((83.5, 36.4), 1.25, fc="white", ec=ARROW, lw=1.0, zorder=6))
    txt(ax, 83.5, 36.45, "+", size=8.5, color=ARROW, weight="bold", z=7)
    txt(ax, 83.5, 39.0, "concat", size=4.8, z=7)
    line(ax, [80.7, 82.3], [36.4, 36.4])
    line(ax, [74.5, 83.5], [32.8, 32.8])
    arrow(ax, (83.5, 32.8), (83.5, 35.15))
    rbox(ax, 86.0, 34.4, 4.8, 4.0, "#efecfa", CE[1], lw=0.8, z=4)
    txt(ax, 88.4, 37.0, "MLP", size=5.3, color=INK, weight="bold", z=5)
    txt(ax, 88.4, 35.4, "head", size=5.3, color=INK, weight="bold", z=5)
    arrow(ax, (84.75, 36.4), (86.0, 36.4))
    rbox(ax, 92.0, 33.7, 6.2, 5.4, FORMULA[0], FORMULA[1], lw=1.0, z=4)
    txt(ax, 95.1, 37.4, "quality", size=5.8, color=INK, weight="bold", z=5)
    txt(ax, 95.1, 35.3, "$\\hat{q}_p$", size=7.0, color=INK, z=5)
    arrow(ax, (90.8, 36.4), (92.0, 36.4))
    rbox(ax, 36.0, 26.6, 37.0, 1.9, "white", RULE, lw=0.6, dashed=True, z=7)
    txt(ax, 54.5, 27.55, f"protocol: {V['seeds']} page-grouped splits · λ chosen on validation", size=5.4, z=8)

    # ================= Stage 3 =================
    y0, h = 8.8, 12.6
    hbox(ax, 12.6, y0, 7.6, h, NEUTRAL, "Query")
    txt(ax, 16.4, 15.6, "text", size=5.7)
    txt(ax, 16.4, 13.6, "question $x$", size=5.7)
    hbox(ax, 22.0, y0, 14.6, h, CAVR, "Retrieval")
    txt(ax, 29.3, 16.3, "BM25 (CPU)", size=5.8)
    txt(ax, 29.3, 14.2, "ColPali scores (GPU)", size=5.8)
    txt(ax, 29.3, 12.0, "every query for CAVR-CE", size=4.9, style="italic")
    hbox(ax, 38.4, y0, 17.2, h, CE, "Route")
    rbox(ax, 39.3, 14.3, 15.4, 2.9, FORMULA[0], FORMULA[1], lw=0.7, z=4)
    txt(ax, 47.0, 15.75, "$p^\\ast = \\arg\\max_p\\, \\hat{q}_p - \\lambda\\,\\bar{c}_p$", size=5.8, color=INK, z=5)
    txt(ax, 47.0, 12.9, "λ set at serving time", size=5.5)
    txt(ax, 47.0, 10.8, f"router: CE {V['ce_ms']:.1f} ms · CAVR {V['cavr_ms']:.1f} ms", size=4.9)
    hbox(ax, 57.4, y0, 18.6, h, HYBRID, "Escalate (optional)")
    rbox(ax, 58.3, 15.0, 16.8, 2.5, FORMULA[0], FORMULA[1], lw=0.7, z=4)
    txt(ax, 66.7, 16.25, "calibrated $P$(good) < τ → hybrid", size=5.0, color=INK, z=5)
    for i, s in enumerate(("pre-generation re-route", "serial", "speculative")):
        pill(ax, 59.6, 12.4 - 1.72 * i, 14.2, 1.42, s, HYBRID[1], size=4.9)
    hbox(ax, 77.8, y0, 20.9, h, NEUTRAL, "Generate")
    txt(ax, 88.25, 16.3, "per-path batching queues", size=5.7)
    txt(ax, 88.25, 14.2, f"{V['gen'].split('-Instruct')[0]} replicas", size=5.7)
    txt(ax, 88.25, 11.9, "→ answer", size=6.3, color=INK, weight="bold")
    for a, b in ((20.2, 22.0), (36.6, 38.4), (55.6, 57.4), (76.0, 77.8)):
        arrow(ax, (a, 15.1), (b, 15.1))
    # serial / speculative escalation feeds back from generation
    arrow(ax, (86.0, 21.4), (70.0, 21.4), color=HYBRID[1], dashed=True, rad=0.28)
    txt(ax, 88.6, 25.25, "answer log-probs (serial / speculative)", size=4.9, color=HYBRID[1], z=7)
    # trained routers are deployed into Route
    line(ax, [33.0, 33.0], [28.4, 23.6], color=CE[1], dashed=True)
    line(ax, [33.0, 45.0], [23.6, 23.6], color=CE[1], dashed=True)
    arrow(ax, (45.0, 23.6), (45.0, 21.4), color=CE[1], dashed=True)
    line(ax, [75.0, 75.0], [28.4, 23.6], color=CE[1], dashed=True)
    line(ax, [75.0, 49.0], [23.6, 23.6], color=CE[1], dashed=True)
    arrow(ax, (49.0, 23.6), (49.0, 21.4), color=CE[1], dashed=True)
    txt(ax, 47.0, 24.6, "trained router (either)", size=5.0, color=CE[1], z=7)

    rbox(ax, 22.0, 1.3, 33.6, 5.0, "#f4f3f0", NEUTRAL[1], dashed=True)
    txt(ax, 38.8, 3.8, "index: GPU-resident → sharded, host-streamed or pooled", size=5.4)
    line(ax, [29.3, 29.3], [8.8, 6.3], color=NEUTRAL[1], dashed=True)
    rbox(ax, 57.4, 1.3, 41.3, 5.0, "#f4f3f0", NEUTRAL[1], dashed=True)
    txt(ax, 78.05, 3.8, "design option, not evaluated: raise λ when queues grow (load shedding)", size=5.3,
        style="italic")
    line(ax, [88.25, 88.25], [8.8, 6.3], color=NEUTRAL[1], dashed=True)

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "method.pdf")
    fig.savefig(FIG / "method.png", dpi=300)
    plt.close(fig)
    print("figure method.pdf / method.png")


if __name__ == "__main__":
    main()
