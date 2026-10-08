"""Training and validation loss figure for CAVR and CAVR-CE, from results/training_curves.json.

(a) CAVR: weighted multi-task loss per split (training solid, validation dashed); the dot marks the epoch whose
    weights are kept (best validation utility; early stopping with patience 10).
(b) CAVR-CE: binary cross-entropy against ANLS targets, mean +- std over splits.
(c) CAVR-CE: validation success AUC per epoch (the model-selection criterion), mean +- std over splits.
Colour = router (CAVR blue, CAVR-CE violet, as in the other figures); line style = training vs validation.
Writes paper_out/figures/training_curves.pdf and .png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import RESULTS, ROOT  # noqa: E402

FIG = ROOT / "paper_out" / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, VIOLET = "#2a78d6", "#4a3aa7"


def style(ax, title):
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def main():
    D = json.load(open(RESULTS / "training_curves.json"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), gridspec_kw={"width_ratios": [1.35, 1, 1]})

    # (a) CAVR, one line pair per split
    ax = axes[0]
    for i, r in enumerate(D["cavr"]):
        ep = np.arange(1, len(r["train_total"]) + 1)
        ax.plot(ep, r["train_total"], color=BLUE, lw=1.1, alpha=0.75, label="training" if i == 0 else None)
        ax.plot(ep, r["val_total"], color=BLUE, lw=1.1, alpha=0.75, ls=(0, (3, 2)),
                label="validation" if i == 0 else None)
        b = r["best_epoch"]
        ax.plot(b + 1, r["val_total"][b], "o", ms=4, color=BLUE, mec="white", mew=0.8, zorder=5,
                label="kept epoch" if i == 0 else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("multi-task loss")
    style(ax, f"(a) CAVR ({len(D['cavr'])} splits)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")

    # (b) CAVR-CE loss, mean +- std over splits
    ax = axes[1]
    tr = np.array([r["train_bce"] for r in D["cavr_ce"]])
    va = np.array([r["val_bce"] for r in D["cavr_ce"]])
    ep = np.arange(1, tr.shape[1] + 1)
    for arr, ls, lab in ((tr, "-", "training"), (va, (0, (3, 2)), "validation")):
        m, s = arr.mean(0), arr.std(0)
        ax.fill_between(ep, m - s, m + s, color=VIOLET, alpha=0.12, lw=0)
        ax.plot(ep, m, color=VIOLET, lw=1.4, ls=ls, marker="o", ms=3, label=lab)
    best = [r["best_epoch"] + 1 for r in D["cavr_ce"]]
    ax.set_xticks(ep)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE vs ANLS target")
    style(ax, "(b) CAVR-CE loss")
    ax.legend(fontsize=6.5, frameon=False, loc="center right")

    # (c) CAVR-CE validation AUC, mean +- std
    ax = axes[2]
    auc = np.array([r["val_auc"] for r in D["cavr_ce"]])
    m, s = auc.mean(0), auc.std(0)
    ax.fill_between(ep, m - s, m + s, color=VIOLET, alpha=0.12, lw=0)
    ax.plot(ep, m, color=VIOLET, lw=1.4, marker="o", ms=3)
    kept = int(np.bincount(best).argmax())
    ax.axvline(kept, color=INK2, lw=0.8, ls=":")
    ax.text(kept + 0.1, m.min(), f" kept epoch\n (most splits: {kept})", fontsize=6.2, color=INK2, va="bottom")
    ax.set_xticks(ep)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation success AUC")
    style(ax, "(c) CAVR-CE selection")

    fig.tight_layout(w_pad=1.2)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "training_curves.pdf")
    fig.savefig(FIG / "training_curves.png", dpi=300)
    plt.close(fig)
    print("figure training_curves.pdf / training_curves.png")


if __name__ == "__main__":
    main()
