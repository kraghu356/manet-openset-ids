"""Figures for the revised manuscript, from results/*.json only.

    python3 analysis/make_figures.py      # writes figures/*.pdf and *.png (600 dpi)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b",
                     "axes.labelcolor": "#0b0b0b", "pdf.fonttype": 42})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
ATT = ["bha", "flood", "gha", "sink", "worm"]
ATT_LBL = ["Black hole", "Flooding", "Grey hole", "Sinkhole", "Wormhole"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig_protocol():
    """Evaluation protocol: seed folds and leave-one-attack-out removal."""
    fig, ax = plt.subplots(figsize=(3.4, 1.75)); ax.set_xlim(0, 20); ax.set_ylim(1.0, 7.6); ax.axis("off")
    cols = {"train": "#86b6ef", "val": "#eda100", "test": "#e34948"}
    folds = [(range(1, 13), range(13, 17), range(17, 21)), (range(5, 17), range(17, 21), range(1, 5)),
             (range(9, 21), range(1, 5), range(5, 9)), (list(range(1, 5)) + list(range(13, 21)), range(5, 9), range(9, 13))]
    for fi, (tr, va, te) in enumerate(folds):
        y = 6.2 - fi * 1.05
        ax.text(-0.2, y + 0.35, f"Fold {fi + 1}", ha="right", va="center", fontsize=7, color=INK)
        for s in range(1, 21):
            part = "train" if s in tr else "val" if s in va else "test"
            ax.add_patch(plt.Rectangle((s - 1 + 0.06, y), 0.88, 0.7, color=cols[part], lw=0))
    for s in (1, 5, 9, 13, 17, 20):
        ax.text(s - 0.5, 7.15, str(s), ha="center", fontsize=6, color=INK2)
    ax.text(10, 7.55, "Simulation seed (one run per seed and scenario)", ha="center", fontsize=6.5, color=INK2)
    for i, (k, lab) in enumerate([("train", "Training"), ("val", "Validation"), ("test", "Test")]):
        ax.add_patch(plt.Rectangle((0.2 + i * 6.6, 1.95), 0.7, 0.5, color=cols[k], lw=0))
        ax.text(1.1 + i * 6.6, 2.2, lab, va="center", fontsize=7)
    ax.text(10, 1.25, "Held-out attack h: every run of scenario h removed from training and validation",
            ha="center", fontsize=6.2, color=INK2)
    save(fig, "fig_protocol")


def fig_auroc(r):
    rows = [("msp_gru", "Softmax confidence"), ("energy_gru", "Energy score"), ("openmax_gru", "OpenMax"),
            ("rf_vote", "Forest vote disagreement"), ("svm_margin", "SVM margin"), ("knn_dist", "k-NN distance"),
            ("autoencoder", "Reconstruction error"), ("nu", "Unheard neighbours ν"), ("hybrid", "Hybrid (Eq. 9)")]
    M = np.array([[r["open_set"][a][m]["auroc"]["mean"] if r["open_set"][a][m]["auroc"]["mean"] is not None
                   else np.nan for a in ATT] for m, _ in rows])
    cmap = LinearSegmentedColormap.from_list("div", ["#a32a2a", "#e34948", "#f0efec", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    show = np.where(np.isnan(M), 0.5, M)
    ax.imshow(show, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; t = "n/a" if np.isnan(v) else f"{v:.2f}"
            dark = (not np.isnan(v)) and (v < 0.2 or v > 0.8)
            ax.text(j, i, t, ha="center", va="center", fontsize=6.8, color="white" if dark else INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(ATT_LBL, fontsize=6.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l in rows], fontsize=6.8)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True); ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    ax.axhline(2.5, color=INK2, lw=0.8); ax.axhline(6.5, color=INK2, lw=0.8)
    save(fig, "fig_auroc")


def fig_observability(r):
    ab = r["autoencoder_ablation"]
    sets = [("F_all", "All 17 inputs", "#2a78d6", ""), ("F_observable", "Observable 15", "#eb6834", "////"),
            ("F_observable_minus_nu", "Observable without ν", "#1baf7a", "....")]
    fig, ax = plt.subplots(figsize=(3.4, 1.9)); w = 0.26; x = np.arange(5)
    for k, (key, lab, c, hatch) in enumerate(sets):
        v = [ab[key][a]["mean"] for a in ATT]; sd = [ab[key][a]["sd"] for a in ATT]
        b = ax.bar(x + (k - 1) * w, v, w - 0.03, color=c, edgecolor="white", lw=0.4, hatch=hatch, label=lab,
                   yerr=sd, error_kw=dict(lw=0.5, capsize=1.2, ecolor=INK2))
    ax.axhline(0.5, color=INK2, lw=0.6, ls="--"); ax.text(4.62, 0.52, "chance", fontsize=5.5, color=INK2, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(ATT_LBL, fontsize=6.8); ax.set_ylim(0, 1.08); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylabel("AUROC, held-out attack", fontsize=7); ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), handlelength=1.4)
    save(fig, "fig_observability")


if __name__ == "__main__":
    r = json.load(open(os.path.join(R, "openset_results.json")))
    fig_protocol(); fig_auroc(r); fig_observability(r)
    print("figures written to", os.path.abspath(OUT))
