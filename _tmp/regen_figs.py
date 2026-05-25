"""Regenerate fig5, fig6, fig7 from the corrected exp5_summary.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DEFAULT_CHUNK_SIZE = 128
DEFAULT_K = 5

PALETTE = {
    "bm25": "#4878CF", "tfidf": "#6ACC65", "dense": "#D65F5F",
    "rag": "#4878CF", "no_rag": "#AAAAAA",
    "concise": "#E29C3F", "instructed": "#4878CF",
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _bar_with_ci(ax, labels, means, lows, highs, colors, title, ylabel="Score"):
    x = np.arange(len(labels))
    width = 0.5
    bars = ax.bar(x, means, width, color=colors, alpha=0.85, zorder=3)
    yerr_lo = [m - l for m, l in zip(means, lows)]
    yerr_hi = [h - m for m, h in zip(means, highs)]
    ax.errorbar(x, means, yerr=[yerr_lo, yerr_hi], fmt="none",
                color="black", capsize=5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(means) * 1.4 + 0.05))
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)


def fig5(data):
    conditions = ["no_rag", "rag"]
    labels = ["No-RAG\n(parametric)", "RAG\n(dense, k=5)"]
    colors = [PALETTE[c] for c in conditions]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    for ax, metric, full_name in zip(axes, ["em", "f1"], ["Exact Match", "Token F1"]):
        means = [data[c][metric]["mean"] for c in conditions]
        lows = [data[c][metric]["lo"] for c in conditions]
        highs = [data[c][metric]["hi"] for c in conditions]
        _bar_with_ci(ax, labels, means, lows, highs, colors, title=full_name, ylabel=full_name)
    fig.suptitle("Experiment 5 — RAG vs No-RAG Baseline\n"
                 f"(Dense, chunk={DEFAULT_CHUNK_SIZE}, k={DEFAULT_K})",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig5_rag_vs_no_rag.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Wrote fig5_rag_vs_no_rag.png")


def fig6(data):
    delta = data.get("delta_analysis", [])
    helps = data["delta_counts"]["helps"]
    hurts = data["delta_counts"]["hurts"]
    ties = data["delta_counts"]["ties"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    wedge_labels = [f"RAG helps\n({helps})", f"Ties\n({ties})", f"RAG hurts\n({hurts})"]
    wedge_sizes = [helps, ties, hurts]
    wedge_colors = ["#4878CF", "#AAAAAA", "#D65F5F"]
    ax.pie(wedge_sizes, labels=wedge_labels, colors=wedge_colors,
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax.set_title("When Does RAG Help vs Hurt?\n(per-question EM comparison)",
                 fontsize=12, fontweight="bold")

    ax = axes[1]
    recalls = [d["recall"] for d in delta]
    rag_ems = [d["rag_em"] for d in delta]
    helps_mask = [d["rag_helps"] for d in delta]
    hurts_mask = [d["rag_hurts"] for d in delta]
    ties_mask = [1 - d["rag_helps"] - d["rag_hurts"] for d in delta]

    rng = np.random.default_rng(42)
    jitter_x = rng.uniform(-0.02, 0.02, len(delta))
    jitter_y = rng.uniform(-0.02, 0.02, len(delta))
    recalls_j = np.array(recalls, dtype=float) + jitter_x
    rag_ems_j = np.array(rag_ems, dtype=float) + jitter_y

    for mask, color, label in [
        (helps_mask, "#4878CF", "RAG helps"),
        (ties_mask, "#AAAAAA", "Tie"),
        (hurts_mask, "#D65F5F", "RAG hurts"),
    ]:
        idx = [i for i, m in enumerate(mask) if m]
        if idx:
            ax.scatter(recalls_j[idx], rag_ems_j[idx],
                       c=color, alpha=0.5, s=25, label=label)

    ax.set_xlabel("Retrieval Recall (answer in retrieved chunks?)", fontsize=10)
    ax.set_ylabel("RAG Exact Match", fontsize=10)
    ax.set_title("Retrieval Recall vs RAG Answer Quality",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig6_error_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Wrote fig6_error_analysis.png")


def fig7(data):
    delta = data.get("delta_analysis", [])
    helps_cases = [d for d in delta if d["rag_helps"] == 1 and d["recall"] == 1.0][:3]
    hurts_cases = [d for d in delta if d["rag_hurts"] == 1][:3]

    fig, ax = plt.subplots(
        figsize=(14, max(4, (len(helps_cases) + len(hurts_cases)) * 1.2))
    )
    ax.axis("off")

    col_labels = ["Category", "Question (truncated)", "Gold",
                  "No-RAG pred", "RAG pred"]
    rows = []
    for d in helps_cases:
        rows.append([
            "RAG helps",
            d["question"][:55] + ("…" if len(d["question"]) > 55 else ""),
            str(d["gold"][0])[:25],
            str(d["no_rag_pred"])[:25],
            str(d["rag_pred"])[:25],
        ])
    for d in hurts_cases:
        rows.append([
            "RAG hurts",
            d["question"][:55] + ("…" if len(d["question"]) > 55 else ""),
            str(d["gold"][0])[:25],
            str(d["no_rag_pred"])[:25],
            str(d["rag_pred"])[:25],
        ])

    if not rows:
        print("Not enough examples for qualitative figure — skipping.")
        plt.close()
        return

    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DDDDDD")
            cell.set_text_props(fontweight="bold")
        elif row <= len(helps_cases):
            cell.set_facecolor("#EBF4FF")
        else:
            cell.set_facecolor("#FFF0EE")

    ax.set_title("Experiment 5 — Qualitative Examples: When RAG Helps vs Hurts",
                 fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig7_qualitative.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Wrote fig7_qualitative.png")


def main():
    with open(RESULTS / "exp5_summary.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    fig5(data)
    fig6(data)
    fig7(data)


if __name__ == "__main__":
    main()
