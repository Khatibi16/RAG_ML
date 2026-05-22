"""
plot_results.py — Generate all figures from experiment result JSON files.

Figures produced
----------------
  fig1_retriever_comparison.png   — EM + F1 bar chart, BM25 / TF-IDF / Dense
  fig2_chunk_size.png             — EM / F1 / Recall@k vs chunk size (line)
  fig3_k_values.png               — EM / F1 / Recall@k vs k  (line)
  fig4_prompt_template.png        — EM + F1 bar chart: concise vs instructed
  fig5_rag_vs_no_rag.png          — EM + F1 grouped bar: RAG vs baseline
  fig6_error_analysis.png         — Scatter: recall→EM; win/loss/tie counts
  fig7_qualitative.png            — Example table of RAG-helps vs RAG-hurts cases

Usage
-----
    python analysis/plot_results.py

All figures are saved to config.FIGURES_DIR as 300 dpi PNG files.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("plot_results")

# ─────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────

PALETTE = {
    "bm25":       "#4878CF",
    "tfidf":      "#6ACC65",
    "dense":      "#D65F5F",
    "rag":        "#4878CF",
    "no_rag":     "#AAAAAA",
    "concise":    "#E29C3F",
    "instructed": "#4878CF",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


def load(name: str) -> Optional[dict]:
    p = config.RESULTS_DIR / f"{name}.json"
    if not p.exists():
        logger.warning("Results file not found: %s  (skip)", p)
        return None
    with open(p) as f:
        return json.load(f)


def savefig(name: str) -> None:
    out = config.FIGURES_DIR / name
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out)


def _bar_with_ci(ax, labels, means, lows, highs, colors, title, ylabel="Score"):
    x     = np.arange(len(labels))
    width = 0.5
    bars  = ax.bar(x, means, width, color=colors, alpha=0.85, zorder=3)
    # 95 % CI error bars
    yerr_lo = [m - l for m, l in zip(means, lows)]
    yerr_hi = [h - m for m, h in zip(means, highs)]
    ax.errorbar(
        x, means,
        yerr=[yerr_lo, yerr_hi],
        fmt="none", color="black", capsize=5, linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(means) * 1.4 + 0.05))
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    # Value labels
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9,
        )


# ─────────────────────────────────────────────────────────────────
# Figure 1 — Retriever comparison
# ─────────────────────────────────────────────────────────────────

def fig1_retriever_comparison():
    data = load("exp1_summary")
    if data is None:
        return
    results = data["results"]

    rtypes  = list(results.keys())
    colors  = [PALETTE.get(r, "#888888") for r in rtypes]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, metric, full_name in zip(axes, ["em", "f1"], ["Exact Match (EM)", "Token F1"]):
        means  = [results[r][metric]["mean"] for r in rtypes]
        lows   = [results[r][metric]["lo"]   for r in rtypes]
        highs  = [results[r][metric]["hi"]   for r in rtypes]
        labels = [r.upper() for r in rtypes]
        _bar_with_ci(ax, labels, means, lows, highs, colors,
                     title=full_name, ylabel=full_name)

    fig.suptitle(
        "Experiment 1 — Retriever Comparison\n"
        f"(chunk={config.DEFAULT_CHUNK_SIZE} words, k={config.DEFAULT_K})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    savefig("fig1_retriever_comparison.png")


# ─────────────────────────────────────────────────────────────────
# Figure 2 — Chunk size
# ─────────────────────────────────────────────────────────────────

def fig2_chunk_size():
    data = load("exp2_summary")
    if data is None:
        return
    results = data["results"]

    sizes   = sorted(int(k) for k in results.keys())
    metrics = ["em", "f1", "recall_at_k"]
    labels  = ["Exact Match", "Token F1", "Recall@k"]
    colors  = ["#4878CF", "#6ACC65", "#D65F5F"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for metric, label, color in zip(metrics, labels, colors):
        means = [results[str(s)][metric]["mean"] for s in sizes]
        lows  = [results[str(s)][metric]["lo"]   for s in sizes]
        highs = [results[str(s)][metric]["hi"]   for s in sizes]
        ax.plot(sizes, means, marker="o", label=label, color=color, linewidth=2)
        ax.fill_between(sizes, lows, highs, alpha=0.15, color=color)

    ax.set_xlabel("Chunk Size (words)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Experiment 2 — Effect of Chunk Size\n"
        f"(Dense retriever, k={config.DEFAULT_K})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(sizes)
    plt.tight_layout()
    savefig("fig2_chunk_size.png")


# ─────────────────────────────────────────────────────────────────
# Figure 3 — k values
# ─────────────────────────────────────────────────────────────────

def fig3_k_values():
    data = load("exp3_summary")
    if data is None:
        return
    results = data["results"]

    ks      = sorted(int(k) for k in results.keys())
    metrics = ["em", "f1", "recall_at_k"]
    labels  = ["Exact Match", "Token F1", "Recall@k"]
    colors  = ["#4878CF", "#6ACC65", "#D65F5F"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for metric, label, color in zip(metrics, labels, colors):
        means = [results[str(k)][metric]["mean"] for k in ks]
        lows  = [results[str(k)][metric]["lo"]   for k in ks]
        highs = [results[str(k)][metric]["hi"]   for k in ks]
        ax.plot(ks, means, marker="o", label=label, color=color, linewidth=2)
        ax.fill_between(ks, lows, highs, alpha=0.15, color=color)

    ax.set_xlabel("Number of Retrieved Passages (k)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Experiment 3 — Effect of Number of Passages (k)\n"
        f"(Dense retriever, chunk={config.DEFAULT_CHUNK_SIZE} words)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(ks)
    plt.tight_layout()
    savefig("fig3_k_values.png")


# ─────────────────────────────────────────────────────────────────
# Figure 4 — Prompt template
# ─────────────────────────────────────────────────────────────────

def fig4_prompt_template():
    data = load("exp4_summary")
    if data is None:
        return
    results = data["results"]

    templates = ["concise", "instructed"]
    colors    = [PALETTE[t] for t in templates]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))

    for ax, metric, full_name in zip(axes, ["em", "f1"], ["Exact Match", "Token F1"]):
        means  = [results[t][metric]["mean"] for t in templates]
        lows   = [results[t][metric]["lo"]   for t in templates]
        highs  = [results[t][metric]["hi"]   for t in templates]
        _bar_with_ci(ax, templates, means, lows, highs, colors,
                     title=full_name, ylabel=full_name)

    fig.suptitle(
        "Experiment 4 — Prompt Template Ablation\n"
        f"(Dense, chunk={config.DEFAULT_CHUNK_SIZE}, k={config.DEFAULT_K})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    savefig("fig4_prompt_template.png")


# ─────────────────────────────────────────────────────────────────
# Figure 5 — RAG vs No-RAG
# ─────────────────────────────────────────────────────────────────

def fig5_rag_vs_no_rag():
    data = load("exp5_summary")
    if data is None:
        return

    conditions = ["no_rag", "rag"]
    labels     = ["No-RAG\n(parametric)", "RAG\n(dense, k=5)"]
    colors     = [PALETTE[c] for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))

    for ax, metric, full_name in zip(axes, ["em", "f1"], ["Exact Match", "Token F1"]):
        means  = [data[c][metric]["mean"] for c in conditions]
        lows   = [data[c][metric]["lo"]   for c in conditions]
        highs  = [data[c][metric]["hi"]   for c in conditions]
        _bar_with_ci(ax, labels, means, lows, highs, colors,
                     title=full_name, ylabel=full_name)

    fig.suptitle(
        "Experiment 5 — RAG vs No-RAG Baseline\n"
        f"(Dense, chunk={config.DEFAULT_CHUNK_SIZE}, k={config.DEFAULT_K})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    savefig("fig5_rag_vs_no_rag.png")


# ─────────────────────────────────────────────────────────────────
# Figure 6 — Error analysis: when does RAG help vs hurt?
# ─────────────────────────────────────────────────────────────────

def fig6_error_analysis():
    data = load("exp5_summary")
    if data is None:
        return

    delta = data.get("delta_analysis", [])
    if not delta:
        logger.warning("No delta_analysis in exp5_summary — skipping fig6.")
        return

    helps = data["delta_counts"]["helps"]
    hurts = data["delta_counts"]["hurts"]
    ties  = data["delta_counts"]["ties"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: win/loss/tie pie ──────────────────────────────────
    ax = axes[0]
    wedge_labels = [f"RAG helps\n({helps})", f"Ties\n({ties})", f"RAG hurts\n({hurts})"]
    wedge_sizes  = [helps, ties, hurts]
    wedge_colors = ["#4878CF", "#AAAAAA", "#D65F5F"]
    ax.pie(
        wedge_sizes, labels=wedge_labels, colors=wedge_colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title("When Does RAG Help vs Hurt?\n(per-question EM comparison)",
                 fontsize=12, fontweight="bold")

    # ── Right: recall vs EM scatter ─────────────────────────────
    ax = axes[1]
    recalls = [d["recall"]   for d in delta]
    rag_ems  = [d["rag_em"]  for d in delta]
    helps_mask = [d["rag_helps"] for d in delta]
    hurts_mask = [d["rag_hurts"] for d in delta]
    ties_mask  = [1 - d["rag_helps"] - d["rag_hurts"] for d in delta]

    # Jitter for visibility
    rng = np.random.default_rng(42)
    jitter_x = rng.uniform(-0.02, 0.02, len(delta))
    jitter_y = rng.uniform(-0.02, 0.02, len(delta))
    recalls_j = np.array(recalls, dtype=float) + jitter_x
    rag_ems_j = np.array(rag_ems,  dtype=float) + jitter_y

    for mask, color, label in [
        (helps_mask, "#4878CF", "RAG helps"),
        (ties_mask,  "#AAAAAA", "Tie"),
        (hurts_mask, "#D65F5F", "RAG hurts"),
    ]:
        idx = [i for i, m in enumerate(mask) if m]
        if idx:
            ax.scatter(
                recalls_j[idx], rag_ems_j[idx],
                c=color, alpha=0.5, s=25, label=label,
            )

    ax.set_xlabel("Retrieval Recall (answer in retrieved chunks?)", fontsize=10)
    ax.set_ylabel("RAG Exact Match", fontsize=10)
    ax.set_title("Retrieval Recall vs RAG Answer Quality", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    savefig("fig6_error_analysis.png")


# ─────────────────────────────────────────────────────────────────
# Figure 7 — Qualitative examples table
# ─────────────────────────────────────────────────────────────────

def fig7_qualitative():
    data = load("exp5_summary")
    if data is None:
        return

    delta = data.get("delta_analysis", [])
    if not delta:
        return

    # Pick 3 cases where RAG clearly helps and 3 where it hurts
    helps_cases = [d for d in delta if d["rag_helps"] == 1 and d["recall"] == 1.0][:3]
    hurts_cases = [d for d in delta if d["rag_hurts"] == 1][:3]

    fig, ax = plt.subplots(figsize=(14, max(4, (len(helps_cases) + len(hurts_cases)) * 1.2)))
    ax.axis("off")

    col_labels = ["Category", "Question (truncated)", "Gold", "No-RAG pred", "RAG pred"]
    rows = []
    for d in helps_cases:
        rows.append([
            "RAG ✅ helps",
            d["question"][:55] + ("…" if len(d["question"]) > 55 else ""),
            str(d["gold"][0])[:25],
            str(d["no_rag_pred"])[:25],
            str(d["rag_pred"])[:25],
        ])
    for d in hurts_cases:
        rows.append([
            "RAG ❌ hurts",
            d["question"][:55] + ("…" if len(d["question"]) > 55 else ""),
            str(d["gold"][0])[:25],
            str(d["no_rag_pred"])[:25],
            str(d["rag_pred"])[:25],
        ])

    if not rows:
        logger.info("Not enough examples for qualitative figure — skipping.")
        plt.close()
        return

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.8)

    # Colour rows by category
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DDDDDD")
            cell.set_text_props(fontweight="bold")
        elif row <= len(helps_cases):
            cell.set_facecolor("#EBF4FF")
        else:
            cell.set_facecolor("#FFF0EE")

    ax.set_title(
        "Experiment 5 — Qualitative Examples: When RAG Helps vs Hurts",
        fontsize=12, fontweight="bold", pad=20,
    )
    plt.tight_layout()
    savefig("fig7_qualitative.png")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    logger.info("Generating all figures…")
    fig1_retriever_comparison()
    fig2_chunk_size()
    fig3_k_values()
    fig4_prompt_template()
    fig5_rag_vs_no_rag()
    fig6_error_analysis()
    fig7_qualitative()
    logger.info("Done — figures in %s", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
