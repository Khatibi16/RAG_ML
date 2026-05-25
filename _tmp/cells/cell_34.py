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
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


def _load_summary(name: str) -> Optional[dict]:
    p = config.RESULTS_DIR / f"{name}.json"
    if not p.exists():
        logger.warning("Results file not found: %s  (skip)", p)
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _savefig(name: str) -> None:
    out = config.FIGURES_DIR / name
    plt.savefig(out, dpi=300, bbox_inches="tight")
    logger.info("Saved → %s", out)


def _bar_with_ci(ax, labels, means, lows, highs, colors, title, ylabel="Score"):
    x     = np.arange(len(labels))
    width = 0.5
    bars  = ax.bar(x, means, width, color=colors, alpha=0.85, zorder=3)
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
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
