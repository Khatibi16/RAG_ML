def fig11_distraction():
    """EM / F1 / Recall@k vs number of non-gold distractors (Exp 9)."""
    data = _load_summary("exp9_summary")
    if data is None:
        return
    results = data["results"]
    ns = sorted(int(k) for k in results.keys())
    if not ns:
        return

    metrics = ["em", "f1", "recall_at_k"]
    labels  = ["Exact Match", "Token F1", "Recall@k"]
    colors  = ["#4878CF", "#6ACC65", "#D65F5F"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for metric, label, color in zip(metrics, labels, colors):
        means = [results[str(n)][metric]["mean"] for n in ns]
        lows  = [results[str(n)][metric]["lo"]   for n in ns]
        highs = [results[str(n)][metric]["hi"]   for n in ns]
        ax.plot(ns, means, marker="o", label=label, color=color, linewidth=2)
        ax.fill_between(ns, lows, highs, alpha=0.15, color=color)

    ax.set_xlabel("Number of non-gold distractor chunks (gold always present)",
                  fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Experiment 9 — Controlled Distraction (Recall@k held at 1)\n"
        f"(Dense, chunk={config.EXP3_CHUNK_SIZE} words, gold position shuffled per question)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(ns)
    plt.tight_layout()
    _savefig("fig11_distraction.png")
    plt.show()


def fig12_significance():
    """Table of paired significance tests (McNemar on EM, paired bootstrap)."""
    data = _load_summary("paired_significance")
    if data is None:
        return
    contrasts = data.get("contrasts", [])
    if not contrasts:
        return

    col_labels = ["Contrast (A − B)", "ΔEM", "McNemar p",
                  "disc. A/B", "ΔF1", "ΔF1 95% CI", "F1 boot p"]
    rows = []
    for c in contrasts:
        em, dem, df1 = (c["em_mcnemar"], c["em_paired_bootstrap"],
                        c["f1_paired_bootstrap"])
        rows.append([
            c["contrast"],
            f"{dem['mean_delta']:+.3f}",
            f"{em['p_value']:.4f}",
            f"{em['a_only']}/{em['b_only']}",
            f"{df1['mean_delta']:+.3f}",
            f"[{df1['lo']:+.3f}, {df1['hi']:+.3f}]",
            f"{df1['p_value']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(13, 0.9 + 0.5 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, _col), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#DDDDDD")
            cell.set_text_props(fontweight="bold")
    ax.set_title(
        "Paired Significance Tests\n"
        "(same questions; McNemar's exact test on EM, paired bootstrap on per-question deltas)",
        fontsize=12, fontweight="bold", pad=18,
    )
    plt.tight_layout()
    _savefig("fig12_significance.png")
    plt.show()
