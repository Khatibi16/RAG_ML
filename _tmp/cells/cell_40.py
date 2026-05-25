def fig6_error_analysis():
    data = _load_summary("exp5_summary")
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

    # Left: win/loss/tie pie
    ax = axes[0]
    wedge_labels = [f"RAG helps\n({helps})", f"Ties\n({ties})", f"RAG hurts\n({hurts})"]
    wedge_sizes  = [helps, ties, hurts]
    wedge_colors = ["#4878CF", "#AAAAAA", "#D65F5F"]
    ax.pie(
        wedge_sizes, labels=wedge_labels, colors=wedge_colors,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10},
    )
    ax.set_title("When Does RAG Help vs Hurt?\n(per-question EM comparison)",
                 fontsize=12, fontweight="bold")

    # Right: recall vs EM scatter
    ax = axes[1]
    recalls = [d["recall"]   for d in delta]
    rag_ems = [d["rag_em"]   for d in delta]
    helps_mask = [d["rag_helps"] for d in delta]
    hurts_mask = [d["rag_hurts"] for d in delta]
    ties_mask  = [1 - d["rag_helps"] - d["rag_hurts"] for d in delta]

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
    _savefig("fig6_error_analysis.png")
    plt.show()


fig6_error_analysis()
