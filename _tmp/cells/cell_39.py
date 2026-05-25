def fig5_rag_vs_no_rag():
    data = _load_summary("exp5_summary")
    if data is None:
        return

    conditions = ["no_rag", "rag"]
    labels     = ["No-RAG\n(parametric)", "RAG\n(dense, k=5)"]
    colors     = [PALETTE[c] for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    for ax, metric, full_name in zip(axes, ["em", "f1"],
                                     ["Exact Match", "Token F1"]):
        means = [data[c][metric]["mean"] for c in conditions]
        lows  = [data[c][metric]["lo"]   for c in conditions]
        highs = [data[c][metric]["hi"]   for c in conditions]
        _bar_with_ci(ax, labels, means, lows, highs, colors,
                     title=full_name, ylabel=full_name)

    fig.suptitle(
        "Experiment 5 — RAG vs No-RAG Baseline\n"
        f"(Dense, chunk={config.DEFAULT_CHUNK_SIZE}, k={config.DEFAULT_K})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    _savefig("fig5_rag_vs_no_rag.png")
    plt.show()


fig5_rag_vs_no_rag()
