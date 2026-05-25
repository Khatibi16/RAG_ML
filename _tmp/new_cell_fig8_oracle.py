def fig8_oracle():
    """No-RAG vs RAG vs Oracle bars (EM + F1)."""
    rag_data    = _load_summary("exp5_summary")
    oracle_data = _load_summary("exp6_oracle")
    if rag_data is None or oracle_data is None:
        return

    conditions = ["no_rag", "rag", "oracle"]
    labels     = ["No-RAG\n(parametric)", "RAG\n(dense, k=5)",
                  "Oracle\n(answer guaranteed)"]
    colors     = ["#AAAAAA", "#4878CF", "#6ACC65"]
    cond_data  = {
        "no_rag": rag_data["no_rag"],
        "rag":    rag_data["rag"],
        "oracle": oracle_data["metrics"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, metric, full_name in zip(axes, ["em", "f1"],
                                     ["Exact Match", "Token F1"]):
        means = [cond_data[c][metric]["mean"] for c in conditions]
        lows  = [cond_data[c][metric]["lo"]   for c in conditions]
        highs = [cond_data[c][metric]["hi"]   for c in conditions]
        _bar_with_ci(ax, labels, means, lows, highs, colors,
                     title=full_name, ylabel=full_name)

    fig.suptitle(
        "Experiment 6 — Oracle Baseline\n"
        f"(Dense, chunk={config.DEFAULT_CHUNK_SIZE}, k={config.DEFAULT_K}; "
        "oracle injects an answer-bearing chunk at rank 1 when missing)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    _savefig("fig8_oracle.png")
    plt.show()


fig8_oracle()
