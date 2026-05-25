def fig1_retriever_comparison():
    data = _load_summary("exp1_summary")
    if data is None:
        return
    results = data["results"]
    rtypes  = list(results.keys())
    colors  = [PALETTE.get(r, "#888888") for r in rtypes]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric, full_name in zip(axes, ["em", "f1"],
                                     ["Exact Match (EM)", "Token F1"]):
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
    _savefig("fig1_retriever_comparison.png")
    plt.show()


fig1_retriever_comparison()
