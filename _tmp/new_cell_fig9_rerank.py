def fig9_rerank():
    """Compare dense (top-5) vs dense+rerank (top-5 of top-50)."""
    exp1 = _load_summary("exp1_summary")
    exp7 = _load_summary("exp7_rerank")
    if exp1 is None or exp7 is None:
        return

    dense_metrics  = exp1["results"]["dense"]
    rerank_metrics = exp7["metrics"]

    labels = ["Dense", "Dense + Rerank"]
    colors = ["#D65F5F", "#7E55CC"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, metric, full_name in zip(
        axes, ["em", "f1", "recall_at_k"],
        ["Exact Match", "Token F1", "Recall@k"],
    ):
        d = dense_metrics.get(metric, {})
        r = rerank_metrics.get(metric, {})
        means = [d.get("mean", 0), r.get("mean", 0)]
        lows  = [d.get("lo",   0), r.get("lo",   0)]
        highs = [d.get("hi",   0), r.get("hi",   0)]
        _bar_with_ci(ax, labels, means, lows, highs, colors,
                     title=full_name, ylabel=full_name)

    fig.suptitle(
        "Experiment 7 — Effect of Cross-Encoder Re-ranking\n"
        f"(Dense top-{config.RERANK_TOP_N} → rerank → top-{config.DEFAULT_K})",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    _savefig("fig9_rerank.png")
    plt.show()


fig9_rerank()
