def fig2_chunk_size():
    data = _load_summary("exp2_summary")
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
    _savefig("fig2_chunk_size.png")
    plt.show()


fig2_chunk_size()
