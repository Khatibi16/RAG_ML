def fig3_k_values():
    data = _load_summary("exp3_summary")
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
    _savefig("fig3_k_values.png")
    plt.show()


fig3_k_values()
