def fig10_distractor_sweep():
    """EM / F1 / Recall@k as a function of the external-distractor count."""
    data = _load_summary("exp8_summary")
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
        means = [results[str(n)]["metrics"][metric]["mean"] for n in ns]
        lows  = [results[str(n)]["metrics"][metric]["lo"]   for n in ns]
        highs = [results[str(n)]["metrics"][metric]["hi"]   for n in ns]
        ax.plot(ns, means, marker="o", label=label, color=color, linewidth=2)
        ax.fill_between(ns, lows, highs, alpha=0.15, color=color)

    ax.set_xscale("symlog", linthresh=500)
    ax.set_xlabel("Number of External Wikipedia Distractors", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Experiment 8 — Effect of External Distractor Count\n"
        f"(Dense, chunk={config.DEFAULT_CHUNK_SIZE}, k={config.DEFAULT_K})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    plt.tight_layout()
    _savefig("fig10_distractor_sweep.png")
    plt.show()


fig10_distractor_sweep()
