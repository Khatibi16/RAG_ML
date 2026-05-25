def fig4_prompt_template():
    data = _load_summary("exp4_summary")
    if data is None:
        return
    results = data["results"]

    templates = ["concise", "instructed"]
    colors    = [PALETTE[t] for t in templates]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    for ax, metric, full_name in zip(axes, ["em", "f1"],
                                     ["Exact Match", "Token F1"]):
        means = [results[t][metric]["mean"] for t in templates]
        lows  = [results[t][metric]["lo"]   for t in templates]
        highs = [results[t][metric]["hi"]   for t in templates]
        _bar_with_ci(ax, templates, means, lows, highs, colors,
                     title=full_name, ylabel=full_name)

    fig.suptitle(
        "Experiment 4 — Prompt Template Ablation\n"
        f"(Dense, chunk={config.DEFAULT_CHUNK_SIZE}, k={config.DEFAULT_K})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    _savefig("fig4_prompt_template.png")
    plt.show()


fig4_prompt_template()
