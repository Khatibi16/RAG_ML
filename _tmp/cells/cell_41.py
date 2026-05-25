def fig7_qualitative():
    data = _load_summary("exp5_summary")
    if data is None:
        return

    delta = data.get("delta_analysis", [])
    if not delta:
        return

    helps_cases = [d for d in delta if d["rag_helps"] == 1 and d["recall"] == 1.0][:3]
    hurts_cases = [d for d in delta if d["rag_hurts"] == 1][:3]

    fig, ax = plt.subplots(
        figsize=(14, max(4, (len(helps_cases) + len(hurts_cases)) * 1.2))
    )
    ax.axis("off")

    col_labels = ["Category", "Question (truncated)", "Gold",
                  "No-RAG pred", "RAG pred"]
    rows = []
    for d in helps_cases:
        rows.append([
            "RAG helps",
            d["question"][:55] + ("…" if len(d["question"]) > 55 else ""),
            str(d["gold"][0])[:25],
            str(d["no_rag_pred"])[:25],
            str(d["rag_pred"])[:25],
        ])
    for d in hurts_cases:
        rows.append([
            "RAG hurts",
            d["question"][:55] + ("…" if len(d["question"]) > 55 else ""),
            str(d["gold"][0])[:25],
            str(d["no_rag_pred"])[:25],
            str(d["rag_pred"])[:25],
        ])

    if not rows:
        logger.info("Not enough examples for qualitative figure — skipping.")
        plt.close()
        return

    tbl = ax.table(
        cellText=rows, colLabels=col_labels, cellLoc="left", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DDDDDD")
            cell.set_text_props(fontweight="bold")
        elif row <= len(helps_cases):
            cell.set_facecolor("#EBF4FF")
        else:
            cell.set_facecolor("#FFF0EE")

    ax.set_title(
        "Experiment 5 — Qualitative Examples: When RAG Helps vs Hurts",
        fontsize=12, fontweight="bold", pad=20,
    )
    plt.tight_layout()
    _savefig("fig7_qualitative.png")
    plt.show()


fig7_qualitative()
