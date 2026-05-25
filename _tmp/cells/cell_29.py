def experiment_5(questions, corpus_docs, generator):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5 — RAG vs No-RAG baseline")
    logger.info("Fixed: retriever=dense  chunk_size=%d  k=%d  prompt=%s",
                config.DEFAULT_CHUNK_SIZE, config.DEFAULT_K, config.DEFAULT_PROMPT)

    chunks    = chunk_corpus(corpus_docs, chunk_size=config.DEFAULT_CHUNK_SIZE)
    retriever = build_retriever(
        "dense", chunks,
        cache_dir=config.CACHE_DIR,
        chunk_size=config.DEFAULT_CHUNK_SIZE,
    )

    if already_done("exp5_rag"):
        rag_result = load_results("exp5_rag")
    else:
        logger.info("Running RAG condition…")
        rag_pipe   = RAGPipeline(retriever, generator, use_retrieval=True)
        rag_result = rag_pipe.run(questions, k=config.DEFAULT_K)
        print_metrics("RAG", rag_result)
        save_results("exp5_rag", rag_result)

    if already_done("exp5_no_rag"):
        no_rag_result = load_results("exp5_no_rag")
    else:
        logger.info("Running No-RAG (parametric-only) condition…")
        no_rag_pipe   = RAGPipeline(None, generator, use_retrieval=False)
        no_rag_result = no_rag_pipe.run(questions, k=config.DEFAULT_K)
        print_metrics("No-RAG", no_rag_result)
        save_results("exp5_no_rag", no_rag_result)

    # Per-question delta analysis
    rag_per    = {ex["qid"]: ex for ex in rag_result["per_example"]}
    no_rag_per = {ex["qid"]: ex for ex in no_rag_result["per_example"]}

    delta_analysis = []
    for qid, rag_ex in rag_per.items():
        if qid not in no_rag_per:
            continue
        no_rag_ex = no_rag_per[qid]
        delta_analysis.append({
            "qid":         qid,
            "question":    rag_ex["question"],
            "gold":        rag_ex["gold"],
            "rag_pred":    rag_ex["prediction"],
            "no_rag_pred": no_rag_ex["prediction"],
            "rag_em":      rag_ex["em"],
            "no_rag_em":   no_rag_ex["em"],
            "rag_f1":      rag_ex["f1"],
            "no_rag_f1":   no_rag_ex["f1"],
            "recall":      rag_ex.get("recall", 0),
            "rag_helps":   int(rag_ex["em"] > no_rag_ex["em"]),
            "rag_hurts":   int(rag_ex["em"] < no_rag_ex["em"]),
        })

    helps = sum(d["rag_helps"] for d in delta_analysis)
    hurts = sum(d["rag_hurts"] for d in delta_analysis)
    ties  = len(delta_analysis) - helps - hurts
    logger.info("RAG helps: %d  RAG hurts: %d  Ties: %d  (out of %d)",
                helps, hurts, ties, len(delta_analysis))

    summary = {
        "rag":    {"em": rag_result["metrics"]["em"],    "f1": rag_result["metrics"]["f1"]},
        "no_rag": {"em": no_rag_result["metrics"]["em"], "f1": no_rag_result["metrics"]["f1"]},
        "delta_counts":   {"helps": helps, "hurts": hurts, "ties": ties},
        "delta_analysis": delta_analysis,
    }
    save_results("exp5_summary", summary)
    return {"rag": rag_result, "no_rag": no_rag_result, "summary": summary}


exp5_results = experiment_5(questions, corpus_docs, generator)
logger.info("All experiments complete. Results in %s", config.RESULTS_DIR)
