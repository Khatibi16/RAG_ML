def experiment_3(questions, corpus_docs, generator):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3 — Number of passages k in %s", config.K_VALUES)
    logger.info("Fixed: retriever=dense  chunk_size=%d  prompt=%s",
                config.DEFAULT_CHUNK_SIZE, config.DEFAULT_PROMPT)

    chunks    = chunk_corpus(corpus_docs, chunk_size=config.DEFAULT_CHUNK_SIZE)
    retriever = build_retriever(
        "dense", chunks,
        cache_dir=config.CACHE_DIR,
        chunk_size=config.DEFAULT_CHUNK_SIZE,
    )

    exp3_results = {}
    for k in config.K_VALUES:
        if already_done(f"exp3_k{k}"):
            exp3_results[k] = load_results(f"exp3_k{k}")
            continue
        logger.info("k = %d", k)
        pipe   = RAGPipeline(retriever, generator)
        result = pipe.run(questions, k=k)
        print_metrics(f"k={k}", result)
        exp3_results[k] = result
        save_results(f"exp3_k{k}", result)

    summary = {}
    for k, result in exp3_results.items():
        m = result["metrics"]
        summary[str(k)] = {
            "em":          m["em"],
            "f1":          m["f1"],
            "recall_at_k": m.get("recall_at_k", {}),
        }
    save_results("exp3_summary", {"results": summary})
    return exp3_results


exp3_results = experiment_3(questions, corpus_docs, generator)
