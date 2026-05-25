def experiment_1(questions, corpus_docs, generator):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1 — Retriever comparison (BM25 / TF-IDF / Dense)")
    logger.info("Fixed: chunk_size=%d  k=%d  prompt=%s",
                config.DEFAULT_CHUNK_SIZE, config.DEFAULT_K, config.DEFAULT_PROMPT)

    chunks = chunk_corpus(corpus_docs, chunk_size=config.DEFAULT_CHUNK_SIZE)
    logger.info("Chunks: %d (size=%d)", len(chunks), config.DEFAULT_CHUNK_SIZE)

    exp1_results = {}
    for rtype in config.RETRIEVER_TYPES:
        if already_done(f"exp1_{rtype}"):
            exp1_results[rtype] = load_results(f"exp1_{rtype}")
            continue
        logger.info("Building retriever: %s", rtype)
        retriever = build_retriever(
            rtype, chunks,
            cache_dir=config.CACHE_DIR,
            chunk_size=config.DEFAULT_CHUNK_SIZE,
        )
        pipe   = RAGPipeline(retriever, generator)
        result = pipe.run(questions, k=config.DEFAULT_K)
        print_metrics(rtype, result)
        exp1_results[rtype] = result
        save_results(f"exp1_{rtype}", result)

    summary = {}
    for rtype, result in exp1_results.items():
        m = result["metrics"]
        summary[rtype] = {
            "em":          m["em"],
            "f1":          m["f1"],
            "recall_at_k": m.get("recall_at_k", {}),
            "timing":      result["timing"],
        }
    save_results("exp1_summary", {"results": summary})
    return exp1_results


exp1_results = experiment_1(questions, corpus_docs, generator)
