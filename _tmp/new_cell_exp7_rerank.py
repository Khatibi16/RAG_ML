def experiment_7_rerank(questions, corpus_docs, generator):
    """Two-stage retrieval: dense top-N → cross-encoder rerank → top-k."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 7 — Cross-encoder reranker on top of dense retrieval")
    logger.info(
        "Fixed: chunk_size=%d  k=%d  prompt=%s  top_n=%d  reranker=%s",
        config.DEFAULT_CHUNK_SIZE, config.DEFAULT_K,
        config.DEFAULT_PROMPT, config.RERANK_TOP_N, config.RERANK_MODEL,
    )

    if already_done("exp7_rerank"):
        return load_results("exp7_rerank")

    chunks = chunk_corpus(corpus_docs, chunk_size=config.DEFAULT_CHUNK_SIZE)
    dense  = build_retriever(
        "dense", chunks,
        cache_dir=config.CACHE_DIR,
        chunk_size=config.DEFAULT_CHUNK_SIZE,
    )
    rerank = RerankRetriever(base=dense)
    rerank.build(chunks)

    pipe   = RAGPipeline(rerank, generator)
    result = pipe.run(questions, k=config.DEFAULT_K)
    print_metrics("dense+rerank", result)
    save_results("exp7_rerank", result)
    return result


exp7_results = experiment_7_rerank(questions, corpus_docs, generator)
