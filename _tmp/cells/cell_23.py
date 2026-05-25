def experiment_2(questions, corpus_docs, generator):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2 — Chunk size ablation")
    logger.info("Fixed: retriever=dense  k=%d  prompt=%s",
                config.DEFAULT_K, config.DEFAULT_PROMPT)

    exp2_results = {}
    for chunk_size in config.CHUNK_SIZES:
        if already_done(f"exp2_chunk{chunk_size}"):
            exp2_results[chunk_size] = load_results(f"exp2_chunk{chunk_size}")
            continue
        logger.info("Chunk size: %d words", chunk_size)
        chunks = chunk_corpus(corpus_docs, chunk_size=chunk_size)
        logger.info("  -> %d chunks", len(chunks))

        retriever = build_retriever(
            "dense", chunks,
            cache_dir=config.CACHE_DIR,
            chunk_size=chunk_size,
        )
        pipe   = RAGPipeline(retriever, generator)
        result = pipe.run(questions, k=config.DEFAULT_K)
        print_metrics(f"chunk={chunk_size}", result)
        exp2_results[chunk_size] = result
        save_results(f"exp2_chunk{chunk_size}", result)

    summary = {}
    for sz, result in exp2_results.items():
        m = result["metrics"]
        summary[str(sz)] = {
            "em":          m["em"],
            "f1":          m["f1"],
            "recall_at_k": m.get("recall_at_k", {}),
            "n_chunks":    result["timing"]["n_questions"],
        }
    save_results("exp2_summary", {"results": summary})
    return exp2_results


exp2_results = experiment_2(questions, corpus_docs, generator)
