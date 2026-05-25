def experiment_8_distractor_sweep(generator):
    """Sweep NUM_WIKI_DISTRACTORS holding everything else fixed.

    For each value, re-load the corpus, re-chunk, re-embed (using a
    distractor-count-specific dense cache), and run the RAG arm only.
    The No-RAG arm doesn't depend on the corpus, so it isn't re-run here.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 8 — Distractor count sweep over %s",
                config.DISTRACTOR_SWEEP_VALUES)

    summary: Dict[str, Any] = {}
    original_n_dist = config.NUM_WIKI_DISTRACTORS

    try:
        for n_dist in config.DISTRACTOR_SWEEP_VALUES:
            tag = f"exp8_d{n_dist}"
            if already_done(tag):
                summary[str(n_dist)] = load_results(tag)
                continue
            logger.info("--- n_distractors = %d ---", n_dist)
            config.NUM_WIKI_DISTRACTORS = n_dist
            questions_d, corpus_d = load_triviaqa(num_questions=config.NUM_QUESTIONS)
            chunks_d = chunk_corpus(corpus_d, chunk_size=config.DEFAULT_CHUNK_SIZE)

            # Distractor-count-specific dense cache so corpora don't collide.
            cache_path = (
                config.CACHE_DIR
                / f"dense_embeddings_{config.DEFAULT_CHUNK_SIZE}w_d{n_dist}.pkl"
            )
            retriever = DenseRetriever(cache_path=cache_path)
            retriever.build(chunks_d)

            pipe   = RAGPipeline(retriever, generator)
            result = pipe.run(questions_d, k=config.DEFAULT_K)
            print_metrics(f"n_dist={n_dist}", result)

            entry = {
                "n_corpus_docs": len(corpus_d),
                "n_chunks":      len(chunks_d),
                "metrics":       result["metrics"],
                "timing":        result["timing"],
            }
            summary[str(n_dist)] = entry
            save_results(tag, entry)
    finally:
        config.NUM_WIKI_DISTRACTORS = original_n_dist

    save_results("exp8_summary", {"results": summary})
    return summary


exp8_results = experiment_8_distractor_sweep(generator)
