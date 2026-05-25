def experiment_4(questions, corpus_docs, generator):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4 — Prompt template: concise vs instructed")
    logger.info("Fixed: retriever=dense  chunk_size=%d  k=%d",
                config.DEFAULT_CHUNK_SIZE, config.DEFAULT_K)

    chunks    = chunk_corpus(corpus_docs, chunk_size=config.DEFAULT_CHUNK_SIZE)
    retriever = build_retriever(
        "dense", chunks,
        cache_dir=config.CACHE_DIR,
        chunk_size=config.DEFAULT_CHUNK_SIZE,
    )

    exp4_results = {}
    for template in ["concise", "instructed"]:
        if already_done(f"exp4_{template}"):
            exp4_results[template] = load_results(f"exp4_{template}")
            continue
        logger.info("Template: %s", template)
        pipe   = RAGPipeline(retriever, generator)
        result = pipe.run(questions, k=config.DEFAULT_K, prompt_template=template)
        print_metrics(f"prompt={template}", result)
        exp4_results[template] = result
        save_results(f"exp4_{template}", result)

    summary = {}
    for tmpl, result in exp4_results.items():
        m = result["metrics"]
        summary[tmpl] = {"em": m["em"], "f1": m["f1"]}
    save_results("exp4_summary", {"results": summary})
    return exp4_results


exp4_results = experiment_4(questions, corpus_docs, generator)
