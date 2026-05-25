def experiment_6_oracle(questions, corpus_docs, generator):
    """Oracle baseline: guarantee an answer-bearing chunk in the top-k."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 6 — Oracle baseline (answer guaranteed in context)")
    logger.info("Fixed: chunk_size=%d  k=%d  prompt=%s",
                config.DEFAULT_CHUNK_SIZE, config.DEFAULT_K, config.DEFAULT_PROMPT)

    if already_done("exp6_oracle"):
        return load_results("exp6_oracle")

    chunks    = chunk_corpus(corpus_docs, chunk_size=config.DEFAULT_CHUNK_SIZE)
    retriever = build_retriever(
        "dense", chunks,
        cache_dir=config.CACHE_DIR,
        chunk_size=config.DEFAULT_CHUNK_SIZE,
    )

    # Stage 1: ordinary dense retrieval.
    q_texts = [q["question"] for q in questions]
    dense_results = retriever.batch_retrieve(q_texts, config.DEFAULT_K)

    # Stage 2: inject an answer-bearing chunk wherever the top-k misses.
    oracle_retrieved: List[List[Dict]] = []
    n_injected = 0
    n_unfixable = 0
    for q, retrieved in zip(questions, dense_results):
        norms = [normalize_answer(a) for a in q["answers"] if a]
        already_has = any(
            any(g and g in normalize_answer(c["text"]) for g in norms)
            for c in retrieved
        )
        if already_has:
            oracle_retrieved.append(retrieved)
            continue
        replacement = None
        for c in chunks:
            if any(g and g in normalize_answer(c["text"]) for g in norms):
                replacement = {**c, "score": float("inf")}
                break
        if replacement is None:
            n_unfixable += 1
            oracle_retrieved.append(retrieved)
        else:
            oracle_retrieved.append(
                [replacement] + retrieved[: config.DEFAULT_K - 1]
            )
            n_injected += 1
    logger.info(
        "Oracle: injected for %d / %d questions (%d had no answer-bearing "
        "chunk anywhere in the corpus).",
        n_injected, len(questions), n_unfixable,
    )

    # Build prompts manually since the pipeline drives retrieval itself.
    prompts: List[Any] = []
    for q, retrieved in zip(questions, oracle_retrieved):
        if retrieved:
            prompts.append(
                build_rag_prompt(q["question"], retrieved, config.DEFAULT_PROMPT)
            )
        else:
            prompts.append(build_no_rag_prompt(q["question"]))
    predictions = generator.generate(prompts)

    gold_answers_list = [q["answers"] for q in questions]
    metrics = evaluate_with_ci(predictions, gold_answers_list, oracle_retrieved)

    per_example = []
    for q, p, g, r, prompt in zip(
        questions, predictions, gold_answers_list, oracle_retrieved, prompts,
    ):
        prompt_str, eff_k, nom_k = generator.render_with_effective_k(prompt)
        per_example.append({
            "qid":         q["qid"],
            "question":    q["question"],
            "gold":        g,
            "prediction":  p,
            "em":          exact_match(p, g),
            "f1":          token_f1(p, g),
            "recall":      retrieval_recall_single(r, g),
            "n_retrieved": len(r),
            "nominal_k":   nom_k,
            "effective_k": eff_k,
            "prompt":      prompt_str,
        })

    result = {
        "metrics":     metrics,
        "per_example": per_example,
        "n_injected":  n_injected,
        "n_unfixable": n_unfixable,
        "timing": {
            "n_questions":  len(questions),
            "cache_hits":   generator.last_cache_hits,
            "cache_misses": generator.last_cache_misses,
        },
    }
    print_metrics("oracle", result)
    save_results("exp6_oracle", result)
    return result


exp6_results = experiment_6_oracle(questions, corpus_docs, generator)
