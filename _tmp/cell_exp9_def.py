def experiment_9_distraction(questions, corpus_docs, generator):
    """Controlled distraction: hold retrieval recall fixed at 1 (a gold
    answer-bearing chunk is always present) and vary the number of *non-gold*
    distractor chunks placed alongside it.

    This isolates the headline question — *when does retrieval hurt?* — from
    retrieval quality. In Experiment 5 RAG almost never hurts, and in
    Experiment 8 the topic-agnostic distractors never out-rank the gold pages
    so Recall@5 is flat; neither isolates pure generator distraction. Here
    every condition contains the answer (Recall@k = 1 by construction), so any
    decline in EM/F1 as N grows is attributable to the generator being
    distracted by competing context, not to a retrieval miss.

    Design
    ------
    * Chunk size = ``EXP3_CHUNK_SIZE`` (48 words) so the largest context
      (gold + max(N) distractors) fits the ``MAX_INPUT_TOKENS`` budget
      untruncated, exactly as in Experiment 3 — no truncation confound.
    * Distractors are the top dense-retrieved chunks for the question that do
      NOT contain the answer: realistic *hard* negatives that a real RAG
      system would actually surface (topically related but answer-free),
      rather than random text.
    * The gold chunk is inserted at a per-question-shuffled position (seeded
      deterministically by qid) so the answer is not always first; as N grows
      the answer is increasingly diluted and buried, which is the regime in
      which distraction is expected to bite.
    * Questions whose answer appears in no chunk anywhere in the corpus are
      excluded (recall = 1 cannot be guaranteed for them); the count is logged
      and held constant across all N so every condition scores the same set.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 9 — Controlled distraction (recall held at 1)")
    logger.info("Fixed: retriever=dense  chunk_size=%d  prompt=%s  N in %s",
                config.EXP3_CHUNK_SIZE, config.DEFAULT_PROMPT,
                config.DISTRACTION_N_VALUES)

    chunks    = chunk_corpus(corpus_docs, chunk_size=config.EXP3_CHUNK_SIZE)
    retriever = build_retriever(
        "dense", chunks,
        cache_dir=config.CACHE_DIR,
        chunk_size=config.EXP3_CHUNK_SIZE,
    )

    # Retrieve one deep candidate pool per question and reuse it across all N.
    max_n   = max(config.DISTRACTION_N_VALUES)
    q_texts = [q["question"] for q in questions]
    pool    = retriever.batch_retrieve(q_texts, max_n + 8)  # headroom for filtering

    def _is_gold(chunk, norms):
        nc = normalize_answer(chunk["text"])
        return any(g and g in nc for g in norms)

    gold_chunks:   List[Optional[Dict]] = []
    distractors:   List[List[Dict]]     = []
    included_mask: List[bool]           = []
    n_excluded = 0
    for q, cands in zip(questions, pool):
        norms = [normalize_answer(a) for a in q["answers"] if a]
        gold  = next((c for c in cands if _is_gold(c, norms)), None)
        if gold is None:  # fall back to a full-corpus scan
            gold = next((c for c in chunks if _is_gold(c, norms)), None)
        if gold is None:
            n_excluded += 1
            gold_chunks.append(None)
            distractors.append([])
            included_mask.append(False)
            continue
        dist = [c for c in cands if not _is_gold(c, norms)][:max_n]
        gold_chunks.append(gold)
        distractors.append(dist)
        included_mask.append(True)
    logger.info(
        "Distraction: %d/%d questions usable (%d excluded — no answer-bearing "
        "chunk in the corpus).",
        sum(included_mask), len(questions), n_excluded,
    )

    summary: Dict[str, Any] = {}
    per_n_results: Dict[int, Dict] = {}
    for n_dist in config.DISTRACTION_N_VALUES:
        tag = f"exp9_n{n_dist}"
        if already_done(tag):
            per_n_results[n_dist] = load_results(tag)
            summary[str(n_dist)]  = per_n_results[n_dist]["metrics"]
            continue
        logger.info("--- N distractors = %d ---", n_dist)
        prompts, golds_list, ctx_list, qids = [], [], [], []
        for q, gold, dist, ok in zip(questions, gold_chunks, distractors, included_mask):
            if not ok:
                continue
            ctx  = [gold] + dist[:n_dist]
            seed = int(hashlib.md5(str(q["qid"]).encode()).hexdigest(), 16) % (2 ** 32)
            order = np.random.default_rng(seed).permutation(len(ctx))
            ctx   = [ctx[i] for i in order]
            prompts.append(build_rag_prompt(q["question"], ctx, config.DEFAULT_PROMPT))
            golds_list.append(q["answers"])
            ctx_list.append(ctx)
            qids.append(q["qid"])

        preds   = generator.generate(prompts)
        metrics = evaluate_with_ci(preds, golds_list, ctx_list)
        per_example = [
            {
                "qid":           qid,
                "prediction":    p,
                "gold":          g,
                "em":            exact_match(p, g),
                "f1":            token_f1(p, g),
                "recall":        retrieval_recall_single(c, g),
                "n_distractors": n_dist,
            }
            for qid, p, g, c in zip(qids, preds, golds_list, ctx_list)
        ]
        result = {
            "metrics":       metrics,
            "per_example":   per_example,
            "n_included":    len(prompts),
            "n_excluded":    n_excluded,
            "n_distractors": n_dist,
            "timing": {
                "n_questions":  len(prompts),
                "cache_hits":   generator.last_cache_hits,
                "cache_misses": generator.last_cache_misses,
            },
        }
        print_metrics(f"N_distractors={n_dist}", result)
        per_n_results[n_dist] = result
        summary[str(n_dist)]  = metrics
        save_results(tag, result)

    save_results("exp9_summary", {"results": summary, "n_excluded": n_excluded})
    return per_n_results
