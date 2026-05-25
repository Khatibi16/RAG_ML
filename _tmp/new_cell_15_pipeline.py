class RAGPipeline:
    """End-to-end RAG pipeline: retrieve -> prompt -> generate -> evaluate."""

    def __init__(
        self,
        retriever: Optional[BaseRetriever],
        generator: Generator,
        use_retrieval: bool = True,
    ):
        self.retriever     = retriever
        self.generator     = generator
        self.use_retrieval = use_retrieval

    def run(
        self,
        questions: List[Dict],
        k: int = config.DEFAULT_K,
        prompt_template: str = config.DEFAULT_PROMPT,
    ) -> Dict:
        """Run the pipeline on a list of question dicts."""
        t_start = time.perf_counter()

        # 1. Retrieval
        retrieved_chunks_list: List[List[Dict]] = []
        if self.use_retrieval and self.retriever is not None:
            q_texts = [q["question"] for q in questions]
            if hasattr(self.retriever, "batch_retrieve"):
                retrieved_chunks_list = self.retriever.batch_retrieve(q_texts, k)
            else:
                retrieved_chunks_list = [
                    self.retriever.retrieve(q, k)
                    for q in tqdm(q_texts, desc="Retrieving")
                ]
        else:
            retrieved_chunks_list = [[] for _ in questions]

        t_retrieval = time.perf_counter() - t_start

        # 2. Prompt construction. RAG prompts come back as
        # (prefix, context, suffix) triples so the generator can middle-
        # truncate; no-RAG prompts are short plain strings.
        prompts: List[Any] = []
        for q, retrieved in zip(questions, retrieved_chunks_list):
            if self.use_retrieval and retrieved:
                prompt = build_rag_prompt(q["question"], retrieved, prompt_template)
            else:
                prompt = build_no_rag_prompt(q["question"])
            prompts.append(prompt)

        # 2b. Sanity log — token-length stats AND effective-k stats.
        #
        # raw_lens    = the prompt as if no truncation happened. Useful for
        #               spotting conditions where the budget is exceeded.
        # final_lens  = the input the generator actually sees (always
        #               <= max_input_tokens). Difference shows how much
        #               context was lost to middle-truncation.
        # n_truncated = how many *RAG* prompts had their context body
        #               clipped to fit budget.
        # effective_k = number of [N] (...) chunk markers that survive in
        #               the rendered prompt — the model has at least the
        #               header + some text for this many of the originally-
        #               retrieved chunks. When the prompt was truncated,
        #               the chunk at the last surviving marker is typically
        #               only partially visible.
        prompt_tok_stats: Dict[str, float] = {}
        per_prompt_eff_k:  List[int] = []
        per_prompt_nom_k:  List[int] = []
        try:
            self.generator._load_tokenizer()
            tok = self.generator._tokenizer
            raw_lens:   List[int] = []
            final_lens: List[int] = []
            n_ctx_trunc = 0
            for p in prompts:
                if isinstance(p, tuple):
                    raw_str = p[0] + p[1] + p[2]
                    final_str, was_trunc, eff_k, nom_k = (
                        self.generator._truncate_middle(p)
                    )
                    raw_lens.append(
                        len(tok.encode(raw_str, truncation=False, add_special_tokens=True))
                    )
                    final_lens.append(
                        len(tok.encode(final_str, truncation=False, add_special_tokens=True))
                    )
                    per_prompt_eff_k.append(eff_k)
                    per_prompt_nom_k.append(nom_k)
                    if was_trunc:
                        n_ctx_trunc += 1
                else:
                    L = len(tok.encode(p, truncation=False, add_special_tokens=True))
                    raw_lens.append(L)
                    final_lens.append(min(L, self.generator.max_input_tokens))
                    per_prompt_eff_k.append(0)  # no retrieval
                    per_prompt_nom_k.append(0)
                    if L > self.generator.max_input_tokens:
                        n_ctx_trunc += 1
            budget = self.generator.max_input_tokens

            # Aggregate effective-k stats only over prompts that actually
            # used retrieval (nominal_k > 0), so the no-RAG arm doesn't
            # drag the mean to 0.
            rag_eff = [
                e for e, n in zip(per_prompt_eff_k, per_prompt_nom_k) if n > 0
            ]
            rag_nom = [n for n in per_prompt_nom_k if n > 0]
            n_partial_k = sum(
                1 for e, n in zip(per_prompt_eff_k, per_prompt_nom_k)
                if n > 0 and e < n
            )

            prompt_tok_stats = {
                "prompt_tokens_mean":   float(np.mean(raw_lens)),
                "prompt_tokens_max":    int(np.max(raw_lens)),
                "prompt_tokens_budget": int(budget),
                "final_tokens_mean":    float(np.mean(final_lens)),
                "final_tokens_max":     int(np.max(final_lens)),
                "prompts_truncated":    int(n_ctx_trunc),
                "n_prompts":            int(len(raw_lens)),
                # Effective-k summary (computed over RAG prompts only).
                "nominal_k":              int(rag_nom[0]) if rag_nom else 0,
                "effective_k_mean":       float(np.mean(rag_eff)) if rag_eff else 0.0,
                "effective_k_min":        int(np.min(rag_eff))   if rag_eff else 0,
                "effective_k_max":        int(np.max(rag_eff))   if rag_eff else 0,
                "n_prompts_partial_k":    int(n_partial_k),
                "n_rag_prompts":          int(len(rag_eff)),
            }
            logger.info(
                "Prompt tokens: raw mean=%.0f max=%d  final mean=%.0f max=%d  "
                "budget=%d  context_truncated=%d/%d",
                prompt_tok_stats["prompt_tokens_mean"],
                prompt_tok_stats["prompt_tokens_max"],
                prompt_tok_stats["final_tokens_mean"],
                prompt_tok_stats["final_tokens_max"],
                prompt_tok_stats["prompt_tokens_budget"],
                prompt_tok_stats["prompts_truncated"],
                prompt_tok_stats["n_prompts"],
            )
            if rag_eff:
                logger.info(
                    "Effective k: nominal=%d  mean=%.2f  min=%d  max=%d  "
                    "partial=%d/%d (chunks visible to the generator)",
                    prompt_tok_stats["nominal_k"],
                    prompt_tok_stats["effective_k_mean"],
                    prompt_tok_stats["effective_k_min"],
                    prompt_tok_stats["effective_k_max"],
                    prompt_tok_stats["n_prompts_partial_k"],
                    prompt_tok_stats["n_rag_prompts"],
                )
        except Exception as e:
            logger.warning("Prompt-length / effective-k logging failed: %s", e)

        # 3. Generation
        t_gen_start = time.perf_counter()
        predictions = self.generator.generate(prompts)
        t_generation = time.perf_counter() - t_gen_start

        # 4. Evaluation
        gold_answers_list = [q["answers"] for q in questions]
        metrics = evaluate_with_ci(
            predictions,
            gold_answers_list,
            retrieved_chunks_list if self.use_retrieval else None,
        )

        # Per-example breakdown
        per_example = []
        for q, pred, golds, retrieved, prompt in zip(
            questions, predictions, gold_answers_list,
            retrieved_chunks_list, prompts,
        ):
            prompt_str, eff_k, nom_k = self.generator.render_with_effective_k(prompt)
            per_example.append({
                "qid":         q["qid"],
                "question":    q["question"],
                "gold":        golds,
                "prediction":  pred,
                "em":          exact_match(pred, golds),
                "f1":          token_f1(pred, golds),
                "recall":      retrieval_recall_single(retrieved, golds),
                "n_retrieved": len(retrieved),
                "nominal_k":   nom_k,
                "effective_k": eff_k,
                "prompt":      prompt_str,
                "top_chunk":   retrieved[0]["text"][:200] if retrieved else "",
            })

        timing = {
            "retrieval_s":  round(t_retrieval, 2),
            "generation_s": round(t_generation, 2),
            "total_s":      round(time.perf_counter() - t_start, 2),
            "n_questions":  len(questions),
            # Cache stats so a near-zero generation_s can be
            # recognised as "everything was cached" rather than
            # "the generator is suspiciously fast".
            "cache_hits":   getattr(self.generator, "last_cache_hits",   None),
            "cache_misses": getattr(self.generator, "last_cache_misses", None),
        }
        timing.update(prompt_tok_stats)

        return {
            "predictions":      predictions,
            "gold_answers":     gold_answers_list,
            "retrieved_chunks": retrieved_chunks_list,
            "prompts":          prompts,
            "metrics":          metrics,
            "per_example":      per_example,
            "timing":           timing,
        }
