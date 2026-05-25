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

        # 2. Prompt construction
        prompts: List[str] = []
        for q, retrieved in zip(questions, retrieved_chunks_list):
            if self.use_retrieval and retrieved:
                prompt = build_rag_prompt(q["question"], retrieved, prompt_template)
            else:
                prompt = build_no_rag_prompt(q["question"])
            prompts.append(prompt)

        # 2b. Sanity log — tokenised prompt lengths.
        #
        # The generation cache is keyed on the *raw* prompt string but the
        # model sees the *truncated* token sequence. If raw-length means
        # cluster around or above max_input_tokens, conditions that should
        # differ (e.g. k=3 vs k=10) collapse to the same encoder input.
        # Logging this here lets a future-you tell "model finds extra
        # context useless" apart from "tokeniser ate the difference".
        prompt_tok_stats: Dict[str, float] = {}
        try:
            if self.generator._tokenizer is None:
                from transformers import AutoTokenizer
                self.generator._tokenizer = AutoTokenizer.from_pretrained(
                    self.generator.model_name
                )
            tok = self.generator._tokenizer
            raw_lens = [
                len(tok.encode(p, truncation=False, add_special_tokens=True))
                for p in prompts
            ]
            budget = self.generator.max_input_tokens
            trunc_count = sum(1 for L in raw_lens if L > budget)
            prompt_tok_stats = {
                "prompt_tokens_mean":   float(np.mean(raw_lens)),
                "prompt_tokens_max":    int(np.max(raw_lens)),
                "prompt_tokens_budget": int(budget),
                "prompts_truncated":    int(trunc_count),
                "n_prompts":            int(len(raw_lens)),
            }
            logger.info(
                "Prompt tokens: mean=%.0f  max=%d  budget=%d  truncated=%d/%d",
                prompt_tok_stats["prompt_tokens_mean"],
                prompt_tok_stats["prompt_tokens_max"],
                prompt_tok_stats["prompt_tokens_budget"],
                prompt_tok_stats["prompts_truncated"],
                prompt_tok_stats["n_prompts"],
            )
        except Exception as e:
            logger.warning("Prompt-length logging failed: %s", e)

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
            per_example.append({
                "qid":         q["qid"],
                "question":    q["question"],
                "gold":        golds,
                "prediction":  pred,
                "em":          exact_match(pred, golds),
                "f1":          token_f1(pred, golds),
                "recall":      retrieval_recall_single(retrieved, golds),
                "n_retrieved": len(retrieved),
                "prompt":      prompt,
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