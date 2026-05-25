# Detects the "[N] (..." chunk markers emitted by build_rag_prompt at the
# start of the context body and after every "\n\n" joiner. We use this to
# report "effective k" — how many of the retrieved chunks survive in the
# input the generator actually sees after middle-truncation.
_CHUNK_MARKER_RE = re.compile(r"(?:^|\n\n)\[(\d+)\] \(")


class Generator:
    """Wrapper around Flan-T5-base for deterministic QA generation.

    Accepts a mixed list of prompts at generate() time:
      * plain `str`                    — tokenised and right-truncated at
                                         `max_input_tokens` (default HF).
      * `(prefix, context, suffix)` tuple — *middle*-truncated so the
                                         prefix and suffix are preserved
                                         verbatim and only the context body
                                         is shortened when budget is tight.
    """

    def __init__(
        self,
        model_name: str = config.GENERATOR_MODEL,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        max_input_tokens: int = config.MAX_INPUT_TOKENS,
        batch_size: int = config.GENERATOR_BATCH_SIZE,
        cache_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.model_name       = model_name
        self.max_new_tokens   = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.batch_size       = batch_size
        self.cache_path       = cache_path or config.CACHE_DIR / "generation_cache.json"
        self._device          = device
        self._model           = None
        self._tokenizer       = None
        self._cache: Dict[str, str] = {}
        # Stats from the most recent generate() call, so that timing can
        # be interpreted (a fast generation_s is meaningless if everything
        # was served from cache).
        self.last_cache_hits:   int = 0
        self.last_cache_misses: int = 0
        # Number of (tuple-form) prompts in the most-recent generate() call
        # whose context body actually had to be middle-truncated to fit
        # max_input_tokens.
        self.last_n_truncated:  int = 0
        self._load_cache()

    def _load_tokenizer(self) -> None:
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM

        device_str = self._device
        if device_str is None:
            if torch.backends.mps.is_available():
                device_str = "mps"
            elif torch.cuda.is_available():
                device_str = "cuda"
            else:
                device_str = "cpu"
        self._device = device_str
        logger.info("Loading generator '%s' on device '%s'…", self.model_name, device_str)

        self._load_tokenizer()
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.to(device_str)
        self._model.eval()
        logger.info("Generator ready.")

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info("Loaded generation cache (%d entries).", len(self._cache))
            except Exception as e:
                logger.warning("Could not load generation cache: %s", e)
                self._cache = {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _cache_key(prompt: str) -> str:
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def _truncate_middle(
        self, parts: Tuple[str, str, str],
    ) -> Tuple[str, bool, int, int]:
        """Middle-truncate a (prefix, context, suffix) triple so the full
        string fits in `max_input_tokens`.

        Returns `(final_prompt, was_truncated, effective_k, nominal_k)`:
          * `was_truncated`  — True iff the context body had to be shortened.
          * `nominal_k`      — number of chunks the caller passed in
                               (counted from `[N] (` markers in `context`).
          * `effective_k`    — number of `[N] (` markers that survive in
                               the *rendered* prompt. Note that when
                               `was_truncated` is True the chunk at the
                               last surviving marker is typically only
                               partially visible — its text may have been
                               cut mid-sentence. Read effective_k as "the
                               model has at least the header + some text
                               for this many of the original chunks".

        Prefix + suffix are tokenised first and reserve their full length;
        the remaining budget is given to the context. If even prefix+suffix
        won't fit (pathological), we fall back to the default right-truncation
        and flag the prompt as truncated.
        """
        self._load_tokenizer()
        prefix, context, suffix = parts
        tok = self._tokenizer
        nominal_k = len(_CHUNK_MARKER_RE.findall(context))

        # add_special_tokens=False on each piece because T5 only appends one
        # </s> at the *end* of the joined sequence; counting it once is enough.
        pre_ids = tok.encode(prefix, add_special_tokens=False)
        suf_ids = tok.encode(suffix, add_special_tokens=False)
        budget  = self.max_input_tokens - len(pre_ids) - len(suf_ids) - 1  # -1 for EOS

        if budget <= 0:
            # Even prefix+suffix won't fit — fall back to right-truncate;
            # we don't know how many chunks survive in this pathological case.
            return prefix + context + suffix, True, 0, nominal_k

        ctx_ids_full = tok.encode(context, add_special_tokens=False)
        if len(ctx_ids_full) <= budget:
            return prefix + context + suffix, False, nominal_k, nominal_k

        ctx_text    = tok.decode(ctx_ids_full[:budget], skip_special_tokens=True)
        effective_k = len(_CHUNK_MARKER_RE.findall(ctx_text))
        return prefix + ctx_text + suffix, True, effective_k, nominal_k

    def render(self, prompt: Any) -> str:
        """Return the exact string that will be tokenised and fed to the
        model — handy for cache-key debugging and for the per-example log
        emitted by the pipeline."""
        if isinstance(prompt, tuple):
            rendered, *_ = self._truncate_middle(prompt)
            return rendered
        return prompt

    def render_with_effective_k(self, prompt: Any) -> Tuple[str, int, int]:
        """Like `render`, but also returns `(effective_k, nominal_k)`. For
        plain-string prompts (no retrieval), both are 0."""
        if isinstance(prompt, tuple):
            rendered, _was_trunc, eff_k, nom_k = self._truncate_middle(prompt)
            return rendered, eff_k, nom_k
        return prompt, 0, 0

    def generate(self, prompts: List[Any]) -> List[str]:
        """Generate answers for a batch of prompts (strings or 3-tuples)."""
        # Render each prompt to its final string form so cache keys match
        # exactly what the model will see.
        rendered: List[str] = []
        n_truncated = 0
        for p in prompts:
            if isinstance(p, tuple):
                final, was_trunc, _eff_k, _nom_k = self._truncate_middle(p)
                rendered.append(final)
                if was_trunc:
                    n_truncated += 1
            else:
                rendered.append(p)
        self.last_n_truncated = n_truncated

        results: List[Optional[str]] = [None] * len(rendered)
        uncached_indices: List[int] = []
        uncached_prompts: List[str] = []

        for i, prompt in enumerate(rendered):
            key = self._cache_key(prompt)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_prompts.append(prompt)

        self.last_cache_hits   = len(rendered) - len(uncached_prompts)
        self.last_cache_misses = len(uncached_prompts)
        logger.info("Generation: %d cached, %d to generate.",
                    self.last_cache_hits, self.last_cache_misses)

        if uncached_prompts:
            import torch
            self._load_model()

            generated: List[str] = []
            for batch_start in tqdm(
                range(0, len(uncached_prompts), self.batch_size),
                desc="Generating",
            ):
                batch = uncached_prompts[batch_start: batch_start + self.batch_size]
                enc = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_tokens,
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}

                with torch.no_grad():
                    out = self._model.generate(
                        **enc,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                    )
                decoded = self._tokenizer.batch_decode(out, skip_special_tokens=True)
                generated.extend(decoded)

            for orig_idx, prompt, answer in zip(uncached_indices, uncached_prompts, generated):
                results[orig_idx] = answer
                self._cache[self._cache_key(prompt)] = answer
            self._save_cache()

        return [r or "" for r in results]


def build_rag_prompt(
    question: str,
    retrieved: List[Dict],
    template: str = config.DEFAULT_PROMPT,
) -> Tuple[str, str, str]:
    """Build a RAG prompt as a (prefix, context, suffix) triple.

    Splitting at the `{context}` placeholder lets the generator
    middle-truncate the context body when the joined prompt would
    overflow `MAX_INPUT_TOKENS` — preserving the question repeats and
    the answer cue (`Short answer:`) verbatim.
    """
    template_str = config.PROMPT_TEMPLATES[template]
    if retrieved:
        context_parts = []
        for rank, chunk in enumerate(retrieved, 1):
            title = chunk.get("title", "Unknown")
            text  = chunk["text"]
            context_parts.append(f"[{rank}] ({title}) {text}")
        context = "\n\n".join(context_parts)
    else:
        context = ""
    pre_template, post_template = template_str.split("{context}", 1)
    prefix = pre_template.format(question=question)
    suffix = post_template.format(question=question)
    return (prefix, context, suffix)


def build_no_rag_prompt(question: str) -> str:
    """Prompt with no retrieved context — pure parametric generation."""
    return config.PROMPT_TEMPLATES["no_context"].format(question=question)
