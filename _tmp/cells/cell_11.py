class Generator:
    """Wrapper around Flan-T5-base for deterministic QA generation."""

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
        self._load_cache()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model     = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
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

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate answers for a batch of prompts."""
        results: List[Optional[str]] = [None] * len(prompts)
        uncached_indices: List[int] = []
        uncached_prompts: List[str] = []

        for i, prompt in enumerate(prompts):
            key = self._cache_key(prompt)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_prompts.append(prompt)

        self.last_cache_hits   = len(prompts) - len(uncached_prompts)
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
) -> str:
    """Construct a RAG prompt from retrieved chunks and a question."""
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
    return template_str.format(context=context, question=question)


def build_no_rag_prompt(question: str) -> str:
    """Prompt with no retrieved context — pure parametric generation."""
    return config.PROMPT_TEMPLATES["no_context"].format(context="", question=question)
