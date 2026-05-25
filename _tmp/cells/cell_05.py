def load_triviaqa(
    num_questions: int = config.NUM_QUESTIONS,
    cache_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load TriviaQA and return (questions, corpus_docs)."""
    if cache_path is None:
        # Include parsing-relevant config so changing the search cap or
        # the distractor count doesn't silently re-use a stale parsed corpus.
        web_cap = getattr(config, "MAX_SEARCH_RESULTS_PER_Q", None)
        web_tag = "all" if web_cap is None else str(web_cap)
        wiki_n  = int(getattr(config, "NUM_WIKI_DISTRACTORS", 0) or 0)
        cache_path = (
            config.CACHE_DIR
            / f"triviaqa_{config.DATASET_CONFIG}_n{num_questions}_w{web_tag}_wd{wiki_n}.pkl"
        )

    if cache_path.exists():
        logger.info("Loading TriviaQA from cache: %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info(
        "Downloading TriviaQA (%s, %s split, first %d questions)…",
        config.DATASET_CONFIG, config.DATASET_SPLIT, num_questions,
    )
    from datasets import load_dataset
    ds = load_dataset(
        config.DATASET_NAME, config.DATASET_CONFIG, split=config.DATASET_SPLIT,
    )

    questions: List[Dict[str, Any]] = []
    corpus_docs: List[Dict[str, Any]] = []
    seen_keys: set = set()

    # (source_name, section_key_in_example, text_field_in_section)
    SOURCE_SPECS = [
        ("wiki", "entity_pages",   "wiki_context"),
        ("web",  "search_results", "search_context"),
    ]
    MIN_DOC_CHARS = 50

    subset = ds.select(range(min(num_questions, len(ds))))

    cap_reached = False
    for idx, example in enumerate(tqdm(subset, desc="Parsing TriviaQA")):
        answers = _extract_answers(example)
        if not answers:
            continue
        questions.append({
            "qid":      example.get("question_id", str(idx)),
            "question": example["question"],
            "answers":  answers,
        })

        if cap_reached:
            continue  # keep question, but stop adding evidence

        web_cap = getattr(config, "MAX_SEARCH_RESULTS_PER_Q", None)
        for src, section_key, text_field in SOURCE_SPECS:
            section = example.get(section_key) or {}
            titles    = section.get("title")    or []
            texts     = section.get(text_field) or []
            filenames = section.get("filename") or []
            n         = max(len(titles), len(texts), len(filenames))
            if src == "web" and web_cap is not None:
                n = min(n, web_cap)

            for i in range(n):
                title    = titles[i]    if i < len(titles)    else ""
                text     = texts[i]     if i < len(texts)     else ""
                filename = filenames[i] if i < len(filenames) else ""
                text = (text or "").strip()
                if len(text) < MIN_DOC_CHARS:
                    continue
                dedup_id = filename or title
                if not dedup_id:
                    continue
                dedup_key = (src, dedup_id)
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                corpus_docs.append({
                    "doc_id": f"{src}_{len(corpus_docs):06d}",
                    "title":  title or f"(untitled {src})",
                    "text":   text,
                    "source": src,
                })
                if len(corpus_docs) >= config.MAX_CORPUS_DOCS:
                    cap_reached = True
                    break
            if cap_reached:
                break
        if cap_reached:
            logger.info("Reached MAX_CORPUS_DOCS=%d — no more docs will be added.",
                        config.MAX_CORPUS_DOCS)

    # External Wikipedia distractors (Issue 2.4 mitigation).
    wiki_n = int(getattr(config, "NUM_WIKI_DISTRACTORS", 0) or 0)
    if wiki_n > 0 and not cap_reached:
        max_chars = int(getattr(config, "WIKI_DISTRACTOR_MAX_CHARS", 2000))
        budget = max(0, config.MAX_CORPUS_DOCS - len(corpus_docs))
        target = min(wiki_n, budget)
        distractors = _load_wiki_distractors(target, max_chars=max_chars)
        for d in distractors:
            dedup_key = ("wiki_distractor", d["title"])
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            d = {**d, "doc_id": f"wiki_distractor_{len(corpus_docs):06d}"}
            corpus_docs.append(d)
            if len(corpus_docs) >= config.MAX_CORPUS_DOCS:
                logger.info("Reached MAX_CORPUS_DOCS=%d while adding wiki distractors.",
                            config.MAX_CORPUS_DOCS)
                break

    n_wiki = sum(1 for d in corpus_docs if d["source"] == "wiki")
    n_web  = sum(1 for d in corpus_docs if d["source"] == "web")
    n_dist = sum(1 for d in corpus_docs if d["source"] == "wiki_distractor")
    logger.info(
        "Loaded %d questions and %d corpus documents (%d wiki + %d web + %d distractor).",
        len(questions), len(corpus_docs), n_wiki, n_web, n_dist,
    )

    result = (questions, corpus_docs)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    logger.info("Saved to cache: %s", cache_path)

    return result


def _extract_answers(example: Dict[str, Any]) -> List[str]:
    """Return all gold answer strings from a TriviaQA example."""
    ans_dict = example.get("answer", {})
    if not ans_dict:
        return []

    answers: List[str] = []
    value = ans_dict.get("value", "")
    if value:
        answers.append(value)
    answers.extend([a for a in (ans_dict.get("aliases") or []) if a])
    answers.extend([a for a in (ans_dict.get("normalized_aliases") or []) if a])

    seen: set = set()
    unique: List[str] = []
    for a in answers:
        low = a.lower()
        if low not in seen:
            seen.add(low)
            unique.append(a)
    return unique


def _load_wiki_distractors(
    n: int,
    max_chars: int = 2000,
    dataset_name: str = "wikimedia/wikipedia",
    dataset_config: str = "20231101.simple",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load N Wikipedia articles to use as external (topic-agnostic) distractors.

    Uses Simple English Wikipedia by default (~200k articles, ~100 MB download)
    so the corpus picks up a realistic open-domain noise floor without paying
    for the full English Wikipedia. Returns docs with source='wiki_distractor';
    each article is truncated to `max_chars` characters so chunk counts stay
    bounded.
    """
    if n <= 0:
        return []
    try:
        from datasets import load_dataset
        logger.info(
            "Loading %d Wikipedia distractor articles (%s / %s)…",
            n, dataset_name, dataset_config,
        )
        ds = load_dataset(dataset_name, dataset_config, split="train")
        # Deterministic sample. shuffle() on the small Simple-English subset
        # is fast; on full English Wikipedia, switch to streaming if needed.
        n_take = min(n, len(ds))
        ds = ds.shuffle(seed=seed).select(range(n_take))

        distractors: List[Dict[str, Any]] = []
        for ex in ds:
            text  = (ex.get("text") or "").strip()
            title = (ex.get("title") or "").strip()
            if not text or len(text) < 50:
                continue
            distractors.append({
                "doc_id": "",  # assigned by caller (depends on insertion order)
                "title":  title or "(Wikipedia)",
                "text":   text[:max_chars],
                "source": "wiki_distractor",
            })
        logger.info("Prepared %d wiki distractor documents.", len(distractors))
        return distractors
    except Exception as e:
        logger.warning(
            "Could not load wiki distractors (%s); proceeding without them.", e,
        )
        return []