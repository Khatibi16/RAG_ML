def save_results(name: str, data: dict) -> Path:
    """Serialise results to JSON in config.RESULTS_DIR (drops heavy fields)."""
    out_path = config.RESULTS_DIR / f"{name}.json"
    slim = {k: v for k, v in data.items() if k not in ("retrieved_chunks", "prompts")}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    logger.info("Results saved → %s", out_path)
    return out_path


def already_done(name: str) -> bool:
    """
    Return True only if (a) config.FORCE_RERUN is False AND (b) the results
    JSON already exists. Default config.FORCE_RERUN=True forces every
    experiment to recompute end-to-end, so methodology edits never get
    masked by a stale results file.
    """
    if getattr(config, "FORCE_RERUN", True):
        return False
    p = config.RESULTS_DIR / f"{name}.json"
    if p.exists():
        logger.info("Skipping %s — already done (delete %s to re-run).",
                    name, p.name)
        return True
    return False


def load_results(name: str) -> dict:
    out_path = config.RESULTS_DIR / f"{name}.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_metrics(label: str, result: dict) -> None:
    m = result["metrics"]
    em  = m.get("em",  {})
    f1  = m.get("f1",  {})
    rec = m.get("recall_at_k", {})
    logger.info(
        "%s — EM %.3f [%.3f-%.3f]  F1 %.3f [%.3f-%.3f]  Recall %.3f [%.3f-%.3f]",
        label,
        em.get("mean", 0),  em.get("lo", 0),  em.get("hi", 0),
        f1.get("mean", 0),  f1.get("lo", 0),  f1.get("hi", 0),
        rec.get("mean", 0), rec.get("lo", 0), rec.get("hi", 0),
    )


def setup():
    """Load data and return (questions, corpus_docs, generator)."""
    logger.info("=" * 60)
    logger.info("Loading TriviaQA (%d questions)…", config.NUM_QUESTIONS)
    questions, corpus_docs = load_triviaqa(num_questions=config.NUM_QUESTIONS)
    logger.info("Corpus: %d documents", len(corpus_docs))
    generator = Generator()
    return questions, corpus_docs, generator
