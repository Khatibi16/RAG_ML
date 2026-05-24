"""
corpus.py — TriviaQA corpus loading and passage extraction.

We use the `rc` configuration of TriviaQA (Joshi et al. 2017), which pairs
each trivia question with two evidence sources: Wikipedia entity pages
(curated) and Web search results (noisier).  Both are pooled into a single
shared retrieval corpus, which makes retrieval genuinely difficult — the
correct passage now competes with web pages that may be only tangentially
related to any sampled question.

Design choices:
  - We pool both ``entity_pages`` (wiki) and ``search_results`` (web) from
    all sampled questions to form one shared retrieval corpus.
  - Documents are deduplicated by ``(source, filename)`` where filename is
    available, otherwise by title.
  - We cache the raw dataset download to avoid re-downloading on each run.
  - Each document is stored as a dict with fields ``doc_id``, ``title``,
    ``text``, ``source`` so the chunker and retrievers have a uniform
    interface.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def load_triviaqa(
    num_questions: int = config.NUM_QUESTIONS,
    cache_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load TriviaQA and return (questions, corpus_docs).

    Parameters
    ----------
    num_questions : int
        How many questions to use. We always take the *first* N from the
        validation split for reproducibility.
    cache_path : Path, optional
        Where to cache the parsed output. Defaults to
        ``config.CACHE_DIR / "triviaqa_{num_questions}.pkl"``.

    Returns
    -------
    questions : list of dicts
        Each dict has keys: ``qid``, ``question``, ``answers`` (list[str]).
    corpus_docs : list of dicts
        Each dict has keys: ``doc_id``, ``title``, ``text``.
        Total document count is capped at config.MAX_CORPUS_DOCS.
    """
    if cache_path is None:
        # Include all parsing-relevant config so changing the search cap
        # doesn't silently re-use a stale parsed corpus.
        web_cap = getattr(config, "MAX_SEARCH_RESULTS_PER_Q", None)
        web_tag = "all" if web_cap is None else str(web_cap)
        cache_path = (
            config.CACHE_DIR
            / f"triviaqa_{config.DATASET_CONFIG}_n{num_questions}_w{web_tag}.pkl"
        )

    if cache_path.exists():
        logger.info("Loading TriviaQA from cache: %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info(
        "Downloading TriviaQA (%s, %s split, first %d questions)…",
        config.DATASET_CONFIG,
        config.DATASET_SPLIT,
        num_questions,
    )

    # Lazy import so the module is importable even without `datasets`
    from datasets import load_dataset

    ds = load_dataset(
        config.DATASET_NAME,
        config.DATASET_CONFIG,
        split=config.DATASET_SPLIT,
    )

    questions: List[Dict[str, Any]] = []
    corpus_docs: List[Dict[str, Any]] = []
    seen_keys: set = set()

    # (source_name, section_key_in_example, text_field_in_section)
    SOURCE_SPECS = [
        ("wiki", "entity_pages",   "wiki_context"),
        ("web",  "search_results", "search_context"),
    ]
    MIN_DOC_CHARS = 50  # filter out empty / single-line search snippets

    subset = ds.select(range(min(num_questions, len(ds))))

    cap_reached = False
    for idx, example in enumerate(tqdm(subset, desc="Parsing TriviaQA")):
        # ── Build question entry ────────────────────────────────
        answers = _extract_answers(example)
        if not answers:
            continue

        q = {
            "qid":      example.get("question_id", str(idx)),
            "question": example["question"],
            "answers":  answers,
        }
        questions.append(q)

        if cap_reached:
            # Still record the question so it's evaluable, but don't add
            # any more docs (the answer's evidence may now be missing).
            continue

        # ── Build corpus from wiki entity pages + web search results ──
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

    n_wiki = sum(1 for d in corpus_docs if d["source"] == "wiki")
    n_web  = sum(1 for d in corpus_docs if d["source"] == "web")
    logger.info(
        "Loaded %d questions and %d corpus documents (%d wiki + %d web).",
        len(questions), len(corpus_docs), n_wiki, n_web,
    )

    result = (questions, corpus_docs)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    logger.info("Saved to cache: %s", cache_path)

    return result


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _extract_answers(example: Dict[str, Any]) -> List[str]:
    """Return all gold answer strings from a TriviaQA example."""
    ans_dict = example.get("answer", {})
    if not ans_dict:
        return []

    answers: List[str] = []

    # Primary value
    value = ans_dict.get("value", "")
    if value:
        answers.append(value)

    # Aliases (alternative surface forms)
    aliases = ans_dict.get("aliases", []) or []
    answers.extend([a for a in aliases if a])

    # Normalised aliases
    norm_aliases = ans_dict.get("normalized_aliases", []) or []
    answers.extend([a for a in norm_aliases if a])

    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for a in answers:
        low = a.lower()
        if low not in seen:
            seen.add(low)
            unique.append(a)

    return unique
