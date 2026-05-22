"""
corpus.py — TriviaQA corpus loading and passage extraction.

We use the `rc.wikipedia` configuration of TriviaQA (Lewis et al. 2020 [42];
Joshi et al. 2017), which pairs each trivia question with a set of Wikipedia
entity pages as evidence. This is the exact benchmark used in all three
reference papers.

Design choices:
  - We pool entity pages from all sampled questions to form one shared
    retrieval corpus. This means relevant pages exist in the corpus, but so do
    many distractor pages — a realistic retrieval setting.
  - We cache the raw dataset download to avoid re-downloading on each run.
  - Each document is stored as a dict with fields 'doc_id', 'title', 'text'
    so the chunker and retrievers have a uniform interface.
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
        cache_path = config.CACHE_DIR / f"triviaqa_{num_questions}.pkl"

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
    seen_titles: set = set()

    subset = ds.select(range(min(num_questions, len(ds))))

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

        # ── Build corpus from entity pages ──────────────────────
        # TriviaQA rc.wikipedia stores Wikipedia pages under
        # example["entity_pages"] (a dict with 'title' and 'wiki_context')
        entity_pages = example.get("entity_pages", {})
        titles = entity_pages.get("title", []) or []
        texts  = entity_pages.get("wiki_context", []) or []

        for title, text in zip(titles, texts):
            if not title or not text:
                continue
            if title in seen_titles:
                continue
            seen_titles.add(title)
            corpus_docs.append({
                "doc_id": f"wiki_{len(corpus_docs):06d}",
                "title":  title,
                "text":   text.strip(),
            })

            if len(corpus_docs) >= config.MAX_CORPUS_DOCS:
                break

        if len(corpus_docs) >= config.MAX_CORPUS_DOCS:
            logger.info("Reached MAX_CORPUS_DOCS=%d — stopping corpus collection.",
                        config.MAX_CORPUS_DOCS)
            break

    logger.info(
        "Loaded %d questions and %d corpus documents.",
        len(questions), len(corpus_docs),
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
