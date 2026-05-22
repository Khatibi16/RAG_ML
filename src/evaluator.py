"""
evaluator.py — Answer quality and retrieval quality metrics.

We follow the TriviaQA / SQuAD evaluation protocol exactly:

Answer metrics
--------------
  * Exact Match (EM): 1 if the predicted answer (after normalisation) equals
    any gold answer, 0 otherwise.  Normalisation: lowercase, strip articles
    ("a", "an", "the"), strip punctuation, collapse whitespace.
  * Token F1: precision/recall of shared tokens between prediction and the
    best-matching gold answer.  Same normalisation applied first.

Retrieval metric
----------------
  * Recall@k: fraction of questions for which the correct answer string
    appears in at least one of the top-k retrieved chunks.  This is the
    standard way to measure retrieval quality when exact relevance labels
    are not available (Lewis et al. 2020 [42]).

Confidence intervals
--------------------
  * 95% bootstrap CI with config.BOOTSTRAP_SAMPLES resamples.  The standard
    deviation across resamples gives the CI half-width.
"""

import re
import string
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Text normalisation (SQuAD / TriviaQA convention)
# ─────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lower, strip articles & punctuation, collapse whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ─────────────────────────────────────────────────────────────────
# Single-example metrics
# ─────────────────────────────────────────────────────────────────

def exact_match(prediction: str, gold_answers: List[str]) -> float:
    """1.0 if normalised prediction matches any normalised gold answer."""
    norm_pred = normalize_answer(prediction)
    return float(any(norm_pred == normalize_answer(g) for g in gold_answers))


def token_f1(prediction: str, gold_answers: List[str]) -> float:
    """Token-level F1 against the best-matching gold answer."""
    pred_tokens = normalize_answer(prediction).split()

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        common      = Counter(pred_tokens) & Counter(gold_tokens)
        n_common    = sum(common.values())

        if n_common == 0:
            continue

        precision = n_common / len(pred_tokens)
        recall    = n_common / len(gold_tokens)
        f1        = 2 * precision * recall / (precision + recall)
        best_f1   = max(best_f1, f1)

    return best_f1


def retrieval_recall_single(
    retrieved_chunks: List[Dict],
    gold_answers: List[str],
) -> float:
    """
    1.0 if any gold answer appears (substring) in any retrieved chunk.
    We use substring match after normalisation — standard for open-retrieval QA.
    """
    norm_answers = [normalize_answer(a) for a in gold_answers]
    for chunk in retrieved_chunks:
        norm_chunk = normalize_answer(chunk["text"])
        if any(a in norm_chunk for a in norm_answers):
            return 1.0
    return 0.0


# ─────────────────────────────────────────────────────────────────
# Corpus-level metrics
# ─────────────────────────────────────────────────────────────────

def compute_metrics(
    predictions: List[str],
    gold_answers_list: List[List[str]],
    retrieved_chunks_list: Optional[List[List[Dict]]] = None,
) -> Dict[str, float]:
    """
    Compute EM, F1, and optionally Recall@k over the full set.

    Parameters
    ----------
    predictions           : list of model-generated answer strings
    gold_answers_list     : list of gold answer lists (one per question)
    retrieved_chunks_list : optional list of retrieved chunk lists

    Returns
    -------
    dict with keys: em, f1, recall_at_k (only if retrieved_chunks_list given)
    """
    assert len(predictions) == len(gold_answers_list), \
        "predictions and gold_answers_list must have the same length"

    em_scores = []
    f1_scores = []

    for pred, golds in zip(predictions, gold_answers_list):
        em_scores.append(exact_match(pred, golds))
        f1_scores.append(token_f1(pred, golds))

    result: Dict[str, float] = {
        "em":  float(np.mean(em_scores)),
        "f1":  float(np.mean(f1_scores)),
        "n":   float(len(predictions)),
    }

    if retrieved_chunks_list is not None:
        recall_scores = [
            retrieval_recall_single(chunks, golds)
            for chunks, golds in zip(retrieved_chunks_list, gold_answers_list)
        ]
        result["recall_at_k"] = float(np.mean(recall_scores))

    return result


def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = config.BOOTSTRAP_SAMPLES,
    seed: int = config.RANDOM_SEED,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval.

    Returns
    -------
    (mean, lower_bound, upper_bound) at the requested CI level.
    """
    rng     = np.random.default_rng(seed)
    arr     = np.array(scores, dtype=float)
    means   = [rng.choice(arr, size=len(arr), replace=True).mean()
               for _ in range(n_bootstrap)]
    alpha   = (1 - ci) / 2
    lo, hi  = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(arr.mean()), float(lo), float(hi)


def evaluate_with_ci(
    predictions: List[str],
    gold_answers_list: List[List[str]],
    retrieved_chunks_list: Optional[List[List[Dict]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Like compute_metrics but also returns 95% CI for each metric.

    Returns
    -------
    dict mapping metric name → {'mean': ..., 'lo': ..., 'hi': ...}
    """
    em_scores = [exact_match(p, g) for p, g in zip(predictions, gold_answers_list)]
    f1_scores = [token_f1(p, g)    for p, g in zip(predictions, gold_answers_list)]

    result: Dict[str, Dict] = {}
    for name, scores in [("em", em_scores), ("f1", f1_scores)]:
        mean, lo, hi = bootstrap_ci(scores)
        result[name] = {"mean": mean, "lo": lo, "hi": hi, "n": len(scores)}

    if retrieved_chunks_list is not None:
        recall_scores = [
            retrieval_recall_single(chunks, golds)
            for chunks, golds in zip(retrieved_chunks_list, gold_answers_list)
        ]
        mean, lo, hi = bootstrap_ci(recall_scores)
        result["recall_at_k"] = {"mean": mean, "lo": lo, "hi": hi, "n": len(recall_scores)}

    return result
