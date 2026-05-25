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
        common = Counter(pred_tokens) & Counter(gold_tokens)
        n_common = sum(common.values())
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
    """1.0 if any gold answer appears (substring) in any retrieved chunk."""
    norm_answers = [normalize_answer(a) for a in gold_answers]
    for chunk in retrieved_chunks:
        norm_chunk = normalize_answer(chunk["text"])
        if any(a in norm_chunk for a in norm_answers):
            return 1.0
    return 0.0


def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = config.BOOTSTRAP_SAMPLES,
    seed: int = config.RANDOM_SEED,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval; returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    arr = np.array(scores, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(arr.mean()), float(lo), float(hi)


def evaluate_with_ci(
    predictions: List[str],
    gold_answers_list: List[List[str]],
    retrieved_chunks_list: Optional[List[List[Dict]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute EM, F1, and (optionally) Recall@k with 95% bootstrap CIs."""
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
