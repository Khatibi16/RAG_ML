from scipy.stats import binomtest


def _aligned_scores(per_a: List[Dict], per_b: List[Dict], field: str):
    """Return paired score arrays (a, b) aligned by qid (shared questions)."""
    bmap = {x["qid"]: x for x in per_b}
    a, b = [], []
    for x in per_a:
        if x["qid"] in bmap:
            a.append(float(x[field]))
            b.append(float(bmap[x["qid"]][field]))
    return np.array(a, dtype=float), np.array(b, dtype=float)


def mcnemar_em(per_a: List[Dict], per_b: List[Dict]) -> Dict[str, Any]:
    """McNemar's exact test on the paired EM hit/miss table.

    Only *discordant* pairs are informative: with the two systems scored on
    the same questions, let n_b = #(A right, B wrong) and n_c = #(A wrong,
    B right). Under H0 (a discordant pair is equally likely to favour A or B)
    n_b ~ Binomial(n_b + n_c, 0.5); the exact two-sided binomial tail is the
    p-value. Concordant pairs (both right / both wrong) carry no signal and
    are correctly ignored — which is exactly why this is more powerful than
    comparing two independent marginal CIs.
    """
    a, b = _aligned_scores(per_a, per_b, "em")
    a_hit, b_hit = a >= 0.5, b >= 0.5
    n_b = int(np.sum(a_hit & ~b_hit))   # A right, B wrong
    n_c = int(np.sum(~a_hit & b_hit))   # A wrong, B right
    n_disc = n_b + n_c
    p = (1.0 if n_disc == 0
         else float(binomtest(min(n_b, n_c), n_disc, 0.5,
                              alternative="two-sided").pvalue))
    return {"a_only": n_b, "b_only": n_c, "n_discordant": n_disc, "p_value": p}


def paired_bootstrap(
    per_a: List[Dict], per_b: List[Dict], field: str,
    n_boot: int = 10000, seed: int = config.RANDOM_SEED,
) -> Dict[str, Any]:
    """Paired bootstrap on the per-question metric delta (A - B).

    We resample QUESTIONS (not the two systems independently) with
    replacement and recompute the mean delta on each resample. Because the
    same resampled questions are scored under both systems, the
    question-difficulty variance that dominates the marginal CIs cancels, so
    the test is sensitive to a real per-question effect even when the two
    marginal CIs overlap heavily. Reports the mean delta, a 95% percentile
    CI, and a two-sided bootstrap p-value 2*min(P(delta*<0), P(delta*>0)).
    """
    a, b  = _aligned_scores(per_a, per_b, field)
    delta = a - b
    n     = len(delta)
    rng   = np.random.default_rng(seed)
    idx   = rng.integers(0, n, size=(n_boot, n))
    boot  = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(boot < 0)), float(np.mean(boot > 0)))
    return {
        "mean_delta": float(delta.mean()),
        "lo": float(lo), "hi": float(hi),
        "p_value": float(min(1.0, p)), "n": n,
    }


def paired_compare(per_a, per_b, label_a, label_b) -> Dict[str, Any]:
    em   = mcnemar_em(per_a, per_b)
    d_em = paired_bootstrap(per_a, per_b, "em")
    d_f1 = paired_bootstrap(per_a, per_b, "f1")
    logger.info(
        "%-13s vs %-13s | dEM=%+.3f [%+.3f,%+.3f] McNemar p=%.4f "
        "(disc %d/%d) | dF1=%+.3f [%+.3f,%+.3f] boot p=%.4f",
        label_a, label_b,
        d_em["mean_delta"], d_em["lo"], d_em["hi"], em["p_value"],
        em["a_only"], em["n_discordant"],
        d_f1["mean_delta"], d_f1["lo"], d_f1["hi"], d_f1["p_value"],
    )
    return {
        "contrast": f"{label_a} - {label_b}",
        "em_mcnemar": em,
        "em_paired_bootstrap": d_em,
        "f1_paired_bootstrap": d_f1,
    }


def run_paired_significance() -> List[Dict[str, Any]]:
    """Paired comparisons for every contrast that shares a question set.

    Marginal bootstrap CIs (as plotted in figs 1/4/5/8/9) answer "is this
    single number reliable?"; they are the WRONG tool for "is A better than
    B?" when A and B are scored on the same questions. These paired tests are
    the correct, more powerful complement.
    """
    logger.info("=" * 60)
    logger.info("Paired significance tests (McNemar on EM + paired bootstrap)")

    def per(name: str) -> List[Dict]:
        return load_results(name)["per_example"]

    contrasts = [
        paired_compare(per("exp1_bm25"),    per("exp1_dense"),  "BM25",         "Dense"),
        paired_compare(per("exp1_tfidf"),   per("exp1_dense"),  "TF-IDF",       "Dense"),
        paired_compare(per("exp1_bm25"),    per("exp1_tfidf"),  "BM25",         "TF-IDF"),
        paired_compare(per("exp4_instructed"), per("exp4_concise"), "Instructed", "Concise"),
        paired_compare(per("exp5_rag"),     per("exp5_no_rag"), "RAG",          "No-RAG"),
        paired_compare(per("exp7_rerank"),  per("exp1_dense"),  "Dense+Rerank", "Dense"),
        paired_compare(per("exp6_oracle"),  per("exp5_rag"),    "Oracle",       "RAG"),
    ]
    save_results("paired_significance", {"contrasts": contrasts})
    return contrasts
