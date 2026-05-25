## 6. Evaluation metrics

Follows the TriviaQA / SQuAD evaluation protocol exactly:

* **Exact Match (EM)** — 1 if the normalised prediction equals any normalised
  gold answer (lowercase, strip articles & punctuation, collapse whitespace).
* **Token F1** — token-level F1 against the best-matching gold answer.
* **Recall@k** — fraction of questions for which any gold answer appears
  (substring) in at least one of the top-k retrieved chunks.
* **95% bootstrap CI** — `BOOTSTRAP_SAMPLES` resamples; percentile CI.
