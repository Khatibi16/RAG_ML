## 4. Retrieval backends

Three backends with a uniform interface (`build(chunks)`, `retrieve(query, k)`):

* **BM25** — `rank-bm25` Okapi BM25. Indexes a token stream produced by
  `_lexical_tokenize` (lowercase, accent-strip, regex `\b\w\w+\b`).
* **TF-IDF** — `sklearn.feature_extraction.text.TfidfVectorizer` configured
  to consume the *same* `_lexical_tokenize` token stream as BM25, unigrams
  only, sublinear TF, no DF filtering. So BM25 vs TF-IDF differs **only**
  in the scoring function — not in tokenisation, vocabulary, or n-gram order.
* **Dense** — `sentence-transformers/all-MiniLM-L6-v2` (22 M parameters,
  384-dim L2-normalised embeddings; cosine = dot product). Embeddings are
  cached to disk and re-used across experiments.
