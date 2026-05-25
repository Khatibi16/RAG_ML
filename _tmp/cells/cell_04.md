## 2. Corpus loading — TriviaQA (`rc`)

We use the `rc` configuration of TriviaQA (Lewis et al. 2020 [42];
Joshi et al. 2017), which pairs each trivia question with **two** evidence
sources: curated Wikipedia entity pages **and** noisy web search results.
We pool both into a single shared retrieval corpus so the retriever has
to find the answer among genuine distractors rather than only hand-picked
evidence.

* Entity pages and web hits from all sampled questions are pooled into one
  shared retrieval corpus, so each question's relevant pages compete against
  many distractors (a realistic setting). The web-hit count per question is
  capped by `config.MAX_SEARCH_RESULTS_PER_Q`.
* Each document carries a `source` field (`"wiki"` or `"web"`) so per-source
  analyses are possible.
* The raw dataset is cached as a pickle file so subsequent runs skip the download.
* Each corpus document has the uniform schema `{doc_id, title, text, source}`.