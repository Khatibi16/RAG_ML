## 16. Experiment 7 — Cross-encoder re-ranking

A bi-encoder dense retriever embeds the query and each chunk
*independently*, so it cannot model interactions between specific
query terms and specific chunk spans. A cross-encoder reads the
`(query, chunk)` pair jointly and produces a single relevance score
— much more accurate per-pair, but quadratic in #chunks so it can't
be used as a first-stage retriever over a 10 k-chunk corpus.

The standard two-stage solution (Gao et al. [44]):
1. **Stage 1 (recall):** dense bi-encoder returns the top-N
   candidates (`RERANK_TOP_N = 50`).
2. **Stage 2 (precision):** cross-encoder
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores every pair
   and the top-k are kept for the generator.

This experiment compares Dense-only (Exp 1) against Dense + Rerank
at the same `k=5`. The interesting questions are:

* Does the reranker raise Recall@5? (i.e. does dense top-50 actually
  contain answer-bearing chunks that the dense top-5 was missing?)
* Does any Recall@5 lift translate into EM/F1 lift, or does the
  generator already use whatever it's given?
