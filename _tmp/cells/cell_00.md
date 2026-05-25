# RAG Performance Analysis on TriviaQA

End-to-end retrieval-augmented generation benchmark on the TriviaQA
(`rc`) dataset (wiki entity pages + web search results pooled into one
corpus).  The pipeline is:

1. **Corpus** — load TriviaQA questions plus their Wikipedia entity pages and web search results, pooled into one shared retrieval corpus.
2. **Chunking** — split pages into fixed-size overlapping word windows.
3. **Retrieval** — BM25, TF-IDF, or dense sentence-transformer embeddings.
4. **Generation** — `google/flan-t5-base` (instruction-tuned, deterministic).
5. **Evaluation** — Exact-Match, Token-F1, Recall@k with 95% bootstrap CIs.

Five experiments are executed end-to-end inside this notebook:

| # | Variable | Fixed |
|---|----------|-------|
| 1 | retriever ∈ {BM25, TF-IDF, Dense} | chunk=128, k=5, instructed prompt |
| 2 | chunk size ∈ {64, 128, 256, 512} | dense, k=5, instructed prompt |
| 3 | k ∈ {1, 3, 5, 10} | dense, chunk=128, instructed prompt |
| 4 | prompt ∈ {concise, instructed} | dense, chunk=128, k=5 |
| 5 | RAG vs No-RAG baseline | dense, chunk=128, k=5, instructed prompt |

All results are cached as JSON to `results/`; figures land in `figures/`.
Cached embeddings and generation outputs live in `data/cache/` so re-running is cheap.