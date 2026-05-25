## 7. End-to-end RAG pipeline

`RAGPipeline` ties retriever + generator + evaluator together.  A single call
to `run(questions, k, prompt_template)` returns predictions, retrieved chunks,
metrics with CIs, per-example breakdowns, and timing.

Passing `use_retrieval=False` (and `retriever=None`) bypasses retrieval and
uses the `no_context` prompt template — used for the no-RAG baseline.
