## 3. Chunking

Chunking is one of the most impactful decisions in a RAG pipeline (Gao et al. 2024 [44]).

**Fixed-size word chunking with overlap** — split a document's text into
windows of W words, stepping by (W − overlap) words each time. Overlap
prevents an answer that straddles a chunk boundary from being lost.

Every chunk has schema `{chunk_id, doc_id, title, text, chunk_idx}`.
