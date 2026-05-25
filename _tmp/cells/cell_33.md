## 15. Analysis — figures

Generate the seven figures used in the report from the cached JSON results.
Each figure is saved at 300 dpi to `figures/` and displayed inline below.

| File | Description |
|------|-------------|
| `fig1_retriever_comparison.png` | EM + F1 for BM25 / TF-IDF / Dense |
| `fig2_chunk_size.png`           | EM / F1 / Recall@k vs chunk size |
| `fig3_k_values.png`             | EM / F1 / Recall@k vs k |
| `fig4_prompt_template.png`      | EM + F1 for concise vs instructed |
| `fig5_rag_vs_no_rag.png`        | EM + F1 grouped bar: RAG vs baseline |
| `fig6_error_analysis.png`       | Pie + scatter: when does RAG help? |
| `fig7_qualitative.png`          | Example table of RAG-helps vs RAG-hurts |
