## 15. Experiment 6 — Oracle baseline (retrieval upper bound)

The previous experiments confound two sources of error: the retriever
failing to find the answer, and the generator failing to use it once
found. The **oracle** baseline disentangles them by *guaranteeing* the
retrieved context contains the answer.

Procedure: run dense retrieval as usual at k=5. For any question whose
top-5 chunks miss the gold answer (recall = 0), find the first chunk in
the corpus that does contain it (normalised substring match) and place
it at rank 1, dropping the lowest-ranked retrieved chunk so k stays at
5. If no chunk in the corpus contains the answer (chunking artefact —
the answer may straddle a window boundary), the question is left as-is.

Interpretation:
- **Oracle EM/F1 − RAG EM/F1** = the gap attributable to *retrieval
  failure*. Closes if the retriever already finds the answer.
- **1 − Oracle EM/F1** = the residual *generator* failure even when
  retrieval is perfect. This is a ceiling on what better retrieval
  alone can buy us.
