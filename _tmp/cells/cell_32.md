**Interpretation.** A small spread across reasonable prompts (~5pp EM)
indicates the No-RAG baseline is robust to prompt formulation, and the
arm can be reported as a parametric-knowledge floor for Flan-T5-base on
TriviaQA-grade facts. Inspection of the saved `sample` predictions
further confirms that the failures are not EM-normalisation artefacts
(answers are short and well-formed) but genuine knowledge gaps — the
model produces confidently-shaped but unrelated entity strings such as
`"henry viii"` for *"PM after Balfour"* or `"johnny mccartney"* for a
70s music question. We therefore (a) adopt the best-scoring template
(`qa_cue`) as the canonical no-context prompt going forward, and
(b) report F1 alongside EM for the No-RAG arm in Experiment 5: at this
EM range, a single-question swing is 1 pp and F1 is the more
noise-tolerant secondary metric.

A large spread (>10pp EM) — not what we observe here — would instead
flag the No-RAG arm as prompt-dependent, in which case the canonical
choice would require justification beyond "we picked the highest".

The full sweep is retained in `results/exp5_no_rag_prompt_sensitivity.json`
for transparency.