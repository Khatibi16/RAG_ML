## 5. Generator — Flan-T5-base

* `google/flan-t5-base` (~250 M params): instruction-tuned T5 that follows
  diverse prompts without few-shot examples and runs on CPU/MPS/CUDA.
* Greedy decoding (`do_sample=False`, `num_beams=1`) for full determinism.
* Predictions are keyed by the MD5 of the *rendered* prompt (after any
  middle-truncation) and persisted to disk so re-runs only regenerate
  prompts that actually changed.
* **Middle-truncation of long prompts.** RAG prompts are constructed as a
  three-piece tuple `(prefix, context, suffix)`. The prefix carries the
  leading question and `Context:` header; the suffix carries the trailing
  question repeat and the answer cue (`Short answer:`). When the joined
  prompt exceeds `MAX_INPUT_TOKENS`, the generator preserves prefix and
  suffix verbatim and trims only the *context body* — so the model never
  loses the answer cue, only the lowest-ranked retrieved passages. The
  no-RAG prompt is short (≤40 tokens) and passes through as a plain
  string with no truncation.
