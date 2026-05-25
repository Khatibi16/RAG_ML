## 5. Generator — Flan-T5-base

* `google/flan-t5-base` (~250 M params): instruction-tuned T5 that follows
  diverse prompts without few-shot examples and runs on CPU/MPS/CUDA.
* Greedy decoding (`do_sample=False`, `num_beams=1`) for full determinism.
* Predictions are keyed by the MD5 of the prompt and persisted to disk so
  re-runs only regenerate prompts that actually changed.
* Prompts are truncated to `MAX_INPUT_TOKENS` to fit T5's 512-token context.
