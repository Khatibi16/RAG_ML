## 8. Experiment helpers

Shared infrastructure used by all five experiments:

* `save_results` / `load_results` — JSON serialisation to `results/`.
  Heavyweight fields (`retrieved_chunks`, `prompts`) are stripped to keep
  files small.
* `already_done` — lets re-running the notebook skip experiments whose
  results are already on disk (delete the JSON to force a re-run).
* `print_metrics` — compact log line with EM / F1 / Recall@k and CIs.
* `setup` — loads TriviaQA, prints corpus stats, instantiates the generator.
