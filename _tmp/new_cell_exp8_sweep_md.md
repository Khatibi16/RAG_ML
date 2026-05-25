## 17. Experiment 8 — Distractor count sweep (noise-floor sensitivity)

The default corpus mixes `NUM_WIKI_DISTRACTORS = 2000` topic-agnostic
Simple-English Wikipedia articles into the retrieval pool. The choice
is somewhat arbitrary: too few and the retrieval task is essentially
"find the gold page among other questions' gold pages"; too many and
the in-batch gold becomes a needle in a haystack.

This experiment sweeps `NUM_WIKI_DISTRACTORS` ∈ `{0, 500, 2000, 5000}`
holding everything else fixed (same questions, same chunking, same
generator, same prompt). For each value we re-load the corpus,
re-build the dense index (with a distractor-count-specific cache),
and re-run the RAG arm. The No-RAG arm is corpus-independent so we
reuse the Experiment 5 result rather than re-running it.

Metrics to watch:

* **Recall@5** — should fall monotonically as the noise floor rises.
* **EM / F1** — fall *only if* Recall@5 falls AND the generator
  actually used the missing chunks. If EM is flat but Recall drops,
  the generator is robust to noisier retrieval; if both fall together,
  the bottleneck is retrieval.

This is the most direct test in the project of *"how much does
retrieval quality matter"* — the variable is the difficulty of the
retrieval task itself, with everything downstream held fixed.
