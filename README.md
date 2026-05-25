# When Does Retrieval-Augmented Generation Help?

**Project 15 — 30562 Machine Learning and Artificial Intelligence**

---

## Table of Contents

1. [Motivation & Research Questions](#1-motivation--research-questions)
2. [Background & Related Work](#2-background--related-work)
3. [System Architecture](#3-system-architecture)
4. [Dataset](#4-dataset)
5. [Experimental Design](#5-experimental-design)
   - [Exp 1 — Retriever Comparison](#exp-1--retriever-comparison-bm25-vs-tf-idf-vs-dense)
   - [Exp 2 — Chunk Size](#exp-2--chunk-size-ablation)
   - [Exp 3 — Number of Passages k](#exp-3--number-of-retrieved-passages-k)
   - [Exp 4 — Prompt Template](#exp-4--prompt-template-ablation)
   - [Exp 5 — RAG vs No-RAG](#exp-5--rag-vs-no-rag)
   - [Exp 6 — Oracle Baseline](#exp-6--oracle-baseline-retrieval-upper-bound)
   - [Exp 7 — Cross-Encoder Re-ranking](#exp-7--cross-encoder-re-ranking)
   - [Exp 8 — Distractor Count Sweep](#exp-8--distractor-count-sweep)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Implementation Details](#7-implementation-details)
8. [Reproducibility](#8-reproducibility)
9. [How to Run](#9-how-to-run)
10. [Results Summary](#10-results-summary)
11. [Limitations](#11-limitations)
12. [References](#12-references)

---

## 1. Motivation & Research Questions

Large language models (LLMs) encode a vast amount of world knowledge in their
parameters during pre-training. Yet this *parametric memory* has well-known
failure modes: it can be outdated, hallucinated, or simply absent for
long-tail facts. Retrieval-Augmented Generation (RAG) augments a generator
with a *non-parametric* external memory — a corpus of text documents — so
that relevant facts can be looked up at inference time.

Despite the conceptual appeal, RAG is not uniformly beneficial:
- Retrieval can be noisy or miss the answer entirely.
- Irrelevant retrieved passages can *distract* the generator.
- Prompting styles strongly influence how well the model uses the context.
- Chunk granularity and retrieval depth are often tuned by intuition, not evidence.

This project provides a **rigorous empirical investigation** of these questions:

| Research Question | Experiment |
|---|---|
| Which retrieval method (lexical vs semantic) produces the best answers? | Exp 1 |
| How does chunk granularity affect retrieval and generation? | Exp 2 |
| Is more context (larger k) always better? | Exp 3 |
| Does the prompt template matter, and by how much? | Exp 4 |
| When does retrieval help — and when does it actively hurt? | Exp 5 |

---

## 2. Background & Related Work

### RAG (Lewis et al., 2020) [42]

Lewis et al. introduce the RAG model family: a parametric generator (BART)
augmented with a dense retriever (DPR) over Wikipedia. During inference,
the top-k retrieved passages are fed as additional context to the generator.
They demonstrate strong results on open-domain QA benchmarks (Natural
Questions, TriviaQA) without task-specific fine-tuning of the retriever.

Key finding: retrieval helps most when the model's parametric memory is
insufficient — i.e., for specific factual questions about named entities,
dates, and quantities.

### Fusion-in-Decoder (Izacard & Grave, 2021) [43]

FiD extends RAG by encoding each retrieved passage *independently* in the
encoder, then concatenating all encoder outputs before the decoder attends
to them. This allows scaling to many more passages without overwhelming the
cross-attention mechanism. FiD consistently outperforms standard RAG when k
is large (≥ 5), suggesting that *how* multiple passages are integrated into
the generator matters as much as *whether* they are retrieved.

### RAG Survey (Gao et al., 2024) [44]

Gao et al. provide a comprehensive taxonomy of the RAG design space: naive
RAG (retrieve then read), advanced RAG (query rewriting, re-ranking), and
modular RAG (iterative, adaptive retrieval). They identify chunking strategy
and embedding model as two of the highest-impact design choices. Our
experiments directly test several of their identified dimensions.

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  RAG Pipeline                            │
│                                                          │
│  Question ──► Retriever ──► Top-k Chunks                │
│                  │                                       │
│           ┌──────┴──────┐                               │
│           │   BM25      │                               │
│           │   TF-IDF    │                               │
│           │   Dense     │                               │
│           └─────────────┘                               │
│                                                          │
│  Question + Top-k Chunks ──► Prompt Template ──► Prompt │
│                                                          │
│  Prompt ──► Flan-T5-base ──► Generated Answer           │
│                                                          │
│  Generated Answer + Gold Answers ──► EM / F1 / Recall   │
└──────────────────────────────────────────────────────────┘
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `config.py` | All hyperparameters (single source of truth) |
| `src/corpus.py` | TriviaQA loading; Wikipedia entity page extraction |
| `src/chunker.py` | Fixed-size word-window chunking with configurable overlap |
| `src/retriever.py` | BM25Retriever, TFIDFRetriever, DenseRetriever classes |
| `src/generator.py` | Flan-T5-base wrapper with deterministic generation + caching |
| `src/evaluator.py` | Exact Match, Token F1, Recall@k, bootstrap CI |
| `src/pipeline.py` | Orchestrates retrieve → prompt → generate → evaluate |
| `experiments/run_experiments.py` | Runs all 5 experiments; writes JSON results |
| `analysis/plot_results.py` | Generates all 7 figures from JSON results |

---

## 4. Dataset

**TriviaQA** (`rc` configuration, Joshi et al. 2017).

We use this benchmark because it is the primary dataset in all three
reference papers (Lewis et al. [42], Izacard & Grave [43], Gao et al. [44]),
enabling direct comparison with published results.

### Why TriviaQA `rc`?

The `rc` (reading comprehension) configuration provides each trivia question
with **two** evidence sources: Wikipedia *entity pages* (curated, one per
named entity mentioned in the question) and *web search results* (snippets
of pages returned by a real search engine for the question). Pooling both
into one corpus gives a realistic retrieval setting:

- The corpus contains **relevant** documents (some entity page or search
  result that actually contains the answer) **and** many **distractor**
  documents (web pages that share topical vocabulary with the question but
  do not contain the answer, plus all the evidence from *other* questions).
- The retriever must identify the answer-bearing passage without knowing
  which question its evidence was originally collected for, and without the
  guarantee that the answer-bearing passage is a hand-picked Wikipedia
  article. Web search results are noisier, so the task is harder than
  `rc.wikipedia`.

### Corpus construction

We take the first **`config.NUM_QUESTIONS`** questions from the validation
split (default **100** for fast iteration; raise to 500–1000 for the final
run). For each question, TriviaQA provides ~2–10 entity Wikipedia pages
plus ~10–50 web search hits; we keep all the wiki pages and the top
`config.MAX_SEARCH_RESULTS_PER_Q` (default **5**) web hits. All of these
are pooled across questions into a single shared retrieval corpus,
deduplicated by `(source, filename)`. Documents shorter than 50 characters
are dropped.

We additionally mix in **`config.NUM_WIKI_DISTRACTORS`** (default **2 000**)
articles from Simple English Wikipedia
(`wikimedia/wikipedia` / `20231101.simple`, deterministically shuffled with
seed 42, truncated to `config.WIKI_DISTRACTOR_MAX_CHARS = 2 000` chars
each) as topic-agnostic external distractors. These articles carry no
question-conditioned signal; their job is to widen the noise floor of the
retrieval pool so the retriever has to find the gold-bearing pages among
documents whose presence is *not* correlated with our test questions. Set
`NUM_WIKI_DISTRACTORS = 0` to fall back to the pure-TriviaQA pool. The
first run additionally downloads ~100 MB of Wikipedia content; the parsed
corpus is then cached as a pickle keyed on `(num_questions, web_cap,
wiki_distractor_count)` so subsequent runs are instant.

At the default settings this produces:

- **~100 questions** with gold answer strings and aliases.
- **~2 700–3 500 documents** in the retrieval corpus: ~700–1 500 from
  TriviaQA (wiki entity pages + web hits) plus ~2 000 external Simple
  Wikipedia distractors. The per-source breakdown is logged at load time;
  overall capped at `config.MAX_CORPUS_DOCS = 5 000`.

Each document carries a `source` field (`"wiki"`, `"web"`, or
`"wiki_distractor"`) so per-source analyses are possible if needed.

The corpus is cached to disk after the first download to avoid
re-downloading. **Note:** the `rc` config is substantially larger than
`rc.wikipedia` (~5–10 GB download), so the first download will take a
while even though we only use the first 100 questions.

### Retrieval setting — what this *is* and what it isn't

We call this setting **pooled-evidence retrieval**: a hybrid between
closed-book QA (only the gold evidence for each question) and the true
open-domain regime used in Lewis et al. [42] (the entire ~21 M-passage
Wikipedia indexed with FAISS). Specifically:

- **Gold evidence is present for every test question.** TriviaQA's
  curated entity pages are designed to contain the answer; pooling
  doesn't remove them, so retrieval is always answerable in principle.
- **In-batch distractors are present.** Each question's retrieval
  competes with the entity pages and web hits collected for the other
  ~99 questions. These are topical, but not selected to be adversarial
  for any specific question.
- **The web-hit half of the in-batch pool is task-conditioned.** Web
  results in TriviaQA were originally retrieved by a search engine
  *using the question itself*, so they are correlated with the question
  by construction — pages that a real search engine already judged
  relevant. This makes that half of the noise easier than realistic
  open-domain noise.
- **External distractors (this implementation) are topic-agnostic.** The
  ~2 000 Simple Wikipedia articles we mix in were sampled with no
  reference to our test questions; they raise the corpus to ~2.5×–4×
  the size of the pure-TriviaQA pool and contribute pure noise.

Because of points 1–3, absolute Recall@k numbers should be read as
**upper bounds on retrieval difficulty**, not estimates of performance
in a full open-domain deployment. Point 4 mitigates but does not close
the gap to Lewis et al.'s setting, which is ~4 orders of magnitude
larger and contains many adversarial near-duplicates per query.

### Answer format

TriviaQA provides multiple gold answers per question (a primary value plus
aliases). We consider a prediction *correct* if it matches **any** gold
answer after normalisation.

---

## 5. Experimental Design

All experiments follow the same evaluation protocol:
- Metrics: Exact Match (EM), Token F1, and Retrieval Recall@k
- 95% bootstrap confidence intervals (1000 resamples) for every reported metric
- Deterministic generation (greedy decoding) for full reproducibility
- Generation is cached so re-runs only regenerate changed prompts

### Exp 1 — Retriever Comparison: BM25 vs TF-IDF vs Dense

**Fixed:** chunk size = 128 words, k = 5, prompt = "instructed"

**Variable:** retrieval method ∈ {BM25, TF-IDF, Dense}

**Prior expectations vs. what we should actually expect on this corpus.**
A common textbook prior is that a dense bi-encoder should outperform
sparse retrieval on QA because questions and answer passages need not
share vocabulary (Lewis et al. [42]; Karpukhin et al., DPR). The
opposite is plausible on TriviaQA `rc` specifically: trivia questions
are dominated by named entities, dates, and titles, and the gold
evidence is Wikipedia entity pages where those exact strings recur —
strong literal overlap → term-frequency matching is already strong, and
an off-the-shelf MiniLM bi-encoder (not fine-tuned on TriviaQA, unlike
DPR) gives up the small generalisation advantage that the dense
direction usually enjoys. So a tie or a slight sparse win at this scale
would not be surprising; this experiment is a test of that.

**Empirical result (this run, n = 100, ~2.7 k-doc corpus):**
BM25 ≈ TF-IDF ≈ 0.70 EM, Dense ≈ 0.67 EM, all three at Recall@5 ≈ 0.94
with heavily overlapping bootstrap CIs. The interpretation is that
retrieval is near-ceiling on this corpus for all three methods and the
generator-side gap dominates the residual error. A larger or more
adversarial corpus (see Experiments 7 and 8), a multi-hop dataset, or
a TriviaQA-fine-tuned dense encoder would all be expected to widen the
dense-vs-sparse spread.

**Methods:** BM25 and TF-IDF consume the **same** token stream produced
by a shared `_lexical_tokenize` (NFKD accent-strip → lowercase → regex
`\b\w\w+\b`). The only remaining difference between the two sparse
retrievers is the scoring function itself — vocabulary, n-gram order,
and DF filtering are held constant.
- **BM25** (Robertson et al.): Okapi BM25 on the shared token stream.
  No stemming.
- **TF-IDF** (sklearn): `TfidfVectorizer` with `tokenizer=_lexical_tokenize`,
  `ngram_range=(1,1)`, sublinear TF, no `min_df` / `max_df` filtering,
  cosine similarity retrieval.
- **Dense** (sentence-transformers): `all-MiniLM-L6-v2` bi-encoder
  (22 M parameters). Query and passage embeddings are L2-normalised; 
  retrieval is by dot product (= cosine similarity). Embeddings cached to disk.

**Measurement:**
- Primary: EM and F1 on the 500-question set
- Secondary: Retrieval Recall@k (answer string substring-matched in top-5 chunks)

---

### Exp 2 — Chunk Size Ablation

**Fixed:** retriever = Dense, k = 5, prompt = "instructed"

**Variable:** chunk size ∈ {64, 128, 256, 512} words

**Hypothesis (two-sided):**
- *Small chunks* (64 words) → high precision but low recall — the answer
  may straddle a chunk boundary or be split across two chunks.
- *Large chunks* (512 words) → high recall but the relevant sentence is
  diluted by surrounding text, making it harder for the generator to
  extract the answer.
- An intermediate size (128–256 words) should optimise the EM/F1 tradeoff.

**Overlap:** 32 words (half the smallest chunk) to reduce boundary effects.

**Measurement:** EM, F1, Recall@k at each chunk size.

---

### Exp 3 — Number of Retrieved Passages (k)

**Fixed:** retriever = Dense, chunk size = 128 words, prompt = "instructed"

**Variable:** k ∈ {1, 3, 5, 10}

**Hypothesis:**
- Recall@k increases monotonically with k (more passages → higher chance
  that one contains the answer).
- EM and F1 may *decrease* for large k if the model is overwhelmed by long
  or distracting context (cf. Izacard & Grave [43] who address this with FiD).
- We expect a sweet spot near k = 5 for Flan-T5-base's 512-token context.

**Measurement:** EM, F1, Recall@k at each k.

---

### Exp 4 — Prompt Template Ablation

**Fixed:** retriever = Dense, chunk size = 128 words, k = 5

**Variable:** prompt template ∈ {"concise", "instructed"}

**Templates:** Both templates use the same question-twice scaffold (the
question is repeated *before* and *after* the context, since T5 truncates
from the end and we want the question to survive long contexts). The only
deliberate contrast is the leading instruction sentence in the
*instructed* variant; the leading sentence is kept short so that
*concise* and *instructed* are length-matched to within ~10 tokens — Exp 4
therefore measures the instruction signal itself, not the cost of
displaced context under truncation.

*Concise:*
```
Question: {question}

Context:
{retrieved passages}

Question: {question}
Answer:
```

*Instructed:*
```
Answer with a short phrase.

Question: {question}

Context:
{retrieved passages}

Question: {question}
Short answer:
```

**Hypothesis:** Instruction-tuned models like Flan-T5 respond better to
explicit directives. The single leading sentence ("Answer with a short
phrase.") plus the "Short answer:" cue should push the model toward
terse phrase-level outputs, improving EM (which requires an exact short
phrase match); without them the model may copy sentences from the
context or generate overly verbose answers.

**Measurement:** EM and F1.

---

### Exp 5 — RAG vs No-RAG

**Fixed:** retriever = Dense, chunk size = 128 words, k = 5, prompt = "instructed"

**Conditions:**
- **RAG:** top-5 retrieved passages included in the prompt.
- **No-RAG (parametric floor):** no context — the model is given only the
  question through a minimal `Q: {question}\nA:` cue. This cue was
  selected from a four-prompt robustness sweep (see notebook §14
  "Prompt sensitivity of the No-RAG baseline"), in which the spread
  across reasonable alternatives was only ~5 pp EM. The arm should
  therefore be read as a **blind-guess floor** for Flan-T5-base on
  TriviaQA-grade questions — inspection of sample predictions confirms
  the model produces confidently-shaped but unrelated entity strings
  (e.g. `"henry viii"` for *"PM after Balfour"*) — not as a measurement
  of its specific parametric knowledge. Flan-T5-base genuinely lacks
  these long-tail facts; *that the floor is so low is itself the
  finding*.

  We report **F1 alongside EM** for this arm: at EM≈0.07 a single
  question swing is 1 pp, so EM is noisy, whereas F1 retains a
  partial-credit signal that distinguishes near-misses from total
  ignorance and is the more stable comparator for the per-question
  delta analysis below.

**Analysis:**
Beyond aggregate metrics, we perform a **per-question delta analysis**:
for each of the 500 questions, we record whether RAG *helps* (EM improves),
*ties*, or *hurts* (EM decreases) relative to No-RAG.

We further categorise the "RAG helps" and "RAG hurts" buckets by retrieval
recall — this lets us test the causal story:
- **RAG helps when recall = 1** → retrieval found the answer, generator used it.
- **RAG hurts when recall = 1** → answer was retrieved but generator was distracted.
- **RAG hurts when recall = 0** → retrieval missed entirely; model got wrong context.

> **Caveat — the "RAG hurts" bucket is bounded by the No-RAG floor.**
> A question can only end up in "RAG hurts" if No-RAG got it right and
> RAG got it wrong. With the No-RAG floor at EM≈0.07 (see above), the
> bucket size is upper-bounded at ~7 questions out of 100 — a handful at
> most — so the further decomposition into "RAG hurts when recall = 1" /
> "RAG hurts when recall = 0" is essentially unpowered. The
> `helps / ties / hurts` split should therefore be read as "where RAG
> adds value on top of a weak parametric floor", not as evidence that
> RAG rarely hurts in general. The per-question **ΔF1** view
> (RAG F1 − No-RAG F1) is more informative at this scale because it
> registers partial-credit regressions that EM cannot; a larger generator
> (e.g. Flan-T5-large/XL) would also raise the floor and give the
> decomposition real statistical power.

---

### Exp 6 — Oracle Baseline (retrieval upper bound)

**Fixed:** retriever = Dense (used for the non-oracle slots), chunk size = 128, k = 5, prompt = "instructed"

**Condition:** Every question is *guaranteed* to have an answer-bearing
chunk in its top-k context. Concretely: run dense retrieval as usual at
k = 5; for any question whose top-5 misses the gold answer (recall = 0),
find the first chunk in the corpus that contains it (normalised substring
match) and place it at rank 1, dropping the lowest-ranked retrieved chunk
so k stays at 5. If no chunk in the corpus contains the answer (chunking
artefact — the answer may straddle a window boundary), the question is
left as-is and flagged.

**Hypothesis & interpretation:**
- **Oracle − RAG** = the share of error attributable to *retrieval
  failure*. If the dense retriever already finds the answer for most
  questions, this gap is small.
- **1 − Oracle** = the *generator* failure under perfect retrieval — a
  ceiling on what better retrieval alone can buy. If Oracle is far below
  1.0, scaling retrieval further has diminishing returns and the
  generator (model size, prompting, fine-tuning) is the binding
  constraint.

**Measurement:** EM, F1, and the number of questions where the oracle
fix was applied (`n_injected`) and where no fix was possible
(`n_unfixable`).

---

### Exp 7 — Cross-Encoder Re-ranking

**Fixed:** chunk size = 128, k = 5 (final), prompt = "instructed"

**Condition:** two-stage retrieval — a dense bi-encoder first stage
returns the top-N candidates (`RERANK_TOP_N = 50`), then a cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores every `(query, chunk)`
pair jointly and the top-k are kept for the generator.

A cross-encoder reads the query and chunk *together* in a single
transformer pass, so it can model term-level interactions a bi-encoder
cannot — but it is quadratic in the number of chunks, so it has to be
used as a *re-ranker* on a small candidate set rather than as a
first-stage retriever. This is the canonical "advanced RAG" topology in
the Gao et al. survey [44].

**Hypotheses:**
- Re-ranking should **raise Recall@5** if the dense top-50 contained
  answer-bearing chunks that the dense top-5 missed (a "good candidates,
  bad ranking" failure mode).
- Whether the Recall@5 lift translates into EM/F1 lift is the
  *interesting* question: if Exp 6 already showed the generator is the
  binding constraint, even a perfect retriever won't help; if Exp 6
  showed retrieval is the bottleneck, re-ranking should help roughly
  in proportion to the Recall@5 gain.

**Measurement:** EM, F1, Recall@5 vs the Exp 1 dense-only numbers.

---

### Exp 8 — Distractor Count Sweep

**Fixed:** retriever = Dense, chunk size = 128, k = 5, prompt = "instructed"

**Variable:** `NUM_WIKI_DISTRACTORS` ∈ {0, 500, 2000, 5000}

The default corpus mixes 2 000 topic-agnostic Simple-English Wikipedia
articles into the retrieval pool. This number is somewhat arbitrary —
too few and retrieval is essentially "find the gold page among other
questions' gold pages"; too many and the gold becomes a needle in a
haystack. This experiment sweeps the distractor count holding everything
else fixed (same questions, same chunking, same generator) and re-runs
the RAG arm only; the No-RAG arm is corpus-independent and is reused
from Experiment 5.

**Hypotheses:**
- **Recall@5** should fall monotonically with the noise floor.
- **EM / F1** falls *only if* (a) Recall@5 falls *and* (b) the generator
  was actually using the now-missing chunks. If EM is flat while
  Recall@5 drops, the generator is robust to noisier retrieval; if both
  drop together, retrieval quality is the binding constraint at scale.

This is the most direct test in the project of *"how much does
retrieval quality matter for end-to-end answer quality"* — the variable
*is* the difficulty of the retrieval task itself.

**Measurement:** EM, F1, Recall@5 at each distractor count, with
absolute corpus sizes and chunk counts reported for context.

---

## 6. Evaluation Metrics

All metrics follow the SQuAD / TriviaQA evaluation convention exactly.

### Answer normalisation

Before any comparison:
1. Lowercase the string.
2. Remove punctuation.
3. Remove articles: "a", "an", "the".
4. Collapse whitespace.

### Exact Match (EM)

```
EM(pred, golds) = 1  if  normalize(pred) == normalize(gold)  for any gold
                 0  otherwise
```

### Token F1

Token-level precision / recall between predicted tokens and gold tokens.
F1 is computed against each gold answer independently; we take the maximum.

```
F1 = 2 * |common tokens| / (|pred tokens| + |gold tokens|)
```

### Retrieval Recall@k

```
Recall@k = 1  if  normalize(answer)  is a substring of  normalize(chunk)
                  for any chunk in the top-k retrieved chunks
           0  otherwise
```

This measures whether the retriever *found* the information, independently
of whether the generator *extracted* it.

> **Caveat — substring matching inflates Recall@k.** Because we use a
> normalised-substring test (not a token / span match), a gold answer that
> happens to be a substring of an unrelated longer token counts as a hit:
> "Black" inside "Blackboard", "Green" inside "Greenland", numeric answers
> like "1066" inside "10665". This biases Recall@k *upward*, with the bias
> concentrated on short / common / numeric answers. The downstream EM and
> F1 numbers are unaffected (they compare against the *generated* answer),
> but the absolute Recall@k values — and the "recall=1 but EM=0" buckets in
> the Exp 5 error analysis — should be read as upper bounds on true
> retrieval success. A cleaner alternative is a token-span match against
> TriviaQA's annotated gold-evidence pages, which we leave as future work.

### Bootstrap Confidence Intervals

We compute 95% CIs using **percentile bootstrap** with 1000 resamples
(Efron & Tibshirani, 1994). For a metric m computed on n examples:

1. Draw n samples with replacement from the per-example scores.
2. Compute the mean of each resample.
3. Report the 2.5th and 97.5th percentiles as the CI bounds.

---

## 7. Implementation Details

### Generator: Flan-T5-base

- Model: `google/flan-t5-base` (~250 M parameters)
- Decoding: greedy (deterministic; no sampling)
- Truncation budget: 1024 input tokens
- **Middle-truncation for long RAG prompts.** RAG prompts are built as
  `(prefix, context, suffix)` triples where the prefix and suffix carry
  the question repeats and the `Short answer:` cue. When the joined
  prompt exceeds 1024 tokens, the generator preserves prefix and suffix
  verbatim and trims only the *context body* — so the answer cue is never
  lost to right-truncation, only the lowest-ranked retrieved passages.
  Prompt-length logs distinguish raw tokens (pre-truncation) from final
  tokens (what the model sees) and report how many prompts had context
  trimmed.
- Maximum new tokens: 32
- Device: auto-detected (MPS → CUDA → CPU)
- All generation results cached to `data/cache/generation_cache.json`,
  keyed by MD5 of the *final* rendered prompt so changing the budget
  invalidates exactly the affected entries.

We choose Flan-T5-base because:
1. It is instruction-tuned, so it follows prompt directives reliably.
2. It runs locally without any API key.
3. Its results are fully reproducible (deterministic decoding).
4. ~250 M parameters is large enough to hold substantial world knowledge
   but small enough to run on a laptop CPU within reasonable time.

### Dense Retriever: all-MiniLM-L6-v2

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384
- Normalised embeddings → retrieval = dot product
- Passage embeddings cached per chunk-size to `data/cache/`

### Chunking

Fixed-size word windows with 32-word overlap. Example for size = 128:

```
Document: [w₁  w₂  …  w₄₀₀]
Chunk 0:  [w₁  … w₁₂₈]
Chunk 1:  [w₉₇ … w₂₂₄]  (overlap: w₉₇–w₁₂₈)
Chunk 2:  [w₁₉₃ … w₃₂₀]
…
```

---

## 8. Reproducibility

| Component | Seed / Determinism mechanism |
|---|---|
| Dataset sampling | Always take first N questions from validation split |
| Chunking | Deterministic word split |
| BM25 / TF-IDF | Deterministic (no random state) |
| Dense embeddings | Deterministic forward pass; cached to disk |
| Generation | Greedy decoding (`do_sample=False`, `num_beams=1`); cached to disk |
| Bootstrap CI | `numpy.random.default_rng(seed=42)` |

Running the experiment pipeline twice should produce **identical results**.

---

## 9. How to Run

### Prerequisites

```bash
# Python 3.9+
pip install -r requirements.txt

# NLTK data (used by rouge-score)
python -c "import nltk; nltk.download('punkt')"
```

### Run all experiments

```bash
python experiments/run_experiments.py
```

This will:
1. Download TriviaQA (first run only; cached afterwards).
2. Run all 5 experiments sequentially.
3. Save results JSON files to `results/`.

**Expected runtime** at the default `NUM_QUESTIONS=100` / `MAX_SEARCH_RESULTS_PER_Q=5`
(Apple Silicon / modern CPU):
- Data loading: 5–15 min first time (the `rc` dataset is large); <10 s from cache afterwards.
- Dense embedding (~1 k docs × 4 chunk sizes): ~2–5 min.
- Generation (100 questions × ~8 conditions): ~5–15 min on CPU, faster on MPS/CUDA. Cache keeps re-runs cheap.
- Total first end-to-end run: ~20–30 min.

To run at "report scale", bump `config.NUM_QUESTIONS` to 500 or 1000 and
optionally raise `MAX_SEARCH_RESULTS_PER_Q`; runtime scales roughly linearly
with both.

### Run a single experiment

```bash
python experiments/run_experiments.py --exp 1   # Retriever comparison only
python experiments/run_experiments.py --exp 5   # RAG vs No-RAG only
```

### Generate figures

```bash
python analysis/plot_results.py
```

Figures are saved to `figures/` as 300 dpi PNG files.

### Verify the pipeline quickly (smoke test)

```python
import config
config.NUM_QUESTIONS = 10   # override for a quick test
from experiments.run_experiments import setup, experiment_5
q, docs, gen = setup()
experiment_5(q, docs, gen)
```

---

## 10. Results Summary

> **Status — results pending regeneration.** The numbers and figures
> previously listed here were produced under an earlier pipeline
> configuration (`rc.wikipedia` corpus, `MAX_INPUT_TOKENS=512`,
> question-at-end prompts). The pipeline has since been updated to fix
> several validity issues:
>
> - dataset switched from `rc.wikipedia` to `rc` (wiki + web evidence
>   pooled into one corpus),
> - `MAX_INPUT_TOKENS` raised to 1024 so that at chunk=128 the encoder
>   actually sees the difference between k=3, k=5, and k=10 instead of
>   receiving an identically-truncated input (Flan-T5-base was
>   pre-trained at 512 but tolerates longer inputs in practice; see the
>   inline comment in `config.MAX_INPUT_TOKENS`),
> - prompts now repeat the question both before and after the context
>   so truncation can't drop it.
>
> Each of those changes invalidates the cached results. Re-run the
> notebook (or `python experiments/run_experiments.py`) to populate
> `results/` and `figures/`, then fill in the tables below.
>
> **Additional pending invalidation (middle-truncation fix).** The
> Generator now middle-truncates long RAG prompts so that the trailing
> `Short answer:` cue is preserved (it was previously being truncated
> away in ~60 % of Exp 1 prompts and 100 % of Exp 3 k=10 prompts). The
> rendered prompt strings for those conditions have changed, so the
> MD5 cache keys no longer match — those entries will be re-generated
> automatically on the next run, but until then the saved
> `results/exp{1,2,3,5}*.json` and the existing figures reflect the
> pre-fix behaviour.
>
> **New experiments added in this pass.** Experiments 6 (oracle), 7
> (cross-encoder re-ranking) and 8 (distractor count sweep) are in
> the notebook but have not been run yet — their `results/exp{6,7,8}*.json`
> and `figures/fig{8,9,10}*.png` will appear after re-running.

### Experiment 1 — Retriever Comparison

| Method | EM | Token F1 | Recall@5 |
|---|---|---|---|
| BM25   | _TBD_ | _TBD_ | _TBD_ |
| TF-IDF | _TBD_ | _TBD_ | _TBD_ |
| Dense  | _TBD_ | _TBD_ | _TBD_ |

### Experiment 2 — Chunk Size

| Chunk (words) | EM | Token F1 | Recall@5 |
|---|---|---|---|
| 64  | _TBD_ | _TBD_ | _TBD_ |
| 128 | _TBD_ | _TBD_ | _TBD_ |
| 256 | _TBD_ | _TBD_ | _TBD_ |
| 512 | _TBD_ | _TBD_ | _TBD_ |

### Experiment 3 — Number of Retrieved Passages (k)

| k | EM | Token F1 | Recall@k |
|---|---|---|---|
| 1  | _TBD_ | _TBD_ | _TBD_ |
| 3  | _TBD_ | _TBD_ | _TBD_ |
| 5  | _TBD_ | _TBD_ | _TBD_ |
| 10 | _TBD_ | _TBD_ | _TBD_ |

### Experiment 4 — Prompt Template

| Template   | EM    | Token F1 |
|---|---|---|
| Concise    | _TBD_ | _TBD_    |
| Instructed | _TBD_ | _TBD_    |

### Experiment 5 — RAG vs No-RAG

| Condition                | EM    | Token F1 |
|---|---|---|
| No-RAG (parametric only) | _TBD_ | _TBD_    |
| RAG (Dense, k = 5)       | _TBD_ | _TBD_    |

Per-question breakdown (n = 500): _TBD helps / TBD hurts / TBD ties_.

### Experiment 6 — Oracle Baseline

| Condition                       | EM    | Token F1 | Recall@5 |
|---|---|---|---|
| RAG (Dense, k = 5)              | _TBD_ | _TBD_    | _TBD_    |
| Oracle (answer guaranteed, k=5) | _TBD_ | _TBD_    | 1.00     |

Injected answer chunk for _TBD / n_ questions; _TBD / n_ unfixable (no
answer-bearing chunk in corpus).

### Experiment 7 — Cross-Encoder Re-ranking

| Method                           | EM    | Token F1 | Recall@5 |
|---|---|---|---|
| Dense (Exp 1)                    | _TBD_ | _TBD_    | _TBD_    |
| Dense + Rerank (top-50 → top-5)  | _TBD_ | _TBD_    | _TBD_    |

### Experiment 8 — Distractor Count Sweep

| `NUM_WIKI_DISTRACTORS` | n_corpus_docs | EM    | Token F1 | Recall@5 |
|---|---|---|---|---|
| 0    | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 500  | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 2000 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 5000 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

Figures in `figures/`:

| Figure | Content |
|---|---|
| `fig1_retriever_comparison.png` | EM + F1 bar chart across retrieval methods with 95% CI |
| `fig2_chunk_size.png`           | EM / F1 / Recall@k vs chunk size (line + CI band) |
| `fig3_k_values.png`             | EM / F1 / Recall@k vs k (line + CI band) |
| `fig4_prompt_template.png`      | EM + F1: concise vs instructed prompts |
| `fig5_rag_vs_no_rag.png`        | EM + F1: RAG vs parametric baseline |
| `fig6_error_analysis.png`       | Pie (helps/hurts/ties) + scatter (recall vs EM) |
| `fig7_qualitative.png`          | Example table: cases where RAG helps vs hurts |
| `fig8_oracle.png`               | EM + F1: No-RAG / RAG / Oracle (Exp 6 — retrieval vs generation gap) |
| `fig9_rerank.png`               | EM + F1 + Recall@5: Dense vs Dense + Cross-Encoder Rerank (Exp 7) |
| `fig10_distractor_sweep.png`    | EM / F1 / Recall@k vs number of external distractors (Exp 8) |

---

## 11. Limitations

1. **Model scale**: Flan-T5-base is a relatively small model. Larger models
   (e.g., Flan-T5-XL, LLaMA-3) may show different patterns — e.g., less
   benefit from retrieval because they have stronger parametric knowledge.

2. **No retriever fine-tuning**: We use off-the-shelf BM25 and
   `all-MiniLM-L6-v2` without any domain-specific fine-tuning. A DPR model
   fine-tuned on TriviaQA (as in Lewis et al. [42]) would likely show a
   larger performance gap between dense and sparse retrieval.

3. **Corpus size and provenance**: At the default fast-iteration settings
   (`NUM_QUESTIONS=100`, `MAX_SEARCH_RESULTS_PER_Q=5`,
   `NUM_WIKI_DISTRACTORS=2000`) the corpus is ~2 700–3 500 documents —
   enough to demonstrate the pipeline and the directional findings, but
   not enough to draw quantitative conclusions. For the final report run,
   raise `NUM_QUESTIONS` and `NUM_WIKI_DISTRACTORS` together. Even at the
   maximum `rc` setting the corpus is much smaller and less diverse than
   the full open-domain setting in Lewis et al. [42] (~21 M Wikipedia
   passages indexed with FAISS). As discussed in §4, the in-batch web
   hits are also task-conditioned by construction, and the external
   Simple-Wikipedia distractors are topic-agnostic rather than
   adversarial — both factors flatter retrieval. Absolute Recall@k
   numbers should be read as **upper bounds**, not as estimates of
   open-domain performance.

4. **No query expansion or re-ranking**: Advanced RAG techniques (HyDE,
   step-back prompting, cross-encoder re-ranking) are not evaluated. These
   could change the relative ranking of retrieval methods.

5. **Single-hop questions only**: TriviaQA questions are mostly single-hop
   (one Wikipedia article). Multi-hop reasoning (e.g., HotpotQA) would
   likely show different patterns for dense vs sparse retrieval.

---

## 12. References

[42] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V.,
     Goyal, N., … Kiela, D. (2020). *Retrieval-Augmented Generation for
     Knowledge-Intensive NLP Tasks*. NeurIPS 2020.

[43] Izacard, G., & Grave, E. (2021). *Leveraging Passage Retrieval with
     Generative Models for Open Domain Question Answering*. EACL 2021.
     (Fusion-in-Decoder)

[44] Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., … Wang, H.
     (2024). *Retrieval-Augmented Generation for Large Language Models:
     A Survey*. arXiv:2312.10997.
