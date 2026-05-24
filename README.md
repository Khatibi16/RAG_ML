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
are dropped. At the default settings this produces:

- **~100 questions** with gold answer strings and aliases.
- **~700–1 500 documents** in the retrieval corpus (the wiki/web
  breakdown is logged at load time, capped overall at
  `config.MAX_CORPUS_DOCS = 5 000`).

Each document carries a `source` field (`"wiki"` or `"web"`) so per-source
analyses are possible if needed.

The corpus is cached to disk after the first download to avoid
re-downloading. **Note:** the `rc` config is substantially larger than
`rc.wikipedia` (~5–10 GB download), so the first download will take a
while even though we only use the first 100 questions.

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

**Hypothesis:** Dense retrieval (semantic matching) will outperform sparse
methods (BM25, TF-IDF) on TriviaQA because trivia questions rarely share
exact vocabulary with their answer passages, making term-based matching
unreliable.

**Methods:**
- **BM25** (Robertson et al.): Okapi BM25 on lowercased whitespace tokens.
  No stemming, to keep the comparison with TF-IDF fair.
- **TF-IDF** (sklearn): TfidfVectorizer with sublinear TF, unigrams +
  bigrams, min/max DF filtering, cosine similarity retrieval.
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

**Templates:**

*Concise:*
```
Context:
{retrieved passages}

Question: {question}
Answer:
```

*Instructed:*
```
You are a factual QA assistant. Read the context and answer the question
with a short phrase only. Do NOT repeat the question. Do NOT write a
sentence; just the answer.

Context:
{retrieved passages}

Question: {question}
Short answer:
```

**Hypothesis:** Instruction-tuned models like Flan-T5 respond better to
explicit directives. Without them, the model may copy sentences from the
context or generate overly verbose answers, hurting EM (which requires an
exact short phrase match).

**Measurement:** EM and F1.

---

### Exp 5 — RAG vs No-RAG

**Fixed:** retriever = Dense, chunk size = 128 words, k = 5, prompt = "instructed"

**Conditions:**
- **RAG:** top-5 retrieved passages included in the prompt.
- **No-RAG (parametric):** no context — the model must answer from its
  parametric knowledge alone using a minimal prompt.

**Analysis:**
Beyond aggregate metrics, we perform a **per-question delta analysis**:
for each of the 500 questions, we record whether RAG *helps* (EM improves),
*ties*, or *hurts* (EM decreases) relative to No-RAG.

We further categorise the "RAG helps" and "RAG hurts" buckets by retrieval
recall — this lets us test the causal story:
- **RAG helps when recall = 1** → retrieval found the answer, generator used it.
- **RAG hurts when recall = 1** → answer was retrieved but generator was distracted.
- **RAG hurts when recall = 0** → retrieval missed entirely; model got wrong context.

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
- Truncation: prompts are truncated to 1024 tokens
- Maximum new tokens: 64
- Device: auto-detected (MPS → CUDA → CPU)
- All generation results cached to `data/cache/generation_cache.json`

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
> configuration (`rc.wikipedia` corpus, `MAX_INPUT_TOKENS=1024`,
> question-at-end prompts). The pipeline has since been updated to fix
> several validity issues:
>
> - dataset switched from `rc.wikipedia` to `rc` (wiki + web evidence
>   pooled into one corpus),
> - `MAX_INPUT_TOKENS` lowered to 512 to match Flan-T5-base's
>   pre-training context,
> - prompts now repeat the question both before and after the context
>   so truncation can't drop it.
>
> Each of those changes invalidates the cached results. Re-run the
> notebook (or `python experiments/run_experiments.py`) to populate
> `results/` and `figures/`, then fill in the tables below.

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
   (`NUM_QUESTIONS=100`, `MAX_SEARCH_RESULTS_PER_Q=5`) the corpus is only
   ~700–1 500 documents — enough to demonstrate the pipeline and the
   directional findings, but not enough to draw quantitative conclusions.
   For the final report run, raise both constants. Even at the maximum
   `rc` setting the corpus is much smaller and less diverse than the
   open-domain settings used in Lewis et al. [42] (~21 M Wikipedia
   passages indexed with FAISS); absolute Recall@k numbers should be read
   in that context.

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
