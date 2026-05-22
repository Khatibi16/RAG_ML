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

**TriviaQA** (`rc.wikipedia` configuration, Joshi et al. 2017).

We use this benchmark because it is the primary dataset in all three
reference papers (Lewis et al. [42], Izacard & Grave [43], Gao et al. [44]),
enabling direct comparison with published results.

### Why TriviaQA `rc.wikipedia`?

The `rc.wikipedia` configuration provides each trivia question paired with a
set of Wikipedia *entity pages* — the documents that Wikipedia's search would
surface for the named entities mentioned in the question. This creates a
realistic retrieval setting:

- The corpus contains **relevant** documents (the entity pages associated
  with the question's answer) **and** many **distractor** documents
  (entity pages from other questions).
- The retriever must identify the relevant pages without knowing which
  question they were associated with.

### Corpus construction

We take the first **500 questions** from the validation split.
For each question, TriviaQA provides a list of entity Wikipedia pages
(typically 2–10 pages). We pool all pages across all questions into a
single shared retrieval corpus. This produces:

- **~500 questions** with gold answer strings and aliases.
- **Up to 6,000 Wikipedia article passages** as the retrieval corpus.

The corpus is cached to disk after the first download to avoid re-downloading.

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

**Expected runtime** (Apple Silicon / modern CPU):
- Data loading: ~5 min (first run; <10 s from cache)
- Dense embedding (6k chunks × 4 chunk sizes): ~15–20 min
- Generation (500 questions × ~6 conditions): ~30–60 min (mostly cached after first run)
- Total first run: ~1–2 hours

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

All results below are from n = 500 TriviaQA `rc.wikipedia` validation questions.
All numbers are **mean [95 % bootstrap CI]**.

---

### Experiment 1 — Retriever Comparison

| Method | EM | Token F1 | Recall@5 |
|---|---|---|---|
| BM25   | 0.176 [0.140–0.210] | 0.229 [0.196–0.263] | 0.688 |
| TF-IDF | 0.192 [0.160–0.228] | 0.253 [0.220–0.289] | 0.772 |
| **Dense** | **0.204 [0.168–0.238]** | **0.271 [0.237–0.308]** | **0.842** |

**Finding:** Dense semantic retrieval dominates across every metric. It raises
Recall@5 by +15.4 pp over BM25 (68.8% → 84.2%), confirming the hypothesis
that trivia questions rarely share exact vocabulary with their Wikipedia evidence.
The ordering matches Lewis et al. (2020) [42].

---

### Experiment 2 — Chunk Size

| Chunk (words) | EM | Token F1 | Recall@5 |
|---|---|---|---|
| 64  | **0.490 [0.446–0.534]** | **0.563 [0.523–0.604]** | 0.784 |
| 128 | 0.204 [0.168–0.238] | 0.271 [0.237–0.308] | 0.842 |
| 256 | 0.016 [0.006–0.028] | 0.061 [0.048–0.075] | 0.888 |
| 512 | 0.016 [0.006–0.028] | 0.063 [0.048–0.079] | 0.922 |

**Finding:** A striking non-monotone trade-off. Recall@k *increases* with chunk
size (more context → higher probability of covering the answer), but EM
*collapses* for chunks ≥ 256 words. The root cause: Flan-T5-base has a
512-token encoder limit. With k = 5 passages of 256 words each, the total
prompt exceeds ~1 280 words (≈ 1 700 tokens), forcing severe truncation that
drops the question itself. Small 64-word chunks keep each passage focused and
the total prompt within the model's context window, yielding the highest EM.

> **Key insight:** Chunk size and k are not independent — their *product*
> determines whether the context fits in the generator's window. This
> interaction is under-reported in the RAG survey literature (Gao et al. [44]).

---

### Experiment 3 — Number of Retrieved Passages (k)

| k | EM | Token F1 | Recall@k |
|---|---|---|---|
| 1  | 0.408 [0.366–0.452] | 0.483 [0.444–0.525] | 0.596 |
| 3  | **0.512 [0.470–0.554]** | **0.590 [0.552–0.629]** | 0.788 |
| 5  | 0.204 [0.168–0.238] | 0.271 [0.237–0.308] | 0.842 |
| 10 | 0.016 [0.006–0.028] | 0.069 [0.054–0.084] | 0.910 |

**Finding:** k = 3 is the empirical sweet spot for Flan-T5-base with 128-word
chunks. Recall continues to grow with k, but EM peaks at k = 3 and then
drops sharply. At k = 10, performance collapses for the same context-window
reason as in Experiment 2. This directly validates the core argument of
Izacard & Grave (2021) [43]: *how* you aggregate many passages into the
generator matters enormously — naive concatenation (as used here) hits a
hard wall at the model's context length. FiD's per-passage encoding removes
this bottleneck.

---

### Experiment 4 — Prompt Template

| Template | EM | Token F1 |
|---|---|---|
| **Concise** | **0.344 [0.304–0.384]** | **0.397 [0.358–0.434]** |
| Instructed | 0.204 [0.168–0.238] | 0.271 [0.237–0.308] |

**Finding:** Counter-intuitively, the shorter "concise" prompt outperforms the
"instructed" prompt. The explanation is again the context window: the
"instructed" template adds ~50 tokens of system instructions before the
retrieved context, which pushes the total prompt length further past the
encoding limit and causes more truncation of the retrieved passages. This
reveals a **prompt length vs. instruction quality** trade-off: better
instructions cost tokens, and those tokens compete directly with retrieved
evidence.

---

### Experiment 5 — RAG vs No-RAG

| Condition | EM | Token F1 |
|---|---|---|
| No-RAG (parametric only) | 0.058 [0.038–0.078] | 0.101 [0.079–0.123] |
| **RAG (Dense, k = 5)** | **0.204 [0.168–0.238]** | **0.271 [0.237–0.308]** |

**RAG gain:** +14.6 pp EM, +17.0 pp F1 — a **3.5× improvement**.

Per-question breakdown (n = 500):
- **RAG helps** (EM improves): 96 questions (19.2 %)
- **RAG hurts** (EM decreases): 23 questions (4.6 %)
- **Ties**: 381 questions (76.2 %)

**Finding:** RAG is unambiguously better than parametric-only generation at
this model scale — Flan-T5-base memorises very little trivia. The per-question
analysis reveals the mechanism: on the 84.2 % of questions where the answer
is retrieved (Recall@5 = 0.842), the model often extracts it correctly; on
the 15.8 % where retrieval misses, the model receives irrelevant context that
sometimes confuses it (the 4.6 % hurt cases). This matches Lewis et al.'s
finding that retrieval quality is the primary bottleneck in RAG systems.

---

### Summary of Key Takeaways

| Finding | Evidence |
|---|---|
| Dense retrieval > sparse methods | Exp 1: +15 pp Recall@5 over BM25 |
| Context window caps performance | Exp 2 & 3: EM collapses at chunk≥256 or k≥10 |
| Chunk size × k interact critically | Exp 2 & 3: product determines context length |
| Shorter prompts can beat longer ones | Exp 4: concise +14 pp EM over instructed |
| RAG gives 3.5× EM improvement | Exp 5: 5.8% → 20.4% over parametric baseline |
| RAG hurts < 5% of the time | Exp 5: 23/500 hurt cases |

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

3. **Corpus size**: We cap the corpus at 6,000 pages for computational
   feasibility. Full TriviaQA uses the entire Wikipedia dump (~21 M passages),
   which changes the retrieval difficulty significantly.

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
