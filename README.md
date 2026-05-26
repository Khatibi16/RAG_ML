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
   - [Exp 9 — Controlled Distraction](#exp-9--controlled-distraction-recall-held-at-1)
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
| Which embedding model produces the best dense retrieval? | Exp 10 |
| How does chunk granularity affect retrieval and generation? | Exp 2 |
| Is more context (larger k) always better? | Exp 3 |
| Does the prompt template matter, and by how much? | Exp 4 |
| When does retrieval help — and when does it actively hurt? | Exp 5, 9, 11 |
| Does better ranking (oracle / cross-encoder) translate into better answers? | Exp 6, 7 |
| How much does the difficulty of the retrieval task itself matter? | Exp 8, 9 |

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

### Notebook structure

The whole project is a single self-contained notebook, `project.ipynb`
(the modular file paths some sections reference do **not** exist). It is
organised into the following sections:

| Notebook section | Responsibility |
|---|---|
| §1 `config` | All hyperparameters (single source of truth) |
| §2 Corpus loading | TriviaQA `rc` loading + external Wikipedia distractors |
| §3 Chunking | Fixed-size word-window chunking with configurable overlap |
| §4 Retrieval backends | BM25, TF-IDF, Dense, and Cross-Encoder rerank classes |
| §5 Generator | Flan-T5-base wrapper: deterministic generation, caching, middle-truncation, context packing |
| §6 Evaluation metrics | Exact Match, Token F1, Recall@k, bootstrap CI |
| §7 Pipeline | Orchestrates retrieve → prompt → generate → evaluate |
| §8 Experiment helpers | JSON save/load, metric logging, shared setup |
| §9–13 Experiments 1–5 | Retriever / chunk / k / prompt / RAG-vs-No-RAG |
| §12b No-RAG prompt selection | Picks the no-context prompt on a held-out dev split, *before* Exp 5 uses it (D6) |
| §14 No-RAG prompt sensitivity | Displays/interprets the dev sweep |
| §15–17 Experiments 6–8 | Oracle, cross-encoder rerank, distractor sweep |
| §17b Experiment 9 | Controlled-distraction experiment (recall held at 1, gold position fixed) |
| §17d Experiment 10 | Embedding-model comparison (dense bi-encoder ablation) |
| §17e Experiment 11 | Forced retrieval miss (recall held at 0) — mirror of Exp 9 |
| §17c Paired significance | McNemar + paired-bootstrap tests across shared-question contrasts |
| §18 Analysis | Generates all figures from the saved JSON results |

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

We draw a **seeded random sample** of **`config.NUM_QUESTIONS`** questions from
the validation split (`QUESTION_SAMPLE_SEED = 42`; default **100** for fast
iteration; raise to 500–1000 for the final run). A random sample is more
representative of TriviaQA than the first N rows, and the seed keeps it
reproducible — the seed is folded into the corpus cache key so changing it
does not silently reuse a stale parse. For each question, TriviaQA provides
~2–10 entity Wikipedia pages
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

- **100 questions** with gold answer strings and aliases.
- **~2 500 documents** in the retrieval corpus: ~525 from TriviaQA (wiki
  entity pages + web hits) plus ~2 000 external Simple Wikipedia distractors
  (observed: 2 524 docs at the default settings). The per-source breakdown is
  logged at load time; overall capped at `config.MAX_CORPUS_DOCS = 10 000`.

Each document carries a `source` field (`"wiki"`, `"web"`, or
`"wiki_distractor"`) so per-source analyses are possible if needed.

The corpus is cached to disk after the first download to avoid
re-downloading. **Note:** the `rc` config is substantially larger than
`rc.wikipedia` (~5–10 GB download), so the first download will take a
while even though we only use a 100-question sample.

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

**Empirical result (n = 100, see §10).** Sparse edges out dense on the point
estimate — BM25 EM 0.69, TF-IDF 0.66, Dense 0.63 — consistent with the
"trivia is literal" prior above. But **none of the three pairwise differences
is statistically significant**: the paired tests (§6, §10) give BM25-vs-Dense
McNemar *p* = 0.26, TF-IDF-vs-Dense *p* = 0.65, BM25-vs-TF-IDF *p* = 0.45.
All three sit at the same Recall@5 = 0.94 — and that tie is *coincidental*,
not a shared retrieved set: each retriever misses a **different** set of 6
questions (the three miss-sets overlap on only 2, with 10 distinct questions
missed across all three). So the honest reading is a three-way tie at the
retrieval ceiling, with the generator-side gap dominating the residual error.
A larger or more adversarial corpus (Experiments 8 and 9), a multi-hop
dataset, or a TriviaQA-fine-tuned dense encoder would be expected to widen
the dense-vs-sparse spread.

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
- Primary: EM and F1 on the evaluation set (`config.NUM_QUESTIONS`, default 100)
- Secondary: Retrieval Recall@k (answer token-span-matched in top-5 chunks)

---

### Exp 2 — Chunk Size Ablation

**Fixed:** retriever = Dense, prompt = "instructed", **context-token budget =
`MAX_INPUT_TOKENS` (1024)**.

**Variable:** chunk size ∈ {64, 128, 256, 512} words

**Why a fixed token budget instead of a fixed `k`.** Under a 1024-token
generator budget a fixed `k` (e.g. `k=5`) does *not* hold the amount of
context constant: 5×64 words ≈ 450 tokens but 5×512 words ≈ 3 600 tokens. The
large-chunk conditions then overflow the budget and are middle-truncated
*mid-chunk*, while the small-chunk conditions under-fill the window — so a
fixed-`k` sweep confounds chunk **granularity** with the **amount of context
the generator actually reads**, and showing 5 chunks of 512 words is simply
impossible at 1024 tokens (no code or budget change fixes that). Instead we
hold the *context-token budget* fixed: we retrieve `PACK_RETRIEVE_N` (= 30)
candidates and **greedily pack whole chunks** into the prompt until the next
one would exceed the budget (`Generator.pack_to_budget`). Every chunk-size
condition then presents ~the same amount of context (~510–630 words in our
runs), nothing is truncated, and the number of chunks `k` becomes a *reported
output* (`packed_k_mean` ≈ 8.28 / 4.40 / 2.29 / 1.29 chunks at 64 / 128 / 256 /
512 words). The experiment therefore isolates the one thing we mean to vary —
how a fixed budget is *sliced*: many small chunks vs few large ones.

**Hypothesis (two-sided), at fixed budget:**
- *Small chunks* (64 words, ~10 of them) → finer-grained retrieval and higher
  Recall@k, but each passage carries little surrounding context and the budget
  is split across many headers.
- *Large chunks* (512 words, ~1 of them) → more self-contained context, but
  far fewer independent passages fit, so a single retrieval miss costs the
  whole window.
- An intermediate size (128–256 words) should optimise the EM/F1 tradeoff.

**Overlap:** 32 words to reduce boundary effects (held constant across sizes).

**Measurement:** EM, F1, Recall@k, and `packed_k_mean` at each chunk size.

---

### Exp 3 — Number of Retrieved Passages (k)

**Fixed:** retriever = Dense, chunk size = **48 words** (`EXP3_CHUNK_SIZE`),
prompt = "instructed"

**Variable:** k ∈ {1, 3, 5, 10}

**Why chunk = 48 here (not the project default of 128).** This experiment
varies `k`, so `k` has to be a genuine input — every passage we ask for must
actually reach the generator. At chunk = 128 the top of the sweep overflows
the 1024-token budget (k=10 ≈ 1 800 tokens) and is middle-truncated, so the
"k=10" condition would really measure a context-window ceiling, not the effect
of retrieval depth. Dropping the chunk size to 48 words makes the entire range
fit untruncated (verified: k=10 ≈ 850–917 tokens < 1024), so the k effect is
measured cleanly. The trade-off — smaller chunks carry slightly less
per-passage context — is held constant across all k values and so does not
confound the comparison.

**Hypothesis:**
- Recall@k increases monotonically with k (more passages → higher chance
  that one contains the answer).
- EM and F1 may *decrease* for large k if the model is distracted by extra
  passages (cf. Izacard & Grave [43] who address this with FiD) — and because
  the whole range now fits the budget, any decrease is attributable to
  distraction rather than to truncation dropping passages.

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
  question through a minimal `Q: {question}\nA:` cue. This cue is
  selected from a four-prompt sweep on a **held-out dev split** (disjoint
  from the eval set — notebook §12b, run *before* Exp 5 so the choice never
  peeks at the test questions; the spread across reasonable alternatives was
  only ~5 pp EM). The arm should therefore be read as a **blind-guess floor**
  for Flan-T5-base on
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
for each of the `NUM_QUESTIONS` (default 100) questions, we record whether
RAG *helps* (EM improves), *ties*, or *hurts* (EM decreases) relative to
No-RAG. At the default settings the split is **57 helps / 1 hurts / 42 ties**
(all 57 helps have retrieval recall = 1; the single hurt also had recall = 1,
i.e. the answer was retrieved but the generator was distracted).

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
> decomposition real statistical power. The empty "RAG hurts when recall = 0"
> cell — unobservable here because gold is almost always retrieved — is
> measured directly in **Experiment 11** (forced retrieval miss), which
> guarantees recall = 0 by construction.

---

### Exp 6 — Oracle Baseline (retrieval upper bound)

**Fixed:** retriever = Dense (used for the non-oracle slots), chunk size = 128, k = 5, prompt = "instructed"

**Condition:** Every question is *guaranteed* to have an answer-bearing
chunk in its top-k context. Concretely: run dense retrieval as usual at
k = 5; for any question whose top-5 misses the gold answer (recall = 0),
find the first chunk in the corpus that contains it (normalised token-span
match, `answer_in_text`) and place it at rank 1, dropping the lowest-ranked
retrieved chunk so k stays at 5. If no chunk in the corpus contains the answer
(chunking
artefact — the answer may straddle a window boundary), the question is
left as-is and flagged.

**Hypothesis & interpretation:**
- **Oracle − RAG** = the share of error attributable to *retrieval
  failure*. If the dense retriever already finds the answer for most
  questions, this gap is small. At n = 100 it is small: dense already has
  Recall@5 = 0.94, so the oracle only injects for 6 questions and EM rises
  just 0.63 → 0.65 (paired ΔEM +0.02; see §10).
- **1 − Oracle** = the *generator* failure given the answer is present.

> **Caveat — this oracle is NOT a true retrieval upper bound.** It only
> *adds* an answer-bearing chunk at rank 1 for the recall-0 questions
> (via the `answer_in_text` token-span match, which avoids the spurious
> partial-word hits a substring match would admit), and leaves ranks 2–5 as
> the raw dense output
> with no re-ordering. It therefore measures "dense retrieval, plus the
> answer forced in when missing", not "the best context any retriever
> could supply". The empirical proof: stronger retrieval beats the oracle —
> the **e5 embedding model (Exp 10) reaches EM 0.73** and the cross-encoder
> re-ranker (Exp 7) **EM 0.71**, both *above* the MiniLM oracle's 0.65. A
> better *retriever / ranking* of genuinely relevant chunks beats merely
> *guaranteeing the answer string is somewhere in the window*. So
> "1 − Oracle" should be read as a *soft* indication that the generator is
> the binding constraint, not as a hard ceiling that better retrieval cannot
> exceed.

**Measurement:** EM, F1, and the number of questions where the oracle
fix was applied (`n_injected`) and where no fix was possible
(`n_unfixable`).

---

### Exp 7 — Cross-Encoder Re-ranking

**Fixed:** chunk size = 128, k = 5 (final), prompt = "instructed"

**Condition:** two-stage retrieval — the **e5** dense bi-encoder
(`intfloat/e5-small-v2`, the strongest first stage from Exp 10) returns
the top-N candidates (`RERANK_TOP_N = 50`), then a cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores every `(query, chunk)`
pair jointly and the top-k are kept for the generator. The first-stage
model is set by `config.RERANK_BASE_MODEL` (with matching
`query:`/`passage:` prefixes), so the reranker re-orders the best
available candidates rather than MiniLM's.

A cross-encoder reads the query and chunk *together* in a single
transformer pass, so it can model term-level interactions a bi-encoder
cannot — but it is quadratic in the number of chunks, so it has to be
used as a *re-ranker* on a small candidate set rather than as a
first-stage retriever. This is the canonical "advanced RAG" topology in
the Gao et al. survey [44].

**Clean ablation.** Because the first stage is now e5 (not MiniLM), the
experiment also scores the **e5 dense-only** arm (`exp7_dense_base`) on the
same questions and compares against *that*, so the reranking effect is
isolated from the embedding-model upgrade — "does reranking add anything on
top of the best bi-encoder?" rather than conflating the two changes.

**Hypotheses:**
- Re-ranking should **raise Recall@5** if the e5 top-50 contained
  answer-bearing chunks that the e5 top-5 missed (a "good candidates,
  bad ranking" failure mode).
- Whether the Recall@5 lift translates into EM/F1 lift is the
  *interesting* question: if Exp 6 already showed the generator is the
  binding constraint, even a perfect retriever won't help; if Exp 6
  showed retrieval is the bottleneck, re-ranking should help roughly
  in proportion to the Recall@5 gain.

**Measurement:** EM, F1, Recall@5 vs the e5 dense-only baseline
(`exp7_dense_base`).

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

**Empirical result (n = 100, see §10): a flat null.** Recall@5 is
**exactly 0.94 at every distractor count** (0 / 500 / 2 000 / 5 000) and EM
barely moves (0.63 → 0.64). The hypothesised monotone decline does *not*
occur — but for an instructive reason rather than because the generator is
noise-robust: the Simple-Wikipedia distractors are **topic-agnostic**, so
the dense retriever never ranks them above the (task-conditioned) gold web
pages, and they exert *zero* pressure on the top-5. The lesson is about the
distractors, not the retriever: a noise floor only tests retrieval if the
noise can actually compete. **Experiment 9 supplies the test this one
cannot** — it uses *hard* (retrieved, topical) distractors and holds recall
fixed so distraction is isolated from retrieval misses.

**Measurement:** EM, F1, Recall@5 at each distractor count, with
absolute corpus sizes and chunk counts reported for context.

---

### Exp 9 — Controlled Distraction (recall held at 1)

**Fixed:** retriever = Dense, chunk size = **48 words** (`EXP3_CHUNK_SIZE`),
prompt = "instructed".

**Variable:** number of *non-gold* distractor chunks
`N ∈ {0, 1, 2, 4, 8}` placed alongside a guaranteed answer-bearing chunk.

**Why this experiment.** The project's headline question is *when does
retrieval hurt?*, but neither Exp 5 nor Exp 8 can isolate the mechanism: in
Exp 5 RAG almost never hurts (1/100), and in Exp 8 the distractors never
compete so Recall@5 is flat. This experiment **fixes Recall@k at 1** by
always including a gold chunk, and varies *only* the amount of competing
context. Any drop in EM/F1 as `N` grows is therefore **pure generator
distraction**, not a retrieval miss — the one regime in which "retrieval
hurts" can be measured cleanly.

**Design.**
- **Hard distractors.** The distractors are the top dense-retrieved chunks
  for the question that do *not* contain the answer — realistic, topical
  near-misses that a real RAG system would actually surface, not random
  text (contrast Exp 8's topic-agnostic noise).
- **Gold position held fixed (first).** The gold chunk is placed at rank 1 in
  every condition, so increasing `N` adds competing context *without* changing
  the gold's depth — this isolates the effect of distractor **count** alone.
  (An earlier version shuffled the gold to a per-`qid`-random slot, but that
  confounded count with depth: a randomly-placed gold sits deeper on average
  as `N` grows, so a measured EM drop could be either effect. Position 0 is the
  only slot definable consistently across the whole range — at `N = 0` the gold
  is the sole chunk — so fixing the gold first is the clean control.) The
  complementary *position* effect ("lost in the middle") — gold depth varied at
  fixed `N` — is left to future work.
- **Budget-clean.** At chunk = 48 even the largest context (gold + 8
  distractors = 9 chunks ≈ 660 tokens) fits the 1024-token budget
  untruncated, so the `N` effect is not confounded with truncation (same
  rationale as Exp 3).
- Questions whose answer appears in no chunk anywhere in the corpus are
  excluded so recall = 1 is guaranteed (at the default settings, **0
  excluded** — all 100 are usable, and the same set is scored at every `N`).
  **Note:** "answer present" uses the same `answer_in_text` **token-span** test
  as Recall@k (§6), so the `Recall@k = 1` guarantee — and the `N = 0`
  single-passage ceiling — hold in the whole-word sense (no spurious
  partial-word / numeric matches). A span match against TriviaQA's annotated
  gold-evidence pages would tighten it further (future work).

**Hypothesis:** Recall@k is 1 by construction at every `N`. EM/F1 should
*decline* as `N` grows if the generator is genuinely distracted by competing
context, even though the answer is always present.

**Measurement:** EM, F1, Recall@k (= 1, a sanity check) at each `N`.

---

### Exp 10 — Embedding-Model Comparison

**Fixed:** chunk size = 128 words, k = 5, prompt = "instructed", corpus fixed.

**Variable:** dense embedding model ∈ {`all-MiniLM-L6-v2`, `bge-small-en-v1.5`}
(downscaled — each extra model adds a cold full-corpus encode of tens of minutes
on CPU; `e5-small-v2` is left in `config.EMBEDDING_MODELS`, commented out, to
re-enable the fuller sweep).

Experiment 1 fixed the dense backbone to `all-MiniLM-L6-v2` and varied the
retrieval *family* (BM25 / TF-IDF / Dense). This experiment isolates the
**embedding model itself** — one of the design axes named in the project brief
— by swapping the dense bi-encoder while holding everything downstream fixed.
All three models are small (384-dim, 22–33 M params) so runtime stays modest;
the `all-MiniLM` row reuses the Experiment-1 dense cache and is therefore free.

**Fairness — instruction prefixes.** `e5` and `bge` are trained with
*asymmetric* query/passage instructions (`"query:"` / `"passage:"` for e5, a
search instruction on the query side for bge) and score poorly if embedded as
raw text. Each model is given its own recommended prefixes
(`config.EMBEDDING_MODELS`); `all-MiniLM` uses none. Without this the
comparison would conflate the model with a missing prefix.

**Hypothesis:** the stronger MTEB-leaderboard encoders (`bge`, `e5`) should
match or beat `all-MiniLM` on Recall@5; whether any recall gain reaches EM/F1
depends on whether the generator (not retrieval) is the binding constraint —
the same question Exp 6/7 probe.

**Measurement:** EM, F1, Recall@5 per embedding model.

---

### Exp 11 — Forced Retrieval Miss (recall held at 0)

**Fixed:** retriever = Dense, chunk size = **48 words** (`EXP3_CHUNK_SIZE`),
prompt = "instructed", k = 5.

**Condition:** the generator is given **only non-gold chunks** — the top
dense-retrieved chunks for the question that do *not* contain the answer — so
the answer is **guaranteed absent (Recall@k = 0 by construction)**. This is
the mirror image of Exp 9: Exp 9 holds recall at 1 and varies distractor
*count* (pure distraction); Exp 11 holds recall at 0 (a genuine retrieval
miss) and asks whether wrong context drags the model *below* its own
parametric knowledge.

**Why this experiment.** TriviaQA `rc` pools every question's gold evidence
into the corpus, so ordinary retrieval almost never misses and the "RAG hurts
when recall = 0" cell of the Exp 5 error analysis is essentially empty.
Forcing the miss is the only way to measure that regime with any power on this
corpus. Together, **Exp 9 (distraction at recall = 1)** and **Exp 11 (miss at
recall = 0)** bound the project's headline question — *when does retrieval
hurt?* — from both sides.

**Hypothesis:** if RAG-with-wrong-context scores **at or below** the No-RAG
parametric floor (Exp 5), wrong retrieved context actively hurts; if it stays
above, the generator is largely ignoring irrelevant context. Reported as a
per-question helps/hurts/ties split against No-RAG, plus a paired
`ForcedMiss − No-RAG` contrast (§6).

**Measurement:** EM, F1, Recall@k (= 0, a sanity check), and the
helps/hurts/ties split vs the No-RAG floor.

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
Recall@k = 1  if  normalize(answer)  is a contiguous TOKEN SUBSEQUENCE of
                  normalize(chunk)  for any chunk in the top-k retrieved chunks
           0  otherwise
```

This measures whether the retriever *found* the information, independently
of whether the generator *extracted* it. The match is a **whole-word token
span** (`answer_in_text`), not a character substring.

> **Why a token-span match (not a substring).** An earlier version used a
> normalised-*substring* test, which fired on spurious partial-word and
> numeric hits — "Black" inside "Blackboard", "Green" inside "Greenland",
> "1066" inside "10665" — biasing Recall@k upward on short / common / numeric
> answers. We now require the gold answer to appear as a *contiguous token
> subsequence* after SQuAD normalisation, which eliminates those false hits
> (the tokens `10665` and `1066` are distinct; `blackboard` no longer contains
> `black`). The **same** `answer_in_text` test is shared by the Exp 6 oracle
> injection and the Exp 9 gold check, so "the answer is present" means the
> same thing everywhere. A still-stricter alternative is a span match against
> TriviaQA's annotated gold-evidence pages, which we leave as future work.

### Bootstrap Confidence Intervals

We compute 95% CIs using **percentile bootstrap** with 1000 resamples
(Efron & Tibshirani, 1994). For a metric m computed on n examples:

1. Draw n samples with replacement from the per-example scores.
2. Compute the mean of each resample.
3. Report the 2.5th and 97.5th percentiles as the CI bounds.

### Paired significance testing

The per-metric bootstrap CI above answers *"is this single number
reliable?"*. It is the **wrong** tool for *"is condition A better than
condition B?"* when A and B are evaluated on the **same** questions, because
the (large) question-to-question difficulty variance inflates each marginal
CI even though it is *shared* by both conditions and cancels under pairing.
Two heavily overlapping marginal CIs can therefore hide a real, consistent
per-question difference. Since every experiment here scores all conditions
on the identical question set, we add two **paired** tests (notebook §17c,
`run_paired_significance`), reported for every shared-question contrast:

- **McNemar's exact test (EM).** Build the 2×2 table of per-question
  hit/miss for the two systems. Only the *discordant* cells matter:
  *b* = #(A right, B wrong), *c* = #(A wrong, B right). Under H₀ a discordant
  pair is equally likely to favour A or B, so *b* ~ Binomial(*b*+*c*, ½);
  the exact two-sided binomial tail is the *p*-value. Concordant pairs (both
  right / both wrong) carry no signal and are correctly ignored — which is
  precisely why this is more powerful than comparing marginal EM CIs.

- **Paired bootstrap (EM and F1 deltas).** Resample *questions* (not the two
  systems independently) with replacement, recompute the mean per-question
  delta (A − B) on each of 10 000 resamples, and report the mean delta, its
  95% percentile CI, and a two-sided bootstrap *p*-value
  `2·min(P(Δ*<0), P(Δ*>0))`.

> **Note — statistical vs practical significance.** A paired test can flag a
> *consistent* effect as significant even when it is *tiny*. The Oracle-vs-RAG
> F1 delta is the clearest example below: only +0.015, but because the oracle
> never *lowers* a question's F1 (it only ever adds an answer-bearing chunk),
> the delta is consistently ≥ 0 and the bootstrap *p* is ≈ 0. Read the
> magnitude (ΔEM/ΔF1) alongside the *p*-value, never the *p*-value alone.

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
- **Context packing (Experiment 2).** When `RAGPipeline.run(..., pack=True)`
  is used, the pipeline instead retrieves a deep candidate set and packs
  whole chunks up to the token budget (`Generator.pack_to_budget`), so the
  prompt fits with no truncation and `k` is reported as an output. The
  middle-truncation path above then acts only as a backstop for the
  fixed-`k` experiments (1, 4–8), where it can still fire at chunk = 128,
  k = 5.
- **Effective-k accounting.** The prompt-length log reports `effective_k` —
  how many retrieved chunks the generator still sees the header (and some
  text) for after truncation. It is counted at the *token* level over the
  original context, because the T5 tokenizer strips newlines on decode, so a
  text-level marker count would otherwise collapse every truncated prompt to
  `effective_k = 1`.
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
| Dataset sampling | Seeded random sample of N questions (`QUESTION_SAMPLE_SEED = 42`), folded into the corpus cache key |
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
```

### Run the experiments

Everything runs from the single notebook `project.ipynb`: open it and run all
cells top to bottom (in Jupyter / VS Code, or
`jupyter nbconvert --to notebook --execute project.ipynb`). This will:
1. Download TriviaQA (first run only; cached to `data/cache/` afterwards).
2. Run all 9 experiments (plus the paired-significance tests) sequentially,
   saving results JSON to `results/`.
3. Generate every figure into `figures/` (300 dpi PNG).

Every experiment recomputes end-to-end on each run and overwrites its
`results/*.json` — there is no results-level skip cache, so editing an
experiment and re-running always reflects the change.

Generation answer caching is controlled by `config.USE_GENERATION_CACHE`
(default **False**). With it off, the generator actually runs on every prompt
each run (greedy decoding is deterministic, so the numbers are identical to a
cached run — a full run just takes the full ~40 min instead of seconds). Set it
**True** for fast plot-only iteration once the results exist: answers are then
cached per *rendered prompt* in `data/cache/generation_cache.json` (keyed by the
exact prompt text), so re-runs only regenerate prompts that actually changed.
Note that key is the prompt text only, so if you cache and then change a
decoding/budget setting that doesn't alter the prompt (e.g. `MAX_NEW_TOKENS`),
delete that file to avoid replaying stale answers.

**Expected runtime** at the default `NUM_QUESTIONS=100` / `MAX_SEARCH_RESULTS_PER_Q=5`
(Apple Silicon / modern CPU):
- Data loading: 5–15 min first time (the `rc` dataset is large); <10 s from cache afterwards.
- Dense embedding (across chunk sizes, including the chunk=48 index for Exp 3): ~3–8 min.
- Generation (100 questions × ~25 conditions across all experiments): ~15–30 min on CPU, faster on MPS/CUDA. Cache keeps re-runs cheap.
- Total first end-to-end run: ~30–45 min.

To run at "report scale", bump `config.NUM_QUESTIONS` to 500 or 1000 and
optionally raise `MAX_SEARCH_RESULTS_PER_Q`; runtime scales roughly linearly
with both.

### Run a single experiment

Each experiment is a self-contained function in the notebook (`experiment_1`,
…, `experiment_9_distraction`, `experiment_10_embedding_models`,
`experiment_11_forced_miss`). After running the setup cells (config, corpus
load, generator) — and, for the No-RAG arm, the dev-split prompt-selection cell
(§12b) — call just the one you want, e.g.
`experiment_5(questions, corpus_docs, generator)`. The paired-significance
tests (`run_paired_significance`) read the saved `results/*.json` and need no
generation, so they can be re-run on their own once the experiments exist.

---

## 10. Results Summary

> **Status — current.** All numbers and figures below were produced by the
> committed notebook in a single end-to-end run at the default settings
> (`NUM_QUESTIONS = 100`, `MAX_SEARCH_RESULTS_PER_Q = 5`,
> `NUM_WIKI_DISTRACTORS = 2000`, `rc` corpus, `MAX_INPUT_TOKENS = 1024`,
> middle-truncation + question-twice prompts, greedy decoding). They live in
> `results/*.json` and `figures/*.png`; re-running the notebook reproduces
> them deterministically. **At n = 100 the bootstrap CIs are wide (±~0.09
> EM), so absolute numbers are directional** — for which differences are
> actually distinguishable, see the *paired-significance* table at the end
> of this section.
>
> **Caveats on the numbers below.** (1) *Question sampling has changed.*
> The committed results were produced with the old *first-N* validation slice;
> the notebook now draws a **seeded random sample** of `NUM_QUESTIONS`
> (`QUESTION_SAMPLE_SEED = 42`, folded into the corpus cache key), which is more
> representative. The next end-to-end run will therefore re-sample, re-parse the
> corpus, and re-embed — every number below will shift slightly (the qualitative
> story is not expected to change). (2) *Experiment 7 now re-ranks on the e5
> first stage* (was MiniLM) and compares against an e5 dense-only baseline
> (`exp7_dense_base`); the Exp 7 row and the rerank line of the paired table
> still show the committed **MiniLM-base** numbers and will refresh on the next
> run. Experiment 10's `e5-small` row is from a run with all three models
> enabled; `e5` is currently commented out in `config.EMBEDDING_MODELS` for
> runtime, but its cache and results are retained (see §5). (3) *Recall@k is
> now a token-span match* (`answer_in_text`, D4) rather than a character
> substring, so absolute Recall@k values will drop slightly (no more
> spurious "Black"⊂"Blackboard" hits) and the Exp 6 oracle / Exp 9 gold
> checks tighten accordingly. (4) *Two methodology changes have no committed
> numbers yet:* the No-RAG prompt is now selected on a held-out dev split
> (§12b, D6) and a new **Experiment 11** (forced retrieval miss, D5) was
> added; both populate on the next end-to-end run.

### Experiment 1 — Retriever Comparison

| Method | EM | Token F1 | Recall@5 |
|---|---|---|---|
| BM25   | 0.69 | 0.730 | 0.94 |
| TF-IDF | 0.66 | 0.700 | 0.94 |
| Dense  | 0.63 | 0.678 | 0.94 |

Sparse leads on the point estimate, but **no pairwise difference is
significant** (paired tests below). The identical Recall@5 = 0.94 is a
coincidence, not a shared retrieved set — each retriever misses a *different*
set of 6 questions (overlap = 2; 10 distinct questions missed in total). Read
this as a three-way tie at the retrieval ceiling.

### Experiment 2 — Chunk Size (fixed token budget)

| Chunk (words) | packed k (mean) | EM | Token F1 | Recall@k |
|---|---|---|---|---|
| 64  | 8.28 | 0.60 | 0.655 | 0.89 |
| 128 | 4.40 | 0.63 | 0.682 | 0.91 |
| 256 | 2.29 | 0.60 | 0.669 | 0.90 |
| 512 | 1.29 | 0.54 | 0.613 | 0.81 |

An intermediate chunk size (128 words) optimises the EM/F1 tradeoff, as
hypothesised; the largest chunks (512) both pack fewest passages and drop
Recall@k sharply.

### Experiment 3 — Number of Retrieved Passages (k, chunk = 48 words)

| k | EM | Token F1 | Recall@k |
|---|---|---|---|
| 1  | 0.35 | 0.384 | 0.36 |
| 3  | 0.49 | 0.544 | 0.67 |
| 5  | 0.53 | 0.579 | 0.78 |
| 10 | 0.59 | 0.646 | 0.85 |

EM, F1 and Recall@k all rise monotonically through k = 10: at this chunk
size the whole range fits the budget untruncated and no *distraction*
penalty appears — "more is better" here, because added passages keep adding
recall faster than they add noise. (Distraction at fixed recall is isolated
in Experiment 9.)

### Experiment 4 — Prompt Template

| Template   | EM    | Token F1 |
|---|---|---|
| Concise    | 0.62 | 0.677 |
| Instructed | 0.63 | 0.678 |

A null result: the leading "Answer with a short phrase." instruction makes no
distinguishable difference (paired ΔEM +0.01, McNemar *p* = 1.0). Flan-T5 is
already terse enough on this task that the explicit cue adds nothing.

### Experiment 5 — RAG vs No-RAG

| Condition                | EM    | Token F1 |
|---|---|---|
| No-RAG (parametric only) | 0.07 | 0.093 |
| RAG (Dense, k = 5)       | 0.63 | 0.678 |

Per-question breakdown (n = 100): **57 helps / 1 hurts / 42 ties**. All 57
"helps" have retrieval recall = 1; the single "hurts" also had recall = 1 (the
answer was retrieved but the generator was distracted). The No-RAG floor is
very low because Flan-T5-base genuinely lacks these long-tail facts — *that
the floor is so low is itself the finding*. This is the project's largest and
most significant effect (paired ΔEM +0.56, McNemar *p* < 1e-4).

### Experiment 6 — Oracle Baseline

| Condition                       | EM    | Token F1 | Recall@5 |
|---|---|---|---|
| RAG (Dense, k = 5)              | 0.63 | 0.678    | 0.94 |
| Oracle (answer guaranteed, k=5) | 0.65 | 0.693    | 1.00 |

Injected answer chunk for **6 / 100** questions; **0 / 100** unfixable. The
oracle barely moves EM (+0.02) because dense retrieval already had Recall@5 =
0.94, so the generator — not retrieval — is the binding constraint here. **But
the oracle is not a true upper bound**: the cross-encoder re-ranker (Exp 7)
scores *higher* (EM 0.71), because re-ranking improves the *quality* of all
five passages whereas the oracle only forces a token-span-matched chunk in at
rank 1 (see the caveat under Exp 6 in §5).

### Experiment 7 — Cross-Encoder Re-ranking

> **Numbers below are the committed MiniLM-base run.** The notebook now
> re-ranks on the **e5** first stage and compares against an e5 dense-only
> baseline (`exp7_dense_base`); these MiniLM-base numbers (rerank vs the Exp 1
> MiniLM dense) will refresh on the next end-to-end run.

| Method                           | EM    | Token F1 | Recall@5 |
|---|---|---|---|
| Dense, MiniLM (Exp 1)            | 0.63 | 0.678    | 0.94 |
| Dense + Rerank (top-50 → top-5)  | 0.71 | 0.784    | 0.99 |

Re-ranking lifts Recall@5 0.94 → 0.99 *and* improves answer quality beyond the
recall gain: paired ΔF1 +0.106 (*p* = 0.003), ΔEM +0.08 (McNemar *p* = 0.057,
borderline). A better *ordering* of genuinely relevant chunks helps the
generator, not just a higher recall count. Note that on the committed numbers
the strongest single end-to-end retriever is actually the **e5 embedding model
(EM 0.73, Exp 10)** — slightly above this MiniLM-base reranker (0.71) — which
is exactly why the experiment now re-ranks *on top of* e5 to test whether the
two gains stack.

### Experiment 8 — Distractor Count Sweep

| `NUM_WIKI_DISTRACTORS` | n_corpus_docs | EM    | Token F1 | Recall@5 |
|---|---|---|---|---|
| 0    | 525  | 0.63 | 0.678 | 0.94 |
| 500  | 1024 | 0.63 | 0.686 | 0.94 |
| 2000 | 2524 | 0.63 | 0.678 | 0.94 |
| 5000 | 5523 | 0.64 | 0.688 | 0.94 |

**A flat null.** Recall@5 is exactly 0.94 at every distractor count and EM is
flat — but because the topic-agnostic Simple-Wikipedia distractors never
out-rank the gold pages, not because the generator is noise-robust. The
proper distraction test (hard distractors, recall held fixed) is Experiment 9.

### Experiment 9 — Controlled Distraction (recall held at 1)

Produced with the current *fixed-first-position* gold design (§5): the gold
chunk is held at rank 1 and only the number of competing non-gold chunks varies.

| N distractors | EM | Token F1 | Recall@k |
|---|---|---|---|
| 0 | 0.68 | 0.718 | 1.00 |
| 1 | 0.59 | 0.633 | 1.00 |
| 2 | 0.57 | 0.635 | 1.00 |
| 4 | 0.53 | 0.584 | 1.00 |
| 8 | 0.50 | 0.565 | 1.00 |

0 / 100 questions excluded (a gold chunk exists for all). **This is where
"retrieval hurts" shows up cleanly.** With the answer guaranteed present
(Recall@k = 1 throughout) and the gold pinned at rank 1, adding hard, topical,
non-gold distractors drives EM down ~18 points (0.68 → 0.50) and F1 ~15 points
(0.718 → 0.565), monotonically across the full range. The N = 0 anchor (0.68)
is also a clean single-passage generator ceiling at this chunk size: even with
a perfect one-chunk context the generator misses ~32 % of questions.

### Experiment 10 — Embedding-Model Comparison

| Embedding model        | EM    | Token F1 | Recall@5 |
|---|---|---|---|
| `all-MiniLM-L6-v2`     | 0.63 | 0.678    | 0.94 |
| `bge-small-en-v1.5`    | 0.65 | 0.717    | 0.94 |
| `e5-small-v2`          | 0.73 | 0.786    | 0.97 |

The stronger MTEB-leaderboard encoders help, and the effect reaches the
generator: **e5 is the single best dense retriever in the project** — it lifts
Recall@5 0.94 → 0.97 and EM 0.63 → 0.73 over the MiniLM baseline (a +0.10 EM
gain, larger than the cross-encoder reranker's). bge sits in between. The
embedding model is thus one of the highest-impact knobs here, consistent with
Gao et al. [44]. **Note:** these numbers are from a run with all three models
enabled; `e5-small-v2` is currently commented out in
`config.EMBEDDING_MODELS` to keep cold-start runtime down, but its embedding
cache and `results/exp10_e5-small.json` are retained, and it is wired in as the
first stage of the Exp 7 reranker (`config.RERANK_BASE_MODEL`).

### Experiment 11 — Forced Retrieval Miss (recall held at 0)

> **Pending — not yet run.** Added after the last end-to-end run; its
> `results/exp11_forced_miss.json` and `figures/fig14_forced_miss.png` populate
> on the next run. Expected outputs: EM / F1 (≈ at or below the No-RAG floor if
> wrong context hurts), Recall@k = 0 (sanity check), and a helps/hurts/ties
> split vs No-RAG plus the `ForcedMiss − No-RAG` paired contrast. This supplies
> the recall = 0 measurement that Exp 5 cannot (gold is almost always
> retrieved), completing the "when does retrieval hurt" picture alongside Exp 9.

### Paired significance (same questions; McNemar on EM, paired bootstrap on deltas)

| Contrast (A − B) | ΔEM | McNemar *p* | disc. (A/B) | ΔF1 | ΔF1 95% CI | F1 boot *p* |
|---|---|---|---|---|---|---|
| BM25 − Dense          | +0.060 | 0.2632 | 13/7  | +0.052 | [−0.034, +0.138] | 0.2294 |
| TF-IDF − Dense        | +0.030 | 0.6476 | 11/8  | +0.022 | [−0.064, +0.105] | 0.6090 |
| BM25 − TF-IDF         | +0.030 | 0.4531 | 5/2   | +0.030 | [−0.016, +0.081] | 0.1980 |
| Instructed − Concise  | +0.010 | 1.0000 | 3/2   | +0.001 | [−0.039, +0.038] | 0.8962 |
| RAG − No-RAG          | +0.560 | 0.0000 | 57/1  | +0.585 | [+0.487, +0.677] | 0.0000 |
| Dense+Rerank − Dense* | +0.080 | 0.0574 | 11/3  | +0.106 | [+0.034, +0.181] | 0.0034 |
| Oracle − RAG          | +0.020 | 0.5000 | 2/0   | +0.015 | [+0.000, +0.040] | 0.0000 |

*The committed rerank contrast is MiniLM-base rerank vs MiniLM dense. The
notebook now computes **e5-base rerank vs e5 dense** (`exp7_rerank` vs
`exp7_dense_base`); this row will refresh on the next run. A new
`ForcedMiss − No-RAG` contrast (Exp 11) will also be added to this table on the
next run.

Reading the table: **only RAG-vs-No-RAG and Dense+Rerank-vs-Dense (on F1) are
statistically distinguishable** at n = 100. The retriever choice (Exp 1) and
the prompt template (Exp 4) are *not* — their apparent orderings are within
sampling noise. Oracle-vs-RAG is the cautionary case: the F1 delta is
"significant" (*p* ≈ 0) but practically negligible (+0.015), because the
oracle only ever *adds* to a question's score, so the sign is consistent even
though the magnitude is tiny (see the note in §6).

---

Figures in `figures/`:

| Figure | Content |
|---|---|
| `fig1_retriever_comparison.png` | EM + F1 bar chart across retrieval methods with 95% CI |
| `fig2_chunk_size.png`           | EM / F1 / Recall@k vs chunk size (whole chunks packed to a fixed token budget) |
| `fig3_k_values.png`             | EM / F1 / Recall@k vs k (chunk = 48 words; line + CI band) |
| `fig4_prompt_template.png`      | EM + F1: concise vs instructed prompts |
| `fig5_rag_vs_no_rag.png`        | EM + F1: RAG vs parametric baseline |
| `fig6_error_analysis.png`       | Pie (helps/hurts/ties) + scatter (recall vs EM) |
| `fig7_qualitative.png`          | Example table: cases where RAG helps vs hurts |
| `fig8_oracle.png`               | EM + F1: No-RAG / RAG / Oracle (Exp 6 — retrieval vs generation gap) |
| `fig9_rerank.png`               | EM + F1 + Recall@5: e5 Dense vs e5 Dense + Cross-Encoder Rerank (Exp 7) |
| `fig10_distractor_sweep.png`    | EM / F1 / Recall@k vs number of external distractors (Exp 8) |
| `fig11_distraction.png`         | EM / F1 / Recall@k vs number of hard non-gold distractors, recall held at 1 (Exp 9) |
| `fig12_significance.png`        | Table of paired tests (McNemar on EM, paired bootstrap on EM/F1 deltas) |
| `fig13_embedding_models.png`    | EM + F1 + Recall@5 across dense embedding models (Exp 10) |
| `fig14_forced_miss.png`         | EM + F1: No-RAG / Forced-miss (non-gold ctx) / RAG (Exp 11) |

---

## 11. Limitations

1. **Model scale**: Flan-T5-base is a relatively small model. Larger models
   (e.g., Flan-T5-XL, LLaMA-3) may show different patterns — e.g., less
   benefit from retrieval because they have stronger parametric knowledge.

2. **Statistical power**: All reported numbers are at `NUM_QUESTIONS = 100`,
   where the marginal bootstrap CIs are ±~0.09 EM. The paired tests (§6, §10)
   show that only the two largest effects (RAG vs No-RAG; rerank vs dense on
   F1) are distinguishable at this scale — the retriever and prompt
   comparisons are within noise. Quantitative claims about the smaller
   effects would require the report-scale run (`NUM_QUESTIONS = 500`–1000).

3. **Corpus size and provenance**: At the default settings the corpus is
   ~2 500 documents (≈525 from TriviaQA + 2 000 Simple-Wikipedia
   distractors). This is enough to demonstrate the pipeline and the
   directional findings, but is much smaller and less diverse than the full
   open-domain setting in Lewis et al. [42] (~21 M Wikipedia passages
   indexed with FAISS). As discussed in §4, the in-batch web hits are
   task-conditioned by construction, and the external Simple-Wikipedia
   distractors are topic-agnostic rather than adversarial (Exp 8 confirms
   they never compete) — both factors flatter retrieval, so absolute
   Recall@k should be read as **upper bounds**, not open-domain estimates.
   Because gold evidence is therefore almost always retrievable, the genuine
   *retrieval-miss* regime (recall = 0) does not arise naturally; **Exp 11**
   forces it to measure that regime directly, but a larger / more adversarial
   open-domain corpus would let it arise on its own.

4. **No retriever fine-tuning**: We use off-the-shelf BM25 and
   `all-MiniLM-L6-v2` with no domain-specific fine-tuning. A DPR model
   fine-tuned on TriviaQA (Lewis et al. [42]) would likely widen the
   dense-vs-sparse gap that is currently a tie (Exp 1).

5. **Limited advanced-RAG coverage**: Cross-encoder re-ranking *is*
   evaluated (Exp 7), but query expansion / rewriting (HyDE, step-back
   prompting) and iterative/adaptive retrieval are not.

6. **Single-hop questions only**: TriviaQA questions are mostly single-hop
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
