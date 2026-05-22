"""
config.py — Central configuration for all RAG experiments.

Every hyperparameter lives here. Changing a value here propagates
through the entire pipeline automatically. No magic constants elsewhere.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
CACHE_DIR   = ROOT_DIR / "data" / "cache"

for _d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────
DATASET_NAME   = "trivia_qa"          # HuggingFace dataset identifier
DATASET_CONFIG = "rc.wikipedia"       # Wikipedia evidence pages (as in Lewis et al.)
DATASET_SPLIT  = "validation"         # validation split (avoid contaminating test)
NUM_QUESTIONS  = 500                  # first N questions for all experiments
                                      # (set lower if memory-constrained; 300 is fast)
MAX_CORPUS_DOCS = 6000                # cap on total Wikipedia pages to index
                                      # (keeps memory manageable; ~300+ pages/question)

# ─────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────
# Experiment 2: vary chunk size to measure its effect on retrieval quality
# and downstream answer quality.
CHUNK_SIZES   = [64, 128, 256, 512]  # words per chunk
CHUNK_OVERLAP = 32                    # word overlap between consecutive chunks
                                      # (half the smallest chunk = generous overlap)
DEFAULT_CHUNK_SIZE = 128              # used in experiments that don't vary chunk size

# ─────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────
# Experiment 1: compare retrieval methods
RETRIEVER_TYPES = ["bm25", "tfidf", "dense"]  # methods to benchmark

# Experiment 3: vary number of retrieved passages (k)
K_VALUES    = [1, 3, 5, 10]          # number of passages returned to the generator
DEFAULT_K   = 5                       # used in experiments that don't vary k

# Dense retrieval model — strong multilingual-capable sentence encoder
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 22M params, fast
DENSE_BATCH_SIZE = 64                 # encoding batch size (reduce if OOM)

# ─────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────
# Flan-T5-base: instruction-tuned T5 variant, no API key needed,
# runs on CPU/MPS, handles multi-passage prompts well.
GENERATOR_MODEL  = "google/flan-t5-base"   # ~250M params
MAX_INPUT_TOKENS = 1024               # truncate prompt to stay within model limits
MAX_NEW_TOKENS   = 64                 # maximum tokens to generate per answer
GENERATOR_BATCH_SIZE = 16            # generation batch size

# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────
BOOTSTRAP_SAMPLES = 1000             # number of bootstrap resamples for 95% CI
RANDOM_SEED       = 42               # reproducibility

# ─────────────────────────────────────────────────────────────────
# Prompting
# ─────────────────────────────────────────────────────────────────
# Experiment 4 (prompting ablation): we test two prompt templates.
# "concise" = minimal instruction (prone to verbosity),
# "instructed" = explicit directive to answer briefly and directly.
PROMPT_TEMPLATES = {
    "concise": (
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "instructed": (
        "You are a factual QA assistant. "
        "Read the context and answer the question with a short phrase only. "
        "Do NOT repeat the question. Do NOT write a sentence; just the answer.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Short answer:"
    ),
    "no_context": (
        "Answer the following question with a short phrase.\n\n"
        "Question: {question}\n"
        "Short answer:"
    ),
}
DEFAULT_PROMPT = "instructed"        # default template across experiments
