class config:
    """Central configuration for all RAG experiments."""

    # ── Paths ────────────────────────────────────────────────
    ROOT_DIR    = Path.cwd()
    DATA_DIR    = ROOT_DIR / "data"
    RESULTS_DIR = ROOT_DIR / "results"
    FIGURES_DIR = ROOT_DIR / "figures"
    CACHE_DIR   = ROOT_DIR / "data" / "cache"

    # ── Dataset ──────────────────────────────────────────────
    DATASET_NAME    = "mandarjoshi/trivia_qa"  # HF requires namespace/name
    DATASET_CONFIG  = "rc"      # Open reading-comprehension config:
                                # pools wiki entity pages AND web search
                                # results into the corpus, so retrieval
                                # competes against genuine distractors
                                # rather than only curated evidence.
    DATASET_SPLIT   = "validation"
    NUM_QUESTIONS   = 100       # Fast-iteration default; bump to 500 (or
                                # 1000) for the final report run. Runtime
                                # scales roughly linearly.
    MAX_CORPUS_DOCS = 10000     # With MAX_SEARCH_RESULTS_PER_Q=5, 100 qs
                                # yield ~700-1500 unique docs; this is a
                                # safety ceiling, not a working limit.
                                # Raised from 5000 so the distractor sweep
                                # (Experiment 8) can run at n_dist=5000
                                # without the cap clipping the distractor
                                # mix.
    MAX_SEARCH_RESULTS_PER_Q = 5  # Take only the top-N web hits per qn.
                                  # rc ships 10-50 search results per qn;
                                  # the long tail mostly adds embedding
                                  # cost.  Set to None for no cap.
    NUM_WIKI_DISTRACTORS = 2000   # External Wikipedia articles mixed into
                                  # the retrieval pool as topic-agnostic
                                  # distractors. The in-batch TriviaQA
                                  # pool alone contains the gold pages
                                  # plus only ~1.5k other-question docs,
                                  # so retrieval is easier here than in a
                                  # real open-domain setting. Adding
                                  # generic Wikipedia paragraphs widens
                                  # the noise floor without breaking
                                  # answerability. Set 0 to disable.
                                  # First-run cost on CPU at this size:
                                  # ~100 MB download, ~1-3 min extra
                                  # embedding time per chunk size.
    WIKI_DISTRACTOR_MAX_CHARS = 2000  # Truncate each distractor article
                                      # to its first N characters; keeps
                                      # chunk counts bounded so the
                                      # distractor pool stays modest.

    # ── Chunking ─────────────────────────────────────────────
    CHUNK_SIZES        = [64, 128, 256, 512]
    CHUNK_OVERLAP      = 32
    DEFAULT_CHUNK_SIZE = 128

    # ── Retrieval ────────────────────────────────────────────
    RETRIEVER_TYPES  = ["bm25", "tfidf", "dense"]
    K_VALUES         = [1, 3, 5, 10]
    DEFAULT_K        = 5
    DENSE_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
    DENSE_BATCH_SIZE = 64

    # ── Re-ranker (Experiment 7) ─────────────────────────────
    # Cross-encoder applied on top of a dense first-stage retrieval. The
    # base retriever returns RERANK_TOP_N candidates; the cross-encoder
    # scores every (query, chunk) pair and selects the top-k for the
    # generator. This is the canonical "advanced RAG" two-stage setup
    # (Gao et al. [44]).
    RERANK_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_N      = 50
    RERANK_BATCH_SIZE = 64

    # ── Distractor sweep (Experiment 8) ──────────────────────
    # Re-runs the RAG arm at multiple distractor counts so we can see how
    # retrieval degrades as the topic-agnostic noise floor rises. The
    # No-RAG arm is corpus-independent, so we reuse it from Experiment 5.
    DISTRACTOR_SWEEP_VALUES = [0, 500, 2000, 5000]

    # ── Generation ───────────────────────────────────────────
    GENERATOR_MODEL      = "google/flan-t5-base"
    MAX_INPUT_TOKENS     = 1024   # Raised from 512: at chunk=128, k>=3
                                  # already overflows a 512-token budget
                                  # (~170 tok/chunk + ~100 tok prompt
                                  # overhead), so the encoder receives an
                                  # identically-truncated input for k=3/5/10
                                  # and Exp 3 cannot resolve k effects.
                                  # Flan-T5 tolerates >=1024 input fine
                                  # in practice. NB: prompts that still
                                  # exceed the budget are *middle*-truncated
                                  # by the Generator (see _truncate_middle):
                                  # the question repeats and the "Short
                                  # answer:" cue are preserved, only the
                                  # context body is shortened.
                                  # Re-runs after this change require
                                  # clearing data/cache/generation_cache.json
                                  # so prompts aren't re-served from the
                                  # earlier 512-token cache.
    MAX_NEW_TOKENS       = 32       # answers are almost always <10 tokens
    GENERATOR_BATCH_SIZE = 16

    # ── Evaluation ───────────────────────────────────────────
    BOOTSTRAP_SAMPLES = 1000
    RANDOM_SEED       = 42

    # ── Run control ──────────────────────────────────────────
    # When True, experiments always recompute even if a results JSON
    # already exists. Set False only if you want plot-only re-runs
    # off cached results.
    FORCE_RERUN = True

    # ── Prompting ────────────────────────────────────────────
    PROMPT_TEMPLATES = {
        # NB: both RAG templates repeat the question *before* and *after*
        # the context. T5 truncation chops from the end, so a question
        # only at the end can be cut off; the leading copy keeps it
        # visible under aggressive truncation while the trailing
        # "Answer:" / "Short answer:" still serves as the generation cue.
        # As of the middle-truncation fix in `Generator._truncate_middle`,
        # the trailing question/cue is also preserved verbatim — only the
        # `{context}` body is shortened when the prompt exceeds budget.
        #
        # The instructed template's leading sentence is intentionally
        # short (~7 tokens) so concise and instructed are length-matched
        # to within ~10 tokens — that way Exp 4 measures the instruction
        # signal itself, not the cost of displaced context.
        "concise": (
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
        "instructed": (
            "Answer with a short phrase.\n\n"
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Short answer:"
        ),
        # The no_context template is the canonical no-RAG (parametric)
        # cue. We adopt T5's "Q:/A:" shape because the prompt-robustness
        # sweep in section 14 ("Prompt sensitivity of the No-RAG
        # baseline") showed it produced the highest EM among four
        # reasonable alternatives. The spread across alternatives is
        # only ~5pp, so the No-RAG arm should be read as a robust
        # parametric *floor* — Flan-T5-base genuinely lacks the
        # TriviaQA-grade facts — not as a measure of its specific
        # knowledge.
        "no_context": "Q: {question}\nA:",
    }
    DEFAULT_PROMPT = "instructed"


for _d in [config.DATA_DIR, config.RESULTS_DIR, config.FIGURES_DIR, config.CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

print("Working directory:", config.ROOT_DIR)
print("Results dir:      ", config.RESULTS_DIR)
print("Figures dir:      ", config.FIGURES_DIR)
print("Cache dir:        ", config.CACHE_DIR)
