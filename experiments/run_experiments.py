"""
run_experiments.py — Master experiment runner.

Five experiments are executed sequentially.  Each writes its results as a
JSON file to config.RESULTS_DIR so that the analysis / plotting script can
be run independently without re-running experiments.

Experiments
-----------
1. Retriever comparison  : BM25 vs TF-IDF vs Dense (fixed chunk=128, k=5)
2. Chunk size            : 64 / 128 / 256 / 512 words  (Dense retriever, k=5)
3. Number of passages k  : k ∈ {1, 3, 5, 10}           (Dense, chunk=128)
4. Prompt template       : "concise" vs "instructed"    (Dense, chunk=128, k=5)
5. RAG vs No-RAG         : with / without retrieval     (Dense, chunk=128, k=5)

Usage
-----
    python experiments/run_experiments.py [--exp {1,2,3,4,5,all}]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Make root importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.corpus    import load_triviaqa
from src.chunker   import chunk_corpus
from src.retriever import build_retriever
from src.generator import Generator
from src.pipeline  import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiments")


# ─────────────────────────────────────────────────────────────────
# Helper: save results
# ─────────────────────────────────────────────────────────────────

def save_results(name: str, data: dict) -> Path:
    """Serialise results to JSON in config.RESULTS_DIR."""
    out_path = config.RESULTS_DIR / f"{name}.json"
    # Strip large non-serialisable objects before saving
    slim = {k: v for k, v in data.items() if k not in ("retrieved_chunks", "prompts")}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    logger.info("Results saved → %s", out_path)
    return out_path


def already_done(name: str) -> bool:
    """Return True if the results file already exists (skip re-running)."""
    p = config.RESULTS_DIR / f"{name}.json"
    if p.exists():
        logger.info("Skipping %s — already done (delete %s to re-run).", name, p.name)
        return True
    return False


def load_results(name: str) -> dict:
    out_path = config.RESULTS_DIR / f"{name}.json"
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_metrics(label: str, result: dict) -> None:
    m = result["metrics"]
    em  = m.get("em",  {})
    f1  = m.get("f1",  {})
    rec = m.get("recall_at_k", {})
    logger.info(
        "%s — EM %.3f [%.3f–%.3f]  F1 %.3f [%.3f–%.3f]  Recall %.3f [%.3f–%.3f]",
        label,
        em.get("mean", 0),  em.get("lo", 0),  em.get("hi", 0),
        f1.get("mean", 0),  f1.get("lo", 0),  f1.get("hi", 0),
        rec.get("mean", 0), rec.get("lo", 0), rec.get("hi", 0),
    )


# ─────────────────────────────────────────────────────────────────
# Shared setup
# ─────────────────────────────────────────────────────────────────

def setup():
    """Load data and return (questions, corpus_docs, generator)."""
    logger.info("=" * 60)
    logger.info("Loading TriviaQA (%d questions)…", config.NUM_QUESTIONS)
    questions, corpus_docs = load_triviaqa(num_questions=config.NUM_QUESTIONS)
    logger.info("Corpus: %d documents", len(corpus_docs))

    generator = Generator()
    return questions, corpus_docs, generator


# ─────────────────────────────────────────────────────────────────
# Experiment 1: Retriever comparison
# ─────────────────────────────────────────────────────────────────

def experiment_1(questions, corpus_docs, generator):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1 — Retriever comparison (BM25 / TF-IDF / Dense)")
    logger.info("Fixed: chunk_size=%d  k=%d  prompt=%s",
                config.DEFAULT_CHUNK_SIZE, config.DEFAULT_K, config.DEFAULT_PROMPT)

    chunks = chunk_corpus(corpus_docs, chunk_size=config.DEFAULT_CHUNK_SIZE)
    logger.info("Chunks: %d (size=%d)", len(chunks), config.DEFAULT_CHUNK_SIZE)

    exp1_results = {}
    for rtype in config.RETRIEVER_TYPES:
        if already_done(f"exp1_{rtype}"):
            exp1_results[rtype] = load_results(f"exp1_{rtype}")
            continue
        logger.info("Building retriever: %s", rtype)
        retriever = build_retriever(
            rtype, chunks,
            cache_dir=config.CACHE_DIR,
            chunk_size=config.DEFAULT_CHUNK_SIZE,
        )
        pipe   = RAGPipeline(retriever, generator)
        result = pipe.run(questions, k=config.DEFAULT_K)
        print_metrics(rtype, result)
        exp1_results[rtype] = result
        save_results(f"exp1_{rtype}", result)

    # Summary table
    summary = {}
    for rtype, result in exp1_results.items():
        m = result["metrics"]
        summary[rtype] = {
            "em":         m["em"],
            "f1":         m["f1"],
            "recall_at_k": m.get("recall_at_k", {}),
            "timing":     result["timing"],
        }
    save_results("exp1_summary", {"results": summary})
    return exp1_results


# ─────────────────────────────────────────────────────────────────
# Experiment 2: Chunk size
# ─────────────────────────────────────────────────────────────────

def experiment_2(questions, corpus_docs, generator):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2 — Chunk size ablation")
    logger.info("Fixed: retriever=dense  k=%d  prompt=%s",
                config.DEFAULT_K, config.DEFAULT_PROMPT)

    exp2_results = {}
    for chunk_size in config.CHUNK_SIZES:
        if already_done(f"exp2_chunk{chunk_size}"):
            exp2_results[chunk_size] = load_results(f"exp2_chunk{chunk_size}")
            continue
        logger.info("Chunk size: %d words", chunk_size)
        chunks = chunk_corpus(corpus_docs, chunk_size=chunk_size)
        logger.info("  → %d chunks", len(chunks))

        retriever = build_retriever(
            "dense", chunks,
            cache_dir=config.CACHE_DIR,
            chunk_size=chunk_size,
        )
        pipe   = RAGPipeline(retriever, generator)
        result = pipe.run(questions, k=config.DEFAULT_K)
        print_metrics(f"chunk={chunk_size}", result)

        exp2_results[chunk_size] = result
        save_results(f"exp2_chunk{chunk_size}", result)

    summary = {}
    for sz, result in exp2_results.items():
        m = result["metrics"]
        summary[str(sz)] = {
            "em":          m["em"],
            "f1":          m["f1"],
            "recall_at_k": m.get("recall_at_k", {}),
            "n_chunks":    result["timing"]["n_questions"],  # repurposed temporarily
        }
    save_results("exp2_summary", {"results": summary})
    return exp2_results


# ─────────────────────────────────────────────────────────────────
# Experiment 3: Number of retrieved passages (k)
# ─────────────────────────────────────────────────────────────────

def experiment_3(questions, corpus_docs, generator):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3 — Number of passages k ∈ %s", config.K_VALUES)
    logger.info("Fixed: retriever=dense  chunk_size=%d  prompt=%s",
                config.DEFAULT_CHUNK_SIZE, config.DEFAULT_PROMPT)

    chunks    = chunk_corpus(corpus_docs, chunk_size=config.DEFAULT_CHUNK_SIZE)
    retriever = build_retriever(
        "dense", chunks,
        cache_dir=config.CACHE_DIR,
        chunk_size=config.DEFAULT_CHUNK_SIZE,
    )

    exp3_results = {}
    for k in config.K_VALUES:
        if already_done(f"exp3_k{k}"):
            exp3_results[k] = load_results(f"exp3_k{k}")
            continue
        logger.info("k = %d", k)
        pipe   = RAGPipeline(retriever, generator)
        result = pipe.run(questions, k=k)
        print_metrics(f"k={k}", result)
        exp3_results[k] = result
        save_results(f"exp3_k{k}", result)

    summary = {}
    for k, result in exp3_results.items():
        m = result["metrics"]
        summary[str(k)] = {
            "em":          m["em"],
            "f1":          m["f1"],
            "recall_at_k": m.get("recall_at_k", {}),
        }
    save_results("exp3_summary", {"results": summary})
    return exp3_results


# ─────────────────────────────────────────────────────────────────
# Experiment 4: Prompt template
# ─────────────────────────────────────────────────────────────────

def experiment_4(questions, corpus_docs, generator):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4 — Prompt template: concise vs instructed")
    logger.info("Fixed: retriever=dense  chunk_size=%d  k=%d",
                config.DEFAULT_CHUNK_SIZE, config.DEFAULT_K)

    chunks    = chunk_corpus(corpus_docs, chunk_size=config.DEFAULT_CHUNK_SIZE)
    retriever = build_retriever(
        "dense", chunks,
        cache_dir=config.CACHE_DIR,
        chunk_size=config.DEFAULT_CHUNK_SIZE,
    )

    exp4_results = {}
    for template in ["concise", "instructed"]:
        if already_done(f"exp4_{template}"):
            exp4_results[template] = load_results(f"exp4_{template}")
            continue
        logger.info("Template: %s", template)
        pipe   = RAGPipeline(retriever, generator)
        result = pipe.run(questions, k=config.DEFAULT_K, prompt_template=template)
        print_metrics(f"prompt={template}", result)
        exp4_results[template] = result
        save_results(f"exp4_{template}", result)

    summary = {}
    for tmpl, result in exp4_results.items():
        m = result["metrics"]
        summary[tmpl] = {"em": m["em"], "f1": m["f1"]}
    save_results("exp4_summary", {"results": summary})
    return exp4_results


# ─────────────────────────────────────────────────────────────────
# Experiment 5: RAG vs No-RAG
# ─────────────────────────────────────────────────────────────────

def experiment_5(questions, corpus_docs, generator):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5 — RAG vs No-RAG baseline")
    logger.info("Fixed: retriever=dense  chunk_size=%d  k=%d  prompt=%s",
                config.DEFAULT_CHUNK_SIZE, config.DEFAULT_K, config.DEFAULT_PROMPT)

    chunks    = chunk_corpus(corpus_docs, chunk_size=config.DEFAULT_CHUNK_SIZE)
    retriever = build_retriever(
        "dense", chunks,
        cache_dir=config.CACHE_DIR,
        chunk_size=config.DEFAULT_CHUNK_SIZE,
    )

    # RAG condition
    if already_done("exp5_rag"):
        rag_result = load_results("exp5_rag")
    else:
        logger.info("Running RAG condition…")
        rag_pipe   = RAGPipeline(retriever, generator, use_retrieval=True)
        rag_result = rag_pipe.run(questions, k=config.DEFAULT_K)
        print_metrics("RAG", rag_result)
        save_results("exp5_rag", rag_result)

    # No-RAG condition
    if already_done("exp5_no_rag"):
        no_rag_result = load_results("exp5_no_rag")
    else:
        logger.info("Running No-RAG (parametric-only) condition…")
        no_rag_pipe   = RAGPipeline(None, generator, use_retrieval=False)
        no_rag_result = no_rag_pipe.run(questions, k=config.DEFAULT_K)
        print_metrics("No-RAG", no_rag_result)
        save_results("exp5_no_rag", no_rag_result)

    # Per-question delta analysis
    rag_per     = {ex["qid"]: ex for ex in rag_result["per_example"]}
    no_rag_per  = {ex["qid"]: ex for ex in no_rag_result["per_example"]}

    delta_analysis = []
    for qid, rag_ex in rag_per.items():
        if qid not in no_rag_per:
            continue
        no_rag_ex = no_rag_per[qid]
        delta_analysis.append({
            "qid":        qid,
            "question":   rag_ex["question"],
            "gold":       rag_ex["gold"],
            "rag_pred":   rag_ex["prediction"],
            "no_rag_pred": no_rag_ex["prediction"],
            "rag_em":     rag_ex["em"],
            "no_rag_em":  no_rag_ex["em"],
            "rag_f1":     rag_ex["f1"],
            "no_rag_f1":  no_rag_ex["f1"],
            "recall":     rag_ex.get("recall", 0),
            "rag_helps":  int(rag_ex["em"] > no_rag_ex["em"]),
            "rag_hurts":  int(rag_ex["em"] < no_rag_ex["em"]),
        })

    # Counts
    helps = sum(d["rag_helps"] for d in delta_analysis)
    hurts = sum(d["rag_hurts"] for d in delta_analysis)
    ties  = len(delta_analysis) - helps - hurts
    logger.info(
        "RAG helps: %d  RAG hurts: %d  Ties: %d  (out of %d)",
        helps, hurts, ties, len(delta_analysis),
    )

    summary = {
        "rag":    {"em": rag_result["metrics"]["em"],    "f1": rag_result["metrics"]["f1"]},
        "no_rag": {"em": no_rag_result["metrics"]["em"], "f1": no_rag_result["metrics"]["f1"]},
        "delta_counts": {"helps": helps, "hurts": hurts, "ties": ties},
        "delta_analysis": delta_analysis,
    }
    save_results("exp5_summary", summary)
    return {"rag": rag_result, "no_rag": no_rag_result, "summary": summary}


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run RAG experiments")
    parser.add_argument(
        "--exp",
        default="all",
        choices=["1", "2", "3", "4", "5", "all"],
        help="Which experiment to run (default: all)",
    )
    args = parser.parse_args()

    questions, corpus_docs, generator = setup()

    if args.exp in ("1", "all"):
        experiment_1(questions, corpus_docs, generator)

    if args.exp in ("2", "all"):
        experiment_2(questions, corpus_docs, generator)

    if args.exp in ("3", "all"):
        experiment_3(questions, corpus_docs, generator)

    if args.exp in ("4", "all"):
        experiment_4(questions, corpus_docs, generator)

    if args.exp in ("5", "all"):
        experiment_5(questions, corpus_docs, generator)

    logger.info("=" * 60)
    logger.info("All requested experiments complete.")
    logger.info("Results saved to: %s", config.RESULTS_DIR)
    logger.info("Run `python analysis/plot_results.py` to generate figures.")


if __name__ == "__main__":
    main()
