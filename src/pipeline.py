"""
pipeline.py — Full RAG pipeline: retrieve → prompt → generate → evaluate.

The RAGPipeline class ties together:
    1. A retriever (BM25 / TF-IDF / Dense)
    2. A generator (Flan-T5-base)
    3. An evaluator (EM + F1 + Recall@k)

It exposes a single `run()` method that processes a list of questions and
returns a structured result dict ready for analysis and plotting.

Usage:
    pipe = RAGPipeline(retriever, generator)
    results = pipe.run(questions, k=5, prompt_template="instructed")
"""

import logging
import time
from typing import Dict, List, Optional

from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.retriever import BaseRetriever
from src.generator import Generator, build_rag_prompt, build_no_rag_prompt
from src.evaluator  import evaluate_with_ci, exact_match, token_f1, retrieval_recall_single

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Parameters
    ----------
    retriever : BaseRetriever
        Already built (indexed) retriever.
    generator : Generator
        Lazy-loaded Flan-T5-base generator.
    use_retrieval : bool
        If False, skip retrieval and use a no-context prompt.  Used to
        establish the RAG-less baseline (Experiment 5).
    """

    def __init__(
        self,
        retriever:     Optional[BaseRetriever],
        generator:     Generator,
        use_retrieval: bool = True,
    ):
        self.retriever     = retriever
        self.generator     = generator
        self.use_retrieval = use_retrieval

    def run(
        self,
        questions:       List[Dict],
        k:               int  = config.DEFAULT_K,
        prompt_template: str  = config.DEFAULT_PROMPT,
    ) -> Dict:
        """
        Run the pipeline on a list of question dicts.

        Parameters
        ----------
        questions       : list of {qid, question, answers}
        k               : number of passages to retrieve per question
        prompt_template : key into config.PROMPT_TEMPLATES

        Returns
        -------
        dict with keys:
            predictions, gold_answers, retrieved_chunks,
            metrics, per_example, timing
        """
        t_start = time.perf_counter()

        # ── Step 1: Retrieval ──────────────────────────────────
        retrieved_chunks_list: List[List[Dict]] = []

        if self.use_retrieval and self.retriever is not None:
            q_texts = [q["question"] for q in questions]

            # Use batch retrieval if the backend supports it efficiently
            if hasattr(self.retriever, "batch_retrieve"):
                retrieved_chunks_list = self.retriever.batch_retrieve(q_texts, k)
            else:
                retrieved_chunks_list = [
                    self.retriever.retrieve(q, k)
                    for q in tqdm(q_texts, desc="Retrieving")
                ]
        else:
            retrieved_chunks_list = [[] for _ in questions]

        t_retrieval = time.perf_counter() - t_start

        # ── Step 2: Prompt construction ────────────────────────
        prompts: List[str] = []
        for q, retrieved in zip(questions, retrieved_chunks_list):
            if self.use_retrieval and retrieved:
                prompt = build_rag_prompt(q["question"], retrieved, prompt_template)
            else:
                prompt = build_no_rag_prompt(q["question"])
            prompts.append(prompt)

        # ── Step 3: Generation ─────────────────────────────────
        t_gen_start = time.perf_counter()
        predictions = self.generator.generate(prompts)
        t_generation = time.perf_counter() - t_gen_start

        # ── Step 4: Evaluation ─────────────────────────────────
        gold_answers_list = [q["answers"] for q in questions]
        metrics = evaluate_with_ci(
            predictions,
            gold_answers_list,
            retrieved_chunks_list if self.use_retrieval else None,
        )

        # ── Per-example breakdown (for qualitative analysis) ───
        per_example = []
        for q, pred, golds, retrieved, prompt in zip(
            questions, predictions, gold_answers_list,
            retrieved_chunks_list, prompts
        ):
            per_example.append({
                "qid":         q["qid"],
                "question":    q["question"],
                "gold":        golds,
                "prediction":  pred,
                "em":          exact_match(pred, golds),
                "f1":          token_f1(pred, golds),
                "recall":      retrieval_recall_single(retrieved, golds),
                "n_retrieved": len(retrieved),
                "prompt":      prompt,
                "top_chunk":   retrieved[0]["text"][:200] if retrieved else "",
            })

        return {
            "predictions":          predictions,
            "gold_answers":         gold_answers_list,
            "retrieved_chunks":     retrieved_chunks_list,
            "prompts":              prompts,
            "metrics":              metrics,
            "per_example":          per_example,
            "timing": {
                "retrieval_s":       round(t_retrieval, 2),
                "generation_s":      round(t_generation, 2),
                "total_s":           round(time.perf_counter() - t_start, 2),
                "n_questions":       len(questions),
                # Cache stats so that a near-zero generation_s can be
                # recognised as "everything was cached" rather than "the
                # generator is suspiciously fast".
                "cache_hits":        getattr(self.generator, "last_cache_hits",   None),
                "cache_misses":      getattr(self.generator, "last_cache_misses", None),
            },
        }
