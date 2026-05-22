"""
retriever.py — Retrieval backends: BM25, TF-IDF, and Dense (sentence-transformers).

All three classes share the same interface:
    retriever.build(chunks)       # index the corpus
    retriever.retrieve(query, k)  # return top-k chunks

We benchmark these three approaches to understand when lexical matching
(BM25, TF-IDF) is sufficient versus when semantic matching (dense) helps.

Design notes:
  - BM25  : rank-bm25 library (Robertson et al. BM25 variant).
            Tokenises on whitespace; no stemming (keeps the comparison
            with TF-IDF fair).
  - TF-IDF: sklearn TfidfVectorizer + cosine similarity.  Captures sub-word
            character n-gram statistics which BM25 does not.
  - Dense : sentence-transformers `all-MiniLM-L6-v2` (22 M params).
            Encodes queries and passages into 384-dim dense vectors; retrieval
            by cosine similarity.  Embeddings are cached to disk.
"""

import logging
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────

class BaseRetriever(ABC):
    """Common interface for all retrieval backends."""

    name: str = "base"

    @abstractmethod
    def build(self, chunks: List[Dict]) -> None:
        """Index the provided chunks."""

    @abstractmethod
    def retrieve(self, query: str, k: int) -> List[Dict]:
        """Return up to k most-relevant chunks for *query*."""

    def batch_retrieve(self, queries: List[str], k: int) -> List[List[Dict]]:
        """Retrieve for a list of queries. Subclasses may override for efficiency."""
        return [self.retrieve(q, k) for q in tqdm(queries, desc=f"{self.name} retrieval")]


# ─────────────────────────────────────────────────────────────────
# BM25
# ─────────────────────────────────────────────────────────────────

class BM25Retriever(BaseRetriever):
    """Sparse retrieval using Okapi BM25 (rank-bm25 library)."""

    name = "bm25"

    def __init__(self):
        self._index   = None
        self._chunks  : List[Dict] = []

    def build(self, chunks: List[Dict]) -> None:
        from rank_bm25 import BM25Okapi
        self._chunks = chunks
        tokenized    = [c["text"].lower().split() for c in chunks]
        self._index  = BM25Okapi(tokenized)
        logger.info("BM25 index built over %d chunks.", len(chunks))

    def retrieve(self, query: str, k: int) -> List[Dict]:
        if self._index is None:
            raise RuntimeError("Call build() before retrieve().")
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        top_k  = int(min(k, len(self._chunks)))
        idx    = np.argsort(scores)[::-1][:top_k]
        return [
            {**self._chunks[i], "score": float(scores[i])}
            for i in idx
        ]


# ─────────────────────────────────────────────────────────────────
# TF-IDF
# ─────────────────────────────────────────────────────────────────

class TFIDFRetriever(BaseRetriever):
    """Sparse retrieval using sklearn TF-IDF + cosine similarity."""

    name = "tfidf"

    def __init__(self):
        self._vectorizer = None
        self._matrix     = None
        self._chunks     : List[Dict] = []

    def build(self, chunks: List[Dict]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._chunks     = chunks
        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,       # log(1 + tf) dampening
            min_df=2,                # ignore terms appearing in < 2 chunks
            max_df=0.90,             # ignore terms in > 90 % of chunks (stop-words)
            ngram_range=(1, 2),      # unigrams + bigrams
            strip_accents="unicode",
        )
        texts        = [c["text"] for c in chunks]
        self._matrix = self._vectorizer.fit_transform(texts)
        logger.info(
            "TF-IDF index built: %d chunks, vocabulary size %d.",
            len(chunks), len(self._vectorizer.vocabulary_),
        )

    def retrieve(self, query: str, k: int) -> List[Dict]:
        if self._vectorizer is None:
            raise RuntimeError("Call build() before retrieve().")
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec  = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_k  = int(min(k, len(self._chunks)))
        idx    = np.argsort(scores)[::-1][:top_k]
        return [
            {**self._chunks[i], "score": float(scores[i])}
            for i in idx
        ]


# ─────────────────────────────────────────────────────────────────
# Dense (sentence-transformers)
# ─────────────────────────────────────────────────────────────────

class DenseRetriever(BaseRetriever):
    """Dense retrieval using sentence-transformers (bi-encoder)."""

    name = "dense"

    def __init__(
        self,
        model_name: str       = config.DENSE_MODEL,
        cache_path: Optional[Path] = None,
        batch_size: int       = config.DENSE_BATCH_SIZE,
    ):
        self._model_name = model_name
        self._cache_path = cache_path
        self._batch_size = batch_size
        self._model      = None
        self._embeddings : Optional[np.ndarray] = None
        self._chunks     : List[Dict] = []

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading dense model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)

    def build(self, chunks: List[Dict]) -> None:
        self._chunks = chunks

        # Try to load cached embeddings first
        if self._cache_path and self._cache_path.exists():
            logger.info("Loading cached dense embeddings from %s", self._cache_path)
            with open(self._cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("n_chunks") == len(chunks):
                self._embeddings = cached["embeddings"]
                logger.info("Cache hit — %d embeddings loaded.", len(chunks))
                return
            logger.warning("Cache mismatch (n_chunks changed) — re-encoding.")

        # Encode
        self._load_model()
        texts = [c["text"] for c in chunks]
        logger.info("Encoding %d chunks with %s…", len(texts), self._model_name)
        self._embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine = dot product after L2 normalisation
            convert_to_numpy=True,
        )

        # Persist cache
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "wb") as f:
                pickle.dump({"n_chunks": len(chunks), "embeddings": self._embeddings}, f)
            logger.info("Dense embeddings cached to %s.", self._cache_path)

        logger.info("Dense index built over %d chunks.", len(chunks))

    def retrieve(self, query: str, k: int) -> List[Dict]:
        if self._embeddings is None:
            raise RuntimeError("Call build() before retrieve().")
        self._load_model()
        q_emb  = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        scores = (self._embeddings @ q_emb.T).flatten()
        top_k  = int(min(k, len(self._chunks)))
        idx    = np.argsort(scores)[::-1][:top_k]
        return [
            {**self._chunks[i], "score": float(scores[i])}
            for i in idx
        ]

    def batch_retrieve(self, queries: List[str], k: int) -> List[List[Dict]]:
        """Vectorised batch retrieval — much faster than one-by-one."""
        if self._embeddings is None:
            raise RuntimeError("Call build() before retrieve().")
        self._load_model()
        logger.info("Dense batch encoding %d queries…", len(queries))
        q_embs = self._model.encode(
            queries,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # (n_queries, n_chunks)
        sim_matrix = q_embs @ self._embeddings.T
        results: List[List[Dict]] = []
        top_k = int(min(k, len(self._chunks)))
        for scores in sim_matrix:
            idx = np.argsort(scores)[::-1][:top_k]
            results.append([
                {**self._chunks[i], "score": float(scores[i])}
                for i in idx
            ])
        return results


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────

def build_retriever(
    retriever_type: str,
    chunks: List[Dict],
    cache_dir: Optional[Path] = None,
    chunk_size: Optional[int] = None,
) -> BaseRetriever:
    """
    Instantiate and build a retriever from a type string.

    Parameters
    ----------
    retriever_type : "bm25" | "tfidf" | "dense"
    chunks         : pre-chunked corpus
    cache_dir      : where to store dense embeddings cache
    chunk_size     : used to name the dense embedding cache file
    """
    if retriever_type == "bm25":
        r = BM25Retriever()
        r.build(chunks)
    elif retriever_type == "tfidf":
        r = TFIDFRetriever()
        r.build(chunks)
    elif retriever_type == "dense":
        suffix = f"_{chunk_size}w" if chunk_size else ""
        cache_path = (
            (cache_dir or config.CACHE_DIR) / f"dense_embeddings{suffix}.pkl"
        )
        r = DenseRetriever(cache_path=cache_path)
        r.build(chunks)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type!r}")

    return r
