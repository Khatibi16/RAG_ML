"""
chunker.py — Document chunking strategies.

Chunking is one of the most impactful decisions in a RAG pipeline (Gao et al.
2024 [44]). We implement:

  1. **Fixed-size word chunking with overlap** — split a document's text into
     windows of W words, stepping by (W - overlap) words each time.
     - Overlap prevents an answer that straddles a chunk boundary from being
       completely lost.
     - We study chunk sizes ∈ {64, 128, 256, 512} words (config.CHUNK_SIZES).

  2. **Sentence chunking** — each sentence becomes its own chunk. This keeps
     semantic units intact but produces very short chunks that may lack context.

Returned chunks have the schema:
    {
        "chunk_id":  str,   # globally unique  "<doc_id>_c<n>"
        "doc_id":    str,   # source document
        "title":     str,   # source document title
        "text":      str,   # the chunk text
        "chunk_idx": int,   # position of this chunk within the document
    }
"""

import re
from typing import Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def chunk_corpus(
    corpus_docs: List[Dict],
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    strategy: str = "word",
) -> List[Dict]:
    """
    Chunk every document in corpus_docs.

    Parameters
    ----------
    corpus_docs : list of dicts (doc_id, title, text)
    chunk_size  : words per chunk (word strategy) or ignored (sentence strategy)
    overlap     : word overlap between consecutive chunks (word strategy only)
    strategy    : "word" | "sentence"

    Returns
    -------
    list of chunk dicts (chunk_id, doc_id, title, text, chunk_idx)
    """
    chunks: List[Dict] = []
    for doc in corpus_docs:
        if strategy == "word":
            doc_chunks = _word_chunks(doc, chunk_size, overlap)
        elif strategy == "sentence":
            doc_chunks = _sentence_chunks(doc)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy!r}")
        chunks.extend(doc_chunks)
    return chunks


def chunk_corpus_sizes(
    corpus_docs: List[Dict],
    sizes: List[int] = None,
    overlap: int = config.CHUNK_OVERLAP,
) -> Dict[int, List[Dict]]:
    """
    Return a mapping {chunk_size: [chunks]} for all sizes in `sizes`.

    This lets the experiment runner pre-compute all chunked corpora in one pass.
    """
    if sizes is None:
        sizes = config.CHUNK_SIZES
    return {sz: chunk_corpus(corpus_docs, sz, overlap, "word") for sz in sizes}


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _word_chunks(
    doc: Dict,
    chunk_size: int,
    overlap: int,
) -> List[Dict]:
    """Split doc['text'] into overlapping word windows."""
    text  = doc["text"]
    words = text.split()

    if not words:
        return []

    # Clamp overlap so it can never exceed chunk_size - 1
    effective_overlap = min(overlap, chunk_size - 1)
    step = chunk_size - effective_overlap

    chunks: List[Dict] = []
    start = 0
    idx   = 0
    while start < len(words):
        end        = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "chunk_id":  f"{doc['doc_id']}_c{idx:04d}",
            "doc_id":    doc["doc_id"],
            "title":     doc["title"],
            "text":      chunk_text,
            "chunk_idx": idx,
        })
        start += step
        idx   += 1
        if end == len(words):
            break

    return chunks


def _sentence_chunks(doc: Dict) -> List[Dict]:
    """One sentence → one chunk."""
    text      = doc["text"]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks: List[Dict] = []
    for idx, sent in enumerate(sentences):
        sent = sent.strip()
        if len(sent) < 10:          # skip very short fragments
            continue
        chunks.append({
            "chunk_id":  f"{doc['doc_id']}_c{idx:04d}",
            "doc_id":    doc["doc_id"],
            "title":     doc["title"],
            "text":      sent,
            "chunk_idx": idx,
        })

    return chunks


# ─────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────

def chunk_stats(chunks: List[Dict]) -> Dict:
    """Return descriptive statistics about chunk lengths (in words)."""
    import numpy as np
    lengths = [len(c["text"].split()) for c in chunks]
    arr     = np.array(lengths, dtype=float)
    return {
        "n_chunks": len(chunks),
        "mean_words": float(arr.mean()),
        "std_words":  float(arr.std()),
        "min_words":  int(arr.min()),
        "max_words":  int(arr.max()),
        "median_words": float(np.median(arr)),
    }
