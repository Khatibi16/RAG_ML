def chunk_corpus(
    corpus_docs: List[Dict],
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[Dict]:
    """Chunk every document in corpus_docs into fixed-size word windows."""
    chunks: List[Dict] = []
    for doc in corpus_docs:
        chunks.extend(_word_chunks(doc, chunk_size, overlap))
    return chunks


def _word_chunks(doc: Dict, chunk_size: int, overlap: int) -> List[Dict]:
    """Split doc['text'] into overlapping word windows."""
    words = doc["text"].split()
    if not words:
        return []

    effective_overlap = min(overlap, chunk_size - 1)
    step = chunk_size - effective_overlap

    chunks: List[Dict] = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append({
            "chunk_id":  f"{doc['doc_id']}_c{idx:04d}",
            "doc_id":    doc["doc_id"],
            "title":     doc["title"],
            "text":      " ".join(words[start:end]),
            "chunk_idx": idx,
        })
        start += step
        idx   += 1
        if end == len(words):
            break
    return chunks


def chunk_stats(chunks: List[Dict]) -> Dict:
    """Return descriptive statistics about chunk lengths (in words)."""
    lengths = [len(c["text"].split()) for c in chunks]
    arr = np.array(lengths, dtype=float)
    return {
        "n_chunks":     len(chunks),
        "mean_words":   float(arr.mean()),
        "std_words":    float(arr.std()),
        "min_words":    int(arr.min()),
        "max_words":    int(arr.max()),
        "median_words": float(np.median(arr)),
    }
