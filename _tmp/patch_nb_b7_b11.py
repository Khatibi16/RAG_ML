"""Patch project.ipynb for B7 (BM25/TF-IDF tokenizer parity) and B11 (dead code).

B7: introduce a shared _lexical_tokenize used by both BM25 and TF-IDF.
    TF-IDF drops ngram (1,2) -> (1,1) and the min_df/max_df DF filters
    so that both retrievers see exactly the same token stream — the only
    remaining difference between them is the scoring function itself.

B11: delete _sentence_chunks and chunk_corpus_sizes (dead code) from
    cell 7, and tighten cell 6 markdown that referred to sentence chunking.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "project.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)


def set_source(cell_idx: int, text: str) -> None:
    """Replace a cell's source with `text` (preserving trailing newlines per line)."""
    lines = text.splitlines(keepends=True)
    nb["cells"][cell_idx]["source"] = lines


# ── B11: cell 6 markdown — chunking description ──────────────────────
set_source(6, """## 3. Chunking

Chunking is one of the most impactful decisions in a RAG pipeline (Gao et al. 2024 [44]).

**Fixed-size word chunking with overlap** — split a document's text into
windows of W words, stepping by (W − overlap) words each time. Overlap
prevents an answer that straddles a chunk boundary from being lost.

Every chunk has schema `{chunk_id, doc_id, title, text, chunk_idx}`.
""")

# ── B11: cell 7 code — chunker, drop dead code ───────────────────────
set_source(7, '''def chunk_corpus(
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
''')


# ── B7: cell 8 markdown — describe shared tokenizer ──────────────────
set_source(8, """## 4. Retrieval backends

Three backends with a uniform interface (`build(chunks)`, `retrieve(query, k)`):

* **BM25** — `rank-bm25` Okapi BM25. Indexes a token stream produced by
  `_lexical_tokenize` (lowercase, accent-strip, regex `\\b\\w\\w+\\b`).
* **TF-IDF** — `sklearn.feature_extraction.text.TfidfVectorizer` configured
  to consume the *same* `_lexical_tokenize` token stream as BM25, unigrams
  only, sublinear TF, no DF filtering. So BM25 vs TF-IDF differs **only**
  in the scoring function — not in tokenisation, vocabulary, or n-gram order.
* **Dense** — `sentence-transformers/all-MiniLM-L6-v2` (22 M parameters,
  384-dim L2-normalised embeddings; cosine = dot product). Embeddings are
  cached to disk and re-used across experiments.
""")

# ── B7: cell 9 code — shared tokenizer + retrievers ──────────────────
set_source(9, '''# Shared lexical tokenizer used by both BM25 and TF-IDF so the two sparse
# retrievers see *identical* token streams. The only remaining difference
# between BM25 and TF-IDF in this notebook is the scoring function itself.
_LEXICAL_TOKEN_RE = re.compile(r"(?u)\\b\\w\\w+\\b")


def _lexical_tokenize(text: str) -> List[str]:
    """Lowercase, strip accents, then extract `\\b\\w\\w+\\b` tokens."""
    import unicodedata
    norm = unicodedata.normalize("NFKD", text)
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    return _LEXICAL_TOKEN_RE.findall(norm.lower())


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
        return [self.retrieve(q, k) for q in tqdm(queries, desc=f"{self.name} retrieval")]


class BM25Retriever(BaseRetriever):
    """Sparse retrieval using Okapi BM25 over the shared lexical tokenizer."""
    name = "bm25"

    def __init__(self):
        self._index = None
        self._chunks: List[Dict] = []

    def build(self, chunks: List[Dict]) -> None:
        from rank_bm25 import BM25Okapi
        self._chunks = chunks
        tokenized = [_lexical_tokenize(c["text"]) for c in chunks]
        self._index = BM25Okapi(tokenized)
        logger.info("BM25 index built over %d chunks.", len(chunks))

    def retrieve(self, query: str, k: int) -> List[Dict]:
        if self._index is None:
            raise RuntimeError("Call build() before retrieve().")
        tokens = _lexical_tokenize(query)
        scores = self._index.get_scores(tokens)
        top_k = int(min(k, len(self._chunks)))
        idx = np.argsort(scores)[::-1][:top_k]
        return [{**self._chunks[i], "score": float(scores[i])} for i in idx]


class TFIDFRetriever(BaseRetriever):
    """Sparse retrieval using sklearn TF-IDF over the shared lexical tokenizer."""
    name = "tfidf"

    def __init__(self):
        self._vectorizer = None
        self._matrix = None
        self._chunks: List[Dict] = []

    def build(self, chunks: List[Dict]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._chunks = chunks
        # Match BM25 exactly: same tokenizer, unigrams only, no DF filtering.
        # We pass `tokenizer=_lexical_tokenize` and disable the analyzer's
        # own lowercasing / token regex.
        self._vectorizer = TfidfVectorizer(
            tokenizer=_lexical_tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 1),
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform([c["text"] for c in chunks])
        logger.info("TF-IDF index built: %d chunks, vocabulary size %d.",
                    len(chunks), len(self._vectorizer.vocabulary_))

    def retrieve(self, query: str, k: int) -> List[Dict]:
        if self._vectorizer is None:
            raise RuntimeError("Call build() before retrieve().")
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_k = int(min(k, len(self._chunks)))
        idx = np.argsort(scores)[::-1][:top_k]
        return [{**self._chunks[i], "score": float(scores[i])} for i in idx]


class DenseRetriever(BaseRetriever):
    """Dense retrieval using sentence-transformers (bi-encoder)."""
    name = "dense"

    def __init__(
        self,
        model_name: str = config.DENSE_MODEL,
        cache_path: Optional[Path] = None,
        batch_size: int = config.DENSE_BATCH_SIZE,
    ):
        self._model_name = model_name
        self._cache_path = cache_path
        self._batch_size = batch_size
        self._model = None
        self._embeddings: Optional[np.ndarray] = None
        self._chunks: List[Dict] = []

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading dense model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)

    def build(self, chunks: List[Dict]) -> None:
        self._chunks = chunks

        if self._cache_path and self._cache_path.exists():
            logger.info("Loading cached dense embeddings from %s", self._cache_path)
            with open(self._cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("n_chunks") == len(chunks):
                self._embeddings = cached["embeddings"]
                logger.info("Cache hit — %d embeddings loaded.", len(chunks))
                return
            logger.warning("Cache mismatch — re-encoding.")

        self._load_model()
        texts = [c["text"] for c in chunks]
        logger.info("Encoding %d chunks with %s…", len(texts), self._model_name)
        self._embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
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
        q_emb = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
        )
        scores = (self._embeddings @ q_emb.T).flatten()
        top_k = int(min(k, len(self._chunks)))
        idx = np.argsort(scores)[::-1][:top_k]
        return [{**self._chunks[i], "score": float(scores[i])} for i in idx]

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
        sim_matrix = q_embs @ self._embeddings.T
        results: List[List[Dict]] = []
        top_k = int(min(k, len(self._chunks)))
        for scores in sim_matrix:
            idx = np.argsort(scores)[::-1][:top_k]
            results.append([
                {**self._chunks[i], "score": float(scores[i])} for i in idx
            ])
        return results


def build_retriever(
    retriever_type: str,
    chunks: List[Dict],
    cache_dir: Optional[Path] = None,
    chunk_size: Optional[int] = None,
) -> BaseRetriever:
    """Instantiate and build a retriever from a type string."""
    if retriever_type == "bm25":
        r = BM25Retriever()
        r.build(chunks)
    elif retriever_type == "tfidf":
        r = TFIDFRetriever()
        r.build(chunks)
    elif retriever_type == "dense":
        suffix = f"_{chunk_size}w" if chunk_size else ""
        cache_path = (cache_dir or config.CACHE_DIR) / f"dense_embeddings{suffix}.pkl"
        r = DenseRetriever(cache_path=cache_path)
        r.build(chunks)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type!r}")
    return r
''')

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patched cells 6, 7, 8, 9.")
