# Shared lexical tokenizer used by both BM25 and TF-IDF so the two sparse
# retrievers see *identical* token streams. The only remaining difference
# between BM25 and TF-IDF in this notebook is the scoring function itself.
_LEXICAL_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def _lexical_tokenize(text: str) -> List[str]:
    """Lowercase, strip accents, then extract `\b\w\w+\b` tokens."""
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


class RerankRetriever(BaseRetriever):
    """Two-stage retrieval: a base retriever returns top-N candidates which
    a cross-encoder re-ranks down to top-k.

    The cross-encoder scores each (query, chunk) pair jointly, so it can
    detect lexical/semantic interactions that a bi-encoder (which embeds
    query and chunk independently) cannot. Standard "advanced RAG" setup
    in the Gao et al. survey [44].
    """
    name = "rerank"

    def __init__(
        self,
        base: BaseRetriever,
        model_name: str = config.RERANK_MODEL,
        top_n:      int  = config.RERANK_TOP_N,
        batch_size: int  = config.RERANK_BATCH_SIZE,
    ):
        self.base        = base
        self._model_name = model_name
        self._top_n      = top_n
        self._batch_size = batch_size
        self._reranker   = None
        self._chunks: List[Dict] = []

    def _load_reranker(self) -> None:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder reranker: %s", self._model_name)
            self._reranker = CrossEncoder(self._model_name)

    def build(self, chunks: List[Dict]) -> None:
        # The base retriever owns the index. Build it here only if the
        # caller didn't already do so (idempotent on most retrievers but
        # in our setup we expect base to be pre-built).
        self._chunks = chunks

    def retrieve(self, query: str, k: int) -> List[Dict]:
        candidates = self.base.retrieve(query, self._top_n)
        if not candidates:
            return []
        self._load_reranker()
        pairs  = [(query, c["text"]) for c in candidates]
        scores = self._reranker.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False,
        )
        order  = np.argsort(scores)[::-1][:k]
        return [{**candidates[i], "score": float(scores[i])} for i in order]

    def batch_retrieve(self, queries: List[str], k: int) -> List[List[Dict]]:
        # Stage 1 — candidate retrieval for every query.
        all_candidates = self.base.batch_retrieve(queries, self._top_n)
        self._load_reranker()
        # Stage 2 — flatten into one big cross-encoder pass.
        flat_pairs: List[Tuple[str, str]] = []
        boundaries: List[int] = [0]
        for q, cands in zip(queries, all_candidates):
            flat_pairs.extend((q, c["text"]) for c in cands)
            boundaries.append(len(flat_pairs))

        logger.info("Cross-encoder scoring %d (query, chunk) pairs…",
                    len(flat_pairs))
        if not flat_pairs:
            return [[] for _ in queries]
        scores = self._reranker.predict(
            flat_pairs, batch_size=self._batch_size, show_progress_bar=True,
        )
        results: List[List[Dict]] = []
        for i, cands in enumerate(all_candidates):
            chunk_scores = np.asarray(scores[boundaries[i]: boundaries[i + 1]])
            if len(chunk_scores) == 0:
                results.append([])
                continue
            order = np.argsort(chunk_scores)[::-1][:k]
            results.append([
                {**cands[j], "score": float(chunk_scores[j])} for j in order
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
