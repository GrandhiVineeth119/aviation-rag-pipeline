"""
hybrid_retriever.py — Level 2: BM25 + Vector + Cross-Encoder Reranker
=====================================================================

Extends the Level 1 vector-only retrieval with:

  1. BM25 keyword retrieval  (catches exact terminology matches)
  2. Vector semantic search  (catches paraphrased / conceptual matches)
  3. Reciprocal Rank Fusion  (merges both candidate lists)
  4. Cross-Encoder Reranker  (re-scores merged candidates for precision)

Flow:
    query ──┬──▶ BM25 top-20 ──┐
            │                   ├──▶ RRF merge ──▶ Cross-Encoder ──▶ top-K
            └──▶ FAISS top-20 ─┘

Why this helps:
  • BM25 excels at exact keyword matching — critical for aviation terms
    like "RVSM", "DALR", "PNR", "SSR" that have specific meanings.
  • Vector search captures semantic similarity — "What is the temperature
    at which air becomes saturated?" matches chunks about "dew point".
  • The cross-encoder reranker provides a final precision pass by scoring
    each (query, chunk) pair with a full transformer attention layer.
"""

import re
import math
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np

from app.document_processor import DocumentChunk


# ══════════════════════════════════════════════════════════════════════════
# BM25 Index
# ══════════════════════════════════════════════════════════════════════════

class BM25Index:
    """
    Okapi BM25 implementation — no external dependencies.
    Stores tokenised chunks and computes BM25 scores at query time.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []      # tokenised docs
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.n_docs: int = 0

    def fit(self, documents: list[str]) -> None:
        """Build the BM25 index from raw text documents."""
        self.corpus = [self._tokenize(d) for d in documents]
        self.n_docs = len(self.corpus)
        self.doc_len = [len(d) for d in self.corpus]
        self.avgdl = sum(self.doc_len) / max(self.n_docs, 1)

        # Document frequencies
        self.doc_freq = defaultdict(int)
        for doc in self.corpus:
            seen = set(doc)
            for term in seen:
                self.doc_freq[term] += 1

        # IDF (with smoothing)
        self.idf = {}
        for term, df in self.doc_freq.items():
            self.idf[term] = math.log(
                (self.n_docs - df + 0.5) / (df + 0.5) + 1.0
            )

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Return [(doc_index, bm25_score)] sorted descending."""
        query_terms = self._tokenize(query)
        scores = np.zeros(self.n_docs)

        for term in query_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, doc in enumerate(self.corpus):
                tf = doc.count(term)
                if tf == 0:
                    continue
                dl = self.doc_len[i]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (num / den)

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenisation."""
        return re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "corpus": self.corpus,
                "doc_len": self.doc_len,
                "avgdl": self.avgdl,
                "idf": dict(self.idf),
                "doc_freq": dict(self.doc_freq),
                "n_docs": self.n_docs,
                "k1": self.k1,
                "b": self.b,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.corpus = d["corpus"]
        self.doc_len = d["doc_len"]
        self.avgdl = d["avgdl"]
        self.idf = d["idf"]
        self.doc_freq = defaultdict(int, d["doc_freq"])
        self.n_docs = d["n_docs"]
        self.k1 = d["k1"]
        self.b = d["b"]


# ══════════════════════════════════════════════════════════════════════════
# Cross-Encoder Reranker
# ══════════════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """
    Wraps a cross-encoder model that scores (query, passage) pairs.
    Much more accurate than bi-encoder cosine similarity because it
    attends to both query and passage jointly.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = None
        self.model_name = model_name

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"  ⚙  Loading reranker: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[dict], top_k: int = 5
    ) -> list[dict]:
        """
        Re-score candidates with the cross-encoder and return top_k.
        Each candidate must have candidate["chunk"].text.
        """
        if not candidates:
            return []

        pairs = [(query, c["chunk"].text) for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]


# ══════════════════════════════════════════════════════════════════════════
# Reciprocal Rank Fusion (RRF)
# ══════════════════════════════════════════════════════════════════════════

def reciprocal_rank_fusion(
    *result_lists: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge multiple ranked result lists using RRF.

    RRF_score(d) = Σ  1 / (k + rank_i(d))

    This is robust to different score scales between BM25 and cosine sim.
    """
    scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            cid = item["chunk"].chunk_id
            scores[cid] += 1.0 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = item

    # Sort by fused score
    sorted_ids = sorted(scores, key=scores.get, reverse=True)

    merged = []
    for cid in sorted_ids:
        entry = chunk_map[cid].copy()
        entry["rrf_score"] = scores[cid]
        merged.append(entry)

    return merged


# ══════════════════════════════════════════════════════════════════════════
# Hybrid Retriever (puts it all together)
# ══════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Full hybrid retrieval pipeline:
        BM25 candidates + Vector candidates → RRF merge → Cross-Encoder rerank
    """

    def __init__(
        self,
        vector_store,           # VectorStore instance
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
    ):
        self.vector_store = vector_store
        self.bm25 = BM25Index()
        self.reranker = CrossEncoderReranker(reranker_model)
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self._bm25_ready = False

    def build_bm25_index(self) -> None:
        """Build BM25 index from the chunks already in the vector store."""
        if not self.vector_store.chunks:
            return
        texts = [c.text for c in self.vector_store.chunks]
        self.bm25.fit(texts)
        self._bm25_ready = True
        print(f"  ✓ BM25 index built ({len(texts)} documents)")

    def save_bm25(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.bm25.save(str(Path(directory) / "bm25.pkl"))

    def load_bm25(self, directory: str) -> None:
        p = Path(directory) / "bm25.pkl"
        if p.exists():
            self.bm25.load(str(p))
            self._bm25_ready = True
            print(f"  BM25 index loaded")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Hybrid search: BM25 + Vector → RRF → Cross-Encoder rerank.
        Falls back to vector-only if BM25 index not built.
        """
        # 1 ── Vector search (top-20 candidates)
        vector_results = self.vector_store.search(query, top_k=20)

        # 2 ── BM25 search (top-20 candidates)
        bm25_results = []
        if self._bm25_ready:
            bm25_hits = self.bm25.search(query, top_k=20)
            chunks = self.vector_store.chunks
            for idx, score in bm25_hits:
                if idx < len(chunks):
                    bm25_results.append({
                        "chunk": chunks[idx],
                        "score": score,
                        "rank": len(bm25_results) + 1,
                    })

        if not bm25_results:
            # Fallback: vector only
            return vector_results[:top_k]

        # 3 ── Reciprocal Rank Fusion
        merged = reciprocal_rank_fusion(vector_results, bm25_results)

        # 4 ── Cross-Encoder Rerank (top candidates)
        reranked = self.reranker.rerank(query, merged[:15], top_k=top_k)

        # Preserve score field for downstream compatibility
        for i, r in enumerate(reranked):
            r["score"] = r.get("rerank_score", r.get("score", 0))
            r["rank"] = i + 1

        return reranked

    @property
    def ready(self) -> bool:
        return self._bm25_ready and self.vector_store.ready
