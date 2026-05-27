"""
vector_store.py — sentence-transformers → FAISS IndexFlatIP (cosine similarity
on L2-normalised vectors).  Persistence via faiss.write_index + pickle.
"""

import json, pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EmbeddingConfig, RetrieverConfig
from app.document_processor import DocumentChunk


class EmbeddingEngine:
    def __init__(self, config: EmbeddingConfig | None = None):
        self.cfg = config or EmbeddingConfig()
        print(f"   Loading embedding model: {self.cfg.model_name}")
        self.model = SentenceTransformer(self.cfg.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=len(texts) > 200,
            normalize_embeddings=self.cfg.normalize,
            convert_to_numpy=True,
        ).astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])


class VectorStore:
    def __init__(self, engine: EmbeddingEngine, cfg: RetrieverConfig | None = None):
        self.engine = engine
        self.cfg = cfg or RetrieverConfig()
        self.index = faiss.IndexFlatIP(engine.cfg.dimension)
        self.chunks: list[DocumentChunk] = []

    # ── index ─────────────────────────────────────────────────────────────
    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        embs = self.engine.encode([c.text for c in chunks])
        self.index.add(embs)
        self.chunks.extend(chunks)
        print(f"   Indexed {len(chunks)} chunks (total: {self.index.ntotal})")
        return len(chunks)

    # ── search ────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        k = min(top_k or self.cfg.top_k, self.index.ntotal)
        q = self.engine.encode_one(query)
        scores, idxs = self.index.search(q, k)
        results = []
        for rank, (sc, ix) in enumerate(zip(scores[0], idxs[0])):
            if ix < 0 or sc < self.cfg.similarity_threshold:
                continue
            results.append({"chunk": self.chunks[ix], "score": float(sc), "rank": rank + 1})
        return results

    # ── persist ───────────────────────────────────────────────────────────
    def save(self, directory: str):
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        with open(d / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        (d / "manifest.json").write_text(json.dumps({
            "total_chunks": len(self.chunks),
            "sources": sorted(set(c.source_file for c in self.chunks)),
        }, indent=2))
        print(f"   Saved → {directory}/")

    def load(self, directory: str):
        d = Path(directory)
        self.index = faiss.read_index(str(d / "index.faiss"))
        with open(d / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        print(f"   Loaded {self.index.ntotal} vectors from {directory}/")

    @property
    def ready(self) -> bool:
        return self.index.ntotal > 0
