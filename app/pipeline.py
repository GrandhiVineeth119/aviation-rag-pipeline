"""
pipeline.py — Main orchestrator.

    Level 1:  PDFs → chunk → embed → FAISS → retrieve → generate → guard
    Level 2:  + BM25 index → hybrid retrieve → cross-encoder rerank

Supports switching between vector-only (baseline) and hybrid retrieval
via the `use_hybrid` parameter, enabling direct A/B metric comparison.
"""

import os, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from app.config import PipelineConfig, REFUSAL_MESSAGE
from app.document_processor import DocumentProcessor
from app.vector_store import EmbeddingEngine, VectorStore
from app.generator import Generator
from app.hallucination_guard import HallucinationGuard
from app.hybrid_retriever import HybridRetriever


@dataclass
class QueryResult:
    query: str
    answer: str
    refused: bool
    confidence: float
    retrieval_mode: str = "vector"
    citations: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    timing_ms: dict = field(default_factory=dict)

    def to_api(self, debug: bool = False) -> dict:
        out = {
            "answer": self.answer,
            "refused": self.refused,
            "citations": self.citations,
            "retrieval_mode": self.retrieval_mode,
        }
        if debug:
            out["confidence"] = round(self.confidence, 3)
            out["verification"] = self.verification
            out["timing_ms"] = self.timing_ms
            out["retrieved_chunks"] = [
                {
                    "chunk_id": r["chunk"].chunk_id,
                    "score": round(r["score"], 4),
                    "rerank_score": round(r.get("rerank_score", 0), 4),
                    "source_file": r["chunk"].source_file,
                    "page_number": r["chunk"].page_number,
                    "text": r["chunk"].text[:400],
                }
                for r in self.retrieved_chunks
            ]
        return out


class RAGPipeline:
    """
    Level 1 (vector-only):
        ┌──────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌────────┐
        │ PDFs │─▶│ Chunk │─▶│ FAISS │─▶│ LLM   │─▶│ Guard  │─▶ Answer
        └──────┘  └───────┘  └───────┘  └───────┘  └────────┘

    Level 2 (hybrid):
        ┌──────┐  ┌───────┐  ┌───────────────┐  ┌───────────┐  ┌─────┐  ┌───────┐
        │ PDFs │─▶│ Chunk │─▶│ FAISS + BM25  │─▶│ RRF Merge │─▶│Rerank│─▶│ LLM   │─▶ ...
        └──────┘  └───────┘  └───────────────┘  └───────────┘  └─────┘  └───────┘
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()
        print("═" * 55)
        print("  Initialising Aviation RAG Pipeline")
        print("═" * 55)
        self.processor = DocumentProcessor(self.cfg.chunking)
        self.engine    = EmbeddingEngine(self.cfg.embedding)
        self.store     = VectorStore(self.engine, self.cfg.retriever)
        self.generator = Generator(self.cfg.generator)
        self.guard     = HallucinationGuard()

        # Level 2: Hybrid retriever
        self.hybrid = HybridRetriever(self.store)

        # Auto-load persisted indices
        vs_dir = self.cfg.vector_store_dir
        if (Path(vs_dir) / "index.faiss").exists():
            self.store.load(vs_dir)
            self.hybrid.load_bm25(vs_dir)

        print("  ✓ Pipeline ready\n")

    # ── INGEST ────────────────────────────────────────────────────────────

    def ingest(self, source: Optional[str] = None) -> dict:
        src = source or self.cfg.documents_dir
        print(f" Ingesting from: {src}")
        t0 = time.time()

        chunks = (
            self.processor.process_directory(src)
            if os.path.isdir(src)
            else self.processor.process_pdf(src)
        )
        if not chunks:
            return {"status": "error", "detail": "No chunks produced"}

        # Level 1: Vector index
        self.store.add_chunks(chunks)
        self.store.save(self.cfg.vector_store_dir)

        # Level 2: BM25 index
        self.hybrid.build_bm25_index()
        self.hybrid.save_bm25(self.cfg.vector_store_dir)

        elapsed = round(time.time() - t0, 2)
        sources = sorted(set(c.source_file for c in chunks))
        stats = {
            "status": "ok",
            "total_chunks": len(chunks),
            "files_processed": len(sources),
            "source_files": sources,
            "elapsed_seconds": elapsed,
        }
        print(f"  ✓ {stats['total_chunks']} chunks from "
              f"{stats['files_processed']} files in {elapsed}s\n")
        return stats

    # ── QUERY ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        debug: bool = False,
        use_hybrid: bool = False,
    ) -> QueryResult:
        """
        Answer a question using the RAG pipeline.

        Parameters
        ----------
        question    : user's question
        debug       : include retrieved chunks + timing in response
        use_hybrid  : True = Level 2 hybrid retrieval
                      False = Level 1 vector-only (baseline)
        """
        timing = {}
        mode = "hybrid" if use_hybrid else "vector"

        # 1 ── Retrieve
        t = time.time()
        if use_hybrid and self.hybrid.ready:
            retrieved = self.hybrid.search(question)
            mode = "hybrid"
        else:
            retrieved = self.store.search(question)
            mode = "vector"
        timing["retrieve"] = round((time.time() - t) * 1000, 1)

        if not retrieved:
            return QueryResult(
                query=question, answer=REFUSAL_MESSAGE,
                refused=True, confidence=0.0, retrieval_mode=mode,
                verification={"reason": "no_chunks_retrieved"},
                timing_ms=timing,
            )

        # 2 ── Generate
        t = time.time()
        gen = self.generator.generate(question, retrieved)
        timing["generate"] = round((time.time() - t) * 1000, 1)

        if gen["refused"]:
            return QueryResult(
                query=question, answer=REFUSAL_MESSAGE,
                refused=True, confidence=0.1, retrieval_mode=mode,
                retrieved_chunks=retrieved,
                verification={"reason": gen.get("_reason", "llm_refused")},
                timing_ms=timing,
            )

        # 3 ── Hallucination guard
        t = time.time()
        guard_result = self.guard.run(
            question, gen["answer"], retrieved,
            self_verify_fn=self.generator.self_verify,
        )
        timing["guard"] = round((time.time() - t) * 1000, 1)

        if not guard_result.passed:
            return QueryResult(
                query=question, answer=REFUSAL_MESSAGE,
                refused=True, confidence=guard_result.confidence,
                retrieval_mode=mode, retrieved_chunks=retrieved,
                verification=guard_result.checks,
                timing_ms=timing,
            )

        # 4 ── Build citations
        sources = []
        seen = set()
        for c in gen["citations"]:
            key = (c["source_file"], c["page_number"])
            if key not in seen:
                seen.add(key)
                sources.append({"file": c["source_file"], "page": c["page_number"]})

        return QueryResult(
            query=question, answer=guard_result.answer,
            refused=False, confidence=guard_result.confidence,
            retrieval_mode=mode,
            citations=gen["citations"], sources=sources,
            retrieved_chunks=retrieved,
            verification=guard_result.checks,
            timing_ms=timing,
        )

    # ── STATS ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "indexed_chunks": self.store.index.ntotal,
            "sources": sorted(set(c.source_file for c in self.store.chunks)),
            "embedding_model": self.cfg.embedding.model_name,
            "llm": f"{self.cfg.generator.provider}/{self.cfg.generator.model}",
            "hybrid_ready": self.hybrid.ready,
        }
