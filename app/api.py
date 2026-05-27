"""
api.py — FastAPI application.

Endpoints:
    GET  /health   → liveness + index stats
    POST /ingest   → process PDFs → FAISS + BM25 indices
    POST /ask      → question → grounded answer + citations
                     supports debug flag + hybrid retrieval toggle
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.pipeline import RAGPipeline
from app.config import PipelineConfig

app = FastAPI(
    title="Aviation Document AI",
    description=(
        "RAG pipeline answering ONLY from provided aviation documents. "
        "Level 1: vector retrieval. Level 2: hybrid BM25 + vector + reranker."
    ),
    version="1.0.0",
)

pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline(PipelineConfig())
    return pipeline


# ── Schemas ───────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    directory: Optional[str] = Field(
        None, description="Path to PDF directory (default: documents/)"
    )

class IngestResponse(BaseModel):
    status: str
    total_chunks: int = 0
    files_processed: int = 0
    source_files: list[str] = []
    elapsed_seconds: float = 0

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The user's question")
    debug: bool = Field(False, description="Include retrieved chunks, timing, verification")
    use_hybrid: bool = Field(
        False,
        description="Level 2: use hybrid BM25+vector+reranker retrieval",
    )

class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[dict] = []
    retrieval_mode: str = "vector"
    # debug fields (present only when debug=True)
    confidence: Optional[float] = None
    verification: Optional[dict] = None
    timing_ms: Optional[dict] = None
    retrieved_chunks: Optional[list[dict]] = None


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check + index statistics."""
    p = get_pipeline()
    return {"status": "healthy", "index_ready": p.store.ready, **p.stats()}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest = IngestRequest()):
    """Process PDFs and build FAISS vector index + BM25 keyword index."""
    p = get_pipeline()
    result = p.ingest(req.directory)
    if result.get("status") == "error":
        raise HTTPException(400, result["detail"])
    return IngestResponse(**result)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Ask a question → grounded answer with citations.

    • debug=true → includes retrieved chunks, timing, verification
    • use_hybrid=true → Level 2 hybrid retrieval (BM25+vector+reranker)

    If the answer cannot be supported, returns:
      "This information is not available in the provided document(s)."
    """
    p = get_pipeline()
    if not p.store.ready:
        raise HTTPException(400, "Index is empty. Call POST /ingest first.")
    qr = p.query(req.question, debug=req.debug, use_hybrid=req.use_hybrid)
    return AskResponse(**qr.to_api(debug=req.debug))


@app.on_event("startup")
def on_startup():
    get_pipeline()
