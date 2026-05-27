# Aviation Document AI — RAG Pipeline

> **AIRMAN AI/ML Intern Technical Assignment**
> Answers aviation questions **strictly from provided documents** using a fully local stack — no API keys, no cloud services, runs 100% on your machine.

---

## Stack (fully local, zero API cost)

| Component | Technology |
|-----------|------------|
| LLM | Ollama + Qwen 2.5 (local) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local) |
| Vector Store | FAISS (local) |
| BM25 Index | Custom Okapi BM25 (no deps) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (local) |
| API | FastAPI |
| Tokenizer | tiktoken |

---

## Quick Start

### Step 1 — Install Ollama and pull the model
```bash
# Download Ollama from https://ollama.com/download
# Then pull Qwen 2.5:
ollama pull qwen2.5

# Keep Ollama running in background:
ollama serve
```

### Step 2 — Clone and install
```bash
git clone https://github.com/<your-username>/aviation-rag-pipeline.git
cd aviation-rag-pipeline

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Configure (no API key needed)
```bash
cp .env.example .env
# .env already set to LLM_PROVIDER=ollama — nothing else to change
```

### Step 4 — Add PDFs to documents/
```
documents/
  Air-Regulation-RK-BALI.pdf
  Meteorology_full_book.pdf
  Instruments.pdf
  11-radio-navigation-2014.pdf
  10-General-Navigation-2014__1_.pdf
  7-Flight-Planning-and-Monitoring-2014.pdf
  6-mass-and-balance-and-performance-2014.pdf
```

### Step 5 — Ingest → Serve → Evaluate
```bash
python ingest.py                                    # build FAISS + BM25 index
python main.py                                      # start API at localhost:8000
python evaluate.py --mode both --out report.md      # run evaluation
```

Open **http://localhost:8000/docs** in your browser to use the interactive Swagger UI.

---

## Architecture

### Level 1 — Vector Retrieval (Baseline)
```
PDFs ──▶ Chunk (512 tok) ──▶ Embed (MiniLM) ──▶ FAISS Index
                                                      │
Answer ◀── Guard (3-layer) ◀── Qwen 2.5 ◀── Top-K Retrieve
```

### Level 2 — Hybrid Retrieval (BM25 + Vector + Reranker)
```
             ┌──▶ BM25 top-20 ─────────┐
Query ───────┤                          ├──▶ RRF Merge ──▶ Cross-Encoder ──▶ Top-K
             └──▶ FAISS top-20 ─────────┘
```

---

## Project Structure

```
├── app/
│   ├── config.py               # All tunable parameters
│   ├── document_processor.py   # PDF → text → token-based chunks
│   ├── vector_store.py         # sentence-transformers + FAISS
│   ├── generator.py            # Ollama/OpenAI/Anthropic with grounding prompt
│   ├── hallucination_guard.py  # 3-layer verification gate
│   ├── hybrid_retriever.py     # Level 2: BM25 + RRF + reranker
│   ├── pipeline.py             # Orchestrator (vector or hybrid mode)
│   └── api.py                  # FastAPI endpoints
├── eval_data/questions.json    # 50-question evaluation set
├── ingest.py                   # CLI: build index
├── evaluate.py                 # CLI: run eval, generate report
├── main.py                     # CLI: start API server
├── report.md                   # Auto-generated evaluation report
├── requirements.txt
├── .env.example                # Copy to .env — no API keys needed
├── Dockerfile
└── docker-compose.yml          # Includes Ollama service
```

---

## API Reference

### `GET /health`
```json
{
  "status": "healthy",
  "index_ready": true,
  "indexed_chunks": 8500,
  "llm": "ollama/qwen2.5",
  "hybrid_ready": true
}
```

### `POST /ingest`
```json
{ "directory": "documents/" }
```

### `POST /ask`
```json
{
  "question": "What is the Dry Adiabatic Lapse Rate?",
  "debug": true,
  "use_hybrid": false
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `question` | required | The question to ask |
| `debug` | false | Include retrieved chunks + timing |
| `use_hybrid` | false | Level 2: BM25 + vector + reranker |

**Answerable response:**
```json
{
  "answer": "The DALR is approximately 3°C per 1000 ft [a1b2c3d4e5f6].",
  "refused": false,
  "citations": [
    {
      "chunk_id": "a1b2c3d4e5f6",
      "source_file": "Meteorology_full_book.pdf",
      "page_number": 45
    }
  ],
  "retrieval_mode": "vector"
}
```

**Refused response (out-of-scope or unanswerable):**
```json
{
  "answer": "This information is not available in the provided document(s).",
  "refused": true,
  "citations": []
}
```

---

## Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 512 tokens | Full aviation paragraph with a self-contained fact |
| Overlap | 64 tokens | Boundary sentences appear in ≥2 chunks |
| Tokenizer | tiktoken cl100k_base | Consistent for dense technical text |
| Min chunk | 50 tokens | Drops trailing fragments |

---

## Hallucination Prevention (3 Layers)

| Layer | What it checks |
|-------|---------------|
| **Retrieval Gate** | Enough high-scoring chunks retrieved? |
| **Coverage Check** | Query key terms present in context? |
| **Self-Verification** | Second LLM call: is every claim supported? |

Plus: temperature=0, system prompt forbids outside knowledge, model outputs `INSUFFICIENT_CONTEXT` when unsure.

---

## Level 2: Hybrid Retrieval

| Component | Purpose |
|-----------|---------|
| BM25 | Exact keyword matching — critical for aviation terms (RVSM, DALR, PNR) |
| FAISS | Semantic similarity — catches paraphrased questions |
| RRF | Merges both result lists without score normalisation |
| Cross-Encoder | Final precision pass scoring each (query, chunk) pair |

---

## Evaluation

```bash
python evaluate.py --mode both --out report.md --json eval_results.json
```

| Category | Count |
|----------|-------|
| Factual | 20 |
| Applied | 16 |
| Reasoning | 10 |
| Out-of-scope (must refuse) | 4 |

Metrics: Retrieval hit-rate · Faithfulness · Hallucination rate · Citation accuracy · Refusal accuracy

---

## Docker (with Ollama)

```bash
# Start everything — Ollama + RAG API
docker-compose up --build

# In a separate terminal, pull the model into the Ollama container:
docker exec -it <ollama-container-id> ollama pull qwen2.5

# Ingest documents:
docker exec -it <rag-container-id> python ingest.py
```

Or run manually (recommended for local dev):
```bash
ollama serve                  # terminal 1
python ingest.py              # terminal 2 (once)
python main.py                # terminal 2
```
