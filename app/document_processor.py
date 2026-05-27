"""
document_processor.py — PDF loading → text extraction → token-based chunking.

Each chunk carries full provenance: source filename, page number, chunk index,
and a deterministic chunk_id so citations are traceable back to the exact
location in the original PDF.
"""

import hashlib, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tiktoken
from PyPDF2 import PdfReader

from app.config import ChunkingConfig


# ── Data model ────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """Atomic retrieval unit with full provenance."""
    chunk_id: str
    text: str
    source_file: str
    source_path: str
    page_number: Optional[int]
    chunk_index: int
    token_count: int
    metadata: dict = field(default_factory=dict)

    def citation(self) -> str:
        s = f"Source: {self.source_file}"
        if self.page_number is not None:
            s += f", Page {self.page_number}"
        return s

    def to_dict(self) -> dict:
        return dict(
            chunk_id=self.chunk_id,
            text=self.text,
            source_file=self.source_file,
            page_number=self.page_number,
            chunk_index=self.chunk_index,
            token_count=self.token_count,
        )


# ── Processor ─────────────────────────────────────────────────────────────

class DocumentProcessor:
    def __init__(self, config: ChunkingConfig | None = None):
        self.cfg = config or ChunkingConfig()
        self.tok = tiktoken.get_encoding("cl100k_base")

    def process_directory(self, directory: str) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for p in sorted(Path(directory).rglob("*.pdf")):
            print(f"  📄 {p.name}")
            chunks.extend(self.process_pdf(str(p)))
        print(f"  ── Total chunks: {len(chunks)}")
        return chunks

    def process_pdf(self, path: str) -> list[DocumentChunk]:
        reader = PdfReader(path)
        fname = Path(path).name
        all_c: list[DocumentChunk] = []

        for pi, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            text = _clean(raw)
            if not text:
                continue
            all_c.extend(self._chunk(text, fname, path, pi + 1, len(all_c)))
        return all_c

    def _chunk(self, text, fname, fpath, page, offset) -> list[DocumentChunk]:
        tokens = self.tok.encode(text)
        out: list[DocumentChunk] = []
        start, idx = 0, offset
        step = max(self.cfg.chunk_size - self.cfg.chunk_overlap, 1)

        while start < len(tokens):
            end = min(start + self.cfg.chunk_size, len(tokens))
            ct = tokens[start:end]
            n = len(ct)
            if n < self.cfg.min_chunk_size and start > 0:
                break
            ctext = self.tok.decode(ct).strip()
            cid = _make_id(fname, page, idx)
            out.append(DocumentChunk(
                chunk_id=cid, text=ctext, source_file=fname,
                source_path=fpath, page_number=page,
                chunk_index=idx, token_count=n,
                metadata={"page": page},
            ))
            idx += 1
            start += step
        return out


def _clean(t: str) -> str:
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


def _make_id(src, page, idx) -> str:
    return hashlib.sha256(f"{src}::{page}::{idx}".encode()).hexdigest()[:12]
