"""
ingest.py — CLI script to process PDFs and build the FAISS index.

Usage:
    python ingest.py                       # default: documents/
    python ingest.py --dir /path/to/pdfs   # custom directory
"""

import argparse
from app.config import PipelineConfig
from app.pipeline import RAGPipeline


def main():
    ap = argparse.ArgumentParser(description="Ingest aviation PDFs into the vector store.")
    ap.add_argument("--dir", default=None, help="Directory containing PDF files")
    args = ap.parse_args()

    cfg = PipelineConfig()
    pipe = RAGPipeline(cfg)
    result = pipe.ingest(args.dir)
    print("\n Ingestion Summary:")
    for k, v in result.items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
