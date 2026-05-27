"""
config.py — Single source of truth for every tunable parameter.
 
Default provider: ollama  (runs locally, no API key needed)
Supported providers: "ollama" | "openai" | "anthropic"
"""
 
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
 
load_dotenv()
 
# ── The exact refusal string required by the assignment ───────────────────
REFUSAL_MESSAGE = "This information is not available in the provided document(s)."
 
 
@dataclass
class ChunkingConfig:
    """
    Chunking strategy rationale:
    • 512 tokens ≈ one dense aviation paragraph — enough context for a
      self-contained fact while keeping embedding quality high.
    • 64-token overlap ensures sentences at chunk boundaries appear in
      at least two chunks, so boundary-spanning questions still hit.
    • min_chunk_size=50 drops trailing runts that add noise.
    """
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
 
 
@dataclass
class EmbeddingConfig:
    model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    dimension: int = 384
    batch_size: int = 64
    normalize: bool = True
 
 
@dataclass
class RetrieverConfig:
    top_k: int = int(os.getenv("TOP_K", "3"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))
 
 
@dataclass
class GeneratorConfig:
    # Default to ollama — no API key needed
    provider: str = os.getenv("LLM_PROVIDER", "ollama")
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: str = "ollama"          # Ollama ignores this but field must exist
    ollama_base_url: str = ""
 
    def __post_init__(self):
        if self.provider == "ollama":
            self.model = self.model or os.getenv("OLLAMA_MODEL", "qwen2.5")
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.api_key = "ollama"  # not used, just a placeholder
        elif self.provider == "openai":
            self.model = self.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "anthropic":
            self.model = self.model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
 
 
@dataclass
class PipelineConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    documents_dir: str = os.getenv("DOCUMENTS_DIR", "documents")
    vector_store_dir: str = os.getenv("VECTOR_STORE_DIR", "vector_store")