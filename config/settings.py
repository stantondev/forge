"""Configuration for Forge."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Ollama model configuration."""
    reasoning_model: str = "deepseek-r1:8b"
    coding_model: str = "deepseek-r1:8b"
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    # Lower temperature for reasoning tasks where consistency matters
    reasoning_temperature: float = 0.3


@dataclass
class ReasoningConfig:
    """Reasoning enhancement configuration."""
    # Number of reasoning paths to sample for self-consistency
    # 2 is a good balance for 16GB Mac — increase to 3-5 on beefier hardware
    consistency_samples: int = 2
    # Whether to enable reflection (model critiques its own output)
    # Adds ~30-60s per query on 8B model
    enable_reflection: bool = True
    # Maximum reflection iterations
    max_reflection_rounds: int = 1
    # Whether to show the reasoning chain in output
    show_reasoning: bool = True


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""
    # Where ChromaDB stores its data
    chroma_persist_dir: str = str(Path.home() / ".local-llm" / "chroma_db")
    # Collection name in ChromaDB
    collection_name: str = "knowledge_base"
    # Number of chunks to retrieve per query
    top_k: int = 5
    # Chunk size for document splitting
    chunk_size: int = 512
    # Chunk overlap
    chunk_overlap: int = 50


@dataclass
class CloudConfig:
    """Cloud GPU burst configuration for on-demand scaling."""
    # Enable cloud burst mode (offload to cloud GPU when available)
    enabled: bool = False
    # Cloud provider: "runpod", "tensordock", "scaleway"
    provider: str = "runpod"
    # API key for the cloud provider (set via FORGE_CLOUD_API_KEY env var)
    api_key: str = ""
    # Preferred GPU type
    gpu_type: str = "RTX 3090"
    # Region preference (EU for GDPR)
    region: str = "EU"
    # Auto-shutdown after N minutes of inactivity
    idle_shutdown_minutes: int = 30
    # Max monthly spend limit (USD) — Forge stops using cloud after this
    monthly_budget_limit: float = 50.0


@dataclass
class AppConfig:
    """Top-level application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
