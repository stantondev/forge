"""
RAG indexer — indexes documents into ChromaDB for retrieval-augmented generation.

Supports: .txt, .md, .py, .js, .ts, .json, .yaml, .toml, .html, .css, .sh
"""

import sys
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from config import AppConfig


SUPPORTED_EXTENSIONS = [
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".yaml", ".yml", ".toml",
    ".html", ".css", ".sh", ".bash",
    ".rs", ".go", ".java", ".c", ".cpp", ".h",
    ".sql", ".csv", ".xml",
]


def create_index(config: AppConfig | None = None) -> tuple:
    """Create or load the ChromaDB-backed vector index."""
    config = config or AppConfig()
    rc = config.rag

    # Ensure persist directory exists
    Path(rc.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    # Set up embeddings and LLM to use local Ollama
    Settings.embed_model = OllamaEmbedding(
        model_name=config.model.embedding_model,
        base_url=config.model.ollama_base_url,
    )
    Settings.llm = Ollama(
        model=config.model.reasoning_model,
        base_url=config.model.ollama_base_url,
        temperature=config.model.temperature,
    )

    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=rc.chroma_persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(rc.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return storage_context, chroma_collection


def index_directory(directory: str, config: AppConfig | None = None) -> int:
    """
    Index all supported files in a directory into the vector store.
    Returns the number of documents indexed.
    """
    config = config or AppConfig()
    rc = config.rag
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Read documents
    reader = SimpleDirectoryReader(
        input_dir=str(dir_path),
        recursive=True,
        required_exts=SUPPORTED_EXTENSIONS,
    )
    documents = reader.load_data()

    if not documents:
        print(f"No supported files found in {directory}")
        return 0

    # Split into chunks
    splitter = SentenceSplitter(
        chunk_size=rc.chunk_size,
        chunk_overlap=rc.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    # Create index
    storage_context, _ = create_index(config)
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
    )

    return len(nodes)


def get_query_engine(config: AppConfig | None = None):
    """Get a query engine for the existing index."""
    config = config or AppConfig()
    rc = config.rag

    storage_context, chroma_collection = create_index(config)

    # Load existing index from ChromaDB
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model,
    )

    return index.as_query_engine(
        similarity_top_k=rc.top_k,
    )


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    if len(sys.argv) < 2:
        console.print("[red]Usage: python -m src.rag.indexer /path/to/documents[/red]")
        sys.exit(1)

    directory = sys.argv[1]
    console.print(f"\n[bold]Indexing documents from:[/bold] {directory}")

    config = AppConfig()
    count = index_directory(directory, config)
    console.print(f"[green]Indexed {count} chunks into ChromaDB[/green]")
    console.print(f"Database stored at: {config.rag.chroma_persist_dir}")
