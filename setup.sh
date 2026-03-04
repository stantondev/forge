#!/bin/bash
set -e

echo "=== Forge Setup ==="
echo ""

# Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ollama
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
else
    echo "Ollama already installed: $(ollama --version)"
fi

# Start Ollama server if not running
if ! pgrep -x "ollama" > /dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 3
fi

# Pull models
echo ""
echo "=== Pulling Models ==="

echo "Pulling DeepSeek-R1 8B (reasoning model)..."
ollama pull deepseek-r1:8b

echo "Pulling nomic-embed-text (embedding model for RAG)..."
ollama pull nomic-embed-text

echo ""
echo "=== Setting up Python environment ==="

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created virtual environment"
fi

source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Forge is ready ==="
echo ""
echo "  Activate env:    source venv/bin/activate"
echo "  Chat (CLI):      PYTHONPATH=. python -m src.agent.cli"
echo "  Chat (Web UI):   docker compose up -d  →  http://localhost:3000"
echo "  API server:      PYTHONPATH=. python -m src.agent.server"
echo "  Index docs:      PYTHONPATH=. python -m src.rag.indexer /path/to/docs"
