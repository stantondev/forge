# Forge

A privacy-first, self-hosted AI assistant with enhanced reasoning. Runs entirely on your hardware — no data leaves your machine.

Forge wraps open-source LLMs (DeepSeek-R1) with reasoning enhancement layers that close the gap with commercial AI services, while keeping your data under your control.

## What Forge does differently

- **Chain-of-thought reasoning** — forces step-by-step thinking on every query
- **Self-consistency** — samples multiple reasoning paths and picks the most reliable answer
- **Self-reflection** — the model critiques and improves its own output before returning it
- **RAG grounding** — answers are grounded in your documents, not just model memory
- **100% private** — everything runs locally or on your own server. Zero cloud dependencies.

## Quick start

### Local (Mac/Linux)

```bash
git clone https://github.com/YOUR_USERNAME/forge.git
cd forge
./setup.sh
source venv/bin/activate
PYTHONPATH=. python -m src.agent.cli
```

### Docker (recommended for servers)

```bash
git clone https://github.com/YOUR_USERNAME/forge.git
cd forge
docker compose up -d

# Pull the models (first time only)
docker exec forge-ollama ollama pull deepseek-r1:8b
docker exec forge-ollama ollama pull nomic-embed-text
```

Then open http://localhost:3000 for the chat interface.

## Architecture

```
┌─────────────────────────────────────┐
│        Open WebUI (Chat UI)         │  ← ChatGPT-like interface
├─────────────────────────────────────┤
│         Forge API Server            │  ← OpenAI-compatible API
├─────────────────────────────────────┤
│     Reasoning Orchestrator (DSPy)   │  ← CoT, self-consistency, reflection
├─────────────────────────────────────┤
│    RAG (LlamaIndex + ChromaDB)      │  ← Document-grounded answers
├─────────────────────────────────────┤
│           Ollama                    │  ← Model inference
├─────────────────────────────────────┤
│      DeepSeek-R1 8B (4-bit)        │  ← Base reasoning model
└─────────────────────────────────────┘
```

## CLI commands

| Command | Description |
|---|---|
| `/quick` | Fast single-pass answer (skip reasoning pipeline) |
| `/reason` | Full reasoning mode: CoT + consistency + reflection |
| `/index /path` | Index a directory of documents for RAG |
| `/model` | Show current model configuration |
| `/config` | Show all configuration settings |
| `/quit` | Exit |

## Index your documents

```bash
# CLI
PYTHONPATH=. python -m src.rag.indexer /path/to/your/documents

# Or from within the CLI
/index /path/to/your/documents
```

Supports: `.txt`, `.md`, `.py`, `.js`, `.ts`, `.json`, `.yaml`, `.html`, `.css`, `.sh`, `.rs`, `.go`, `.java`, `.c`, `.cpp`, `.sql`, `.csv`, `.xml`

## Deploy to a GPU server (Hetzner)

For faster inference and always-on availability, deploy to a Hetzner GPU server (~€190/month for an RTX 4000):

```bash
ssh root@YOUR_SERVER_IP 'bash -s' < scripts/deploy-hetzner.sh
```

See [scripts/deploy-hetzner.sh](scripts/deploy-hetzner.sh) for full setup details.

## Configuration

Edit `config/settings.py` to adjust:

- **Model**: Switch between models (`deepseek-r1:8b`, `deepseek-r1:14b`, etc.)
- **Reasoning depth**: Number of consistency samples, reflection rounds
- **RAG settings**: Chunk size, retrieval count, database location

## Hardware requirements

| Setup | RAM | Runs |
|---|---|---|
| Mac M1/M2/M3 16GB | 16GB unified | 8B models comfortably |
| Mac 32GB+ | 32GB+ unified | 14B-30B models |
| GPU server (20GB VRAM) | 20GB+ VRAM | 8B-14B models fast |
| GPU server (48GB VRAM) | 48GB+ VRAM | 70B models |

## License

MIT — use it however you want.
