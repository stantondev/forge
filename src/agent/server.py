"""
Forge API server — OpenAI-compatible API that wraps the reasoning orchestrator.

This lets Open WebUI (or any OpenAI-compatible client) use Forge's
enhanced reasoning pipeline transparently.
"""

import json
import os
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

from config import AppConfig
from src.agent.core import Agent


config = AppConfig()

# Allow overriding Ollama URL via environment variable (for Docker)
ollama_url = os.environ.get("OLLAMA_BASE_URL", config.model.ollama_base_url)
config.model.ollama_base_url = ollama_url

agent = None


def get_agent():
    global agent
    if agent is None:
        agent = Agent(config)
    return agent


class ForgeHandler(BaseHTTPRequestHandler):
    """Handles OpenAI-compatible chat completions API requests."""

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self.handle_chat_completion()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/v1/models":
            self.handle_models()
        elif self.path == "/health":
            self.send_json({"status": "ok"})
        else:
            self.send_error(404)

    def handle_models(self):
        self.send_json({
            "object": "list",
            "data": [
                {
                    "id": "forge-reasoning",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "forge",
                },
                {
                    "id": "forge-quick",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "forge",
                },
            ],
        })

    def handle_chat_completion(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        request = json.loads(body)

        messages = request.get("messages", [])
        model = request.get("model", "forge-reasoning")
        use_reasoning = "quick" not in model

        # Extract the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            self.send_error(400, "No user message found")
            return

        a = get_agent()
        result = a.query(user_message, use_reasoning=use_reasoning)

        # Build response with reasoning shown if enabled
        response_text = result["answer"]
        if config.reasoning.show_reasoning and result.get("reasoning") and use_reasoning:
            response_text = (
                f"**Reasoning:**\n{result['reasoning']}\n\n"
                f"---\n\n{result['answer']}"
            )

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(user_message.split()) + len(response_text.split()),
            },
        }

        self.send_json(response)

    def send_json(self, data):
        response = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        print(f"[Forge] {args[0]}")


def main():
    port = int(os.environ.get("FORGE_PORT", "8000"))
    server = HTTPServer(("0.0.0.0", port), ForgeHandler)
    print(f"Forge API server running on http://0.0.0.0:{port}")
    print(f"  Models: forge-reasoning (full pipeline), forge-quick (single-pass)")
    print(f"  Ollama: {config.model.ollama_base_url}")
    print(f"  Reasoning: {config.reasoning.consistency_samples} samples, reflection={'on' if config.reasoning.enable_reflection else 'off'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
