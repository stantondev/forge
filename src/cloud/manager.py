"""
Cloud GPU burst manager — spins up on-demand GPU instances for heavy
inference and auto-shuts them down when idle.

Supports RunPod (recommended for cost) with EU region preference.

The idea: your Mac handles casual queries for free. When you need
speed or heavier models, Forge spins up a cloud GPU, runs the query
there, and shuts it down after you're done. Pay-per-second billing
means you only pay for what you use.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from config import CloudConfig


@dataclass
class CloudInstance:
    """Tracks a running cloud GPU instance."""
    instance_id: str
    provider: str
    gpu_type: str
    region: str
    ollama_url: str
    started_at: float
    last_used_at: float
    hourly_rate: float


SPEND_LOG_PATH = Path.home() / ".local-llm" / "cloud_spend.json"


class CloudManager:
    """
    Manages on-demand cloud GPU instances for burst inference.

    Usage:
        manager = CloudManager(config.cloud)
        url = manager.get_ollama_url()  # spins up if needed
        # ... use url for inference ...
        manager.mark_used()  # reset idle timer
        manager.check_idle()  # call periodically to auto-shutdown
    """

    def __init__(self, config: CloudConfig):
        self.config = config
        self.instance: CloudInstance | None = None
        self.api_key = config.api_key or os.environ.get("FORGE_CLOUD_API_KEY", "")
        self._load_spend()

    def _load_spend(self):
        """Load monthly spend tracking."""
        self.monthly_spend = 0.0
        self.spend_month = time.strftime("%Y-%m")
        if SPEND_LOG_PATH.exists():
            data = json.loads(SPEND_LOG_PATH.read_text())
            if data.get("month") == self.spend_month:
                self.monthly_spend = data.get("total", 0.0)

    def _save_spend(self):
        """Save monthly spend tracking."""
        SPEND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPEND_LOG_PATH.write_text(json.dumps({
            "month": self.spend_month,
            "total": round(self.monthly_spend, 2),
        }))

    def _check_budget(self):
        """Check if we're within monthly budget."""
        if self.monthly_spend >= self.config.monthly_budget_limit:
            raise BudgetExceededError(
                f"Monthly cloud budget of ${self.config.monthly_budget_limit:.2f} reached "
                f"(spent: ${self.monthly_spend:.2f}). "
                f"Falling back to local inference."
            )

    def is_available(self) -> bool:
        """Check if cloud burst is configured and within budget."""
        if not self.config.enabled:
            return False
        if not self.api_key:
            return False
        try:
            self._check_budget()
            return True
        except BudgetExceededError:
            return False

    def get_ollama_url(self) -> str:
        """
        Get the Ollama URL for a cloud instance.
        Spins up a new instance if none is running.
        """
        self._check_budget()

        if self.instance:
            # Check if instance is still alive
            try:
                resp = httpx.get(f"{self.instance.ollama_url}/api/tags", timeout=5)
                if resp.status_code == 200:
                    return self.instance.ollama_url
            except Exception:
                self.instance = None

        # Need to spin up a new instance
        return self._start_instance()

    def _start_instance(self) -> str:
        """Start a new cloud GPU instance. Currently supports RunPod."""
        if self.config.provider == "runpod":
            return self._start_runpod()
        else:
            raise ValueError(f"Unsupported cloud provider: {self.config.provider}")

    def _start_runpod(self) -> str:
        """Start a RunPod GPU pod with Ollama."""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Create a pod with Ollama pre-installed
        payload = {
            "name": "forge-burst",
            "imageName": "ollama/ollama:latest",
            "gpuTypeId": "NVIDIA GeForce RTX 3090",
            "cloudType": "COMMUNITY",  # cheaper than SECURE
            "volumeInGb": 20,
            "containerDiskInGb": 20,
            "ports": "11434/http",
            "env": {},
        }

        # Prefer EU region if configured
        if self.config.region == "EU":
            payload["dataCenterId"] = "EU-RO-1"  # Romania datacenter

        resp = httpx.post(
            "https://api.runpod.io/v2/pods",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        pod_id = data["id"]
        # RunPod pods take a moment to start
        ollama_url = f"https://{pod_id}-11434.proxy.runpod.net"

        # Wait for the pod to be ready
        for _ in range(60):
            try:
                check = httpx.get(f"{ollama_url}/api/tags", timeout=5)
                if check.status_code == 200:
                    break
            except Exception:
                time.sleep(5)

        self.instance = CloudInstance(
            instance_id=pod_id,
            provider="runpod",
            gpu_type=self.config.gpu_type,
            region=self.config.region,
            ollama_url=ollama_url,
            started_at=time.time(),
            last_used_at=time.time(),
            hourly_rate=0.20,  # approximate RTX 3090 rate
        )

        # Pull the model on the cloud instance
        httpx.post(
            f"{ollama_url}/api/pull",
            json={"name": "deepseek-r1:8b"},
            timeout=300,
        )

        return ollama_url

    def mark_used(self):
        """Mark the cloud instance as recently used (resets idle timer)."""
        if self.instance:
            self.instance.last_used_at = time.time()

    def check_idle(self):
        """Shutdown the instance if it's been idle too long."""
        if not self.instance:
            return

        idle_seconds = time.time() - self.instance.last_used_at
        if idle_seconds > self.config.idle_shutdown_minutes * 60:
            self.shutdown()

    def shutdown(self):
        """Shutdown the cloud instance and record spending."""
        if not self.instance:
            return

        # Calculate cost
        runtime_hours = (time.time() - self.instance.started_at) / 3600
        cost = runtime_hours * self.instance.hourly_rate
        self.monthly_spend += cost
        self._save_spend()

        # Terminate the instance
        try:
            if self.instance.provider == "runpod":
                headers = {"Authorization": f"Bearer {self.api_key}"}
                httpx.delete(
                    f"https://api.runpod.io/v2/pods/{self.instance.instance_id}",
                    headers=headers,
                    timeout=15,
                )
        except Exception:
            pass  # Best effort shutdown

        print(f"[Forge Cloud] Shutdown {self.instance.instance_id} "
              f"(runtime: {runtime_hours:.1f}h, cost: ${cost:.2f}, "
              f"month total: ${self.monthly_spend:.2f})")
        self.instance = None

    def get_status(self) -> dict:
        """Get current cloud status."""
        return {
            "enabled": self.config.enabled,
            "provider": self.config.provider,
            "instance_running": self.instance is not None,
            "monthly_spend": round(self.monthly_spend, 2),
            "budget_limit": self.config.monthly_budget_limit,
            "budget_remaining": round(
                self.config.monthly_budget_limit - self.monthly_spend, 2
            ),
        }


class BudgetExceededError(Exception):
    """Raised when monthly cloud budget is exceeded."""
    pass
