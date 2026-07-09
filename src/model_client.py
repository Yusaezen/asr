"""
model_client.py — GPU-ready, timeout raised to 600s.

Fix 1: timeout raised from 120s → 600s (eliminates Ollama timeouts on GPU).
Fix 2: comment updated — no longer macOS-only, works on Ubuntu + CUDA.
"""
import requests
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "mistral:7b-instruct-q4_K_M"


def check_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def list_available_models() -> list:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
        return []
    except Exception:
        return []


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Call Ollama /api/generate and return the full response string.
    Timeout raised to 600s for GPU generation (was 120s → caused timeouts on
    complex GSM8K questions with 4+ steps on CPU).
    """
    if not check_ollama_running():
        raise ConnectionError(
            "Ollama is not running. Start it with: ollama serve"
        )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=600,          # raised from 120 → 600
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"Ollama timed out generating response for model {model}. "
            f"On GPU this should not happen — check that Ollama is using the GPU "
            f"(run: ollama ps  or  nvidia-smi while generating)."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}")