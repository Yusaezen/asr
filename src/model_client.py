import requests
import json
import logging
from typing import Optional

# For Ollama, on macOS

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral:7b-instruct-q4_K_M"


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
    Temperature kept low (0.2) for consistent structured output.
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
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Ollama timed out generating response for model {model}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}")
