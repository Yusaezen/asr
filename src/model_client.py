"""
model_client.py — HuggingFace backend (replaces Ollama).

Previously this module called the Ollama HTTP API at localhost:11434.
That endpoint is an opaque text-in/text-out interface: it does NOT expose
token-level hidden states, which are required by HiddenStateExtractor for
UHead scoring.

The replacement uses the HuggingFace model already loaded by
confidence/model_loader.py (Mistral-7B-Instruct, 4-bit NF4 via bitsandbytes).
Because both generation (here) and UHead scoring (scorer.py) now go through
the same model instance, hidden-state distributions at training time and
inference time are guaranteed to be consistent.

Public API is unchanged so that generation_harness.py, batch_runner.py, and
any other callers do not need to be modified:
    generate(prompt, model, temperature, max_tokens, system_prompt) -> str
    DEFAULT_MODEL                                                    -> str

Removed (Ollama-specific, no longer applicable):
    check_ollama_running()  — kept as a no-op stub for import compatibility
    list_available_models() — kept as a stub returning [DEFAULT_MODEL]
"""
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure confidence/ is importable when this module is called from src/
sys.path.insert(0, str(Path(__file__).parent))

from confidence.model_loader import generate_text, MODEL_ID

logger = logging.getLogger(__name__)

# Expose the HuggingFace model ID as DEFAULT_MODEL so call sites
# (generation_harness.py, batch_runner.py) that reference model_client.DEFAULT_MODEL
# continue to work without changes.
DEFAULT_MODEL = MODEL_ID


# ── Compatibility stubs (previously Ollama-specific) ──────────────────────────

def check_ollama_running() -> bool:
    """
    Stub retained for import compatibility with generation_harness.py.
    Always returns True — Ollama is no longer used; the HuggingFace model
    is loaded on first call to generate().
    """
    return True


def list_available_models() -> list:
    """
    Stub retained for import compatibility.
    Returns the single HuggingFace model in use.
    """
    return [DEFAULT_MODEL]


# ── Main generation function ───────────────────────────────────────────────────

def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,       # kept for API compatibility; value ignored
    temperature: float = 0.2,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Generate a response using the unified HuggingFace model.

    The `model` argument is accepted for API compatibility but is ignored —
    the model loaded by confidence/model_loader.py (MODEL_ID) is always used.

    Delegates entirely to confidence.model_loader.generate_text() so that
    the HuggingFace singleton is shared between generation and UHead scoring,
    ensuring consistent hidden-state distributions.
    """
    if model != DEFAULT_MODEL:
        logger.warning(
            f"model_client.generate() received model='{model}' but the "
            f"HuggingFace backend always uses '{DEFAULT_MODEL}'. "
            f"The requested model identifier will be ignored."
        )

    logger.info(f"Generating via HuggingFace ({DEFAULT_MODEL}) — temperature={temperature}")
    return generate_text(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_new_tokens=max_tokens,
    )