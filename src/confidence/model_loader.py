"""
confidence/model_loader.py — Unified HuggingFace model singleton (4-bit quantized).

Replaces the previous float16 loader. Now loads Mistral-7B via bitsandbytes
4-bit quantization (BitsAndBytesConfig) so the memory footprint is ~4-5 GB —
comparable to the previous Ollama setup — while keeping the full HuggingFace
Transformers interface: forward hooks, output_hidden_states, and model.generate()
all work identically.

Why 4-bit instead of Ollama:
    Ollama is an opaque HTTP endpoint. It does not expose token-level hidden
    states, which are required by HiddenStateExtractor for UHead scoring.
    Loading quantized via HuggingFace keeps both generation and UHead scoring
    on the same model instance, eliminating the hidden-state distribution
    mismatch that would arise from training on HF float16 hidden states but
    scoring on Ollama-generated steps.

Design notes:
    - Singleton pattern (_model / _tokenizer) — model is loaded once and reused
      across both generation (generate_text) and scoring (scorer.py via extractor).
    - 4-bit NF4 quantization via BitsAndBytesConfig. Falls back to float16 if
      bitsandbytes is not installed (e.g. macOS / CPU-only machines).
    - device_map="auto" lets accelerate place layers across available devices.
    - generate_text() mirrors the old model_client.generate() signature so that
      model_client.py can delegate here transparently.
"""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

_model     = None
_tokenizer = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_bnb_config():
    """
    Returns a BitsAndBytesConfig for 4-bit NF4 quantization, or None if
    bitsandbytes is unavailable (CPU / macOS without CUDA).
    """
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,   # nested quantization saves ~0.4 GB
            bnb_4bit_compute_dtype=torch.float16,
        )
    except (ImportError, ValueError):
        logger.warning(
            "bitsandbytes not available — falling back to float16. "
            "Install with: pip install bitsandbytes"
        )
        return None


def get_model_and_tokenizer():
    """
    Returns the singleton (model, tokenizer) pair.

    First call: loads Mistral-7B in 4-bit quantization if bitsandbytes is
    available, otherwise in float16. Subsequent calls return cached instances.
    """
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    device = get_device()
    bnb_config = _build_bnb_config()

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _tokenizer.pad_token = _tokenizer.eos_token

    if bnb_config is not None:
        logger.info(f"Loading {MODEL_ID} with 4-bit NF4 quantization (bitsandbytes)...")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",          # accelerate handles multi-device placement
        )
    else:
        logger.info(f"Loading {MODEL_ID} on {device} in float16 (no bitsandbytes)...")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            device_map="auto",
        )

    _model.eval()
    logger.info("Model loaded and ready.")
    return _model, _tokenizer


def generate_text(
    prompt: str,
    system_prompt: str = None,
    temperature: float = 0.2,
    max_new_tokens: int = 1024,
) -> str:
    """
    Generate a response using the loaded HuggingFace model.

    Mirrors the interface of the old model_client.generate() so that
    model_client.py can delegate here without changing any call sites.

    Args:
        prompt:        The user-facing prompt string.
        system_prompt: Optional system-level instruction prepended in Mistral
                       instruct format ([INST] <<SYS>> ... <</SYS>> [/INST]).
        temperature:   Sampling temperature (0.0 = greedy).
        max_new_tokens: Maximum tokens to generate (not counting the prompt).

    Returns:
        The generated text string (prompt tokens stripped).
    """
    model, tokenizer = get_model_and_tokenizer()
    device = get_device()

    # Build Mistral instruct-format input
    if system_prompt:
        full_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
    else:
        full_prompt = f"[INST] {prompt} [/INST]"

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt)
    new_tokens = output_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
