"""
Singleton HF model + tokenizer loader.
Loads Mistral-7B-Instruct-v0.2 in float16 on MPS (M2 Mac).
Call get_model_and_tokenizer() — subsequent calls return cached instance.
"""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

_model = None
_tokenizer = None


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model_and_tokenizer():
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    device = get_device()
    logger.info(f"Loading {MODEL_ID} on {device} in float16...")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    _model.eval()
    logger.info("Model loaded.")
    return _model, _tokenizer