"""
confidence/model_loader.py — GPU-ready singleton loader.

Fix 3: device detection now correctly prioritises CUDA (Ubuntu GPU machine).
       The original code used device_map=device which fails on CPU without
       accelerate — changed to device_map='auto' so HuggingFace handles
       placement correctly whether on CPU, CUDA, or MPS.
Fix 4: torch_dtype deprecation warning silenced (use dtype= instead).
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
        dtype=torch.float16,      # fix: dtype= not torch_dtype=
        device_map="auto",        # fix: auto handles GPU/CPU placement correctly
    )
    _model.eval()
    logger.info("Model loaded.")
    return _model, _tokenizer
