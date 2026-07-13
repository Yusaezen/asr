"""
uhead/scorer.py

Inference-time UHead confidence scorer.
Runs frozen Mistral, extracts layer -1 hidden state at last token of each
step via forward hook. No output_hidden_states=True needed — hook handles it.
"""
import logging
import torch
from pathlib import Path
from typing import List, Optional

from confidence.model_loader import get_model_and_tokenizer, get_device
from uhead.model import UHead
from uhead.extractor import HiddenStateExtractor

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"
FINETUNE_CKPT = OUTPUTS_DIR / "uhead_finetuned.pt"
PRETRAIN_CKPT = OUTPUTS_DIR / "uhead_pretrained.pt"

_uhead: Optional[UHead] = None


def _load_uhead() -> UHead:
    global _uhead
    if _uhead is not None:
        return _uhead

    _uhead = UHead()
    ckpt = FINETUNE_CKPT if FINETUNE_CKPT.exists() else PRETRAIN_CKPT
    if ckpt.exists():
        _uhead.load_state_dict(torch.load(ckpt, map_location="cpu"))
        logger.info(f"UHead loaded from {ckpt}")
    else:
        logger.warning(
            "No UHead checkpoint found — scores will be random.\n"
            "Run: python uhead/train.py --pretrain"
        )
    _uhead.eval()
    return _uhead


def score_steps_uhead(question: str, steps: List[dict]) -> List[float]:
    """
    Returns UHead confidence scores (one per step) in [0, 1].
    Single forward pass per step — no sampling overhead.
    """
    model, tokenizer = get_model_and_tokenizer()
    device = get_device()
    uhead = _load_uhead()

    scores = []
    extractor = HiddenStateExtractor(model)
    context = f"Question: {question}\n"

    with extractor:
        for step in sorted(steps, key=lambda s: s["step_id"]):
            step_text = f"Step {step['step_id']}: {step['content']}"
            full_text = context + step_text

            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                _ = model(**inputs)  # hook fires here; output_hidden_states NOT needed

            state = extractor.last_token_state().unsqueeze(0).float()  # [1, 4096] on CPU

            with torch.no_grad():
                score = uhead(state).item()

            scores.append(round(score, 4))
            logger.debug(f"Step {step['step_id']}: uhead_score={score:.4f}")

            context += step_text + "\n"
            extractor._states.clear()  # safe inside `with` — hook still registered

    return scores
