"""
Log-probability scorer.

For each step, computes the mean token log-probability of the step's
content tokens, conditioned on the full prompt + prior steps as context.

Higher mean log-prob → model is more "certain" about that step.
Score is normalised to [0, 1] via sigmoid for comparability.
"""
import logging
import math
import torch
from typing import List

from model_loader import get_model_and_tokenizer, get_device

logger = logging.getLogger(__name__)


def _mean_logprob_for_text(
    context: str,
    continuation: str,
    model,
    tokenizer,
    device: str,
) -> float:
    """
    Compute mean token log-probability of `continuation` given `context`.
    """
    context_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    continuation_ids = tokenizer.encode(
        continuation, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    full_ids = torch.cat([context_ids, continuation_ids], dim=1)

    with torch.no_grad():
        outputs = model(full_ids, labels=full_ids)
        # outputs.logits shape: [1, seq_len, vocab_size]
        logits = outputs.logits

    # Shift: logits[i] predicts token[i+1]
    shift_logits = logits[0, :-1, :]  # [seq_len-1, vocab_size]
    shift_labels = full_ids[0, 1:]    # [seq_len-1]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Extract log-prob for each actual token
    token_log_probs = log_probs[
        torch.arange(len(shift_labels)), shift_labels
    ]

    # Only average over continuation tokens
    n_ctx = context_ids.shape[1]
    continuation_log_probs = token_log_probs[n_ctx - 1:]

    if len(continuation_log_probs) == 0:
        return 0.0

    mean_lp = continuation_log_probs.mean().item()
    return mean_lp


def _sigmoid(x: float) -> float:
    """Map log-prob (typically negative) to [0,1]."""
    return 1.0 / (1.0 + math.exp(-x))


def score_steps_logprob(
    question: str,
    steps: List[dict],
) -> List[float]:
    """
    Returns a list of confidence scores (one per step) using mean token
    log-probability. Scores are in [0, 1].

    steps: list of dicts with keys 'step_id', 'content'
    """
    model, tokenizer = get_model_and_tokenizer()
    device = get_device()

    scores = []
    # Build up context incrementally: question + all prior steps
    context = f"Question: {question}\n"

    for step in steps:
        step_text = f"Step {step['step_id']}: {step['content']}\n"
        mean_lp = _mean_logprob_for_text(
            context=context,
            continuation=step_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        score = _sigmoid(mean_lp)
        scores.append(round(score, 4))
        logger.debug(f"Step {step['step_id']}: mean_lp={mean_lp:.4f}, score={score:.4f}")
        # Extend context with this step for next iteration
        context += step_text

    return scores