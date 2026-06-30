"""
Sampling-based disagreement scorer.

For each step, generate k=3 continuations from the model at temperature=0.8.
Compute pairwise BERTScore F1 across the k continuations.
Confidence = mean pairwise BERTScore (high agreement → high confidence).

Low pairwise similarity → model is uncertain → low confidence.
"""
import logging
from itertools import combinations
from typing import List

import torch
from bert_score import score as bertscore
from model_loader import get_model_and_tokenizer, get_device

logger = logging.getLogger(__name__)

K = 3
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 80


def _generate_continuations(
    context: str,
    model,
    tokenizer,
    device: str,
    k: int = K,
) -> List[str]:
    """
    Generate k continuations from `context` at high temperature.
    Returns list of k decoded strings (continuation only, not context).
    """
    inputs = tokenizer(context, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    continuations = []
    for _ in range(k):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_tokens = output[0][input_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        continuations.append(decoded)

    return continuations


def _pairwise_bertscore(texts: List[str]) -> float:
    """
    Compute mean pairwise BERTScore F1 across all pairs in texts.
    Returns a float in [0, 1].
    """
    pairs = list(combinations(range(len(texts)), 2))
    if not pairs:
        return 1.0

    refs, cands = [], []
    for i, j in pairs:
        cands.append(texts[i])
        refs.append(texts[j])

    # BERTScore returns P, R, F1 tensors
    _, _, f1 = bertscore(
        cands,
        refs,
        lang="en",
        verbose=False,
        device=get_device() if get_device() != "mps" else "cpu",
        # BERTScore doesn't support MPS; fall back to CPU
    )
    return f1.mean().item()


def score_steps_sampling(
    question: str,
    steps: List[dict],
) -> List[float]:
    """
    Returns a list of confidence scores (one per step) using sampling
    disagreement + BERTScore. Scores are in [0, 1].

    steps: list of dicts with keys 'step_id', 'content'
    """
    model, tokenizer = get_model_and_tokenizer()
    device = get_device()

    scores = []
    context = f"Question: {question}\n"

    for step in steps:
        # Build the context up to (but not including) this step
        prompt = context + f"Step {step['step_id']}:"

        continuations = _generate_continuations(
            context=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            k=K,
        )

        score = _pairwise_bertscore(continuations)
        scores.append(round(score, 4))
        logger.debug(
            f"Step {step['step_id']}: BERTScore agreement={score:.4f}\n"
            f"  Continuations: {continuations}"
        )

        # Extend context with the actual (generated) step content
        context += f"Step {step['step_id']}: {step['content']}\n"

    return scores