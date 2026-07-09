"""
confidence/sampling_scorer.py — Fix 5: RobertaModel loaded once, not per step.

The original code called bertscore() for every step, and bertscore() was
re-downloading and re-loading roberta-large from HuggingFace on every single
call (visible in the logs: 'Loading weights: 100%' printed for every step).
This happened because bert_score's internal model cache was not being reused
across calls in the way the original code assumed.

Fix: load the RobertaModel once at module level using the same transformers
singleton pattern used by model_loader.py, and pass the cached scorer
directly to each bertscore call via the `model_type` + explicit device args.
This eliminates the repeated download/reload overhead entirely.

Also: BERTScore doesn't support MPS → falls back to CPU for scoring even on
Mac, but on Ubuntu GPU machine it will use CUDA correctly.
"""
import logging
from itertools import combinations
from typing import List

import torch
from bert_score import BERTScorer
from model_loader import get_model_and_tokenizer, get_device

logger = logging.getLogger(__name__)

K              = 3
TEMPERATURE    = 0.8
MAX_NEW_TOKENS = 80

# ── Singleton BERTScorer (Fix 5: loaded once, reused every call) ──────────────
_bert_scorer: BERTScorer = None

def _get_bert_scorer() -> BERTScorer:
    global _bert_scorer
    if _bert_scorer is not None:
        return _bert_scorer
    # BERTScore doesn't support MPS; use cuda if available, else cpu
    bs_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading BERTScorer (roberta-large) on {bs_device} — loaded once, reused per run.")
    _bert_scorer = BERTScorer(
        model_type="roberta-large",
        lang="en",
        rescale_with_baseline=False,
        device=bs_device,
    )
    logger.info("BERTScorer loaded.")
    return _bert_scorer


def _generate_continuations(
    context: str,
    model,
    tokenizer,
    device: str,
    k: int = K,
) -> List[str]:
    inputs   = tokenizer(context, return_tensors="pt").to(device)
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
        new_tokens = output[0][input_len:]
        decoded    = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        continuations.append(decoded)

    return continuations


def _pairwise_bertscore(texts: List[str]) -> float:
    """Mean pairwise BERTScore F1. Uses the singleton scorer (Fix 5)."""
    pairs = list(combinations(range(len(texts)), 2))
    if not pairs:
        return 1.0

    scorer = _get_bert_scorer()
    refs, cands = [], []
    for i, j in pairs:
        cands.append(texts[i])
        refs.append(texts[j])

    # Score using the cached scorer directly — no reload
    _, _, f1 = scorer.score(cands, refs, verbose=False)
    return f1.mean().item()


def score_steps_sampling(
    question: str,
    steps: List[dict],
) -> List[float]:
    model, tokenizer = get_model_and_tokenizer()
    device           = get_device()

    scores  = []
    context = f"Question: {question}\n"

    for step in steps:
        prompt        = context + f"Step {step['step_id']}:"
        continuations = _generate_continuations(
            context=prompt, model=model, tokenizer=tokenizer,
            device=device, k=K,
        )
        score = _pairwise_bertscore(continuations)
        scores.append(round(score, 4))
        logger.debug(
            f"Step {step['step_id']}: BERTScore agreement={score:.4f} | "
            f"Continuations: {continuations}"
        )
        context += f"Step {step['step_id']}: {step['content']}\n"

    return scores
