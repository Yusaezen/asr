"""
confidence/scorer_interface.py  (updated for Phase 2 UHead)

Primary method: "uhead"
Ablation baselines (commented out in driver — re-enable to compare):
  "logprob"  — mean token log-probability (Phase 2 baseline)
  "sampling" — k=3 BERTScore disagreement (Phase 2 baseline)
  "both"     — runs logprob + sampling together (ablation only)
"""
import logging
import sys
from pathlib import Path
from typing import List, Union, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

ScorerMethod = str  # "uhead" | "logprob" | "sampling" | "both"


def score_steps(
    question: str,
    steps: List[dict],
    method: ScorerMethod = "uhead",
) -> Union[List[float], Dict[str, List[float]]]:
    """
    Args:
        question: Original question string.
        steps:    List of step dicts with keys 'step_id', 'content'.
        method:   "uhead" (default) | "logprob" | "sampling" | "both"

    Returns:
        List[float]              for "uhead", "logprob", "sampling"
        Dict[str, List[float]]   for "both"
    """

    if method == "uhead":
        from uhead.scorer import score_steps_uhead
        return score_steps_uhead(question, steps)

    # ── Ablation baselines ────────────────────────────────────────────────────
    # Uncomment the relevant block to re-enable baseline comparison.

    elif method == "logprob":
        from logprob_scorer import score_steps_logprob
        return score_steps_logprob(question, steps)

    elif method == "sampling":
        from sampling_scorer import score_steps_sampling
        return score_steps_sampling(question, steps)

    elif method == "both":
        # Runs logprob + sampling (ablation pair — does NOT include uhead)
        from logprob_scorer import score_steps_logprob
        from sampling_scorer import score_steps_sampling
        logger.info("Running logprob scorer (ablation)...")
        lp = score_steps_logprob(question, steps)
        logger.info("Running sampling scorer (ablation)...")
        samp = score_steps_sampling(question, steps)
        return {"logprob": lp, "sampling": samp}

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose: uhead | logprob | sampling | both"
        )