"""
Unified confidence scoring interface.

Usage:
    from scorer_interface import score_steps

    scores = score_steps(question, steps, method="logprob")
    scores = score_steps(question, steps, method="sampling")
    scores = score_steps(question, steps, method="both")  # returns dict

`method="both"` runs both scorers and returns:
    {
        "logprob": [...],
        "sampling": [...],
    }
"""
import logging
from typing import List, Union, Dict

logger = logging.getLogger(__name__)

ScorerMethod = str  # "logprob" | "sampling" | "both"


def score_steps(
    question: str,
    steps: List[dict],
    method: ScorerMethod = "both",
) -> Union[List[float], Dict[str, List[float]]]:
    """
    Args:
        question: The original question string.
        steps:    List of step dicts with keys 'step_id', 'content'.
        method:   "logprob" | "sampling" | "both"

    Returns:
        List[float] if method is "logprob" or "sampling".
        Dict[str, List[float]] if method is "both".
    """
    if method == "logprob":
        from logprob_scorer import score_steps_logprob
        return score_steps_logprob(question, steps)

    elif method == "sampling":
        from sampling_scorer import score_steps_sampling
        return score_steps_sampling(question, steps)

    elif method == "both":
        from logprob_scorer import score_steps_logprob
        from sampling_scorer import score_steps_sampling
        logger.info("Running logprob scorer...")
        lp_scores = score_steps_logprob(question, steps)
        logger.info("Running sampling scorer...")
        samp_scores = score_steps_sampling(question, steps)
        return {"logprob": lp_scores, "sampling": samp_scores}

    else:
        raise ValueError(f"Unknown method '{method}'. Choose: logprob | sampling | both")