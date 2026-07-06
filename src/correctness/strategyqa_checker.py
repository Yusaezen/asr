"""
StrategyQA correctness checker — Phase 2, Intern 2.

Method: NLI ENTAILMENT (same DeBERTa-v3-mnli model as hotpotqa_checker.py),
reusing the identical classify/margin logic for consistency across the two
NLI-based checkers (so Intern 3's calibration comparison isn't confounded by
two different decision rules).

Gold reference preference order: `facts` (evidence sentences) if the loaded
sample has them, else `decomposition` (gold sub-questions) as a fallback —
mirrors the "gold supporting fact or decomposition sub-step" wording in the
Phase 2 spec directly.
"""
import logging
from typing import List, Dict

from nli_model import nli_probs
from hotpotqa_checker import _classify  # shared decision rule, not duplicated

logger = logging.getLogger(__name__)


def check_strategyqa_correctness(
    steps: List[dict],
    facts: List[str],
    decomposition: List[str],
) -> Dict:
    """
    Args:
        steps: list of {'step_id', 'content', ...} — the SAME steps Intern 1
               scored (from {dataset}_confidence_scores.json).
        facts: gold evidence sentences for this question (preferred gold ref).
        decomposition: gold sub-question decomposition (fallback gold ref if
               `facts` is empty).

    Returns:
        {
          "per_step": [ ... same shape as hotpotqa_checker ... ],
          "gold_reference_used": "facts" | "decomposition" | "none",
          "gold_sentences_found": int,
        }
    """
    gold_sentences = facts if facts else decomposition
    ref_used = "facts" if facts else ("decomposition" if decomposition else "none")

    if not gold_sentences:
        logger.warning("No facts or decomposition available — cannot NLI-check this sample.")
        return {
            "per_step": [
                {"step_id": s["step_id"], "label": "unknown",
                 "method": "nli_entailment", "ambiguous": True,
                 "best_evidence": None, "entailment_prob": None,
                 "neutral_prob": None, "contradiction_prob": None}
                for s in steps
            ],
            "gold_reference_used": "none",
            "gold_sentences_found": 0,
        }

    per_step = []
    for step in sorted(steps, key=lambda s: s["step_id"]):
        hypothesis = step.get("content", "")
        best = {"entailment": -1.0, "neutral": 0.0, "contradiction": 0.0, "sentence": None}
        for gold_sent in gold_sentences:
            probs = nli_probs(premise=gold_sent, hypothesis=hypothesis)
            if probs["entailment"] > best["entailment"]:
                best = {**probs, "sentence": gold_sent}

        label, ambiguous = _classify(
            best["entailment"], best["neutral"], best["contradiction"]
        )
        per_step.append({
            "step_id": step["step_id"],
            "label": label,
            "method": "nli_entailment",
            "ambiguous": ambiguous,
            "best_evidence": best["sentence"],
            "entailment_prob": round(best["entailment"], 4),
            "neutral_prob": round(best["neutral"], 4),
            "contradiction_prob": round(best["contradiction"], 4),
        })

    return {
        "per_step": per_step,
        "gold_reference_used": ref_used,
        "gold_sentences_found": len(gold_sentences),
    }
