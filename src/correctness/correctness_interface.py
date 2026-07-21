"""
Unified correctness-checking interface — Phase 2, Intern 2.

Mirrors the style of confidence/scorer_interface.py (Intern 1) so the two
packages feel consistent to anyone reading both.

Usage:
    from correctness_interface import score_correctness

    result = score_correctness(
        dataset="gsm8k",
        steps=steps,
        gold={"gold_solution": "...", "ground_truth": "72"},
    )
    result = score_correctness(
        dataset="hotpotqa",
        steps=steps,
        gold={"sample_id": "5a8b57f2..."},
    )
    result = score_correctness(
        dataset="strategyqa",
        steps=steps,
        gold={"facts": [...], "decomposition": [...]},
    )

Returns a dict with a "per_step" list — see each checker module for the
exact per-step fields (they differ slightly by method: exact_match_numeric
vs nli_entailment — Intern 3's merge step should key on step_id regardless
of method).
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def score_correctness(dataset: str, steps: List[dict], gold: Dict) -> Dict:
    dataset = dataset.lower()

    if dataset == "gsm8k":
        from gsm8k_checker import check_gsm8k_correctness
        return check_gsm8k_correctness(
            steps=steps,
            gold_solution=gold.get("gold_solution", ""),
            model_final_answer=gold.get("model_final_answer", ""),
            ground_truth=gold.get("ground_truth", ""),
        )

    elif dataset == "hotpotqa":
        from hotpotqa_checker import check_hotpotqa_correctness
        return check_hotpotqa_correctness(
            sample_id=gold.get("sample_id", ""),
            steps=steps,
            split=gold.get("split", "validation"),
        )

    elif dataset == "strategyqa":
        from strategyqa_checker import check_strategyqa_correctness
        return check_strategyqa_correctness(
            steps=steps,
            facts=gold.get("facts", []),
            decomposition=gold.get("decomposition", []),
        )
    # After the hotpotqa elif block, add:
    elif dataset == "musique":
        from musique_checker import check_musique_correctness
        return check_musique_correctness(
            sample_id=gold.get("sample_id", ""),
            steps=steps,
            split=gold.get("split", "validation"),
        )

    else:
        raise ValueError(
            f"Unknown dataset '{dataset}' for correctness checking. "
            f"Choose from: gsm8k | hotpotqa | strategyqa. "
            f"(musique has no gold intermediate-step annotation format yet — "
            f"flag to the team if it's needed for Phase 2.)"
        )
