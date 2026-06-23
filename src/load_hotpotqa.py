"""
Loads HotpotQA from HuggingFace datasets.
Returns a list of {id, question, answer, gold_complexity, hotpot_type,
supporting_titles} dicts.

gold_complexity is a CROSS-CHECK signal (not an override). HotpotQA labels each
question 'bridge' (sequential entity-hopping) or 'comparison' (parallel entity
contrast). We map that structure to a tier as a heuristic:
   comparison -> medium   (two independent lookups, then compare)
   bridge     -> complex  (sequential dependency: hop-2 needs hop-1's result)
The classifier still runs independently; batch_runner stores both for comparison.
NOTE: this bridge->complex / comparison->medium mapping is a deliberate heuristic,
not ground truth — flag for the team to validate in Phase 3.
"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_TYPE_TO_COMPLEXITY = {
    "comparison": "medium",
    "bridge": "complex",
}


def load_hotpotqa(split: str = "validation", limit: Optional[int] = None) -> List[Dict]:
    """
    split: 'train' | 'validation'
    limit: cap number of samples (useful for pilot runs)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info(f"Loading HotpotQA [{split}] from HuggingFace...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    samples = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        htype = row.get("type", "bridge")
        # supporting_facts.title = the gold entities the reasoning hops through
        sup = row.get("supporting_facts", {})
        titles = list(dict.fromkeys(sup.get("title", []))) if isinstance(sup, dict) else []
        samples.append({
            "id": row.get("id", str(i)),
            "question": row["question"],
            "answer": row.get("answer", ""),
            "gold_complexity": _TYPE_TO_COMPLEXITY.get(htype, "complex"),
            "hotpot_type": htype,
            "supporting_titles": titles,
        })

    logger.info(f"Loaded {len(samples)} HotpotQA samples.")
    return samples


if __name__ == "__main__":
    for s in load_hotpotqa(split="validation", limit=5):
        print(f"[{s['hotpot_type']} | {s['gold_complexity']}] {s['question'][:70]}")
        print(f"  Answer: {s['answer']}  | hops: {s['supporting_titles']}\n")
