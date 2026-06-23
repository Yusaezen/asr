"""
Loads GSM8K from HuggingFace datasets.
Returns a list of {id, question, answer, gold_answer, gold_complexity} dicts.

gold_complexity is a CROSS-CHECK signal (not an override): GSM8K gold solutions
embed each calculation as a <<a*b=c>> annotation, so the number of annotations
≈ number of reasoning steps. We bucket that into simple/medium/complex.
The classifier still runs independently; batch_runner stores both for comparison
(per the Phase-3 cross-check plan).
"""
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# step-count thresholds -> complexity tier (heuristic; team can tune)
_SIMPLE_MAX = 2     # <=2 calc steps  -> simple
_MEDIUM_MAX = 4     # 3-4 calc steps  -> medium ; >4 -> complex


def _gsm8k_complexity(answer_text: str) -> str:
    """Estimate complexity from the count of <<...>> calculator annotations."""
    n_steps = len(re.findall(r"<<.*?>>", answer_text))
    if n_steps == 0:
        # fall back to counting newline-separated reasoning lines
        n_steps = len([ln for ln in answer_text.split("\n") if ln.strip()])
    if n_steps <= _SIMPLE_MAX:
        return "simple"
    if n_steps <= _MEDIUM_MAX:
        return "medium"
    return "complex"


def _final_answer(answer_text: str) -> str:
    """GSM8K gold answers end with '#### <number>'."""
    m = re.search(r"####\s*(.+)", answer_text)
    return m.group(1).strip() if m else answer_text.strip()


def load_gsm8k(split: str = "test", limit: Optional[int] = None) -> List[Dict]:
    """
    split: 'train' | 'test'
    limit: cap number of samples (useful for pilot runs)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info(f"Loading GSM8K [{split}] from HuggingFace...")
    ds = load_dataset("openai/gsm8k", "main", split=split)
    samples = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        answer_text = row.get("answer", "")
        samples.append({
            "id": str(i),
            "question": row["question"],
            "answer": _final_answer(answer_text),     # clean final answer
            "gold_solution": answer_text,             # full worked solution (for step grading)
            "gold_complexity": _gsm8k_complexity(answer_text),
        })

    logger.info(f"Loaded {len(samples)} GSM8K samples.")
    return samples


if __name__ == "__main__":
    for s in load_gsm8k(split="test", limit=5):
        print(f"[{s['id']}] ({s['gold_complexity']}) {s['question'][:70]}")
        print(f"  Answer: {s['answer']}\n")
