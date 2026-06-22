"""
Loads MuSiQue from HuggingFace datasets.
Returns a list of {id, question, answer} dicts for use in batch generation.
"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def load_musique(split: str = "validation", limit: Optional[int] = None) -> List[Dict]:
    """
    split: 'train' | 'validation'
    limit: cap number of samples (useful for pilot runs)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info(f"Loading MuSiQue [{split}] from HuggingFace...")
    ds = load_dataset("Salesforce/ContextualBench", "MuSiQue", split="validation") 
    samples = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        samples.append({
            "id": row.get("id", str(i)),
            "question": row["question"],
            "answer": row.get("answer", ""),
        })

    logger.info(f"Loaded {len(samples)} MuSiQue samples.")
    return samples


if __name__ == "__main__":
    samples = load_musique(split="validation", limit=5)
    for s in samples:
        print(f"[{s['id']}] {s['question']}")
        print(f"  Answer: {s['answer']}\n")
