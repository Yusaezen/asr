"""
Loads StrategyQA — needed for Phase 2 correctness checking (Intern 2).

StrategyQA has no single universally-mirrored HF dataset id, so this loader
tries a small list of known mirrors in order (same defensive pattern already
used in this repo for MuSiQue in fetch_exemplars.py). First one that loads
successfully wins.

Returns a list of dicts:
    id, question, answer ('yes'/'no'), facts (list[str] evidence sentences),
    decomposition (list[str] sub-questions, if available), gold_complexity.

gold_complexity is a CROSS-CHECK signal only (same convention as
load_gsm8k.py / load_hotpotqa.py) — derived from len(decomposition).
"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Tried in order; first that loads successfully is used. Documented here so
# the team can see exactly which mirror ended up being used (printed at load time).
_MIRRORS = [
    ("ChilleD/StrategyQA", None),
    ("wics/strategy-qa", None),
    ("tasksource/strategy-qa", None),
]

_SIMPLE_MAX = 2   # <=2 decomposition steps -> simple
_MEDIUM_MAX = 4   # 3-4 -> medium ; >4 -> complex


def _complexity_from_decomposition(decomposition: List[str]) -> str:
    n = len(decomposition) if decomposition else 0
    if n == 0:
        return "unknown"
    if n <= _SIMPLE_MAX:
        return "simple"
    if n <= _MEDIUM_MAX:
        return "medium"
    return "complex"


def _normalize_row(row: dict, idx: int) -> Optional[Dict]:
    """Best-effort normalization across the differing mirror schemas."""
    question = row.get("question")
    if not question:
        return None

    # answer may be bool, 'yes'/'no', or int(0/1) depending on mirror
    raw_answer = row.get("answer")
    if isinstance(raw_answer, bool):
        answer = "yes" if raw_answer else "no"
    elif isinstance(raw_answer, (int, float)):
        answer = "yes" if raw_answer else "no"
    elif isinstance(raw_answer, str):
        answer = raw_answer.strip().lower()
    else:
        answer = "unknown"

    facts = row.get("facts") or row.get("evidence") or []
    if isinstance(facts, str):
        facts = [facts]
    # some mirrors nest facts as list-of-list (one list per annotator) — flatten
    flat_facts = []
    for f in facts:
        if isinstance(f, list):
            flat_facts.extend(f)
        else:
            flat_facts.append(f)

    decomposition = row.get("decomposition") or []
    if isinstance(decomposition, str):
        decomposition = [decomposition]

    return {
        "id": str(row.get("qid", row.get("id", idx))),
        "question": question,
        "answer": answer,
        "facts": [f for f in flat_facts if isinstance(f, str) and f.strip()],
        "decomposition": [d for d in decomposition if isinstance(d, str) and d.strip()],
        "gold_complexity": _complexity_from_decomposition(decomposition),
    }


def load_strategyqa(split: str = "test", limit: Optional[int] = None) -> List[Dict]:
    """
    split: 'test' | 'train' (falls back to 'train' if the mirror has no test split)
    limit: cap number of samples (useful for pilot runs)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    last_err = None
    for repo, cfg in _MIRRORS:
        for try_split in ([split, "train"] if split != "train" else ["train"]):
            try:
                logger.info(f"Trying StrategyQA mirror '{repo}' split='{try_split}'...")
                if cfg:
                    ds = load_dataset(repo, cfg, split=try_split)
                else:
                    ds = load_dataset(repo, split=try_split)
                logger.info(f"Loaded StrategyQA from '{repo}' (split={try_split}).")
                samples = []
                for i, row in enumerate(ds):
                    if limit and i >= limit:
                        break
                    norm = _normalize_row(dict(row), i)
                    if norm:
                        samples.append(norm)
                logger.info(f"Loaded {len(samples)} StrategyQA samples.")
                return samples
            except Exception as e:
                last_err = e
                logger.warning(f"  mirror '{repo}' split='{try_split}' failed: {str(e)[:100]}")
                continue

    raise RuntimeError(
        f"Could not load StrategyQA from any known mirror. Last error: {last_err}\n"
        f"Tell the team — a mirror id may need updating in load_strategyqa.py._MIRRORS."
    )


if __name__ == "__main__":
    for s in load_strategyqa(split="test", limit=5):
        print(f"[{s['id']}] ({s['gold_complexity']}) {s['question'][:70]}")
        print(f"  Answer: {s['answer']}  | facts: {len(s['facts'])}  | decomp: {s['decomposition']}\n")
