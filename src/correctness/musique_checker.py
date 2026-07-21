"""
MuSiQue correctness checker — Developed to replace HotPotQA-based fine-tuning. We will be using HotPotQA and GSM8K as the standard datasets for measuring performance.

Method: NLI ENTAILMENT (same DeBERTa-v3-mnli + _classify logic as hotpotqa_checker)

MuSiQue schema (allenai/musique, validation split):
  - id:                  str
  - question:            str
  - answer:              str
  - paragraphs:          list of {idx, title, paragraph_text, is_supporting}
  - question_decomposition: list of {id, question, answer, paragraph_support_idx}

Gold reference strategy:
  1. Extract supporting paragraph sentences (is_supporting=True) — direct evidence
  2. Also extract decomposed sub-question answers as additional reference strings
  Best-match NLI across all references per step (same as hotpotqa_checker).

sample_id alignment: MuSiQue uses content-based ids (e.g. '2hop__123456_789')
set by load_musique.py from row['id'] — robust, not positional.
"""
import logging
from typing import List, Dict

from nli_model import nli_probs
from hotpotqa_checker import _classify  # shared decision rule

logger = logging.getLogger(__name__)

_gold_cache: Dict[str, List[str]] = {}
_dataset_cache = None


def _load_raw_musique(split: str = "validation"):
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache
    from datasets import load_dataset
    logger.info(f"Loading raw MuSiQue [{split}] for gold supporting sentences...")
    ds = load_dataset("allenai/musique", split=split)
    _dataset_cache = {row["id"]: row for row in ds}
    logger.info(f"Indexed {len(_dataset_cache)} MuSiQue rows.")
    return _dataset_cache


def get_gold_supporting_sentences(sample_id: str, split: str = "validation") -> List[str]:
    """
    Returns gold reference strings for a MuSiQue sample:
      - All sentences from supporting paragraphs (is_supporting=True)
      - All decomposed sub-question answers (non-empty)
    Cached after first fetch.
    """
    if sample_id in _gold_cache:
        return _gold_cache[sample_id]

    ds_by_id = _load_raw_musique(split=split)
    row = ds_by_id.get(sample_id)
    if row is None:
        logger.warning(f"MuSiQue id '{sample_id}' not found in split='{split}'.")
        _gold_cache[sample_id] = []
        return []

    gold_sentences = []

    # 1. Supporting paragraph sentences
    for para in row.get("paragraphs", []):
        if para.get("is_supporting", False):
            text = para.get("paragraph_text", "").strip()
            if text:
                # Split into sentences on period boundaries for finer NLI granularity
                sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
                gold_sentences.extend(sentences)

    # 2. Decomposed sub-question answers as additional reference strings
    for decomp in row.get("question_decomposition", []):
        ans = decomp.get("answer", "").strip()
        q = decomp.get("question", "").strip()
        if ans:
            # Combine sub-question + answer for richer entailment context
            gold_sentences.append(f"{q} {ans}".strip() if q else ans)

    _gold_cache[sample_id] = gold_sentences
    return gold_sentences


def check_musique_correctness(
    sample_id: str,
    steps: List[dict],
    split: str = "validation",
) -> Dict:
    """
    Args:
        sample_id: MuSiQue question id from batch_results_musique.json
        steps:     list of {'step_id', 'content', ...}
        split:     HF split to fetch gold from

    Returns:
        {
          "per_step": [
              {"step_id", "label", "method": "nli_entailment", "ambiguous",
               "best_evidence", "entailment_prob", "neutral_prob",
               "contradiction_prob"}, ...
          ],
          "gold_sentences_found": int,
        }
    """
    gold_sentences = get_gold_supporting_sentences(sample_id, split=split)

    if not gold_sentences:
        logger.warning(f"No gold sentences found for MuSiQue id {sample_id}.")
        return {
            "per_step": [
                {"step_id": s["step_id"], "label": "unknown",
                 "method": "nli_entailment", "ambiguous": True,
                 "best_evidence": None, "entailment_prob": None,
                 "neutral_prob": None, "contradiction_prob": None}
                for s in steps
            ],
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
        "gold_sentences_found": len(gold_sentences),
    }