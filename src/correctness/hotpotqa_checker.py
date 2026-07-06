"""
HotpotQA correctness checker — Phase 2, Intern 2.

Method: NLI ENTAILMENT (DeBERTa-v3-mnli) — is each model reasoning step
entailed by a gold supporting-fact sentence?

WHY a separate gold-fact fetch instead of reusing load_hotpotqa.py:
load_hotpotqa.py (Phase 1, already merged) only keeps `supporting_titles`
(the entity/article titles the gold reasoning hops through) — it does NOT
keep the actual gold SENTENCE TEXT, because Phase 1 only needed titles for
the complexity cross-check. NLI needs real sentence text to compare against,
so this module fetches supporting_facts + context directly from the raw
HotpotQA dataset and caches it by sample id, without touching the merged
Phase 1 loader (avoids destabilizing code Phase 1 already depends on).

Ambiguity rule (per the Phase-2 spec — flag anything not a clear
entailment/contradiction for manual review):
    entailment_prob > 0.5 and entailment_prob - contradiction_prob > MARGIN
        -> "correct"
    contradiction_prob > 0.5 and contradiction_prob - entailment_prob > MARGIN
        -> "incorrect"
    otherwise (neutral dominant, or no clear margin)
        -> "ambiguous"  (ambiguous=True, needs manual review)
"""
import logging
from typing import List, Dict, Optional

from nli_model import nli_probs

logger = logging.getLogger(__name__)

MARGIN = 0.15  # required probability gap between top two classes to be "clear"

# lazy, module-level cache: {sample_id: {"gold_sentences": [str, ...]}}
_gold_cache: Dict[str, List[str]] = {}
_dataset_cache = None  # the raw HF dataset, loaded once


def _load_raw_hotpotqa(split: str = "validation"):
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache
    from datasets import load_dataset
    logger.info(f"Loading raw HotpotQA [{split}] for gold supporting-fact text...")
    _dataset_cache = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    # index by id for O(1)-ish lookup (HF Dataset supports .filter but that's
    # slow per-call; build a dict once)
    logger.info("Indexing HotpotQA rows by id...")
    _dataset_cache = {row["id"]: row for row in _dataset_cache}
    return _dataset_cache


def get_gold_supporting_sentences(sample_id: str, split: str = "validation") -> List[str]:
    """
    Returns the list of actual gold supporting-fact SENTENCES (text, not just
    titles) for a given HotpotQA question id. Cached after first computation.
    """
    if sample_id in _gold_cache:
        return _gold_cache[sample_id]

    ds_by_id = _load_raw_hotpotqa(split=split)
    row = ds_by_id.get(sample_id)
    if row is None:
        logger.warning(f"HotpotQA id {sample_id} not found in split='{split}'.")
        _gold_cache[sample_id] = []
        return []

    context = row.get("context", {})
    ctx_titles = context.get("title", [])
    ctx_sentences = context.get("sentences", [])
    title_to_sents = dict(zip(ctx_titles, ctx_sentences))

    sup = row.get("supporting_facts", {})
    sup_titles = sup.get("title", [])
    sup_sent_ids = sup.get("sent_id", [])

    gold_sentences = []
    for title, sent_idx in zip(sup_titles, sup_sent_ids):
        sents = title_to_sents.get(title, [])
        if 0 <= sent_idx < len(sents):
            gold_sentences.append(sents[sent_idx])

    _gold_cache[sample_id] = gold_sentences
    return gold_sentences


def _classify(entailment: float, neutral: float, contradiction: float) -> tuple:
    """
    Returns (label, ambiguous: bool).

    Margin is checked against the STRONGEST competing class (not just
    contradiction specifically) — e.g. entailment=0.55 vs neutral=0.40 is a
    close call and should be flagged ambiguous even though contradiction is
    low, because neutral being that competitive means the model isn't
    confidently entailing either. This was caught by a unit test during
    development (entailment=0.55/neutral=0.40/contradiction=0.05 was wrongly
    accepted as a clear 'correct' before this fix).
    """
    if entailment > 0.5:
        runner_up = max(neutral, contradiction)
        if entailment - runner_up > MARGIN:
            return "correct", False
        return "ambiguous", True
    if contradiction > 0.5:
        runner_up = max(neutral, entailment)
        if contradiction - runner_up > MARGIN:
            return "incorrect", False
        return "ambiguous", True
    return "ambiguous", True


def check_hotpotqa_correctness(
    sample_id: str,
    steps: List[dict],
    split: str = "validation",
) -> Dict:
    """
    Args:
        sample_id: HotpotQA question id (matches 'sample_id' in
                   {dataset}_confidence_scores.json, which was set from the
                   loader's 'id' field via batch_runner).
        steps: list of {'step_id', 'content', ...} — the SAME steps Intern 1
               scored, so step_id alignment is automatic.
        split: which HotpotQA split to pull gold supporting facts from.

    Returns:
        {
          "per_step": [
              {"step_id", "label", "method": "nli_entailment", "ambiguous",
               "best_evidence": str|None, "entailment_prob", "neutral_prob",
               "contradiction_prob"}, ...
          ],
          "gold_sentences_found": int,
        }
    """
    gold_sentences = get_gold_supporting_sentences(sample_id, split=split)

    if not gold_sentences:
        logger.warning(f"No gold supporting sentences found for {sample_id}.")
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
        # Check this step against EVERY gold supporting sentence, keep the
        # best (highest-entailment) match — HotpotQA supporting facts aren't
        # ordered to correspond to arbitrary model step counts, so best-match
        # is the safe choice rather than assuming positional alignment.
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
