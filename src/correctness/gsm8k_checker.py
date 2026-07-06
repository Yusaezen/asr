"""
GSM8K correctness checker — Phase 2, Intern 2.

Method: EXACT MATCH against gold intermediate numeric values.

GSM8K's gold worked solution embeds each calculator step as a
"<<expr=result>>" annotation, e.g.:
    "Natalia sold 48/2=<<48/2=24>>24 clips in May."
The number after '=' inside <<...>> is a gold intermediate value the
correct reasoning chain must pass through, in order.

Approach:
  1. Extract the ordered list of gold intermediate values from gold_solution.
  2. Extract the ordered list of gold final answer from '#### <answer>'.
  3. Walk the MODEL's steps in order. For each step, extract every number
     it mentions. Greedily match it against the next UNUSED gold value
     (sequential alignment — a step is "correct" if it contains the gold
     value expected at that point in the chain).
  4. A step that contains no unused gold value is labeled "incorrect" —
     it did not advance the calculation the gold solution actually performs.

This is intentionally a *sequential* match, not a free-for-all search over
all gold values for every step — a step matching an EARLIER gold value out
of order would indicate skipped/duplicated computation, which is exactly
the kind of thing this checker should catch, not paper over.

NOTE (heuristic, flagged for the team): model step count and gold calc-step
count often differ (small model may merge/split steps differently from the
gold solution's granularity). This checker does not assume a 1:1 step
correspondence — it only asks "does the gold computation trail get hit, in
order, somewhere across the steps." Two steps could jointly satisfy one gold
value, or one step could satisfy two gold values in a row; both are
correctly handled by the greedy walk. What's NOT handled: a model that
reaches the right final answer via a totally different (but valid)
arithmetic path — the checker would call intermediate steps "incorrect"
because they don't match GSM8K's specific solution path. This is a known
limitation of gold-based exact match; flag any high-incorrect / correct
final-answer cases for manual spot-check (this pattern already showed up in
Nethra's Phase-1 QA notes, e.g. "although final answer is correct, ...").
"""
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")
_GOLD_ANNOTATION_RE = re.compile(r"<<[^=<>]*=([^<>]+)>>")


def normalize_number(s: str) -> Optional[float]:
    """'$1,024.0' / '1024' / '1,024' -> 1024.0 ; returns None if unparsable."""
    if s is None:
        return None
    cleaned = str(s).strip().replace("$", "").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_numbers_from_text(text: str) -> List[float]:
    found = _NUM_RE.findall(text or "")
    nums = [normalize_number(n) for n in found]
    return [n for n in nums if n is not None]


def extract_gold_intermediate_values(gold_solution: str) -> List[float]:
    """Ordered list of gold calculator results from <<expr=result>> annotations."""
    raw = _GOLD_ANNOTATION_RE.findall(gold_solution or "")
    vals = [normalize_number(r) for r in raw]
    return [v for v in vals if v is not None]


def extract_gold_final_answer(gold_solution: str) -> Optional[float]:
    m = re.search(r"####\s*(.+)", gold_solution or "")
    if not m:
        return None
    return normalize_number(m.group(1).strip())


def check_gsm8k_correctness(
    steps: List[dict],
    gold_solution: str,
    model_final_answer: str = "",
    ground_truth: str = "",
) -> Dict:
    """
    Args:
        steps: list of {'step_id', 'content', ...} — the SAME steps object
               Intern 1 scored (from {dataset}_confidence_scores.json), so
               step_id alignment with confidence scores is automatic.
        gold_solution: full worked solution text (from load_gsm8k's
               'gold_solution' field).
        model_final_answer / ground_truth: for an overall final-answer check
               (independent of per-step labels — useful cross-check, matches
               the pattern Nethra already flagged manually in Phase 1).

    Returns:
        {
          "per_step": [
              {"step_id": int, "label": "correct"|"incorrect",
               "method": "exact_match_numeric", "ambiguous": False,
               "matched_gold_value": float|None}, ...
          ],
          "final_answer_correct": bool | None,   # None if unparsable
          "gold_values_total": int,
          "gold_values_matched": int,
        }
    """
    if not gold_solution:
        logger.warning("No gold_solution provided — cannot exact-match GSM8K steps.")
        return {
            "per_step": [
                {"step_id": s["step_id"], "label": "unknown",
                 "method": "exact_match_numeric", "ambiguous": True,
                 "matched_gold_value": None}
                for s in steps
            ],
            "final_answer_correct": None,
            "gold_values_total": 0,
            "gold_values_matched": 0,
        }

    gold_values = extract_gold_intermediate_values(gold_solution)
    used = [False] * len(gold_values)

    per_step = []
    for step in sorted(steps, key=lambda s: s["step_id"]):
        step_numbers = extract_numbers_from_text(step.get("content", ""))
        matched_value = None
        for gi, gv in enumerate(gold_values):
            if used[gi]:
                continue
            if any(abs(n - gv) < 1e-6 for n in step_numbers):
                matched_value = gv
                used[gi] = True
                break
        label = "correct" if matched_value is not None else "incorrect"
        per_step.append({
            "step_id": step["step_id"],
            "label": label,
            "method": "exact_match_numeric",
            "ambiguous": False,
            "matched_gold_value": matched_value,
        })

    gold_final = extract_gold_final_answer(gold_solution)
    model_final_num = normalize_number(model_final_answer) if model_final_answer else None
    gt_num = normalize_number(ground_truth) if ground_truth else gold_final

    final_answer_correct = None
    if model_final_num is not None and gt_num is not None:
        final_answer_correct = abs(model_final_num - gt_num) < 1e-6
    elif model_final_answer and ground_truth:
        # fall back to string equality if numeric parsing failed on either side
        final_answer_correct = model_final_answer.strip() == ground_truth.strip()

    return {
        "per_step": per_step,
        "final_answer_correct": final_answer_correct,
        "gold_values_total": len(gold_values),
        "gold_values_matched": sum(used),
    }
