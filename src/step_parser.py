import json
import logging
import re
from typing import Optional

from schema import CoTDraft, ReasoningStep
from fallback import parse_by_delimiter, parse_by_sentences, extract_final_answer

logger = logging.getLogger(__name__)


def _strip_markdown_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def parse_model_output(
    raw_output: str,
    question: str,
    complexity: str,
    dataset: str = "musique",
) -> CoTDraft:
    """
    Attempt to parse model output into a CoTDraft.
    Priority: JSON schema parse → delimiter fallback → sentence fallback.
    """
    raw_clean = _strip_markdown_fences(raw_output)

    # ── 1. Try strict JSON parse ──────────────────────────────────────────────
    try:
        data = json.loads(raw_clean)
        draft = CoTDraft(**data)
        draft.raw_output = raw_output
        draft.parse_method = "schema"
        logger.info("Parsed via JSON schema.")
        return draft
    except Exception as e:
        logger.warning(f"JSON parse failed: {e}. Trying delimiter fallback.")

    # ── 2. Delimiter fallback ('Step N:') ─────────────────────────────────────
    steps_raw = parse_by_delimiter(raw_output)
    parse_method = "fallback_delimiter"

    if not steps_raw:
        logger.warning("Delimiter fallback failed. Trying sentence segmentation.")
        steps_raw = parse_by_sentences(raw_output)
        parse_method = "fallback_sentence"

    steps = [ReasoningStep(**s) for s in steps_raw] if steps_raw else []
    final_answer = extract_final_answer(raw_output)

    return CoTDraft(
        question=question,
        dataset=dataset,
        complexity=complexity,
        steps=steps,
        final_answer=final_answer,
        raw_output=raw_output,
        parse_method=parse_method,
    )
