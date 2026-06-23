import json
import logging
import re
from typing import Optional

from schema import CoTDraft, ReasoningStep
from fallback import (
    repair_and_parse_json,
    parse_by_delimiter,
    parse_by_sentences,
    extract_final_answer,
    strip_markdown_fences,
)

logger = logging.getLogger(__name__)


def parse_model_output(
    raw_output: str,
    question: str,
    complexity: str,
    dataset: str = "musique",
) -> CoTDraft:
    r"""
    Parse priority:
      1. strict JSON
      2. repaired JSON  (fixes Mistral '\_' illegal-escape quirk) <-- recovers steps
      3. delimiter fallback ('Step N:')
      4. sentence fallback (last resort)
    """
    raw_clean = strip_markdown_fences(raw_output)

    # ── 1. strict JSON ────────────────────────────────────────────────────────
    try:
        data = json.loads(raw_clean)
        draft = CoTDraft(**data)
        draft.raw_output = raw_output
        draft.parse_method = "schema"
        logger.info("Parsed via JSON schema.")
        return draft
    except Exception as e:
        logger.warning(f"Strict JSON failed: {e}. Trying JSON repair.")

    # ── 2. repaired JSON (THE pilot fix) ──────────────────────────────────────
    data = repair_and_parse_json(raw_output)
    if data and isinstance(data, dict) and data.get("steps"):
        try:
            # the model sometimes nests the whole schema inside content; data
            # here is the top-level object, so build directly.
            draft = CoTDraft(
                question=data.get("question", question),
                dataset=data.get("dataset", dataset),
                complexity=data.get("complexity", complexity),
                steps=[ReasoningStep(**s) for s in data["steps"]],
                final_answer=str(data.get("final_answer", "")),
                raw_output=raw_output,
                parse_method="schema_repaired",
            )
            logger.info("Parsed via repaired JSON (recovered illegal escapes).")
            return draft
        except Exception as e:
            logger.warning(f"Repaired JSON had bad structure: {e}. Falling through.")

    # ── 3. delimiter fallback ─────────────────────────────────────────────────
    steps_raw = parse_by_delimiter(raw_output)
    parse_method = "fallback_delimiter"

    # ── 4. sentence fallback ──────────────────────────────────────────────────
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