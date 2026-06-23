r"""
fallback.py  — Intern 2 deliverable (Fallback Logic)
Hardened against the failure modes observed in the MuSiQue pilot.

Parse priority used by step_parser:
    1. strict json.loads            (handled in step_parser)
    2. repair_and_parse_json  <-- NEW: fixes illegal escapes etc., re-parses
    3. parse_by_delimiter           ('Step N:' patterns)
    4. parse_by_sentences           (last resort)

PILOT FINDING (why repair_and_parse_json exists):
    Mistral emits Markdown-style escaped underscores in its JSON keys/strings,
    e.g.  "step\\_id"  and  "final\\_answer".  '\_' is NOT a legal JSON escape,
    so json.loads dies at that exact character (the recurring
    'Invalid \escape: line 7 column 6' seen on 7/10 pilot samples).
    The JSON is otherwise structurally valid, so we repair the illegal
    escapes and recover the FULL step list instead of falling through to
    sentence-splitting (which loses the step structure entirely).
r"""
import re
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def _clean(text: str) -> str:
    return text.strip().strip('"').strip()


# ── 0. Markdown fence stripper (shared) ────────────────────────────────────────
def strip_markdown_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


# ── 1. JSON repair (the main pilot fix) ────────────────────────────────────────
# Legal JSON escapes after a backslash:  " \ / b f n r t u
# Anything else (\_  \$  \(  \%  ...) is illegal and is what breaks Mistral output.
_ILLEGAL_ESCAPE = re.compile(r'\\([^"\\/bfnrtu])')

def repair_json_string(raw: str) -> str:
    r"""
    Repair the common, recoverable defects seen in small-model JSON:
      - illegal backslash escapes (\_ , \$ , \( ...) -> drop the backslash
      - markdown code fences
      - leading/trailing prose around the JSON object
    Returns a string that is *much* more likely to parse. Does NOT guarantee
    valid JSON — caller still wraps json.loads in try/except.
    r"""
    s = strip_markdown_fences(raw)

    # Keep only the outermost {...} block if the model added prose around it.
    first, last = s.find("{"), s.rfind("}")
    if first != -1 and last != -1 and last > first:
        s = s[first:last + 1]

    # Fix illegal escapes: '\_' -> '_', '\$' -> '$', etc.
    # (we simply remove the stray backslash before a non-escape char)
    s = _ILLEGAL_ESCAPE.sub(r"\1", s)

    return s


def repair_and_parse_json(raw: str) -> Optional[Dict]:
    """
    Attempt to repair then parse. Returns the parsed dict on success,
    or None if it still isn't valid JSON.
    """
    repaired = repair_json_string(raw)
    try:
        return json.loads(repaired)
    except Exception as e:
        logger.debug(f"repair_and_parse_json still failed: {e}")
        return None


# ── 2. Delimiter fallback ('Step N:' / 'N.') ───────────────────────────────────
def parse_by_delimiter(raw: str) -> List[Dict]:
    pattern = re.compile(
        r"(?:Step\s*(\d+)[:\.\)]\s*|^(\d+)[:\.\)]\s+)", re.MULTILINE | re.IGNORECASE
    )
    matches = list(pattern.finditer(raw))
    if not matches:
        return []
    steps = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        content = _clean(raw[start:end])
        if content:
            step_num = int(match.group(1) or match.group(2))
            steps.append({"step_id": step_num, "content": content, "confidence": 1.0})
    return steps


# ── 3. Sentence fallback (last resort) ─────────────────────────────────────────
def parse_by_sentences(raw: str, min_length: int = 20) -> List[Dict]:
    # If raw still looks like a JSON blob, strip the structural noise first so
    # we don't emit '{' and '"steps": [' as if they were reasoning sentences.
    cleaned = re.sub(r'[{}\[\]]', ' ', raw)
    cleaned = re.sub(r'"\w+"\s*:', ' ', cleaned)   # drop  "key":
    sentences = re.split(r"(?<=[.!?])\s+", cleaned.strip())
    steps = []
    step_id = 1
    for sent in sentences:
        sent = _clean(sent)
        if len(sent) >= min_length:
            steps.append({"step_id": step_id, "content": sent, "confidence": 1.0})
            step_id += 1
    return steps


# ── 4. Final-answer extractor ──────────────────────────────────────────────────
def extract_final_answer(raw: str) -> str:
    # First: if a repaired JSON has final_answer, prefer that.
    parsed = repair_and_parse_json(raw)
    if parsed and isinstance(parsed, dict):
        fa = parsed.get("final_answer")
        if fa and not str(fa).startswith("<"):   # ignore skeleton placeholders
            return _clean(str(fa))

    patterns = [
        r"(?:final answer|answer)[\":\s]+(.+)",
        r"therefore[,\s]+(?:the answer is\s*)?(.+)",
        r"(?:thus|hence)[,\s]+(.+)",
        r"in conclusion[,\s]+(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            return _clean(m.group(1).split("\n")[0])
    sentences = re.split(r"(?<=[.!?])\s+", raw.strip())
    return _clean(sentences[-1]) if sentences else ""