import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def _clean(text: str) -> str:
    return text.strip().strip('"').strip()


def parse_by_delimiter(raw: str) -> List[Dict]:
    """
    Primary fallback: detect 'Step N:' or 'N.' patterns in raw text.
    Returns list of {step_id, content} dicts.
    """
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


def parse_by_sentences(raw: str, min_length: int = 20) -> List[Dict]:
    """
    Last-resort fallback: split on sentence boundaries.
    Filters very short fragments.
    """
    sentences = re.split(r"(?<=[.!?])\s+", raw.strip())
    steps = []
    step_id = 1
    for sent in sentences:
        sent = _clean(sent)
        if len(sent) >= min_length:
            steps.append({"step_id": step_id, "content": sent, "confidence": 1.0})
            step_id += 1
    return steps


def extract_final_answer(raw: str) -> str:
    """
    Try to find a final answer line in raw text output.
    Looks for patterns like 'Answer:', 'Final answer:', 'Therefore, ...'
    """
    patterns = [
        r"(?:final answer|answer)[:\s]+(.+)",
        r"therefore[,\s]+(?:the answer is\s*)?(.+)",
        r"(?:thus|hence)[,\s]+(.+)",
        r"in conclusion[,\s]+(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            return _clean(m.group(1).split("\n")[0])
    # last sentence as fallback
    sentences = re.split(r"(?<=[.!?])\s+", raw.strip())
    return _clean(sentences[-1]) if sentences else ""
