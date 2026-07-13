"""
generation_harness.py — HuggingFace backend (Ollama removed).

Previously this module used Ollama (via model_client.py) for CoT generation.
Ollama is an opaque HTTP endpoint that does not expose hidden states, which
are required by UHead for step-level confidence scoring.

model_client.py now delegates to confidence/model_loader.py, which loads
Mistral-7B-Instruct in 4-bit NF4 quantization via bitsandbytes. This gives
the same ~4-5 GB memory footprint as Ollama while keeping full HuggingFace
Transformers access (forward hooks, output_hidden_states, model.generate).

Complexity classifier (Fix 2, unchanged):
    GSM8K uses a step-count heuristic based on the number of distinct numeric
    values in the question. All other datasets use the original keyword heuristic.
"""
import json
import logging
import re
import sys
import os
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from model_client import generate, check_ollama_running, list_available_models, DEFAULT_MODEL
from step_parser import parse_model_output
from prompts import build_prompt, SYSTEM_PROMPT as DATASET_SYSTEM_PROMPT
from schema import CoTDraft

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful multi-hop reasoning assistant.
Your task is to answer questions by breaking down your reasoning into clear, numbered steps.
You MUST respond with ONLY valid JSON — no prose, no markdown fences.
Follow the exact schema provided."""


def _build_prompt(question: str, complexity: str, n_steps: int,
                  answer: Optional[str] = None) -> str:
    schema_example = {
        "question": question,
        "dataset": "musique",
        "complexity": complexity,
        "steps": [
            {"step_id": 1, "content": "<first reasoning step>",  "confidence": 0.9},
            {"step_id": 2, "content": "<second reasoning step>", "confidence": 0.85},
            {"step_id": n_steps, "content": "<final reasoning step>", "confidence": 0.9},
        ],
        "final_answer": answer if answer else "<corresponding-answer-of-the-question>",
        "raw_output": "",
        "parse_method": "schema",
    }
    return (
        f"Question: {question}\n\n"
        f"This is a {complexity} multi-hop question requiring {n_steps} reasoning steps.\n"
        "Each step should bridge ONE inference.\n\n"
        f"Respond ONLY with this JSON structure:\n{json.dumps(schema_example, indent=2)}"
    )


# ── Numeric-count heuristic for GSM8K ─────────────────────────────────────────
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")

def _classify_gsm8k(question: str) -> tuple:
    """
    GSM8K-specific classifier. Counts distinct numeric values mentioned —
    more numbers usually means more arithmetic steps needed.
    simple ≤ 2 numbers (e.g. 'A robe takes 2 bolts and half that …')
    medium  3-4 numbers
    complex ≥ 5 numbers
    """
    nums = _NUM_RE.findall(question)
    n = len(set(nums))
    if n <= 2:
        return "simple", 2
    elif n <= 4:
        return "medium", 3
    else:
        return "complex", 4


# ── Original keyword heuristic for QA datasets ────────────────────────────────
_HOP_KEYWORDS = [
    "who", "where", "when", "which", "what", "born", "died",
    "founded", "located", "worked", "directed", "produced",
]

def _classify_qa(question: str) -> tuple:
    q_lower = question.lower()
    hop_count = sum(1 for kw in _HOP_KEYWORDS if kw in q_lower)
    word_count = len(question.split())
    if word_count < 15 and hop_count <= 2:
        return "medium", 2
    elif word_count < 30 and hop_count <= 4:
        return "complex", 3
    else:
        return "complex", 4


def classify_complexity(question: str, dataset: str = "musique") -> tuple:
    """
    Dataset-aware complexity classifier.
    GSM8K uses numeric-count heuristic; all others use keyword heuristic.
    Returns (complexity_label, suggested_n_steps).
    """
    if dataset.lower() == "gsm8k":
        return _classify_gsm8k(question)
    return _classify_qa(question)


def run(
    question: str,
    model: str = DEFAULT_MODEL,
    dataset: str = "musique",
    temperature: float = 0.2,
    output_path: Optional[str] = None,
) -> dict:
    # HuggingFace model loads lazily on first call — no connectivity check needed.
    # The `model` argument is accepted for API compatibility; model_client.py
    # always uses the HuggingFace singleton (Mistral-7B-Instruct-v0.2, 4-bit NF4).

    # Fix 2: pass dataset into classifier
    complexity, n_steps = classify_complexity(question, dataset=dataset)
    logger.info(f"Classified as [{complexity}] — expecting {n_steps} steps.")

    prompt = build_prompt(
        question, dataset=dataset, complexity=complexity, n_steps=n_steps
    )

    logger.info(f"Generating CoT draft with {model}...")
    raw_output = generate(
        prompt=prompt,
        model=model,
        temperature=temperature,
        system_prompt=DATASET_SYSTEM_PROMPT,
    )
    logger.info("Generation complete. Parsing output...")

    draft: CoTDraft = parse_model_output(
        raw_output=raw_output,
        question=question,
        complexity=complexity,
        dataset=dataset,
    )

    result = draft.model_dump()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Output saved to {output_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR — CoT Draft Generator")
    parser.add_argument("--question", type=str)
    parser.add_argument("--model",   type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--output",  type=str, default=None)
    parser.add_argument("--demo",    action="store_true")
    args = parser.parse_args()

    if args.demo:
        question = (
            "Who was the father of the person who directed the film "
            "that won the Academy Award for Best Picture in 1994?"
        )
    elif args.question:
        question = args.question
    else:
        parser.print_help()
        sys.exit(1)

    output_path = args.output or "../outputs/cot_draft.json"
    result = run(question=question, model=args.model,
                 dataset=args.dataset, output_path=output_path)
    print(json.dumps(result, indent=2))
