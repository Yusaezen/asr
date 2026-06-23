import json
import logging
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

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a careful multi-hop reasoning assistant.
Your task is to answer questions by breaking down your reasoning into clear, numbered steps.
You MUST respond with ONLY valid JSON — no prose, no markdown fences.
Follow the exact schema provided."""

def _build_prompt(question: str, complexity: str, n_steps: int, answer: Optional[str]=None) -> str:
    schema_example = {
        "question": question,
        "dataset": "musique",
        "complexity": complexity,
        "steps": [
            {"step_id": 1, "content": "<first reasoning step>", "confidence": 0.9},
            {"step_id": 2, "content": "<second reasoning step>", "confidence": 0.85},
            {"step_id": n_steps, "content": "<final reasoning step>", "confidence": 0.9},
        ],
        "final_answer": answer if answer else "<corresponding-answer-of-the-question",
        "raw_output": "",
        "parse_method": "schema",
    }

    return f"""Question: {question}

This is a {complexity} multi-hop question requiring {n_steps} reasoning steps.
Each step should bridge ONE inference (e.g. identify an entity, retrieve a fact, make a connection).

Respond ONLY with this JSON structure (no extra text):
{json.dumps(schema_example, indent=2)}"""


def classify_complexity(question: str) -> tuple[str, int]:
    """
    Heuristic complexity classifier based on question length and hop indicators.
    Returns (complexity_label, suggested_n_steps).
    For Phase 1 all MuSiQue questions are multi-hop — treated as medium/complex.
    """
    hop_keywords = ["who", "where", "when", "which", "what", "born", "died",
                    "founded", "located", "worked", "directed", "produced"]
    q_lower = question.lower()
    hop_count = sum(1 for kw in hop_keywords if kw in q_lower)
    word_count = len(question.split())

    if word_count < 15 and hop_count <= 2:
        return "medium", 2
    elif word_count < 30 and hop_count <= 4:
        return "complex", 3
    else:
        return "complex", 4


def run(
    question: str,
    model: str = DEFAULT_MODEL,
    dataset: str = "musique",
    temperature: float = 0.2,
    output_path: Optional[str] = None,
) -> dict:
    """
    Main entry point.
    Takes a question string, returns a CoTDraft as a dict (JSON-serialisable).
    """
    if not check_ollama_running():
        raise ConnectionError(
            "Ollama is not running. Please run: ollama serve\n"
            f"Then pull the model: ollama pull {model}"
        )

    available = list_available_models()
    if model not in available:
        logger.warning(
            f"Model '{model}' not found in Ollama. Available: {available}\n"
            f"Pull it with: ollama pull {model}"
        )

    complexity, n_steps = classify_complexity(question)
    logger.info(f"Classified as [{complexity}] — expecting {n_steps} steps.")

    prompt = build_prompt(question, dataset=dataset, complexity=complexity, n_steps=n_steps)

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


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR — CoT Draft Generator")
    parser.add_argument("--question", type=str, help="Question string")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON output")
    parser.add_argument("--demo", action="store_true", help="Run with a sample MuSiQue question")
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

    output_path = args.output or f"../outputs/cot_draft.json"

    result = run(question=question, model=args.model, output_path=output_path)
    print(json.dumps(result, indent=2))