"""
prompts.py  — Intern 2 deliverable (Prompt Design)
Dataset-aware few-shot prompt construction for the ASR CoT drafter.

Plugs into generation_harness.run(): replaces the single hardcoded
_build_prompt with a dataset-routed version. Matches schema.py exactly
(step_id / content / confidence ; final_answer).

Exemplar facts:
  - GSM8K   : pure arithmetic (self-evident, no external facts).
  - HotpotQA: real questions + gold answers from hotpot_qa 'distractor'
              validation (hard level), verified via fetch_exemplars.py.
  - MuSiQue : real questions + gold answers from dgslibisey/MuSiQue
              validation; atomic steps taken from the gold
              question_decomposition, verified via fetch_exemplars.py.

Design choices (defensible to the team):
  - One SHARED json schema across datasets  -> parser stays identical.
  - 4 few-shot exemplars per dataset         -> enough to fix the
    step-granularity pattern, light on context/latency.
  - Exemplars demonstrate ATOMIC steps (one operation / one hop each),
    because the whole downstream pipeline depends on clean step
    separation (SSR itself flags step-splitting as the weak point).
"""
from typing import Optional
import json

# ── Shared system prompt (works for all three datasets) ────────────────────────
SYSTEM_PROMPT = """You are a careful step-by-step reasoning assistant.
You break your reasoning into atomic, individually-checkable steps.
Each step performs exactly ONE inference — one calculation, one fact lookup,
or one comparison — never several at once.
You MUST respond with ONLY valid JSON matching the schema. No prose, no markdown fences."""

# ── Per-dataset guidance: tells the model what an "atomic step" means here ──────
_DATASET_GUIDANCE = {
    "gsm8k": (
        "This is a grade-school math problem. Each step is ONE arithmetic "
        "operation, stating the calculation and its numeric result "
        "(e.g. '48 / 2 = 24'). Do not combine two calculations in one step."
    ),
    "hotpotqa": (
        "This is a multi-hop factual question. Each step resolves ONE entity "
        "or retrieves ONE fact that bridges toward the answer. For comparison "
        "questions, give each entity its own step, then a final comparison step."
    ),
    "musique": (
        "This is a multi-hop question requiring a chain of linked facts. Each "
        "step resolves ONE hop — identify one entity or one attribute that the "
        "next hop depends on. Do not skip intermediate hops."
    ),
}

# ── Four atomic-step exemplars per dataset (facts verified against gold) ────────
_EXEMPLARS = {
    "gsm8k": [
        {
            "question": "Natalia sold clips to 48 friends in April, then sold half as many in May. How many clips did she sell altogether?",
            "complexity": "medium",
            "steps": [
                {"step_id": 1, "content": "April sales = 48 clips.", "confidence": 0.95},
                {"step_id": 2, "content": "May sales = half of April = 48 / 2 = 24 clips.", "confidence": 0.95},
                {"step_id": 3, "content": "Total = 48 + 24 = 72 clips.", "confidence": 0.95},
            ],
            "final_answer": "72",
        },
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. How much did she earn?",
            "complexity": "medium",
            "steps": [
                {"step_id": 1, "content": "Per-minute rate = 12 / 60 = $0.20 per minute.", "confidence": 0.9},
                {"step_id": 2, "content": "Earnings = 0.20 * 50 = $10.", "confidence": 0.95},
            ],
            "final_answer": "10",
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?",
            "complexity": "simple",
            "steps": [
                {"step_id": 1, "content": "Blue fiber = 2 bolts.", "confidence": 0.95},
                {"step_id": 2, "content": "White fiber = half of blue = 2 / 2 = 1 bolt.", "confidence": 0.95},
                {"step_id": 3, "content": "Total = 2 + 1 = 3 bolts.", "confidence": 0.95},
            ],
            "final_answer": "3",
        },
        {
            "question": "Betty needs $100 for a wallet and has half of it. Her parents give her $15 and her grandparents give twice as much as her parents. How much more does she need?",
            "complexity": "complex",
            "steps": [
                {"step_id": 1, "content": "Betty's starting money = half of 100 = $50.", "confidence": 0.95},
                {"step_id": 2, "content": "Grandparents give twice the parents' $15 = 2 * 15 = $30.", "confidence": 0.9},
                {"step_id": 3, "content": "Total after gifts = 50 + 15 + 30 = $95.", "confidence": 0.95},
                {"step_id": 4, "content": "Still needed = 100 - 95 = $5.", "confidence": 0.95},
            ],
            "final_answer": "5",
        },
    ],
    # ── HotpotQA: real hard-level questions, verified gold answers ──────────────
    "hotpotqa": [
        {
            # BRIDGE — gold hops: Kiss and Tell (1945 film) -> Shirley Temple
            "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
            "complexity": "medium",
            "steps": [
                {"step_id": 1, "content": "The woman who portrayed Corliss Archer in the film Kiss and Tell was Shirley Temple.", "confidence": 0.9},
                {"step_id": 2, "content": "Shirley Temple held the government position of Chief of Protocol of the United States.", "confidence": 0.85},
            ],
            "final_answer": "Chief of Protocol",
        },
        {
            # COMPARISON — gold hops: Scott Derrickson, Ed Wood
            "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
            "complexity": "medium",
            "steps": [
                {"step_id": 1, "content": "Scott Derrickson is an American director.", "confidence": 0.85},
                {"step_id": 2, "content": "Ed Wood was an American director.", "confidence": 0.85},
                {"step_id": 3, "content": "Both are American, so they share the same nationality.", "confidence": 0.95},
            ],
            "final_answer": "yes",
        },
        {
            # BRIDGE — gold hops: Big Stone Gap (film) -> Adriana Trigiani
            "question": "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?",
            "complexity": "medium",
            "steps": [
                {"step_id": 1, "content": "The romantic comedy Big Stone Gap was directed by Adriana Trigiani.", "confidence": 0.9},
                {"step_id": 2, "content": "Adriana Trigiani is based in Greenwich Village, New York City.", "confidence": 0.85},
            ],
            "final_answer": "Greenwich Village, New York City",
        },
        {
            # COMPARISON — gold hops: Laleli Mosque, Esma Sultan Mansion
            "question": "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
            "complexity": "complex",
            "steps": [
                {"step_id": 1, "content": "The Laleli Mosque is located in the Laleli neighborhood of Fatih, Istanbul.", "confidence": 0.8},
                {"step_id": 2, "content": "The Esma Sultan Mansion is located in the Ortakoy neighborhood of Istanbul.", "confidence": 0.8},
                {"step_id": 3, "content": "Laleli and Ortakoy are different neighborhoods, so they are not in the same one.", "confidence": 0.95},
            ],
            "final_answer": "no",
        },
    ],
    # ── MuSiQue: real questions, atomic steps from gold decomposition ───────────
    "musique": [
        {
            # gold: Green >> performer -> Steve Hillage ; #1 >> spouse -> Miquette Giraudy
            "question": "Who is the spouse of the Green performer?",
            "complexity": "medium",
            "steps": [
                {"step_id": 1, "content": "The performer of Green is Steve Hillage.", "confidence": 0.85},
                {"step_id": 2, "content": "Steve Hillage's spouse is Miquette Giraudy.", "confidence": 0.85},
            ],
            "final_answer": "Miquette Giraudy",
        },
        {
            # gold: UHF >> distributed by -> Orion Pictures ; #1 >> founded by -> Mike Medavoy
            "question": "Who founded the company that distributed the film UHF?",
            "complexity": "medium",
            "steps": [
                {"step_id": 1, "content": "The film UHF was distributed by Orion Pictures.", "confidence": 0.85},
                {"step_id": 2, "content": "Orion Pictures was founded by Mike Medavoy.", "confidence": 0.85},
            ],
            "final_answer": "Mike Medavoy",
        },
        {
            # gold: Ulrich Walter >> employer -> German Aerospace Center ; #1 >> HQ -> Cologne
            "question": "Where is Ulrich Walter's employer headquartered?",
            "complexity": "medium",
            "steps": [
                {"step_id": 1, "content": "Ulrich Walter's employer is the German Aerospace Center.", "confidence": 0.85},
                {"step_id": 2, "content": "The German Aerospace Center is headquartered in Cologne.", "confidence": 0.85},
            ],
            "final_answer": "Cologne",
        },
        {
            # gold: Caroline LeRoy >> spouse -> Daniel Webster ; #1 >> child -> Fletcher Webster
            "question": "Who is the child of Caroline LeRoy's spouse?",
            "complexity": "complex",
            "steps": [
                {"step_id": 1, "content": "Caroline LeRoy's spouse is Daniel Webster.", "confidence": 0.85},
                {"step_id": 2, "content": "Daniel Webster's child is Fletcher Webster.", "confidence": 0.8},
            ],
            "final_answer": "Fletcher Webster",
        },
    ],
}


def _format_exemplar(ex: dict, dataset: str) -> str:
    """Render one exemplar as a Q + JSON answer block."""
    payload = {
        "question": ex["question"],
        "dataset": dataset,
        "complexity": ex["complexity"],
        "steps": ex["steps"],
        "final_answer": ex["final_answer"],
    }
    return f"Question: {ex['question']}\n{json.dumps(payload, indent=2)}"


def build_prompt(
    question: str,
    dataset: str,
    complexity: str,
    n_steps: int,
    n_shots: int = 4,
) -> str:
    """
    Dataset-routed few-shot prompt.
    `dataset` ∈ {gsm8k, hotpotqa, musique}. Falls back to musique guidance
    if an unknown dataset is passed (keeps the harness from crashing).
    """
    ds = dataset.lower()
    guidance = _DATASET_GUIDANCE.get(ds, _DATASET_GUIDANCE["musique"])
    exemplars = _EXEMPLARS.get(ds, _EXEMPLARS["musique"])[:n_shots]

    shots_block = "\n\n".join(_format_exemplar(ex, ds) for ex in exemplars)

    skeleton = {
        "question": question,
        "dataset": ds,
        "complexity": complexity,
        "steps": [
            {"step_id": 1, "content": "<one atomic inference>", "confidence": 0.9},
            {"step_id": n_steps, "content": "<final atomic inference>", "confidence": 0.9},
        ],
        "final_answer": "<final answer>",
    }

    return f"""{guidance}

Here are worked examples. Study how each step performs exactly ONE inference:

{shots_block}

Now answer this question in the SAME JSON format.
It is a {complexity} question; aim for about {n_steps} atomic steps
(use more or fewer if the reasoning genuinely needs it).

Question: {question}

Respond with ONLY this JSON structure, no extra text:
{json.dumps(skeleton, indent=2)}"""