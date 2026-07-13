"""
confidence/run_scoring.py  (updated for Phase 2 UHead)

Loads batch_results_{dataset}.json, runs UHead confidence scorer per step,
saves annotated output for Intern 3's calibration analysis.

Usage:
    python run_scoring.py --dataset hotpotqa --n 500
    python run_scoring.py --dataset gsm8k --n 200

    # Ablation baselines (slow — only for comparison):
    # python run_scoring.py --dataset gsm8k --method logprob --n 50
    # python run_scoring.py --dataset gsm8k --method sampling --n 50
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from confidence.scorer_interface import score_steps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"


def load_batch_results(dataset: str) -> list:
    path = OUTPUTS_DIR / f"batch_results_{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No batch results at {path}.\n"
            f"Run: python batch_runner.py --dataset {dataset} --n 700"
        )
    with open(path) as f:
        return json.load(f)


def run_scoring(dataset: str, n: int, method: str = "uhead"):
    OUTPUTS_DIR.mkdir(exist_ok=True)
    samples = load_batch_results(dataset)[:n]
    logger.info(f"Scoring {len(samples)} {dataset} samples with method={method}")

    results = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        steps = sample.get("steps", [])
        sample_id = sample.get("sample_id", str(i))

        logger.info(f"[{i+1}/{len(samples)}] {question[:60]}...")

        if not steps:
            logger.warning(f"No steps for {sample_id}, skipping.")
            continue

        try:
            scores = score_steps(question, steps, method=method)
        except Exception as e:
            logger.error(f"Scoring failed for {sample_id}: {e}")
            scores = {"error": str(e)}

        annotated_steps = []
        for step in steps:
            idx = step["step_id"] - 1
            entry = {"step_id": step["step_id"], "content": step["content"]}

            if method == "uhead" and isinstance(scores, list):
                entry["confidence_uhead"] = scores[idx] if idx < len(scores) else None

            elif method == "both" and isinstance(scores, dict):
                # Ablation: logprob + sampling scores together
                entry["confidence_logprob"] = scores["logprob"][idx] if idx < len(scores["logprob"]) else None
                entry["confidence_sampling"] = scores["sampling"][idx] if idx < len(scores["sampling"]) else None

            elif isinstance(scores, list):
                # logprob or sampling individually
                entry[f"confidence_{method}"] = scores[idx] if idx < len(scores) else None

            annotated_steps.append(entry)

        results.append({
            "sample_id": sample_id,
            "dataset": dataset,
            "question": question,
            "complexity": sample.get("complexity", "unknown"),
            "final_answer": sample.get("final_answer", ""),
            "ground_truth": sample.get("ground_truth", ""),
            "parse_method": sample.get("parse_method", "unknown"),
            "steps": annotated_steps,
        })

    out_path = OUTPUTS_DIR / f"{dataset}_confidence_scores.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved {len(results)} scored samples → {out_path}")

    # Summary stats
    all_scores = [
        s.get("confidence_uhead") or s.get("confidence_logprob") or s.get(f"confidence_{method}")
        for r in results for s in r["steps"]
    ]
    all_scores = [s for s in all_scores if s is not None]
    if all_scores:
        logger.info(
            f"Confidence ({method}) — "
            f"mean: {sum(all_scores)/len(all_scores):.4f}, "
            f"min: {min(all_scores):.4f}, "
            f"max: {max(all_scores):.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "hotpotqa", "musique"], required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument(
        "--method",
        choices=["uhead", "logprob", "sampling", "both"],
        default="uhead",
        help="'uhead' is primary. logprob/sampling/both are ablation baselines."
    )
    args = parser.parse_args()
    run_scoring(dataset=args.dataset, n=args.n, method=args.method)