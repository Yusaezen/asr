"""
confidence/run_scoring.py — Fix 6: default n raised to 50.

Now that GPU eliminates timeouts and the RobertaModel reload is fixed,
running on 50 samples is feasible and gives Intern 3 enough data for
meaningful calibration analysis (ECE, reliability diagrams).
Default kept at 50; pass --n to override.
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
            f"No batch results found at {path}.\n"
            f"Run Phase 1 first: python batch_runner.py --dataset {dataset} --n 50"
        )
    with open(path) as f:
        return json.load(f)


def run_scoring(dataset: str, n: int, method: str):
    OUTPUTS_DIR.mkdir(exist_ok=True)

    all_samples = load_batch_results(dataset)
    # Skip samples that errored in Phase 1 (no steps)
    samples = [s for s in all_samples if s.get("steps")][:n]
    logger.info(f"Scoring {len(samples)} samples from {dataset} using method={method}")

    results = []

    for i, sample in enumerate(samples):
        question  = sample["question"]
        steps     = sample.get("steps", [])
        sample_id = sample.get("sample_id", str(i))

        logger.info(f"[{i+1}/{len(samples)}] {question[:60]}...")

        if not steps:
            logger.warning(f"No steps for sample {sample_id}, skipping.")
            continue

        try:
            scores = score_steps(question, steps, method=method)
        except Exception as e:
            logger.error(f"Scoring failed for sample {sample_id}: {e}", exc_info=True)
            scores = {"error": str(e)}

        annotated_steps = []
        for step in steps:
            entry = {"step_id": step["step_id"], "content": step["content"]}
            if method == "both" and isinstance(scores, dict) and "logprob" in scores:
                idx = step["step_id"] - 1
                entry["confidence_logprob"]  = scores["logprob"][idx]  if idx < len(scores["logprob"])  else None
                entry["confidence_sampling"] = scores["sampling"][idx] if idx < len(scores["sampling"]) else None
            elif isinstance(scores, list):
                idx = step["step_id"] - 1
                entry[f"confidence_{method}"] = scores[idx] if idx < len(scores) else None
            annotated_steps.append(entry)

        results.append({
            "sample_id":   sample_id,
            "dataset":     dataset,
            "question":    question,
            "complexity":  sample.get("complexity",   "unknown"),
            "final_answer": sample.get("final_answer", ""),
            "ground_truth": sample.get("ground_truth", ""),
            "parse_method": sample.get("parse_method", "unknown"),
            "steps":        annotated_steps,
        })

    out_path = OUTPUTS_DIR / f"{dataset}_confidence_scores.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved {len(results)} scored samples → {out_path}")

    if results:
        all_lp, all_samp = [], []
        for r in results:
            for s in r["steps"]:
                if s.get("confidence_logprob")  is not None: all_lp.append(s["confidence_logprob"])
                if s.get("confidence_sampling") is not None: all_samp.append(s["confidence_sampling"])
        if all_lp:
            logger.info(f"Logprob scores  — mean: {sum(all_lp)/len(all_lp):.4f}, "
                        f"min: {min(all_lp):.4f}, max: {max(all_lp):.4f}")
        if all_samp:
            logger.info(f"Sampling scores — mean: {sum(all_samp)/len(all_samp):.4f}, "
                        f"min: {min(all_samp):.4f}, max: {max(all_samp):.4f}")
        logger.info(f"Next: run Intern 2 correctness checker — "
                    f"python correctness/run_correctness.py --dataset {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "hotpotqa", "musique"], default="gsm8k")
    parser.add_argument("--n",      type=int, default=50,   help="Samples to score (default 50 on GPU)")
    parser.add_argument("--method", choices=["logprob", "sampling", "both"], default="both")
    args = parser.parse_args()

    run_scoring(dataset=args.dataset, n=args.n, method=args.method)
