"""
Batch pilot generation — Intern 1 + Intern 2 handoff point.
Runs generation_harness over N MuSiQue questions and saves:
  - outputs/batch_results.json   (all results)
  - outputs/batch_failures.json  (failed/fallback parses for Intern 3 QA)
"""
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from generation_harness import run
from load_musique import load_musique
from model_client import DEFAULT_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def run_batch(
    n_samples: int = 10,
    split: str = "validation",
    model: str = DEFAULT_MODEL,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    samples = load_musique(split=split, limit=n_samples)
    results = []
    failures = []

    for i, sample in enumerate(samples):
        logger.info(f"[{i+1}/{len(samples)}] Processing: {sample['question'][:60]}...")
        try:
            result = run(
                question=sample["question"],
                model=model,
                dataset="musique",
                output_path=None,  # batch saves at end
            )
            result["ground_truth"] = sample["answer"]
            result["sample_id"] = sample["id"]

            results.append(result)

            if result["parse_method"] != "schema":
                failures.append({
                    "sample_id": sample["id"],
                    "question": sample["question"],
                    "parse_method": result["parse_method"],
                    "n_steps_parsed": len(result["steps"]),
                    "raw_output_snippet": result["raw_output"][:300],
                })

        except Exception as e:
            logger.error(f"Failed on sample {sample['id']}: {e}")
            failures.append({
                "sample_id": sample["id"],
                "question": sample["question"],
                "error": str(e),
                "parse_method": "error",
            })

    # Save outputs
    results_path = OUTPUT_DIR / "batch_results.json"
    failures_path = OUTPUT_DIR / "batch_failures.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2)

    # Summary
    schema_ok = sum(1 for r in results if r.get("parse_method") == "schema")
    logger.info("=" * 50)
    logger.info(f"Batch complete: {len(results)}/{n_samples} succeeded")
    logger.info(f"  Schema parse:    {schema_ok}")
    logger.info(f"  Fallback parses: {len(results) - schema_ok}")
    logger.info(f"  Errors:          {n_samples - len(results)}")
    logger.info(f"Results → {results_path}")
    logger.info(f"Failures → {failures_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    run_batch(n_samples=args.n, split=args.split, model=args.model)
