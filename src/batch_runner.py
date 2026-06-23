"""
Batch pilot generation — Intern 1 + Intern 2 handoff point.
Runs generation_harness over N questions from a chosen dataset and saves:
  - outputs/batch_results.json   (all results)
  - outputs/batch_failures.json  (true failures only, for Intern 3 QA)

Datasets: musique (default) | gsm8k | hotpotqa
Each loader attaches a gold_complexity tag used as an independent cross-check
against the classifier's complexity decision (stored side by side, not overriding).
"""
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from generation_harness import run
from load_musique import load_musique
from load_gsm8k import load_gsm8k
from load_hotpotqa import load_hotpotqa
from model_client import DEFAULT_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# dataset name -> (loader fn, default split)
_LOADERS = {
    "musique":  (load_musique,  "validation"),
    "gsm8k":    (load_gsm8k,    "test"),
    "hotpotqa": (load_hotpotqa, "validation"),
}


def run_batch(
    n_samples: int = 10,
    dataset: str = "musique",
    split: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    dataset = dataset.lower()
    if dataset not in _LOADERS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(_LOADERS)}.")
    loader, default_split = _LOADERS[dataset]
    split = split or default_split

    samples = loader(split=split, limit=n_samples)
    results = []
    failures = []

    for i, sample in enumerate(samples):
        logger.info(f"[{i+1}/{len(samples)}] ({dataset}) {sample['question'][:55]}...")
        try:
            result = run(
                question=sample["question"],
                model=model,
                dataset=dataset,                 # <-- routes prompts.build_prompt
                output_path=None,
            )
            result["ground_truth"] = sample["answer"]
            result["sample_id"] = sample["id"]
            # cross-check signal: gold complexity vs classifier complexity
            result["gold_complexity"] = sample.get("gold_complexity", "unknown")

            results.append(result)

            # a result is only a TRUE failure if it didn't reach structured steps
            if result["parse_method"] not in ("schema", "schema_repaired"):
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

    # Save outputs (dataset-tagged filenames so runs don't overwrite each other)
    results_path = OUTPUT_DIR / f"batch_results_{dataset}.json"
    failures_path = OUTPUT_DIR / f"batch_failures_{dataset}.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2)

    # Summary (honest categories: schema + repaired both recover full structure)
    schema_ok = sum(1 for r in results if r.get("parse_method") == "schema")
    repaired  = sum(1 for r in results if r.get("parse_method") == "schema_repaired")
    delim     = sum(1 for r in results if r.get("parse_method") == "fallback_delimiter")
    sentence  = sum(1 for r in results if r.get("parse_method") == "fallback_sentence")
    structured_ok = schema_ok + repaired
    # cross-check: how often classifier complexity matched the gold tier
    matched = sum(1 for r in results
                  if r.get("complexity") and r.get("complexity") == r.get("gold_complexity"))
    logger.info("=" * 50)
    logger.info(f"Batch complete ({dataset}): {len(results)}/{n_samples} succeeded")
    logger.info(f"  Structured (schema + repaired): {structured_ok}")
    logger.info(f"    - clean schema:      {schema_ok}")
    logger.info(f"    - repaired schema:   {repaired}")
    logger.info(f"  True fallbacks:        {delim + sentence}")
    logger.info(f"    - delimiter:         {delim}")
    logger.info(f"    - sentence:          {sentence}")
    logger.info(f"  Errors:                {n_samples - len(results)}")
    logger.info(f"  Classifier vs gold complexity match: {matched}/{len(results)}")
    logger.info(f"Results → {results_path}")
    logger.info(f"Failures → {failures_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--dataset", type=str, default="musique",
                        choices=["musique", "gsm8k", "hotpotqa"],
                        help="Which dataset to run")
    parser.add_argument("--split", type=str, default=None,
                        help="Override the dataset's default split")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    run_batch(n_samples=args.n, dataset=args.dataset, split=args.split, model=args.model)