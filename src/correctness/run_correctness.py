"""
Phase 2 Intern 2 — Ground-truth step correctness pipeline runner.

Loads existing confidence-scored steps from {dataset}_confidence_scores.json
(produced by Intern 1's run_scoring.py), attaches gold data per dataset, runs
the appropriate correctness checker, and saves per-step correct/incorrect/
ambiguous labels for Intern 3 to merge with the confidence scores.

Usage:
    python run_correctness.py --dataset gsm8k
    python run_correctness.py --dataset hotpotqa
    python run_correctness.py --dataset strategyqa --n 5

NOTE on id alignment (read before running):
  - GSM8K:      sample_id is a positional index (str(i) into the loaded
                split, set by load_gsm8k.py / batch_runner.py in Phase 1).
                This script reloads the SAME split to rebuild that mapping,
                so ordering must match Phase 1's load (true unless the HF
                dataset changes upstream or a shuffle is introduced anywhere
                in the pipeline — neither is currently the case).
  - HotpotQA:   sample_id is the dataset's own content-based question id
                (e.g. '5a8b57f2...'), NOT positional — robust regardless of
                load order. Handled internally by hotpotqa_checker.py.
  - StrategyQA: same positional caveat as GSM8K unless the mirror provides a
                real 'qid' field (checked first in load_strategyqa.py).
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # -> src/
sys.path.insert(0, str(Path(__file__).parent))          # -> src/correctness/

from correctness_interface import score_correctness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"

# How many gold rows to (re)load when building id-lookup dicts for datasets
# with positional ids (gsm8k, strategyqa). Generous default; raise via
# --gold-limit if your Phase-1 pilot ran deeper into the split than this.
DEFAULT_GOLD_LIMIT = 500


def load_confidence_scores(dataset: str) -> list:
    path = OUTPUTS_DIR / f"{dataset}_confidence_scores.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No confidence scores found at {path}.\n"
            f"Run Intern 1's scorer first: python confidence/run_scoring.py --dataset {dataset}"
        )
    with open(path) as f:
        return json.load(f)


def _build_gsm8k_gold_index(split: str, limit: int) -> dict:
    from load_gsm8k import load_gsm8k
    logger.info(f"Building GSM8K gold index (split={split}, limit={limit})...")
    rows = load_gsm8k(split=split, limit=limit)
    return {r["id"]: r for r in rows}


def _build_strategyqa_gold_index(split: str, limit: int) -> dict:
    from load_strategyqa import load_strategyqa
    logger.info(f"Building StrategyQA gold index (split={split}, limit={limit})...")
    rows = load_strategyqa(split=split, limit=limit)
    return {r["id"]: r for r in rows}


def run_correctness(dataset: str, n: int = None, gold_limit: int = DEFAULT_GOLD_LIMIT,
                     hotpot_split: str = "validation", gold_split: str = None):
    OUTPUTS_DIR.mkdir(exist_ok=True)
    dataset = dataset.lower()

    samples = load_confidence_scores(dataset)
    if n:
        samples = samples[:n]
    logger.info(f"Running correctness checks on {len(samples)} {dataset} samples.")

    # Pre-build gold index for datasets that need positional lookup.
    gsm8k_gold_idx = None
    strategyqa_gold_idx = None
    if dataset == "gsm8k":
        gsm8k_gold_idx = _build_gsm8k_gold_index(gold_split or "test", gold_limit)
    elif dataset == "strategyqa":
        strategyqa_gold_idx = _build_strategyqa_gold_index(gold_split or "test", gold_limit)

    results = []
    n_correct = n_incorrect = n_ambiguous = n_unknown = 0
    n_final_correct = n_final_total = 0

    for i, sample in enumerate(samples):
        sample_id = sample.get("sample_id", str(i))
        steps = sample.get("steps", [])
        logger.info(f"[{i+1}/{len(samples)}] {sample_id} — {sample.get('question','')[:55]}...")

        if not steps:
            logger.warning(f"No steps for {sample_id}, skipping.")
            continue

        try:
            if dataset == "gsm8k":
                gold_row = gsm8k_gold_idx.get(sample_id, {})
                gold = {
                    "gold_solution": gold_row.get("gold_solution", ""),
                    "model_final_answer": sample.get("final_answer", ""),
                    "ground_truth": sample.get("ground_truth", gold_row.get("answer", "")),
                }
            elif dataset == "hotpotqa":
                gold = {"sample_id": sample_id, "split": hotpot_split}
            elif dataset == "strategyqa":
                gold_row = strategyqa_gold_idx.get(sample_id, {})
                gold = {
                    "facts": gold_row.get("facts", []),
                    "decomposition": gold_row.get("decomposition", []),
                }
            elif dataset == "musique":
                gold = {"sample_id": sample_id, "split": "validation"}
            else:
                raise ValueError(f"Unsupported dataset '{dataset}'.")

            result = score_correctness(dataset=dataset, steps=steps, gold=gold)

        except Exception as e:
            logger.error(f"Correctness check failed for {sample_id}: {e}")
            result = {
                "per_step": [
                    {"step_id": s["step_id"], "label": "unknown", "method": "error",
                     "ambiguous": True, "error": str(e)}
                    for s in steps
                ]
            }

        for s in result["per_step"]:
            if s["label"] == "correct":
                n_correct += 1
            elif s["label"] == "incorrect":
                n_incorrect += 1
            elif s["label"] == "ambiguous":
                n_ambiguous += 1
            else:
                n_unknown += 1

        if dataset == "gsm8k" and result.get("final_answer_correct") is not None:
            n_final_total += 1
            if result["final_answer_correct"]:
                n_final_correct += 1

        results.append({
            "sample_id": sample_id,
            "dataset": dataset,
            "question": sample.get("question", ""),
            **result,
        })

    out_path = OUTPUTS_DIR / f"{dataset}_correctness_labels.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    total = n_correct + n_incorrect + n_ambiguous + n_unknown
    logger.info("=" * 50)
    logger.info(f"Correctness check complete ({dataset}): {len(results)} samples, {total} steps")
    logger.info(f"  correct:    {n_correct}")
    logger.info(f"  incorrect:  {n_incorrect}")
    logger.info(f"  ambiguous:  {n_ambiguous}  (flagged for manual review)")
    logger.info(f"  unknown:    {n_unknown}  (missing gold data)")
    if n_final_total:
        logger.info(f"  final-answer accuracy: {n_final_correct}/{n_final_total} "
                    f"({100*n_final_correct/n_final_total:.1f}%)")
    logger.info(f"Saved -> {out_path}")
    logger.info(f"Next: Intern 3 merges this with {dataset}_confidence_scores.json on (sample_id, step_id).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "hotpotqa", "strategyqa", "musique"], required=True)
    parser.add_argument("--n", type=int, default=None, help="Limit number of samples (default: all)")
    parser.add_argument("--gold-limit", type=int, default=DEFAULT_GOLD_LIMIT,
                        help="How many gold rows to index for positional-id datasets (gsm8k/strategyqa)")
    parser.add_argument("--hotpot-split", type=str, default="validation")
    parser.add_argument("--gold-split", type=str, default=None,
                        help="Override gold split for gsm8k/strategyqa (default: test)")
    args = parser.parse_args()

    run_correctness(
        dataset=args.dataset,
        n=args.n,
        gold_limit=args.gold_limit,
        hotpot_split=args.hotpot_split,
        gold_split=args.gold_split,
    )
