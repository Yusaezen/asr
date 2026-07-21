"""
uhead/domain_dataset.py

Builds fine-tuning Dataset from Phase 2 correctness labels joined with
confidence_scores.json to get actual step content (fix: previously used
question as proxy, which produced wrong hidden states).

Join key: (sample_id, step_id)
"""
import json
import logging
import torch
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from typing import List, Dict

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"


def _label_to_float(label: str):
    if label == "correct":
        return 1.0
    elif label == "incorrect":
        return 0.0
    return None  # skip ambiguous / unknown


def _build_step_content_index(dataset: str) -> Dict:
    """
    Builds {sample_id: {step_id: content}} from confidence_scores.json.
    This is the source of truth for step text.
    """
    path = OUTPUTS_DIR / f"{dataset}_confidence_scores.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Confidence scores not found: {path}\n"
            f"Run: python confidence/run_scoring.py --dataset {dataset}"
        )
    with open(path) as f:
        samples = json.load(f)

    index = {}
    for sample in samples:
        sid = sample["sample_id"]
        question = sample["question"]
        index[sid] = {
            "_question": question,
            **{s["step_id"]: s["content"] for s in sample.get("steps", [])}
        }
    return index


def build_domain_cache(
    model,
    tokenizer,
    device: str,
    dataset: str,
) -> Path:
    from uhead.extractor import HiddenStateExtractor

    labels_path = OUTPUTS_DIR / f"{dataset}_correctness_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Correctness labels not found: {labels_path}\n"
            f"Run: python correctness/run_correctness.py --dataset {dataset}"
        )

    with open(labels_path) as f:
        samples = json.load(f)

    # Join with confidence scores to get step content
    step_content_index = _build_step_content_index(dataset)

    cache_path = OUTPUTS_DIR / f"domain_hidden_states_{dataset}.pt"
    extractor = HiddenStateExtractor(model)
    model.eval()

    all_states, all_labels = [], []
    skipped = 0
    missing_content = 0

    logger.info(f"Extracting domain hidden states for {dataset}...")
    with extractor:
        for sample in samples:
            sample_id = sample.get("sample_id", "")
            question = sample.get("question", "")
            step_map = step_content_index.get(sample_id, {})

            for step_entry in sample.get("per_step", []):
                label_str = step_entry.get("label", "")
                label = _label_to_float(label_str)
                if label is None:
                    skipped += 1
                    continue

                step_id = step_entry["step_id"]
                step_content = step_map.get(step_id)

                if not step_content:
                    logger.debug(f"Missing content for {sample_id} step {step_id} — skipping.")
                    missing_content += 1
                    continue

                # Full context: question + prior steps + this step
                context = f"Question: {question}\n"
                for sid in sorted(k for k in step_map if isinstance(k, int) and k < step_id):
                    context += f"Step {sid}: {step_map[sid]}\n"
                text = context + f"Step {step_id}: {step_content}"

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)

                with torch.no_grad():
                    _ = model(**inputs)  # hook fires, output_hidden_states not needed

                state = extractor.last_token_state()
                all_states.append(state)
                all_labels.append(torch.tensor(label, dtype=torch.float32))
                extractor._states.clear()

    if not all_states:
        raise RuntimeError(
            f"No valid steps extracted for {dataset}. "
            f"Check that correctness labels and confidence scores are aligned."
        )

    states_tensor = torch.stack(all_states)
    labels_tensor = torch.stack(all_labels)
    torch.save({"states": states_tensor, "labels": labels_tensor}, cache_path)
    logger.info(
        f"Domain cache saved → {cache_path}\n"
        f"  Steps extracted: {len(all_states)}\n"
        f"  Skipped (ambiguous/unknown): {skipped}\n"
        f"  Skipped (missing content):   {missing_content}"
    )
    return cache_path


class DomainDataset(Dataset):
    def __init__(self, cache_path: Path):
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.states = data["states"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]


def load_domain_datasets(datasets: List[str] = ("musique",)) -> ConcatDataset:
    parts = []
    for ds in datasets:
        cache_path = OUTPUTS_DIR / f"domain_hidden_states_{ds}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Domain cache missing for {ds}: {cache_path}\n"
                f"Run: python uhead/train.py --finetune-cache --dataset {ds}"
            )
        parts.append(DomainDataset(cache_path))
        logger.info(f"Loaded domain cache: {ds} ({len(parts[-1])} steps)")
    return ConcatDataset(parts)
