"""
uhead/prm_dataset.py

Loads trl-lib/prm800k and builds hidden state cache for UHead pretraining.

Actual schema (trl-lib/prm800k):
    - prompt:      str   — the math problem
    - completions: list of str — one string per reasoning step
    - labels:      list of bool — True=correct, False=incorrect per step

Cache: outputs/prm800k_hidden_states.pt
  → {"states": [N, 4096], "labels": [N]}  (one row per step, not per problem)
"""
import logging
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent.parent.parent / "outputs" / "prm800k_hidden_states.pt"
HF_DATASET_ID = "trl-lib/prm800k"


def build_hidden_state_cache(
    model,
    tokenizer,
    device: str,
    limit: int = 50000,
    split: str = "train",
) -> None:
    """
    Iterates PRM800K problems. For each problem, iterates its steps.
    Builds cumulative context (prompt + prior steps + current step),
    runs through frozen Mistral, extracts last-token hidden state from layer -1.
    Stops after `limit` total steps (not problems).
    """
    from datasets import load_dataset
    from uhead.extractor import HiddenStateExtractor

    logger.info(f"Loading {HF_DATASET_ID} [{split}] (limit={limit} steps)...")
    ds = load_dataset(HF_DATASET_ID, split=split)

    CACHE_PATH.parent.mkdir(exist_ok=True)
    extractor = HiddenStateExtractor(model)
    model.eval()

    all_states, all_labels = [], []
    skipped = 0
    total_steps = 0

    logger.info("Extracting hidden states from PRM800K steps...")
    with extractor:
        for row_idx, row in enumerate(ds):
            if total_steps >= limit:
                break

            prompt = row.get("prompt", "")
            completions = row.get("completions", [])   # list of step strings
            labels = row.get("labels", [])             # list of bool

            if not completions or not labels:
                skipped += 1
                continue

            # Build context cumulatively: prompt + all steps up to current
            context = prompt.strip() + "\n"

            for step_idx, (step_text, label) in enumerate(zip(completions, labels)):
                if total_steps >= limit:
                    break

                if label is None:
                    skipped += 1
                    continue

                full_text = context + f"Step {step_idx + 1}: {step_text.strip()}"

                inputs = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)

                with torch.no_grad():
                    _ = model(**inputs)   # hook fires

                state = extractor.last_token_state()   # [hidden_dim]
                all_states.append(state.float())
                all_labels.append(torch.tensor(1.0 if label else 0.0))
                extractor._states.clear()

                # Extend context for next step
                context += f"Step {step_idx + 1}: {step_text.strip()}\n"
                total_steps += 1

            if row_idx % 200 == 0:
                logger.info(f"  Row {row_idx}, steps so far: {total_steps}")

    if not all_states:
        raise RuntimeError("No steps extracted — check dataset schema.")

    states_tensor = torch.stack(all_states)
    labels_tensor = torch.stack(all_labels)
    torch.save({"states": states_tensor, "labels": labels_tensor}, CACHE_PATH)
    logger.info(
        f"Cache saved → {CACHE_PATH}\n"
        f"  Steps cached: {len(all_states)}, skipped: {skipped}"
    )


class PRM800KDataset(Dataset):
    def __init__(self, cache_path: Path = CACHE_PATH):
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache not found at {cache_path}.\n"
                f"Run: python uhead/train.py --build-cache"
            )
        data = torch.load(cache_path, map_location="cpu")
        self.states = data["states"]
        self.labels = data["labels"]
        logger.info(f"Loaded PRM800K cache: {len(self.states)} steps.")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]
