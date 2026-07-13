"""
uhead/prm_dataset.py

Loads PRM800K from HuggingFace and builds a PyTorch Dataset that yields
(hidden_state, label) pairs for UHead training.

Since PRM800K provides text steps (not hidden states), we extract hidden
states on-the-fly during a pre-processing pass using the frozen Mistral model,
then cache them to disk so training epochs don't re-run inference.

Cache path: outputs/prm800k_hidden_states.pt
"""
import logging
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent.parent.parent / "outputs" / "prm800k_hidden_states.pt"
HF_DATASET_ID = "rawsh/mirrorgemma-2-2b-prm800k-data"  # PRM800K-compatible mirror
# Alternative if above fails: "lkevinzc/prm800k-v2"


def _rating_to_label(rating) -> Optional[float]:
    """PRM800K uses +1 (correct) / -1 (incorrect) / 0 (neutral). Map to binary."""
    if rating == 1:
        return 1.0
    elif rating == -1:
        return 0.0
    return None  # skip neutral


def build_hidden_state_cache(
    model,
    tokenizer,
    device: str,
    limit: int = 50000,
    split: str = "train",
) -> None:
    """
    Runs a single forward pass per step, extracts last-token hidden state
    from layer -1, and saves (states, labels) tensors to CACHE_PATH.

    limit: cap at this many samples (full PRM800K ~800K; 50K is enough for
           a strong baseline and fits in reasonable time on M2).
    """
    from datasets import load_dataset
    from uhead.extractor import HiddenStateExtractor

    logger.info(f"Loading PRM800K [{split}] from HuggingFace (limit={limit})...")
    try:
        ds = load_dataset(HF_DATASET_ID, split=split)
    except Exception:
        logger.warning(f"Primary PRM800K source failed, trying fallback...")
        ds = load_dataset("lkevinzc/prm800k-v2", split=split)

    CACHE_PATH.parent.mkdir(exist_ok=True)
    extractor = HiddenStateExtractor(model)
    model.eval()

    all_states, all_labels = [], []
    skipped = 0

    logger.info("Extracting hidden states from PRM800K steps...")
    with extractor:
        for i, row in enumerate(ds):
            if i >= limit:
                break

            # PRM800K schema: 'steps' is a list of {'text': str, 'rating': int}
            steps = row.get("steps", [])
            if not steps:
                # Some mirrors use 'completions' or flat fields
                text = row.get("text") or row.get("step", "")
                rating = row.get("label") or row.get("rating")
                steps = [{"text": text, "rating": rating}] if text else []

            for step in steps:
                text = step.get("text", "").strip()
                rating = step.get("rating")
                label = _rating_to_label(rating)

                if not text or label is None:
                    skipped += 1
                    continue

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(device)

                with torch.no_grad():
                    _ = model(**inputs, output_hidden_states=True)

                state = extractor.last_token_state()   # [hidden_dim]
                all_states.append(state)
                all_labels.append(torch.tensor(label, dtype=torch.float32))

            if i % 500 == 0:
                logger.info(f"  Processed {i} rows, {len(all_states)} steps so far...")

    states_tensor = torch.stack(all_states)    # [N, hidden_dim]
    labels_tensor = torch.stack(all_labels)    # [N]

    torch.save({"states": states_tensor, "labels": labels_tensor}, CACHE_PATH)
    logger.info(
        f"Cache saved → {CACHE_PATH}\n"
        f"  Total steps: {len(all_states)}, skipped (neutral/empty): {skipped}"
    )


class PRM800KDataset(Dataset):
    """
    Loads pre-extracted (hidden_state, label) pairs from cache.
    Call build_hidden_state_cache() first if cache doesn't exist.
    """
    def __init__(self, cache_path: Path = CACHE_PATH):
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache not found at {cache_path}.\n"
                f"Run: python uhead/train.py --build-cache"
            )
        data = torch.load(cache_path, map_location="cpu")
        self.states = data["states"]   # [N, hidden_dim]
        self.labels = data["labels"]   # [N]
        logger.info(f"Loaded PRM800K cache: {len(self.states)} steps.")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]
