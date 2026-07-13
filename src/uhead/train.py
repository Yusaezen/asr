"""
uhead/train.py

Two-stage UHead training:
  Stage 1: Pretrain on PRM800K (math reasoning, large scale)
  Stage 2: Fine-tune on Phase 2 domain labels (GSM8K + HotpotQA)

Mistral 7B weights are fully frozen throughout. Only UHead MLP trains.

Usage:
    # Step 1: build PRM800K hidden state cache (one-time, ~2-3 hrs on M2)
    python uhead/train.py --build-cache --cache-limit 50000

    # Step 2: pretrain UHead on PRM800K cache
    python uhead/train.py --pretrain --epochs 3

    # Step 3: build domain cache from Phase 2 correctness labels
    python uhead/train.py --finetune-cache --dataset gsm8k
    python uhead/train.py --finetune-cache --dataset hotpotqa

    # Step 4: fine-tune UHead on domain labels
    python uhead/train.py --finetune --epochs 2

    # Checkpoints saved to: outputs/uhead_pretrained.pt
    #                        outputs/uhead_finetuned.pt
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from confidence.model_loader import get_model_and_tokenizer, get_device
from uhead.model import UHead
from uhead.prm_dataset import PRM800KDataset, build_hidden_state_cache, CACHE_PATH
from uhead.domain_dataset import build_domain_cache, load_domain_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"
PRETRAIN_CKPT = OUTPUTS_DIR / "uhead_pretrained.pt"
FINETUNE_CKPT = OUTPUTS_DIR / "uhead_finetuned.pt"


def _pos_weight(dataset) -> torch.Tensor:
    """Compute BCE pos_weight to handle class imbalance (correct >> incorrect)."""
    labels = torch.stack([dataset[i][1] for i in range(len(dataset))])
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    weight = n_neg / max(n_pos, 1)
    logger.info(f"Class balance — pos: {int(n_pos)}, neg: {int(n_neg)}, pos_weight: {weight:.2f}")
    return torch.tensor([weight])


def _train_loop(
    uhead: UHead,
    dataset,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    ckpt_path: Path,
    val_split: float = 0.1,
):
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Train/val split
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    pos_weight = _pos_weight(train_ds).to(device)
    criterion = nn.BCELoss(weight=None)   # pos_weight via BCEWithLogitsLoss alternative
    # Using BCELoss directly since UHead outputs sigmoid already
    # Manually scale positive loss:
    criterion_weighted = lambda pred, target: (
        -(pos_weight[0] * target * torch.log(pred + 1e-8) +
          (1 - target) * torch.log(1 - pred + 1e-8)).mean()
    )

    uhead = uhead.to(device)
    optimizer = torch.optim.AdamW(uhead.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        uhead.train()
        train_loss = 0.0
        for states, labels in train_loader:
            states, labels = states.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = uhead(states).squeeze(1)
            loss = criterion_weighted(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uhead.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Val ──
        uhead.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for states, labels in val_loader:
                states, labels = states.to(device), labels.to(device)
                preds = uhead(states).squeeze(1)
                val_loss += criterion_weighted(preds, labels).item()
                predicted = (preds > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += len(labels)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        acc = correct / total if total > 0 else 0.0
        logger.info(
            f"Epoch {epoch}/{epochs} — train_loss: {avg_train:.4f}, "
            f"val_loss: {avg_val:.4f}, val_acc: {acc:.4f}"
        )
        scheduler.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(uhead.state_dict(), ckpt_path)
            logger.info(f"  ✓ Checkpoint saved → {ckpt_path}")


def pretrain(epochs: int = 3, lr: float = 1e-4, batch_size: int = 256):
    device = get_device()
    dataset = PRM800KDataset()
    uhead = UHead()
    logger.info(f"Pretraining UHead on PRM800K ({len(dataset)} steps)...")
    _train_loop(uhead, dataset, epochs, lr, batch_size, device, PRETRAIN_CKPT)
    logger.info("Pretraining complete.")


def finetune(epochs: int = 2, lr: float = 5e-5, batch_size: int = 32):
    device = get_device()
    dataset = load_domain_datasets(["gsm8k", "hotpotqa"])
    logger.info(f"Fine-tuning UHead on domain labels ({len(dataset)} steps)...")

    uhead = UHead()
    if PRETRAIN_CKPT.exists():
        uhead.load_state_dict(torch.load(PRETRAIN_CKPT, map_location="cpu"))
        logger.info(f"Loaded pretrained weights from {PRETRAIN_CKPT}")
    else:
        logger.warning("No pretrained checkpoint found — fine-tuning from scratch.")

    _train_loop(uhead, dataset, epochs, lr, batch_size, device, FINETUNE_CKPT)
    logger.info("Fine-tuning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-cache", action="store_true",
                        help="Extract PRM800K hidden states and cache to disk")
    parser.add_argument("--cache-limit", type=int, default=50000)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--finetune-cache", action="store_true",
                        help="Extract domain hidden states for fine-tuning")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "hotpotqa"])
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.build_cache:
        model, tokenizer = get_model_and_tokenizer()
        device = get_device()
        build_hidden_state_cache(model, tokenizer, device, limit=args.cache_limit)

    elif args.pretrain:
        pretrain(epochs=args.epochs or 3)

    elif args.finetune_cache:
        model, tokenizer = get_model_and_tokenizer()
        device = get_device()
        build_domain_cache(model, tokenizer, device, dataset=args.dataset)

    elif args.finetune:
        finetune(epochs=args.epochs or 2)

    else:
        parser.print_help()
