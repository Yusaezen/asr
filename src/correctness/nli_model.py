"""
Singleton NLI model loader — DeBERTa-v3-mnli.

Used by hotpotqa_checker.py and strategyqa_checker.py to check whether a
model-generated reasoning step is entailed by a gold reference sentence
(supporting fact / decomposition sub-step).

Kept deliberately independent of confidence/model_loader.py (different model,
different task head) rather than sharing code across intern boundaries —
simpler to keep this package self-contained.

Checkpoint: cross-encoder/nli-deberta-v3-base
  - Sequence-pair classifier fine-tuned on MNLI.
  - We read label order from model.config.id2label rather than assuming a
    fixed index, so this is robust if the checkpoint is swapped later.
"""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"

_model = None
_tokenizer = None
_label2idx = None


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_nli_model():
    global _model, _tokenizer, _label2idx

    if _model is not None:
        return _model, _tokenizer, _label2idx

    device = get_device()
    logger.info(f"Loading NLI model {NLI_MODEL_ID} on {device}...")

    _tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    _model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID)
    _model.to(device)
    _model.eval()

    # Build a normalized label -> index map (labels are case-inconsistent
    # across checkpoints, e.g. "ENTAILMENT" vs "entailment").
    id2label = _model.config.id2label
    _label2idx = {v.lower(): k for k, v in id2label.items()}
    for needed in ("entailment", "neutral", "contradiction"):
        if needed not in _label2idx:
            logger.warning(
                f"NLI model label set missing '{needed}' (got {list(_label2idx)}). "
                f"Downstream entailment logic may misbehave — check the checkpoint."
            )

    logger.info(f"NLI model loaded. Label map: {_label2idx}")
    return _model, _tokenizer, _label2idx


def nli_probs(premise: str, hypothesis: str) -> dict:
    """
    Returns {'entailment': p, 'neutral': p, 'contradiction': p} for
    premise -> hypothesis (does `premise` entail `hypothesis`?).

    In our usage: premise = gold reference sentence (supporting fact /
    decomposition sub-step), hypothesis = the model's reasoning step content.
    """
    model, tokenizer, label2idx = get_nli_model()
    device = get_device()

    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt", truncation=True, max_length=256,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    probs = torch.softmax(logits, dim=-1)

    out = {}
    for label in ("entailment", "neutral", "contradiction"):
        idx = label2idx.get(label)
        out[label] = probs[idx].item() if idx is not None else 0.0
    return out
