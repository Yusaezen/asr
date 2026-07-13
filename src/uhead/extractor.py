"""
uhead/extractor.py

Forward hook on Mistral's last transformer layer to capture hidden states.
Handles both 3D [batch, seq_len, hidden_dim] and 2D [seq_len, hidden_dim]
outputs — the latter occurs with 4-bit quantization via bitsandbytes.
"""
import torch
from typing import List


class HiddenStateExtractor:
    def __init__(self, model):
        self._layer = model.model.layers[-1]
        self._hook = None
        self._states: List[torch.Tensor] = []

    def _hook_fn(self, module, input, output):
        # output is a tuple; first element is the hidden state tensor
        tensor = output[0].detach().cpu()
        # Normalise to 3D: [batch, seq_len, hidden_dim]
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        self._states.append(tensor)

    def __enter__(self):
        self._states = []
        self._hook = self._layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *args):
        if self._hook:
            self._hook.remove()
            self._hook = None

    def last_token_state(self, call_idx: int = -1) -> torch.Tensor:
        """Hidden state at last token of most recent forward call. Shape: [hidden_dim]."""
        tensor = self._states[call_idx]   # [batch, seq_len, hidden_dim]
        return tensor[0, -1, :]           # [hidden_dim]
