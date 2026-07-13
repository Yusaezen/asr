"""
uhead/model.py

Two-layer MLP confidence head trained on top of frozen Mistral 7B hidden states.
Input:  hidden state at final token of a reasoning step (dim=4096 for Mistral 7B)
Output: scalar in [0,1] — estimated correctness probability for that step.
"""
import torch
import torch.nn as nn


MISTRAL_HIDDEN_DIM = 4096


class UHead(nn.Module):
    def __init__(self, input_dim: int = MISTRAL_HIDDEN_DIM, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, input_dim] → [batch, 1]"""
        return self.net(x)
