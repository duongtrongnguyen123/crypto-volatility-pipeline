"""LSTM regressor for next-window volatility prediction."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMVolatility(nn.Module):
    """Many-to-one LSTM: a sequence of feature vectors -> one scalar volatility.

    Args:
        input_size:  number of input features per timestep.
        hidden_size: LSTM hidden dimension.
        num_layers:  stacked LSTM layers.
        dropout:     dropout between LSTM layers (ignored if num_layers == 1).
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # last timestep hidden state
        return self.head(last).squeeze(-1)  # [B]

    @property
    def hparams(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }
