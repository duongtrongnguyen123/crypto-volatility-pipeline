"""LSTM regressor for next-window volatility prediction.

The model reads a sequence of feature vectors and predicts a single scalar
(the next window's volatility). Rather than relying solely on the final hidden
state, it learns an additive-attention pooling over all time steps so that
informative windows anywhere in the sequence can drive the prediction. A small
MLP head maps the pooled context to the scalar output.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention pooling over LSTM time steps.

    Scores each time step with a small MLP, softmaxes the scores into weights,
    and returns the weight-averaged context vector [B, H].
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, L, H]
        energy = torch.tanh(self.proj(seq))        # [B, L, H]
        scores = self.score(energy).squeeze(-1)    # [B, L]
        weights = torch.softmax(scores, dim=1)     # [B, L]
        context = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)  # [B, H]
        return context


class LSTMVolatility(nn.Module):
    """Many-to-one LSTM: a sequence of feature vectors -> one scalar volatility.

    The LSTM outputs are pooled (attention by default, last-step otherwise),
    passed through dropout, and mapped to a scalar by a small MLP head.

    Args:
        input_size:    number of input features per timestep.
        hidden_size:   LSTM hidden dimension.
        num_layers:    stacked LSTM layers.
        dropout:       dropout between LSTM layers and before the head.
        use_attention: pool time steps with additive attention (True) or use
                       only the final hidden state (False).
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(hidden_size) if use_attention else None
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier/orthogonal init for stable LSTM training."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set the forget-gate bias to 1 for better early-epoch memory.
                hh = param.size(0)
                param.data[hh // 4 : hh // 2].fill_(1.0)
        for module in list(self.head.modules()) + (
            list(self.attention.modules()) if self.attention is not None else []
        ):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        out, (h_n, _) = self.lstm(x)
        if self.attention is not None:
            pooled = self.attention(out)   # [B, H]
        else:
            pooled = out[:, -1, :]         # last timestep hidden state
        pooled = self.drop(pooled)
        return self.head(pooled).squeeze(-1)  # [B]

    @property
    def hparams(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_attention": self.use_attention,
        }


if __name__ == "__main__":
    # Self-test: build, run a random batch, round-trip through hparams.
    torch.manual_seed(0)
    model = LSTMVolatility(input_size=11)
    x = torch.randn(4, 24, 11)
    y = model(x)
    assert y.shape == (4,), f"expected [4], got {tuple(y.shape)}"

    rebuilt = LSTMVolatility(**model.hparams)
    rebuilt.load_state_dict(model.state_dict())  # state_dict must align
    y2 = rebuilt(x)
    assert y2.shape == (4,), f"rebuilt expected [4], got {tuple(y2.shape)}"

    print(f"OK  output shape {tuple(y.shape)}  hparams={model.hparams}")
