from typing import Optional

import torch as th
from torch import nn
import lightning.pytorch as pl

from zerograsp.nets.mha import MHA


def _make_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "silu":
        return nn.SiLU(inplace=False)
    raise ValueError(f"Unsupported feed-forward activation: {name}")


class FeedForward(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        activation: str,
        hidden_layer_multiplier: int,
    ) -> None:
        super().__init__()
        hidden_dim = dim_model * hidden_layer_multiplier
        self.fc1 = nn.Linear(dim_model, hidden_dim)
        self.activation = _make_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class CrossAttentionLayerNorm(nn.Module):
    def __init__(self, dim_model: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim_model)
        self.context_norm = nn.LayerNorm(dim_model)

    def forward(
        self, x: th.Tensor, y: Optional[th.Tensor] = None
    ) -> tuple[th.Tensor, Optional[th.Tensor]]:
        return self.query_norm(x), self.context_norm(y) if y is not None else None


class SelfAttentionLayerNorm(nn.Module):
    def __init__(self, dim_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.norm(x)


class MAEBlock(pl.LightningModule):
    def __init__(self, config, attn_type="self") -> None:
        super(MAEBlock, self).__init__()

        self.attn = MHA(config, attn_type)
        self.feedforward = FeedForward(
            config.dim_mae,
            config.ff_dropout,
            config.ff_activation,
            config.ff_hidden_layer_multiplier,
        )
        if attn_type == "self":
            self.attn_norm = SelfAttentionLayerNorm(config.dim_mae)
        else:
            self.attn_norm = CrossAttentionLayerNorm(config.dim_mae)
        self.ff_norm = nn.LayerNorm(config.dim_mae)
        self.attn_type = attn_type
        self.pe_type = config.pe_type

        if self.pe_type == "cpe":
            self.cpe = nn.Conv1d(
                config.dim_mae, config.dim_mae, kernel_size=5, padding=2, stride=1
            )
            self.ln = nn.LayerNorm(config.dim_mae)

    def forward(
        self,
        x: th.Tensor,
        x_index: th.Tensor,
        attn_lengths,
        y: Optional[th.Tensor] = None,
        y_index: Optional[th.Tensor] = None,
    ):
        if self.attn_type == "self":
            if self.pe_type == "cpe":
                x = x + self.ln(self.cpe(x.transpose(1, 2)).transpose(1, 2))
            x = x + self.attn(
                self.attn_norm(x),
                x_index=x_index,
                attn_lengths=attn_lengths,
            )
        else:
            if y is None or y_index is None:
                raise ValueError("Cross-attention requires both y and y_index.")
            if self.pe_type == "cpe":
                x = x + self.ln(self.cpe(x.transpose(1, 2)).transpose(1, 2))
                y = y + self.ln(self.cpe(y.transpose(1, 2)).transpose(1, 2))
            x_norm, y_norm = self.attn_norm(x, y)
            x = x + self.attn(
                x_norm,
                y=y_norm,
                x_index=x_index,
                y_index=y_index,
            )
        x = x + self.feedforward(self.ff_norm(x))
        return x
