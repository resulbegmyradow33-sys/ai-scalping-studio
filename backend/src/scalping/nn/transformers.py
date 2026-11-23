"""Compact Transformer classifiers for windowed bar sequences.

Two variants share one encoder:

* ``transformer`` — bidirectional self-attention over the whole window, mean
  pooled. Good for offline classification where the window ends at the
  prediction bar.
* ``causal_transformer`` — an upper-triangular attention mask and last-token
  pooling. ``forward_sequence`` is causal in ``t``: step ``t`` never reads
  ``t + 1``, tested like the TCN.

The configuration guard is deliberately conservative for an 8 GB Mac mini:
``hidden`` must divide by the head count, sequences are capped, and layer count
is bounded. A small model that trains is worth more than a large one that
swaps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = ["CausalTransformer", "TransformerClassifier", "TransformerConfig"]


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Architecture guard rails; defaults fit a small GPU-less machine."""

    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 4
    max_sequence: int = 512

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if not 1 <= self.n_layers <= 8:
            raise ValueError("n_layers must be in [1, 8]")
        if self.ff_mult < 1:
            raise ValueError("ff_mult must be >= 1")
        if not 8 <= self.max_sequence <= 4096:
            raise ValueError("max_sequence must be in [8, 4096]")

    def check(self, *, hidden: int, window: int) -> None:
        if hidden % self.n_heads != 0:
            raise ValueError(
                f"hidden ({hidden}) must be divisible by n_heads ({self.n_heads})"
            )
        if window > self.max_sequence:
            raise ValueError(
                f"window ({window}) exceeds max_sequence ({self.max_sequence})"
            )


class _TransformerBase(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        window: int,
        n_classes: int,
        hidden: int,
        dropout: float,
        config: TransformerConfig,
    ) -> None:
        super().__init__()
        config.check(hidden=hidden, window=window)
        self.n_features = n_features
        self.window = window
        self.config = config
        self.input = nn.Linear(n_features, hidden)
        self.position = nn.Parameter(torch.zeros(1, window, hidden))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.n_heads,
            dim_feedforward=hidden * config.ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_classes)

    def _mask(self, length: int, device: torch.device) -> Tensor | None:
        raise NotImplementedError

    def forward_sequence(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[2] != self.n_features:
            raise ValueError(
                f"expected (batch, window, {self.n_features}), got {tuple(x.shape)}"
            )
        if x.shape[1] > self.window:
            raise ValueError(f"sequence longer than the trained window ({self.window})")
        hidden = self.input(x) + self.position[:, : x.shape[1]]
        hidden = self.encoder(hidden, mask=self._mask(x.shape[1], x.device))
        return self.norm(hidden)

    def forward(self, x: Tensor) -> Tensor:
        sequence = self.forward_sequence(x)
        return self.head(self._pool(sequence))

    def _pool(self, sequence: Tensor) -> Tensor:
        raise NotImplementedError


class TransformerClassifier(_TransformerBase):
    """Bidirectional encoder with mean pooling."""

    def _mask(self, length: int, device: torch.device) -> Tensor | None:
        return None

    def _pool(self, sequence: Tensor) -> Tensor:
        return sequence.mean(dim=1)


class CausalTransformer(_TransformerBase):
    """Causal encoder with last-token pooling."""

    def _mask(self, length: int, device: torch.device) -> Tensor | None:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def _pool(self, sequence: Tensor) -> Tensor:
        return sequence[:, -1]


def build_transformer(
    *,
    n_features: int,
    window: int,
    n_classes: int,
    hidden: int,
    dropout: float,
    causal: bool,
    config: TransformerConfig | None = None,
) -> nn.Module:
    cfg = TransformerConfig() if config is None else config
    if causal:
        return CausalTransformer(
            n_features=n_features,
            window=window,
            n_classes=n_classes,
            hidden=hidden,
            dropout=dropout,
            config=cfg,
        )
    return TransformerClassifier(
        n_features=n_features,
        window=window,
        n_classes=n_classes,
        hidden=hidden,
        dropout=dropout,
        config=cfg,
    )
