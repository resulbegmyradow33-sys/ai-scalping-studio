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

__all__ = [
    "CausalTransformer",
    "PatchTSTClassifier",
    "TransformerClassifier",
    "TransformerConfig",
]


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

class PatchTSTClassifier(nn.Module):
    """PatchTST-style patching: each patch becomes one encoder token.

    The window is split into non-overlapping (or strided) patches; every patch
    is flattened to ``patch_len * features`` and projected to the model width.
    Attention runs over patches, then mean pooling feeds the classifier.

    A trailing partial patch is dropped: ``n_patches = (window - patch_len) //
    stride + 1``. That is visible in ``n_patches`` rather than silently padded
    with fabricated bars.
    """

    def __init__(
        self,
        *,
        n_features: int,
        window: int,
        n_classes: int,
        hidden: int,
        dropout: float,
        config: TransformerConfig,
        patch_len: int = 8,
        stride: int | None = None,
    ) -> None:
        super().__init__()
        if patch_len < 1:
            raise ValueError("patch_len must be >= 1")
        step = patch_len if stride is None else stride
        if step < 1:
            raise ValueError("stride must be >= 1")
        if window < patch_len:
            raise ValueError(f"window ({window}) is shorter than patch_len ({patch_len})")
        n_patches = (window - patch_len) // step + 1
        # For PatchTST the sequence that attention sees is made of patches.
        config.check(hidden=hidden, window=n_patches)
        self.n_features = n_features
        self.window = window
        self.patch_len = patch_len
        self.stride = step
        self.n_patches = n_patches
        self.input = nn.Linear(patch_len * n_features, hidden)
        self.position = nn.Parameter(torch.zeros(1, n_patches, hidden))
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

    def _patches(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[2] != self.n_features:
            raise ValueError(
                f"expected (batch, window, {self.n_features}), got {tuple(x.shape)}"
            )
        if x.shape[1] != self.window:
            raise ValueError(f"expected a fixed window of {self.window}, got {x.shape[1]}")
        # (batch, features, patches, patch_len) -> (batch, patches, features*patch_len)
        patches = x.transpose(1, 2).unfold(2, self.patch_len, self.stride)
        return patches.permute(0, 2, 1, 3).reshape(x.shape[0], self.n_patches, -1)

    def forward_sequence(self, x: Tensor) -> Tensor:
        tokens = self.input(self._patches(x)) + self.position
        return self.norm(self.encoder(tokens))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.forward_sequence(x).mean(dim=1))


def build_patchtst(
    *,
    n_features: int,
    window: int,
    n_classes: int,
    hidden: int,
    dropout: float = 0.1,
    patch_len: int = 8,
    stride: int | None = None,
    config: TransformerConfig | None = None,
) -> nn.Module:
    return PatchTSTClassifier(
        n_features=n_features,
        window=window,
        n_classes=n_classes,
        hidden=hidden,
        dropout=dropout,
        config=TransformerConfig() if config is None else config,
        patch_len=patch_len,
        stride=stride,
    )

