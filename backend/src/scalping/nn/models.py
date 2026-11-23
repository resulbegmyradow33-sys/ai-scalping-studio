"""Sequence model zoo for the neural-network research framework.

Every model consumes ``(batch, window, features)`` float32 and returns logits
``(batch, classes)``. The zoo is intentionally small: enough architectural
families to answer "does sequencing help?" without turning Phase 6 into a model
zoo contest.

Only :class:`CausalTCN` promises causality inside the window: its convolutions
are left-padded, so the representation at step ``t`` never reads ``t + 1``.
That property is tested by mutating the tail of the input. The other models are
free to look both ways inside the window — the window itself already ends at the
prediction time, so this is not look-ahead.
"""

from __future__ import annotations

from typing import Final, Literal

import torch
from torch import Tensor, nn

__all__ = [
    "MODEL_NAMES",
    "CausalTCN",
    "ModelSpec",
    "build_model",
    "model_names",
]

ModelName = Literal[
    "mlp",
    "deep_mlp",
    "residual_mlp",
    "rnn",
    "lstm",
    "gru",
    "cnn",
    "dilated_cnn",
    "cnn_lstm",
    "tcn",
    "transformer",
    "causal_transformer",
]

MODEL_NAMES: Final[tuple[ModelName, ...]] = (
    "mlp",
    "deep_mlp",
    "residual_mlp",
    "rnn",
    "lstm",
    "gru",
    "cnn",
    "dilated_cnn",
    "cnn_lstm",
    "tcn",
    "transformer",
    "causal_transformer",
)


class ModelSpec:
    """Shared constructor arguments; kept as a dataclass for readable tests."""

    def __init__(
        self,
        *,
        n_features: int,
        window: int = 60,
        n_classes: int = 3,
        hidden: int = 64,
        dropout: float = 0.1,
        seed: int = 0,
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if hidden < 1:
            raise ValueError("hidden must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if seed < 0:
            raise ValueError("seed must be >= 0")
        self.n_features = n_features
        self.window = window
        self.n_classes = n_classes
        self.hidden = hidden
        self.dropout = dropout
        self.seed = seed


def _check_input(x: Tensor, n_features: int) -> None:
    if x.ndim != 3:
        raise ValueError(f"expected (batch, window, features), got shape {tuple(x.shape)}")
    if x.shape[2] != n_features:
        raise ValueError(
            f"expected {n_features} features, got {x.shape[2]}"
        )


class _MLP(nn.Module):
    def __init__(self, spec: ModelSpec, *, depth: int) -> None:
        super().__init__()
        self.spec = spec
        layers: list[nn.Module] = [nn.Flatten()]
        width = spec.window * spec.n_features
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(width, spec.hidden),
                    nn.ReLU(),
                    nn.Dropout(spec.dropout),
                ]
            )
            width = spec.hidden
        layers.append(nn.Linear(width, spec.n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        _check_input(x, self.spec.n_features)
        return self.net(x)


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.body(x))


class _ResidualMLP(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        width = spec.window * spec.n_features
        self.input = nn.Sequential(nn.Flatten(), nn.Linear(width, spec.hidden), nn.ReLU())
        self.blocks = nn.Sequential(
            _ResidualBlock(spec.hidden, spec.dropout),
            _ResidualBlock(spec.hidden, spec.dropout),
        )
        self.head = nn.Linear(spec.hidden, spec.n_classes)

    def forward(self, x: Tensor) -> Tensor:
        _check_input(x, self.spec.n_features)
        return self.head(self.blocks(self.input(x)))


class _Recurrent(nn.Module):
    def __init__(self, spec: ModelSpec, *, kind: str) -> None:
        super().__init__()
        self.spec = spec
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = cls(
            input_size=spec.n_features,
            hidden_size=spec.hidden,
            batch_first=True,
        )
        self.drop = nn.Dropout(spec.dropout)
        self.head = nn.Linear(spec.hidden, spec.n_classes)

    def forward(self, x: Tensor) -> Tensor:
        _check_input(x, self.spec.n_features)
        output, _ = self.rnn(x)
        return self.head(self.drop(output[:, -1, :]))


class _CNN(nn.Module):
    def __init__(self, spec: ModelSpec, *, dilations: tuple[int, ...]) -> None:
        super().__init__()
        self.spec = spec
        layers: list[nn.Module] = []
        channels = spec.n_features
        for dilation in dilations:
            layers.extend(
                [
                    nn.Conv1d(
                        channels,
                        spec.hidden,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.ReLU(),
                ]
            )
            channels = spec.hidden
        self.encoder = nn.Sequential(*layers)
        self.drop = nn.Dropout(spec.dropout)
        self.head = nn.Linear(channels, spec.n_classes)

    def forward(self, x: Tensor) -> Tensor:
        _check_input(x, self.spec.n_features)
        # (batch, window, features) -> (batch, features, window)
        encoded = self.encoder(x.transpose(1, 2))
        return self.head(self.drop(encoded[:, :, -1]))


class _CNN_LSTM(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.conv = nn.Sequential(
            nn.Conv1d(spec.n_features, spec.hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=spec.hidden, hidden_size=spec.hidden, batch_first=True)
        self.drop = nn.Dropout(spec.dropout)
        self.head = nn.Linear(spec.hidden, spec.n_classes)

    def forward(self, x: Tensor) -> Tensor:
        _check_input(x, self.spec.n_features)
        encoded = self.conv(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm(encoded)
        return self.head(self.drop(output[:, -1, :]))


class _CausalConv1d(nn.Module):
    """Left-padded convolution: output at ``t`` reads inputs ``<= t`` only."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.left = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        padded = torch.nn.functional.pad(x, (self.left, 0))
        return self.conv(padded)


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel, dilation)
        self.conv2 = _CausalConv1d(channels, channels, kernel, dilation)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.drop(self.act(self.conv1(x)))
        out = self.drop(self.act(self.conv2(out)))
        return out + x


class CausalTCN(nn.Module):
    """Dilated causal TCN with a configurable receptive field.

    ``receptive_field = 1 + 2 * (kernel - 1) * sum(dilations)``: two
    convolutions per residual block, each adding ``(kernel - 1) * dilation``
    past bars.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        kernel: int = 3,
        levels: int = 3,
    ) -> None:
        super().__init__()
        if kernel < 2:
            raise ValueError("kernel must be >= 2")
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.spec = spec
        self.kernel = kernel
        self.dilations = tuple(2**level for level in range(levels))
        self.receptive_field = 1 + 2 * (kernel - 1) * sum(self.dilations)
        self.input = _CausalConv1d(spec.n_features, spec.hidden, 1, 1)
        self.blocks = nn.Sequential(
            *[
                _TCNBlock(spec.hidden, kernel, dilation, spec.dropout)
                for dilation in self.dilations
            ]
        )
        self.head = nn.Linear(spec.hidden, spec.n_classes)

    def forward_sequence(self, x: Tensor) -> Tensor:
        """Per-step hidden states ``(batch, window, hidden)``; causal in ``t``."""
        _check_input(x, self.spec.n_features)
        hidden = self.input(x.transpose(1, 2))
        hidden = self.blocks(hidden)
        return hidden.transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        sequence = self.forward_sequence(x)
        return self.head(sequence[:, -1, :])


def build_model(
    name: ModelName,
    *,
    n_features: int,
    window: int = 60,
    n_classes: int = 3,
    hidden: int = 64,
    dropout: float = 0.1,
    seed: int = 0,
) -> nn.Module:
    """Build one model with deterministic initialisation for ``seed``."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    spec = ModelSpec(
        n_features=n_features,
        window=window,
        n_classes=n_classes,
        hidden=hidden,
        dropout=dropout,
        seed=seed,
    )
    torch.manual_seed(spec.seed)
    if name == "mlp":
        return _MLP(spec, depth=1)
    if name == "deep_mlp":
        return _MLP(spec, depth=4)
    if name == "residual_mlp":
        return _ResidualMLP(spec)
    if name in ("rnn", "lstm", "gru"):
        return _Recurrent(spec, kind=name)
    if name == "cnn":
        return _CNN(spec, dilations=(1,))
    if name == "dilated_cnn":
        return _CNN(spec, dilations=(1, 2, 4))
    if name == "cnn_lstm":
        return _CNN_LSTM(spec)
    if name == "tcn":
        return CausalTCN(spec)
    from scalping.nn.transformers import build_transformer

    return build_transformer(
        n_features=spec.n_features,
        window=spec.window,
        n_classes=spec.n_classes,
        hidden=spec.hidden,
        dropout=spec.dropout,
        causal=name == "causal_transformer",
    )


def model_names() -> tuple[str, ...]:
    """Registered model names as plain strings (for CLI/API choices)."""
    return tuple(MODEL_NAMES)
