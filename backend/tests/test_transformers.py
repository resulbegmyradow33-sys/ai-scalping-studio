"""Transformer models: guard rails, causality and a memory smoke test."""

from __future__ import annotations

import pytest
import torch

from scalping.nn import build_model
from scalping.nn.transformers import (
    CausalTransformer,
    PatchTSTClassifier,
    TransformerConfig,
    build_patchtst,
)

N_FEATURES = 5
WINDOW = 24
N_CLASSES = 3


def tensor(window: int = WINDOW) -> torch.Tensor:
    torch.manual_seed(5)
    return torch.randn(3, window, N_FEATURES)


def test_encoder_and_causal_variant_return_logits() -> None:
    for name in ("transformer", "causal_transformer"):
        model = build_model(
            name,  # type: ignore[arg-type]
            n_features=N_FEATURES,
            window=WINDOW,
            n_classes=N_CLASSES,
            hidden=16,
        ).eval()
        with torch.no_grad():
            logits = model(tensor())
        assert logits.shape == (3, N_CLASSES)
        assert torch.isfinite(logits).all()


def test_config_guard_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="divisible"):
        build_model(
            "transformer",
            n_features=N_FEATURES,
            window=WINDOW,
            hidden=10,
        )
    with pytest.raises(ValueError, match="max_sequence"):
        build_model(
            "transformer",
            n_features=N_FEATURES,
            window=600,
            hidden=16,
        )
    with pytest.raises(ValueError, match="n_layers"):
        TransformerConfig(n_layers=9)
    with pytest.raises(ValueError, match="max_sequence"):
        TransformerConfig(max_sequence=4)
    with pytest.raises(ValueError, match="n_heads"):
        TransformerConfig(n_heads=0)


def test_causal_transformer_never_reads_the_future() -> None:
    model = build_model(
        "causal_transformer",
        n_features=N_FEATURES,
        window=WINDOW,
        n_classes=N_CLASSES,
        hidden=8,
    ).eval()
    assert isinstance(model, CausalTransformer)
    base = tensor()
    mutated = base.clone()
    mutated[:, 10:, :] += 3.0

    with torch.no_grad():
        left = model.forward_sequence(base)
        right = model.forward_sequence(mutated)

    torch.testing.assert_close(left[:, :10], right[:, :10])
    assert not torch.equal(left[:, 10:], right[:, 10:])


def test_bidirectional_encoder_may_use_the_whole_window() -> None:
    model = build_model(
        "transformer",
        n_features=N_FEATURES,
        window=WINDOW,
        n_classes=N_CLASSES,
        hidden=8,
    ).eval()
    base = tensor()
    mutated = base.clone()
    mutated[:, 10:, :] += 3.0

    with torch.no_grad():
        left = model.forward_sequence(base)
        right = model.forward_sequence(mutated)

    # Bidirectional attention means an earlier step sees the changed suffix.
    assert not torch.equal(left[:, :10], right[:, :10])


def test_default_model_stays_small() -> None:
    model = build_model(
        "transformer",
        n_features=59,
        window=60,
        n_classes=3,
        hidden=32,
        dropout=0.1,
    )
    parameters = sum(value.numel() for value in model.parameters())

    assert parameters < 2_000_000


def test_patchtst_patch_math_and_output() -> None:
    model = build_model(
        "patchtst",
        n_features=N_FEATURES,
        window=24,
        n_classes=N_CLASSES,
        hidden=8,
    ).eval()
    assert isinstance(model, PatchTSTClassifier)
    assert (model.patch_len, model.stride, model.n_patches) == (8, 8, 3)

    with torch.no_grad():
        logits = model(tensor(24))
    assert logits.shape == (3, N_CLASSES)
    assert torch.isfinite(logits).all()

    strided = build_patchtst(
        n_features=N_FEATURES,
        window=24,
        n_classes=N_CLASSES,
        hidden=8,
        patch_len=8,
        stride=4,
    )
    assert strided.n_patches == 5


def test_patchtst_drops_a_trailing_partial_patch() -> None:
    model = build_patchtst(
        n_features=N_FEATURES,
        window=25,
        n_classes=N_CLASSES,
        hidden=8,
        patch_len=8,
    ).eval()

    assert model.n_patches == 3  # bars 24 is not a full patch and is not padded
    with torch.no_grad():
        assert model(tensor(25)).shape == (3, N_CLASSES)


def test_patchtst_guard_rails() -> None:
    with pytest.raises(ValueError, match="shorter than patch_len"):
        build_patchtst(
            n_features=N_FEATURES, window=4, n_classes=2, hidden=8, patch_len=8
        )
    with pytest.raises(ValueError, match="patch_len"):
        build_patchtst(
            n_features=N_FEATURES, window=8, n_classes=2, hidden=8, patch_len=0
        )
    with pytest.raises(ValueError, match="stride"):
        build_patchtst(
            n_features=N_FEATURES, window=8, n_classes=2, hidden=8, stride=0
        )
    with pytest.raises(ValueError, match="max_sequence"):
        build_patchtst(
            n_features=N_FEATURES,
            window=100,
            n_classes=2,
            hidden=8,
            patch_len=10,
            config=TransformerConfig(max_sequence=8),
        )
    model = build_patchtst(
        n_features=N_FEATURES, window=16, n_classes=2, hidden=8
    )
    with pytest.raises(ValueError, match="fixed window"):
        model(tensor(24))


def test_patchtst_stays_small() -> None:
    model = build_patchtst(
        n_features=59, window=60, n_classes=3, hidden=32, dropout=0.1
    )
    assert sum(value.numel() for value in model.parameters()) < 2_000_000
