"""Model zoo: shapes, determinism and TCN causality."""

from __future__ import annotations

import pytest
import torch

from scalping.nn import MODEL_NAMES, ModelSpec, build_model
from scalping.nn.models import CausalTCN

N_FEATURES = 5
WINDOW = 30
N_CLASSES = 3


def feature_tensor(batch: int = 4, window: int = WINDOW) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(batch, window, N_FEATURES)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_returns_logits(name: str) -> None:
    model = build_model(
        name,  # type: ignore[arg-type]
        n_features=N_FEATURES,
        window=WINDOW,
        n_classes=N_CLASSES,
        hidden=16,
    ).eval()
    x = feature_tensor()

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (4, N_CLASSES)
    assert torch.isfinite(logits).all()
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_initialisation_is_seeded(name: str) -> None:
    kwargs = {
        "n_features": N_FEATURES,
        "window": WINDOW,
        "n_classes": N_CLASSES,
        "hidden": 16,
        "seed": 7,
    }
    first = build_model(name, **kwargs).eval()  # type: ignore[arg-type]
    second = build_model(name, **kwargs).eval()  # type: ignore[arg-type]
    x = feature_tensor()

    with torch.no_grad():
        assert torch.equal(first(x), second(x))


def test_different_seeds_change_the_weights() -> None:
    kwargs = {"n_features": N_FEATURES, "window": WINDOW, "n_classes": N_CLASSES, "hidden": 16}
    first = build_model("mlp", seed=1, **kwargs)
    second = build_model("mlp", seed=2, **kwargs)

    left = dict(first.named_parameters())
    right = dict(second.named_parameters())
    assert any(not torch.equal(left[key], right[key]) for key in left)


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("spectral", n_features=N_FEATURES)  # type: ignore[arg-type]


def test_input_contract_is_enforced() -> None:
    model = build_model("gru", n_features=N_FEATURES, hidden=8).eval()

    with pytest.raises(ValueError, match="batch, window, features"):
        model(torch.randn(4, WINDOW))
    with pytest.raises(ValueError, match="features"):
        model(torch.randn(4, WINDOW, N_FEATURES + 1))


def test_tcn_receptive_field_matches_the_formula() -> None:
    spec = ModelSpec(n_features=N_FEATURES, window=WINDOW, hidden=8)
    model = CausalTCN(spec, kernel=3, levels=3)

    # 1 + 2 * (kernel - 1) * (1 + 2 + 4)
    assert model.receptive_field == 29
    assert model.dilations == (1, 2, 4)


def test_tcn_never_reads_the_future_inside_the_window() -> None:
    spec = ModelSpec(n_features=N_FEATURES, window=WINDOW, hidden=8)
    model = CausalTCN(spec, kernel=3, levels=2).eval()
    base = feature_tensor(window=WINDOW)
    mutated = base.clone()
    mutated[:, 10:, :] += 5.0

    with torch.no_grad():
        left = model.forward_sequence(base)
        right = model.forward_sequence(mutated)

    # Step t reads inputs 0..t; mutating from 10 changes step 10 onward.
    torch.testing.assert_close(left[:, :10], right[:, :10])
    assert not torch.equal(left[:, 10:], right[:, 10:])


def test_bad_model_spec_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_features"):
        ModelSpec(n_features=0)
    with pytest.raises(ValueError, match="window"):
        ModelSpec(n_features=1, window=0)
    with pytest.raises(ValueError, match="n_classes"):
        ModelSpec(n_features=1, n_classes=1)
    with pytest.raises(ValueError, match="dropout"):
        ModelSpec(n_features=1, dropout=1.0)


def test_build_model_fails_closed_when_a_name_has_no_branch(monkeypatch) -> None:
    from scalping.nn import models

    monkeypatch.setattr(models, "MODEL_NAMES", (*models.MODEL_NAMES, "ghost"))
    with pytest.raises(AssertionError, match="unhandled model 'ghost'"):
        models.build_model("ghost", n_features=N_FEATURES, window=WINDOW)  # type: ignore[arg-type]
