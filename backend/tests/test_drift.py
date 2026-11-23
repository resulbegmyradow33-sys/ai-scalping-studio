"""Drift diagnostics: PSI, KS, feature reports and prediction/calibration drift."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scalping.registry import (
    DriftReport,
    FeatureDrift,
    ks_statistic,
    population_stability_index,
    prediction_drift,
)
from scalping.registry.drift import drift_from_datasets


def _samples(shift: float, size: int = 2_000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    reference = rng.normal(0.0, 1.0, size)
    current = rng.normal(shift, 1.0, size)
    return reference, current


# ------------------------------------------------------------------------ PSI


def test_psi_identical_samples_are_exactly_zero() -> None:
    reference, _ = _samples(0.0)
    assert population_stability_index(reference, reference.copy()) == 0.0
    assert population_stability_index(reference, reference) == 0.0
    assert population_stability_index([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == 0.0


def test_psi_known_shift_crosses_the_default_threshold() -> None:
    reference, current = _samples(2.0)
    psi = population_stability_index(reference, current)
    assert psi > 0.2
    assert np.isfinite(psi)


def test_psi_small_shift_stays_below_the_threshold() -> None:
    reference, _ = _samples(0.0)
    assert population_stability_index(reference, reference + 1e-9) < 0.01


def test_psi_constant_reference() -> None:
    constant = [5.0] * 20
    assert population_stability_index(constant, constant) == 0.0
    assert population_stability_index(constant, [6.0] * 20) > 0.2
    assert population_stability_index(constant, [4.0] * 20) > 0.2
    mixed = [5.0] * 10 + [6.0] * 10
    assert population_stability_index(constant, mixed) > 0.2


def test_psi_drops_non_finite_values() -> None:
    reference = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    current = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    assert population_stability_index(reference, current) == 0.0
    assert population_stability_index(reference, np.append(current, np.inf)) == 0.0


def test_psi_empty_or_all_non_finite_raises() -> None:
    with pytest.raises(ValueError, match="no finite values"):
        population_stability_index([], [1.0])
    with pytest.raises(ValueError, match="no finite values"):
        population_stability_index([1.0], [np.nan])
    with pytest.raises(ValueError, match="one-dimensional"):
        population_stability_index(np.zeros((2, 2)), [1.0])


def test_psi_requires_at_least_two_bins() -> None:
    with pytest.raises(ValueError, match="bins"):
        population_stability_index([1.0, 2.0], [1.0, 2.0], bins=1)
    with pytest.raises(TypeError):
        population_stability_index([1.0, 2.0], [1.0, 2.0], bins=2.5)  # type: ignore[arg-type]


def test_psi_is_deterministic() -> None:
    reference, current = _samples(1.0)
    first = population_stability_index(reference, current)
    second = population_stability_index(reference, current)
    assert first == second


# ------------------------------------------------------------------------- KS


def test_ks_disjoint_samples() -> None:
    d, p = ks_statistic([1, 2, 3, 4], [5, 6, 7, 8])
    assert d == 1.0
    assert 0.0 <= p < 0.05


def test_ks_hand_computed_partial_overlap() -> None:
    # CDFs: 1/3 vs 0 at 1, 2/3 vs 1/3 at 2, 1 vs 2/3 at 3, 1 vs 1 at 4.
    d, p = ks_statistic([1, 2, 3], [2, 3, 4])
    assert d == pytest.approx(1.0 / 3.0)
    assert 0.0 < p < 1.0


def test_ks_identical_samples() -> None:
    reference, _ = _samples(0.0)
    d, p = ks_statistic(reference, reference)
    assert d == 0.0
    assert p == 1.0


def test_ks_empty_raises() -> None:
    with pytest.raises(ValueError, match="no finite values"):
        ks_statistic([], [1.0])
    with pytest.raises(ValueError, match="no finite values"):
        ks_statistic([1.0], [np.nan])


def test_ks_is_deterministic() -> None:
    reference, current = _samples(0.5)
    assert ks_statistic(reference, current) == ks_statistic(reference, current)


# ------------------------------------------------------------------ features


def _bundle(offset: float = 0.0, size: int = 1_000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "atr": rng.normal(0.0, 1.0, size),
        "spread": rng.normal(0.0, 1.0, size) + offset,
        "flow": rng.normal(0.0, 1.0, size),
    }


def test_drift_from_datasets_flags_the_shifted_column() -> None:
    report = drift_from_datasets(_bundle(), _bundle(offset=2.0))
    assert isinstance(report, DriftReport)
    assert report.drifted == ("spread",)
    assert report.features["spread"].psi is not None
    assert report.features["spread"].psi > 0.2
    assert report.features["atr"].psi is not None
    assert report.features["atr"].psi < 0.2
    assert "1 drifted" in report.summary
    assert "psi > 0.2" in report.summary


def test_drift_from_datasets_no_drift() -> None:
    report = drift_from_datasets(_bundle(), _bundle())
    assert report.drifted == ()
    assert "none drifted" in report.summary
    assert report.notes == ()


def test_drift_from_datasets_counts_dropped_rows() -> None:
    current = _bundle()
    current["flow"][:5] = np.nan
    current["flow"][5] = np.inf
    report = drift_from_datasets(_bundle(), current)

    feature = report.features["flow"]
    assert feature.dropped_current == 6
    assert feature.dropped_reference == 0
    assert feature.psi is not None
    assert feature.note is not None and "dropped 0 reference / 6 current" in feature.note
    assert any(note.startswith("flow:") for note in report.notes)


def test_drift_from_datasets_all_nan_feature_is_skipped_not_stable() -> None:
    current = _bundle(size=10)
    current["flow"] = np.full(10, np.nan)
    report = drift_from_datasets(_bundle(size=10), current)

    feature = report.features["flow"]
    assert feature.psi is None
    assert feature.ks is None
    assert feature.note is not None and "no finite values" in feature.note
    assert "flow" not in report.drifted
    assert "1 skipped" in report.summary


def test_drift_from_datasets_handles_empty_columns() -> None:
    report = drift_from_datasets(
        {"a": np.asarray([], dtype=float)}, {"a": np.asarray([], dtype=float)}
    )
    assert report.features["a"].psi is None
    assert report.drifted == ()


def test_drift_from_datasets_selects_columns() -> None:
    report = drift_from_datasets(_bundle(), _bundle(offset=2.0), columns=["spread"])
    assert list(report.features) == ["spread"]
    assert report.drifted == ("spread",)


def test_drift_from_datasets_rejects_a_schema_mismatch() -> None:
    current = _bundle()
    del current["flow"]
    with pytest.raises(ValueError, match="missing from a bundle"):
        drift_from_datasets(_bundle(), current)


def test_drift_from_datasets_rejects_a_non_finite_threshold() -> None:
    with pytest.raises(ValueError, match="psi_threshold"):
        drift_from_datasets(_bundle(), _bundle(), psi_threshold=-1.0)


def test_drift_from_datasets_accepts_feature_frame_like_bundles() -> None:
    names = ["atr", "spread"]
    reference = SimpleNamespace(X=_bundle()["atr"][:, None].repeat(2, axis=1), columns=names)
    shifted = _bundle(offset=3.0)
    current = SimpleNamespace(X=shifted["atr"][:, None].repeat(2, axis=1), columns=names)
    report = drift_from_datasets(reference, current)
    assert set(report.features) == set(names)
    # Both columns are identical arrays in this stub, so neither can drift.
    assert report.drifted == ()


def test_drift_report_to_dict_is_json_serializable() -> None:
    report = drift_from_datasets(_bundle(), _bundle(offset=2.0))
    decoded = json.loads(json.dumps(report.to_dict(), sort_keys=True))
    assert decoded["drifted"] == ["spread"]
    assert decoded["features"]["spread"]["psi"] > 0.2
    assert decoded["summary"] == report.summary


def test_feature_drift_to_dict() -> None:
    feature = FeatureDrift(feature="x", psi=0.3, ks=0.4, ks_pvalue=0.01)
    assert feature.to_dict()["feature"] == "x"
    assert feature.to_dict()["psi"] == 0.3


# --------------------------------------------------------- predictions/E​CE


def _probabilities(sharpness: float, size: int = 600, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet([1.0, 1.0, 1.0], size=size)
    powered = raw**sharpness
    return powered / powered.sum(axis=1, keepdims=True)


def test_prediction_drift_detects_a_confidence_shift() -> None:
    reference = _probabilities(1.0)
    current = _probabilities(0.2)
    outcomes = np.random.default_rng(4).choice([-1, 1], size=600)

    report = prediction_drift(
        reference,
        current,
        reference_outcomes=outcomes,
        current_outcomes=outcomes,
    )
    assert report.drifted is True
    assert report.psi > 0.2
    assert 0.0 <= report.ks <= 1.0
    assert report.mean_ece_reference is not None
    assert report.mean_ece_current is not None
    assert report.ece_delta is not None
    assert "drifted" in report.summary


def test_prediction_drift_without_outcomes_reports_no_ece() -> None:
    reference = _probabilities(1.0)
    report = prediction_drift(reference, reference.copy())
    assert report.psi == 0.0
    assert report.drifted is False
    assert report.mean_ece_reference is None
    assert report.ece_delta is None
    assert any("outcomes" in note for note in report.notes)


def test_prediction_drift_identical_probabilities_and_outcomes() -> None:
    probabilities = _probabilities(1.0)
    outcomes = np.random.default_rng(5).choice([-1, 1], size=600)
    report = prediction_drift(
        probabilities,
        probabilities,
        reference_outcomes=outcomes,
        current_outcomes=outcomes,
    )
    assert report.psi == 0.0
    assert report.ks == 0.0
    assert report.ece_delta == pytest.approx(0.0)


def test_prediction_drift_requires_paired_outcomes() -> None:
    probabilities = _probabilities(1.0)
    with pytest.raises(ValueError, match="both samples or neither"):
        prediction_drift(probabilities, probabilities, reference_outcomes=np.ones(600))


def test_prediction_drift_validates_shapes_and_values() -> None:
    probabilities = _probabilities(1.0)
    with pytest.raises(ValueError, match="two-dimensional"):
        prediction_drift(probabilities[:, 0], probabilities[:, 0])
    with pytest.raises(ValueError, match="3 classes"):
        prediction_drift(probabilities[:, :2], probabilities[:, :2])
    with pytest.raises(ValueError, match="non-finite"):
        bad = probabilities.copy()
        bad[0, 0] = np.nan
        prediction_drift(bad, probabilities)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prediction_drift(probabilities * 2.0, probabilities)
    with pytest.raises(ValueError, match="sum to 1"):
        prediction_drift(np.full((4, 3), 0.5), np.full((4, 3), 0.5))


def test_prediction_drift_to_dict_and_determinism() -> None:
    reference = _probabilities(1.0, seed=9)
    current = _probabilities(0.3, seed=9)
    first = prediction_drift(reference, current)
    second = prediction_drift(reference, current)
    assert first == second

    payload: dict[str, Any] = first.to_dict()
    assert json.loads(json.dumps(payload))["drifted"] is True
    assert payload["summary"] == first.summary
