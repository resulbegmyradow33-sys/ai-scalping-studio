"""Probability averaging and the policy-ensemble wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.rl.ensemble import PolicyEnsemble, average_probabilities


def test_equal_weights_average_distributions() -> None:
    combined = average_probabilities(
        [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]
    )

    np.testing.assert_allclose(combined, [0.4, 0.2, 0.4])
    assert combined.sum() == pytest.approx(1.0)


def test_weights_are_normalised() -> None:
    combined = average_probabilities(
        [[0.8, 0.1, 0.1], [0.2, 0.2, 0.6]], weights=[3.0, 1.0]
    )

    np.testing.assert_allclose(combined, [0.65, 0.125, 0.225])


def test_unnormalised_members_are_normalised() -> None:
    combined = average_probabilities([[4.0, 1.0], [1.0, 4.0]])

    np.testing.assert_allclose(combined, [0.5, 0.5])


@pytest.mark.parametrize(
    "distributions",
    [
        [],
        [[0.5, 0.5], [0.2, 0.2, 0.6]],
        [[-0.1, 1.1]],
        [[np.nan, 1.0]],
        [[0.0, 0.0]],
    ],
)
def test_bad_distributions_are_rejected(distributions) -> None:
    with pytest.raises(ValueError):
        average_probabilities(distributions)


def test_bad_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="match members"):
        average_probabilities([[0.5, 0.5]], weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        average_probabilities([[0.5, 0.5]], weights=[-1.0])
    with pytest.raises(ValueError, match="positive"):
        average_probabilities([[0.5, 0.5]], weights=[0.0])


def test_policy_ensemble_acts_and_ties_go_low() -> None:
    first = lambda obs: np.array([0.6, 0.2, 0.2])
    second = lambda obs: np.array([0.2, 0.2, 0.6])
    ensemble = PolicyEnsemble([first, second])

    assert ensemble.act(np.zeros(3)) == 0
    np.testing.assert_allclose(ensemble.probabilities(np.zeros(3)), [0.4, 0.2, 0.4])
    assert ensemble.to_dict() == {"members": 2, "weights": None}

    tie = PolicyEnsemble([lambda obs: [0.5, 0.5]])
    assert tie.act(np.zeros(1)) == 0


def test_policy_ensemble_guard_rails() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PolicyEnsemble([])
    with pytest.raises(TypeError, match="callable"):
        PolicyEnsemble([1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="match members"):
        PolicyEnsemble([lambda obs: [0.5, 0.5]], weights=[1.0, 2.0])
