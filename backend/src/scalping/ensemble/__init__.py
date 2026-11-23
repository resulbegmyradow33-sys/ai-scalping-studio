"""Ensemble combination and the decision engine (Phase 9, slice 1).

Members are combined only on out-of-fold predictions. The module provides:

- :func:`~scalping.ensemble.core.fit_oof` — purged expanding-window OOF
  probabilities, using the fold geometry from :mod:`scalping.ml.cv`;
- :class:`~scalping.ensemble.core.SoftVotingEnsemble`,
  :class:`~scalping.ensemble.core.WeightedVotingEnsemble`,
  :class:`~scalping.ensemble.core.HardVotingEnsemble` and
  :class:`~scalping.ensemble.core.StackedEnsemble`;
- :func:`~scalping.ensemble.core.disagreement` and
  :func:`~scalping.ensemble.core.compare_to_best_member` for evidence;
- :class:`~scalping.ensemble.decision.DecisionEngine`, which maps a probability
  distribution and a disagreement score to ``BUY``, ``SELL``, ``HOLD`` or
  ``NO_TRADE`` with an explicit reason.

Not in this slice: regime-dependent member selection, temperature scaling,
drift/OOD scoring and persistence. Prediction metrics are not a trading edge;
the simulator and cost gate still decide that.
"""

from __future__ import annotations

from scalping.ensemble.core import (
    Ensemble,
    EnsembleConfig,
    HardVotingEnsemble,
    MemberFactory,
    OOFResult,
    SoftVotingEnsemble,
    StackedEnsemble,
    WeightedVotingEnsemble,
    WeightingMode,
    compare_to_best_member,
    disagreement,
    fit_oof,
)
from scalping.ensemble.decision import Action, Decision, DecisionEngine, Position

__all__ = [
    "Action",
    "Decision",
    "DecisionEngine",
    "Ensemble",
    "EnsembleConfig",
    "HardVotingEnsemble",
    "MemberFactory",
    "OOFResult",
    "Position",
    "SoftVotingEnsemble",
    "StackedEnsemble",
    "WeightedVotingEnsemble",
    "WeightingMode",
    "compare_to_best_member",
    "disagreement",
    "fit_oof",
]
