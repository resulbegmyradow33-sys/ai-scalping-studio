"""Multiple-testing and data-snooping control for one discovery winner.

A search that evaluates ``m`` candidates and keeps the best one has already
looked at ``m`` chances to get lucky. The winner's mean R is therefore an
upward-biased estimate, and an ordinary one-sided test at ``alpha = 0.05``
answers the wrong question: it asks whether *this* candidate looks good, not
whether the *search* produced something good. This module adds two explicit,
documented corrections and returns a screening report. It never promotes.

What is computed
----------------
1. **Family-wise significance.** The one-sided test is a Student-t test of
   ``H0: mean R <= 0`` over the per-trade ``r_multiple`` values with ``n - 1``
   degrees of freedom. The per-test alpha is Bonferroni's
   ``family_alpha / m`` and, for reference, Sidak's
   ``1 - (1 - family_alpha) ** (1 / m)``. The reported and gating value is the
   more conservative of the two (Bonferroni), because Bonferroni controls the
   family-wise error rate without assuming independent trials. The adjusted
   significance threshold in R is ``t_crit(n - 1, 1 - adjusted_alpha) * se``.

2. **Deflated effect floor.** Statistical significance is not economic
   substance. The required mean R is raised by a winner's-curse penalty scaled
   by the standard error::

       search_term     = sqrt(2 * ln(max(trial_count, 1)))
       complexity_term = 0.25 * ln(1 + complexity)
       penalty_r       = min(1.0, 0.5 * se * (search_term + complexity_term))
       deflated_r      = max(min_effect, significance_threshold_r + penalty_r)

   ``sqrt(2 ln m)`` is the leading order of the expected maximum of ``m``
   independent standard normals, the same intuition behind a deflated Sharpe
   ratio. It is a heuristic, not a calibrated probability. The complexity term
   charges for a larger hypothesis space, the ``0.5`` factor keeps the penalty
   from double-counting the critical value, and the ``1.0 R`` cap stops an
   extreme trial count from producing an unbounded requirement. The result is
   deliberately coarse: no decimal of it should be read as precision.

A candidate passes only when the observed mean R is at least the deflated
threshold. The naive confidence interval is reported for context; it is a
seeded percentile bootstrap that ignores selection and serial dependence.

What this does not do
---------------------
* It does not promote, rank or select. :func:`promotion_advice` is advice only;
  the named, versioned promotion policy remains the gate.
* It cannot see unrecorded trials. If the search space was changed, stopped
  adaptively or partly discarded, the trial count is wrong and the correction
  cannot repair it.
* It gives no look-elsewhere guarantee over an uncountable or unbounded search
  space. It corrects the trials the caller declares.
* It assumes trade R multiples are iid enough for a t-test and a bootstrap.
  Streak clustering, regime dependence and cost changes can make the interval
  too narrow.
* It is not a p-hacking panacea. A result that passes is a candidate for
  independent out-of-sample evidence, not evidence of a durable edge.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from scalping.types import Trade

__all__ = [
    "DEFAULT_FOLDS",
    "MAX_SEARCH_BUDGET",
    "MIN_SELECTION_TRADES",
    "MIN_TRADES_PER_FOLD",
    "SELECTION_CAVEAT",
    "PromotionAdvice",
    "SelectionConfig",
    "SelectionError",
    "SelectionReport",
    "promotion_advice",
    "selection_report",
]

#: Default contiguous folds used only to check that every part of the sample is
#: populated; they are not an out-of-sample test.
DEFAULT_FOLDS: Final[int] = 3

#: A fold needs at least this many trades before it says anything about a part
#: of the sample. This mirrors the evidence bundle's default.
MIN_TRADES_PER_FOLD: Final[int] = 5

#: Absolute floor for a mean-R test, independent of the fold count.
MIN_SELECTION_TRADES: Final[int] = 10

#: Refuse an absurd search budget before alpha underflows or the report becomes
#: meaningless. One million declared trials is already far beyond this project.
MAX_SEARCH_BUDGET: Final[int] = 1_000_000

#: Refuse a fold count that cannot be populated; a diagnostic, not a fine grid.
MAX_FOLDS: Final[int] = 100

#: A structural complexity beyond this is a data error, not a plausible DSL.
_MAX_COMPLEXITY: Final[int] = 1_000_000

#: Bound the bootstrap seed so a nonsense integer cannot be handed to NumPy.
_MAX_SEED: Final[int] = 2**63 - 1

#: Seeded percentile-bootstrap draws for the naive interval.
_NAIVE_BOOTSTRAP_SAMPLES: Final[int] = 1000

#: Penalty coefficients. They are stated, bounded and heuristic.
_PENALTY_SCALE: Final[float] = 0.5
_COMPLEXITY_WEIGHT: Final[float] = 0.25
_MAX_SELECTION_PENALTY_R: Final[float] = 1.0

_TOLERANCE: Final[float] = 1e-12

SELECTION_CAVEAT: Final[str] = (
    "Screening control, not a promotion. The correction trusts the recorded trial "
    "count: unrecorded trials, adaptive stopping, a changed search space or a "
    "stopped search can still invalidate it. Bonferroni gates and does not assume "
    "independent trials; Sidak is reference only. The bootstrap interval is iid "
    "and ignores serial dependence. The deflated floor is a heuristic winner's-"
    "curse penalty, not a calibrated probability. The versioned promotion policy "
    "remains the gate."
)


class SelectionError(ValueError):
    """Selection input is invalid or too small; ``code`` is machine-readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Search exposure and the correction the report should apply.

    ``trial_count`` is how many candidates the search actually scored before it
    produced this winner, including rejected ones. ``search_budget`` is the cap
    the run declared; a trial count above it is an inconsistent record and is
    refused. ``family_alpha`` is the family-wise error rate, not the per-test
    rate. ``min_effect`` is the economically meaningful mean R before any
    deflation; it is a floor, never a target. ``complexity`` is the candidate's
    structural cost and only enters through the documented logarithm.
    ``folds`` is the number of contiguous chunks used for the minimum-sample
    check. ``seed`` makes the naive bootstrap reproducible.

    Construction does not validate the numbers; :func:`selection_report` does,
    so the refusal carries a stable code instead of a bare ``ValueError``.
    """

    trial_count: int = 1
    search_budget: int = 100
    family_alpha: float = 0.05
    min_effect: float = 0.0
    complexity: int = 0
    folds: int = DEFAULT_FOLDS
    seed: int = 0


@dataclass(frozen=True, slots=True)
class SelectionReport:
    """Screening statistics for one candidate, with every input that produced them."""

    trades: int
    folds: int
    fold_trades: tuple[int, ...]
    mean_r: float
    std_r: float
    standard_error: float
    t_statistic: float
    naive_p_value: float
    bonferroni_p_value: float
    naive_ci: tuple[float, float]
    trial_count: int
    search_budget: int
    complexity: int
    seed: int
    family_alpha: float
    bonferroni_alpha: float
    sidak_alpha: float
    adjusted_alpha: float
    significance_threshold_r: float
    deflated_threshold_r: float
    min_effect: float
    required_trades: int
    passed: bool
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe payload; tuples become lists and no value is NaN or infinity."""
        return {
            "trades": self.trades,
            "folds": self.folds,
            "fold_trades": list(self.fold_trades),
            "mean_r": self.mean_r,
            "std_r": self.std_r,
            "standard_error": self.standard_error,
            "t_statistic": self.t_statistic,
            "naive_p_value": self.naive_p_value,
            "bonferroni_p_value": self.bonferroni_p_value,
            "naive_ci": [self.naive_ci[0], self.naive_ci[1]],
            "trial_count": self.trial_count,
            "search_budget": self.search_budget,
            "complexity": self.complexity,
            "seed": self.seed,
            "family_alpha": self.family_alpha,
            "bonferroni_alpha": self.bonferroni_alpha,
            "sidak_alpha": self.sidak_alpha,
            "adjusted_alpha": self.adjusted_alpha,
            "significance_threshold_r": self.significance_threshold_r,
            "deflated_threshold_r": self.deflated_threshold_r,
            "min_effect": self.min_effect,
            "required_trades": self.required_trades,
            "passed": self.passed,
            "caveat": self.caveat,
        }


@dataclass(frozen=True, slots=True)
class PromotionAdvice:
    """A recommendation only; it cannot promote and carries no policy authority."""

    recommendation: str
    reasons: tuple[str, ...]
    passed: bool
    caveat: str

    def __post_init__(self) -> None:
        if self.recommendation not in {"accept", "reject", "insufficient_evidence"}:
            raise ValueError(f"unknown recommendation {self.recommendation!r}")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")
        if not isinstance(self.caveat, str) or not self.caveat.strip():
            raise ValueError("advice caveat must be a non-empty string")
        if not self.reasons:
            raise ValueError("promotion advice must explain itself")
        for reason in self.reasons:
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("advice reasons must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "reasons": list(self.reasons),
            "passed": self.passed,
            "caveat": self.caveat,
        }


def selection_report(trades: Sequence[Trade], config: SelectionConfig) -> SelectionReport:
    """Apply the declared corrections to one candidate's closed trades.

    ``trades`` must be in chronological order; the fold check splits that order
    into contiguous chunks and the sample floor is chosen so every chunk holds
    at least ``MIN_TRADES_PER_FOLD`` trades. The report is deterministic for
    identical inputs, including the seeded bootstrap.

    Raises :class:`SelectionError` with a stable code for an invalid exposure,
    a non-finite R multiple, a sample below ``max(folds * 5, 10)`` or a
    degenerate zero-variance sample. A zero-variance sample has no usable
    standard error, so it is refused rather than reported as infinitely
    significant.
    """
    _validate_config(config)
    values = _trade_returns(trades)
    count = len(values)
    required = _required_trades(config.folds)
    if count < required:
        raise SelectionError(
            "insufficient_trades",
            f"need at least {required} trades for {config.folds} fold(s) (got {count})",
        )
    fold_trades = _fold_sizes(count, config.folds)
    data = np.asarray(values, dtype=np.float64)
    mean_r = _finite(float(np.mean(data)), where="mean R")
    std_r = _finite(float(np.std(data, ddof=1)), where="sample standard deviation")
    if std_r <= 0.0:
        raise SelectionError(
            "degenerate_sample",
            "all R multiples are identical; the standard error is zero and a "
            "t-test would be undefined",
        )
    standard_error = _finite(std_r / math.sqrt(count), where="standard error")
    t_statistic = _finite(mean_r / standard_error, where="t statistic")
    degrees_of_freedom = count - 1
    naive_p_value = _finite(_student_t_sf(t_statistic, degrees_of_freedom), where="naive p-value")

    trial_count = config.trial_count
    bonferroni_alpha = _finite(config.family_alpha / trial_count, where="Bonferroni alpha")
    sidak_alpha = _finite(
        -math.expm1(math.log1p(-config.family_alpha) / trial_count),
        where="Sidak alpha",
    )
    adjusted_alpha = min(bonferroni_alpha, sidak_alpha)
    t_critical = _student_t_ppf(1.0 - adjusted_alpha, degrees_of_freedom)
    significance_threshold_r = _finite(t_critical * standard_error, where="significance threshold")

    search_term = math.sqrt(2.0 * math.log(max(trial_count, 1)))
    complexity_term = _COMPLEXITY_WEIGHT * math.log1p(config.complexity)
    penalty_r = min(
        _MAX_SELECTION_PENALTY_R,
        _PENALTY_SCALE * standard_error * (search_term + complexity_term),
    )
    deflated_threshold_r = _finite(
        max(config.min_effect, significance_threshold_r + penalty_r),
        where="deflated threshold",
    )
    bonferroni_p_value = _finite(min(1.0, trial_count * naive_p_value), where="Bonferroni p-value")
    passed = bool(
        mean_r >= deflated_threshold_r and bonferroni_p_value <= config.family_alpha + _TOLERANCE
    )

    confidence = 1.0 - config.family_alpha
    naive_ci = _bootstrap_mean_ci(
        data,
        samples=_NAIVE_BOOTSTRAP_SAMPLES,
        seed=config.seed,
        confidence=confidence,
    )

    return SelectionReport(
        trades=count,
        folds=config.folds,
        fold_trades=fold_trades,
        mean_r=mean_r,
        std_r=std_r,
        standard_error=standard_error,
        t_statistic=t_statistic,
        naive_p_value=naive_p_value,
        bonferroni_p_value=bonferroni_p_value,
        naive_ci=naive_ci,
        trial_count=trial_count,
        search_budget=config.search_budget,
        complexity=config.complexity,
        seed=config.seed,
        family_alpha=config.family_alpha,
        bonferroni_alpha=bonferroni_alpha,
        sidak_alpha=sidak_alpha,
        adjusted_alpha=adjusted_alpha,
        significance_threshold_r=significance_threshold_r,
        deflated_threshold_r=deflated_threshold_r,
        min_effect=config.min_effect,
        required_trades=required,
        passed=passed,
        caveat=SELECTION_CAVEAT,
    )


def promotion_advice(report: SelectionReport, *, min_trades: int | None = None) -> PromotionAdvice:
    """Recommend ``accept``, ``reject`` or ``insufficient_evidence``.

    This is a recommendation, never a promotion. Even ``accept`` means only
    "this screening control did not stop the candidate"; the versioned promotion
    policy still judges identity, folds, locked out-of-sample evidence, cost
    stress, regime coverage and the champion comparison.

    ``reject`` is used only when the naive interval rules out the configured
    minimum effect, which is evidence against the required edge.
    ``insufficient_evidence`` is used when the threshold is not met but the
    interval still leaves the required effect possible, including a sample
    below the configured floor. That distinction keeps statistical
    non-significance from being reported as proof of no edge.
    """
    if not isinstance(report, SelectionReport):
        raise TypeError(f"report must be a SelectionReport, got {type(report).__name__}")
    if min_trades is None:
        required = report.required_trades
    else:
        if isinstance(min_trades, bool) or not isinstance(min_trades, int):
            raise TypeError("min_trades must be an int or None")
        if min_trades < 1:
            raise ValueError("min_trades must be >= 1")
        required = max(report.required_trades, min_trades)

    if report.trades < required:
        return PromotionAdvice(
            recommendation="insufficient_evidence",
            reasons=(
                (
                    f"{report.trades} trades are below the {required}-trade floor; "
                    "the screening control cannot separate edge from noise"
                ),
                "The versioned promotion policy remains the gate; this is advice only.",
            ),
            passed=False,
            caveat=report.caveat,
        )

    if report.passed:
        return PromotionAdvice(
            recommendation="accept",
            reasons=(
                (
                    f"mean R {report.mean_r:.4f} meets the deflated floor "
                    f"{report.deflated_threshold_r:.4f}"
                ),
                (
                    f"Bonferroni-adjusted p {report.bonferroni_p_value:.6f} is within "
                    f"family alpha {report.family_alpha:.4f}"
                ),
                (
                    "This is a screening recommendation only; the versioned promotion "
                    "policy remains the gate."
                ),
            ),
            passed=True,
            caveat=report.caveat,
        )

    reasons: list[str] = []
    if report.mean_r < report.significance_threshold_r:
        reasons.append(
            f"mean R {report.mean_r:.4f} is below the multiple-testing significance "
            f"threshold {report.significance_threshold_r:.4f}"
        )
    elif report.mean_r < report.deflated_threshold_r:
        reasons.append(
            f"mean R {report.mean_r:.4f} is significant after correction but below "
            f"the deflated effect floor {report.deflated_threshold_r:.4f}"
        )
    else:
        reasons.append(
            f"Bonferroni-adjusted p {report.bonferroni_p_value:.6f} exceeds family "
            f"alpha {report.family_alpha:.4f}"
        )

    if report.naive_ci[1] < report.min_effect:
        reasons.append(
            f"the naive interval upper bound {report.naive_ci[1]:.4f} rules out the "
            f"minimum effect {report.min_effect:.4f}"
        )
        recommendation = "reject"
    else:
        reasons.append(
            "the naive interval still leaves the minimum effect possible, so the "
            "result is not sufficient to reject or accept"
        )
        recommendation = "insufficient_evidence"
    reasons.append("The versioned promotion policy remains the gate; this is advice only.")

    return PromotionAdvice(
        recommendation=recommendation,
        reasons=tuple(reasons),
        passed=False,
        caveat=report.caveat,
    )


# --- validation ---------------------------------------------------------------


def _validate_config(config: SelectionConfig) -> None:
    if not isinstance(config, SelectionConfig):
        raise TypeError(f"config must be a SelectionConfig, got {type(config).__name__}")
    for name in ("trial_count", "search_budget", "complexity", "folds", "seed"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise SelectionError(
                f"invalid_{name}", f"{name} must be an int, got {type(value).__name__}"
            )
    if config.trial_count < 1:
        raise SelectionError(
            "invalid_trial_count",
            f"trial_count must be >= 1, got {config.trial_count}",
        )
    if config.search_budget < 1 or config.search_budget > MAX_SEARCH_BUDGET:
        raise SelectionError(
            "invalid_search_budget",
            f"search_budget must be in [1, {MAX_SEARCH_BUDGET}], got {config.search_budget}",
        )
    if config.trial_count > config.search_budget:
        raise SelectionError(
            "trial_count_exceeds_budget",
            f"trial_count {config.trial_count} exceeds search_budget {config.search_budget}",
        )
    alpha = config.family_alpha
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or not 0.0 < float(alpha) < 0.5
    ):
        raise SelectionError(
            "invalid_family_alpha",
            f"family_alpha must be finite and in (0, 0.5), got {alpha!r}",
        )
    effect = config.min_effect
    if (
        isinstance(effect, bool)
        or not isinstance(effect, (int, float))
        or not math.isfinite(float(effect))
        or float(effect) < 0.0
    ):
        raise SelectionError(
            "invalid_min_effect",
            f"min_effect must be finite and >= 0, got {effect!r}",
        )
    if config.complexity < 0 or config.complexity > _MAX_COMPLEXITY:
        raise SelectionError(
            "invalid_complexity",
            f"complexity must be in [0, {_MAX_COMPLEXITY}], got {config.complexity}",
        )
    if config.folds < 1 or config.folds > MAX_FOLDS:
        raise SelectionError(
            "invalid_folds",
            f"folds must be in [1, {MAX_FOLDS}], got {config.folds}",
        )
    if config.seed < 0 or config.seed > _MAX_SEED:
        raise SelectionError(
            "invalid_seed",
            f"seed must be in [0, {_MAX_SEED}], got {config.seed}",
        )


def _trade_returns(trades: Sequence[Trade]) -> list[float]:
    try:
        items = list(trades)
    except TypeError as error:
        raise SelectionError(
            "invalid_trades", "trades must be a sequence of Trade values"
        ) from error
    if not items:
        raise SelectionError("insufficient_trades", "no trades were supplied")
    values: list[float] = []
    for index, trade in enumerate(items):
        if not isinstance(trade, Trade):
            raise SelectionError(
                "invalid_trades",
                f"trade {index} is {type(trade).__name__}, not a Trade",
            )
        raw = trade.r_multiple
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise SelectionError(
                "invalid_trade_return",
                f"trade {index} r_multiple is {type(raw).__name__}, not a number",
            )
        value = float(raw)
        if not math.isfinite(value):
            raise SelectionError(
                "non_finite_trade_return",
                f"trade {index} r_multiple is not finite ({raw!r})",
            )
        values.append(value)
    return values


def _required_trades(folds: int) -> int:
    return max(folds * MIN_TRADES_PER_FOLD, MIN_SELECTION_TRADES)


def _fold_sizes(count: int, folds: int) -> tuple[int, ...]:
    base, remainder = divmod(count, folds)
    return tuple(base + (1 if index < remainder else 0) for index in range(folds))


def _finite(value: float, *, where: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SelectionError("non_finite_statistic", f"{where} is not finite ({number!r})")
    return number


# --- Student-t distribution (no SciPy dependency) -----------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (Lentz method)."""
    max_iterations = 300
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` for positive ``a`` and ``b``."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("betainc parameters must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_sf(t: float, degrees_of_freedom: int) -> float:
    """Upper-tail probability ``P(T > t)`` for a Student-t variate."""
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be >= 1")
    if not math.isfinite(t):
        raise ValueError("t must be finite")
    if t == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + t * t)
    tail = 0.5 * _betainc(0.5 * degrees_of_freedom, 0.5, x)
    tail = min(1.0, max(0.0, tail))
    return tail if t > 0.0 else 1.0 - tail


def _student_t_ppf(probability: float, degrees_of_freedom: int) -> float:
    """Quantile of the Student-t distribution for ``0.5 < probability < 1``."""
    if not 0.5 < probability < 1.0:
        raise ValueError("probability must be in (0.5, 1)")
    target = 1.0 - probability
    low, high = 0.0, 1.0
    while _student_t_sf(high, degrees_of_freedom) > target:
        high *= 2.0
        if high > 1.0e12:
            raise SelectionError("non_finite_statistic", "Student-t quantile did not bracket")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _student_t_sf(middle, degrees_of_freedom) > target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


# --- naive bootstrap ----------------------------------------------------------


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    """Seeded percentile bootstrap of the mean; iid and unadjusted for selection."""
    if values.ndim != 1 or values.size == 0:
        raise SelectionError("insufficient_trades", "bootstrap needs at least one value")
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    outcomes = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = values[rng.integers(0, values.size, values.size)]
        outcomes[index] = float(np.mean(draw))
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(outcomes, [tail, 100.0 - tail])
    return (
        _finite(float(low), where="naive CI lower bound"),
        _finite(float(high), where="naive CI upper bound"),
    )
