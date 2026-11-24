"""Promotion-grade evidence for one DSL candidate: folds, stress, regimes, CIs.

A validation search winner is a candidate, not a champion. This module builds
the evidence a later promotion policy can inspect: several non-overlapping
chronological folds cut from the **train + validation** timeline only, each
evaluated with the same :class:`~scalping.discovery.evaluate.DslEvaluator`
semantics (net of costs, risk engine, weekly session), a cost-stress rerun per
fold, regime attribution at each trade's entry and an iid bootstrap confidence
interval over trade R multiples.

What it guarantees
------------------
* **No test rows.** The fold timeline ends at the dataset manifest's validation
  end. The final test split is never sliced, never named to an evaluator and
  never read from the feature frame. A dataset id naming test/holdout is
  refused.
* **Expanding, non-overlapping folds.** Block 0 is initial history. Fold ``k``
  evaluates the next contiguous block of the discovery timeline, keeps a
  purge/embargo gap immediately before it, and its training window is every row
  from the discovery start up to that gap. Boundaries depend only on row counts
  and configuration, so the same bundle always cuts the same folds.
* **Fail closed.** Fewer than three folds, an empty validation/test split, a
  fold below the trade minimum, a missing or no-op stress multiplier, missing
  regime labels, a non-finite number or any identity mismatch raises
  :class:`EvidenceError` with a stable code. No partial bundle is returned.
* **Not a promotion.** A bundle records evidence; it does not select, promote
  or claim profitability. The locked final OOS call and the multiple-testing
  correction are deliberately still missing (see
  ``docs/backtesting-methodology.md``).

Fold semantics
--------------
Rows are positions in the ``DatasetBundle`` arrays (the first bundle row is the
source frame's ``features.valid_from``). Test blocks tile the discovery timeline
after block 0, so they cannot overlap. For fold ``k``::

    train: [0, test_start - gap)
    gap:   purge + embargo bars immediately before the test block
    test:  [test_start, test_end)

``gap = max(config.purge_bars, manifest.purge_bars) +
max(config.embargo_bars, manifest.embargo_bars)``. The manifest already raised
its purge to at least the label horizon, so a fold boundary never re-introduces
label overlap the dataset builder removed. The dataset's own purged rows (the
gap the builder inserted between train and validation) remain in the timeline
as market bars; no label is read from them and they are not excluded from a
fold's test block.

The candidate is fixed (a DSL policy has no fitted parameters), so the training
range is recorded for provenance rather than refitted. "Expanding window" means
every later fold's recorded training history contains all earlier test blocks
except the trailing purge/embargo rows that sit immediately before the new test
block.

Confidence intervals
--------------------
Two 95% intervals are reported from the pooled per-trade ``r_multiple`` values:
one for mean R (expectancy) and one for total R. The resampling is a plain iid
bootstrap with replacement, ``bootstrap_samples`` draws and a fixed seed. Trade
R multiples are already net of costs; the bootstrap does not re-price them. Iid
is a stated simplification: it destroys streak clustering, so an interval can
look tighter than the trade sequence supports. A block bootstrap is a later
extension, not hidden here.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any, Final

import numpy as np
import polars as pl

from scalping.backtest.metrics import Metrics
from scalping.config import Settings
from scalping.datasets.builder import DatasetBundle
from scalping.discovery.evaluate import DslEvaluator
from scalping.discovery.protocols import canonical_digest, canonical_json
from scalping.features.build import FeatureFrame
from scalping.regime.rules import classify
from scalping.regime.strategy import UNKNOWN_REGIME
from scalping.strategy.dsl.nodes import DslValidationError, StrategySpec
from scalping.types import Trade

__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "MIN_EVIDENCE_FOLDS",
    "AggregateEvidence",
    "CandidateEvidence",
    "EvidenceConfig",
    "EvidenceError",
    "EvidencePolicy",
    "FoldEvidence",
    "LockedOOSEvidence",
    "RegimeEvidence",
    "SplitBoundaries",
    "StressEvidence",
    "bootstrap_ci",
    "build_evidence",
    "evidence_checksum",
    "evidence_from_dict",
    "locked_oos_checksum",
    "per_trade_regimes",
    "regime_labels",
    "regime_rows",
    "settings_digest",
    "slice_frame",
    "verify_evidence",
]

#: Layout version of :meth:`CandidateEvidence.to_dict`; bump on a breaking shape.
#: Version 3 adds the optional frozen ``locked_oos`` section produced by
#: :func:`scalping.discovery.holdout.attach_holdout`.
EVIDENCE_SCHEMA_VERSION: Final[int] = 3

#: A promotion policy needs at least three folds to say anything about stability.
MIN_EVIDENCE_FOLDS: Final[int] = 3

#: Dataset id tokens that name the untouched test/holdout split. Mirrors the
#: guard in :mod:`scalping.discovery.evaluate` without importing its private name.
_HOLDOUT_TOKENS: Final[frozenset[str]] = frozenset(
    {"test", "tests", "holdout", "final", "locked", "oos", "unseen", "outofsample"}
)

_TOKEN_SPLIT: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")

_BOOTSTRAP_STATISTICS: Final[frozenset[str]] = frozenset({"mean", "sum"})


class EvidenceError(ValueError):
    """Evidence cannot be built or verified; ``code`` is the machine-readable reason."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


# --- configuration ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    """Inputs a fold-level evidence build needs beyond the candidate and bundle.

    ``code_version`` is required: an evidence bundle without a code version
    cannot be replayed or invalidated. ``complexity`` is optional and, when set,
    must equal the spec's own structural cost so a caller cannot relabel it.
    ``trial_count`` and ``search_budget`` are the search exposure that produced
    the candidate; they belong in the identity even though they do not change
    the arithmetic.

    ``folds`` accepts any positive count at construction, but
    :func:`build_evidence` enforces :data:`MIN_EVIDENCE_FOLDS` so the refusal
    carries the evidence code ``insufficient_folds`` instead of a bare
    ``ValueError``. ``regime_coverage_min`` defaults to 1.0: every trade must be
    attributed to a regime unless a caller explicitly accepts less.
    """

    code_version: str
    symbol: str = "XAUUSD"
    timeframe: str = "M1"
    folds: int = 3
    purge_bars: int = 0
    embargo_bars: int = 0
    min_trades_per_fold: int = 5
    cost_stress: tuple[float, ...] = (1.5, 2.0)
    regime_coverage_min: float = 1.0
    bootstrap_samples: int = 1000
    bootstrap_confidence: float = 0.95
    seed: int = 0
    search_budget: int = 100
    trial_count: int = 1
    complexity: int | None = None
    window: int = 60

    def __post_init__(self) -> None:
        for name in ("code_version", "symbol", "timeframe"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty str")
        for name, minimum in (
            ("folds", 1),
            ("min_trades_per_fold", 1),
            ("bootstrap_samples", 1),
            ("search_budget", 1),
            ("trial_count", 1),
            ("window", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}, got {value}")
        for name in ("purge_bars", "embargo_bars", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.complexity is not None:
            if isinstance(self.complexity, bool) or not isinstance(self.complexity, int):
                raise TypeError("complexity must be an int or None")
            if self.complexity < 0:
                raise ValueError("complexity must be >= 0")
        if (
            not math.isfinite(float(self.regime_coverage_min))
            or not 0.0 <= float(self.regime_coverage_min) <= 1.0
        ):
            raise ValueError("regime_coverage_min must be a fraction in [0, 1]")
        if (
            not math.isfinite(float(self.bootstrap_confidence))
            or not 0.0 < float(self.bootstrap_confidence) < 1.0
        ):
            raise ValueError("bootstrap_confidence must be in (0, 1)")

        multipliers: list[float] = []
        for raw in self.cost_stress:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"cost_stress multipliers must be numbers, got {raw!r}")
            number = float(raw)
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"cost_stress multipliers must be finite and > 0, got {raw!r}")
            multipliers.append(number)
        labels = [f"x{value:g}" for value in multipliers]
        if len(set(labels)) != len(labels):
            raise ValueError("cost_stress multipliers must be distinct")
        object.__setattr__(self, "cost_stress", tuple(multipliers))


# --- immutable result types ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    """The fold/CI policy a bundle was built under, recorded for replay."""

    folds: int
    purge_bars: int
    embargo_bars: int
    min_trades_per_fold: int
    cost_stress: tuple[float, ...]
    regime_coverage_min: float
    bootstrap_samples: int
    bootstrap_confidence: float
    window: int

    def __post_init__(self) -> None:
        # A bundle decoded from stored JSON has lists where the dataclass has
        # tuples; normalise so a round-trip compares equal and the checksum is
        # stable regardless of the source.
        if isinstance(self.cost_stress, list):
            object.__setattr__(
                self, "cost_stress", tuple(float(value) for value in self.cost_stress)
            )


@dataclass(frozen=True, slots=True)
class RegimeEvidence:
    """One regime bucket: trades whose entry bar carried that label."""

    regime: str
    trades: int
    net_profit: float
    win_rate: float
    expectancy_r: float
    total_r: float


@dataclass(frozen=True, slots=True)
class StressEvidence:
    """One cost-stress rerun of a fold's fixed candidate."""

    multiplier: float
    trades: int
    net_profit: float
    profit_factor: float
    max_drawdown: float
    expectancy_r: float


@dataclass(frozen=True, slots=True)
class FoldEvidence:
    """One chronological test block and everything measured on it.

    ``train_start_row``/``test_end_row`` are bundle-row offsets, half-open.
    The timestamps are the first and last included bar of each range, ISO-8601
    UTC, so a reader can locate the fold without the bundle arrays.

    ``net_profit`` is the simulator's marked equity change, so it includes an
    open position at the fold's last bar. A per-regime ``net_profit`` sums
    closed-trade P&L only; the two need not add up exactly.
    """

    fold: int
    train_start_row: int
    train_end_row: int
    test_start_row: int
    test_end_row: int
    train_rows: int
    test_rows: int
    train_start_ts: str
    train_end_ts: str
    test_start_ts: str
    test_end_ts: str
    purge_bars: int
    embargo_bars: int
    trades: int
    net_profit: float
    profit_factor: float
    win_rate: float
    expectancy_r: float
    max_drawdown: float
    regimes: tuple[RegimeEvidence, ...]
    stress: tuple[StressEvidence, ...]


@dataclass(frozen=True, slots=True)
class AggregateEvidence:
    """Cross-fold summary.

    ``max_drawdown`` is the worst fold's drawdown, not a pooled equity curve:
    folds are separate simulations and their equity paths are not chained.
    ``positive_fold_fraction`` is the share of folds with positive net profit.
    """

    total_trades: int
    total_r: float
    expectancy_r: float
    profit_factor: float
    max_drawdown: float
    positive_folds: int
    positive_fold_fraction: float
    expectancy_r_ci95_low: float
    expectancy_r_ci95_high: float
    total_r_ci95_low: float
    total_r_ci95_high: float
    bootstrap_samples: int
    bootstrap_confidence: float
    bootstrap_method: str
    regime_coverage: float
    regime_trades: tuple[RegimeEvidence, ...]


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    """Dataset split edges as half-open ``DatasetBundle`` row offsets.

    ``train_start_row`` is the first bundle row, ``validation_end_row`` is where
    the discovery timeline stops and ``test_start_row`` is the first untouched
    test row. A promotion policy can therefore tell a selection fold from the
    locked holdout without re-deriving the manifest mapping.
    """

    train_start_row: int
    train_end_row: int
    validation_start_row: int
    validation_end_row: int
    test_start_row: int
    test_end_row: int


@dataclass(frozen=True, slots=True)
class LockedOOSEvidence:
    """The single authorised result of spending the locked final holdout.

    The section is a real dataclass field on :class:`CandidateEvidence`, not a
    dictionary a subclass can inject through ``to_dict``. Its own ``checksum``
    covers every field, and the promotion policy additionally requires its
    identity, policy and test-row range to match the bundle.
    """

    checksum: str
    evidence_checksum: str
    dataset_id: str
    dataset_checksum: str
    symbol: str
    timeframe: str
    spec_hash: str
    spec_version: int
    complexity: int
    code_version: str
    settings_digest: str
    policy_id: str
    policy_version: int
    test_start_row: int
    test_end_row: int
    test_rows: int
    test_start_ts: str
    test_end_ts: str
    trades: int
    net_profit: float
    profit_factor: float
    win_rate: float
    expectancy_r: float
    total_r: float
    max_drawdown: float
    expectancy_r_ci95_low: float
    expectancy_r_ci95_high: float
    total_r_ci95_low: float
    total_r_ci95_high: float
    bootstrap_samples: int
    bootstrap_confidence: float
    bootstrap_method: str
    regime_coverage: float
    regimes: tuple[RegimeEvidence, ...] = ()
    stress: tuple[StressEvidence, ...] = ()
    created_at: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "checksum",
            "evidence_checksum",
            "dataset_id",
            "dataset_checksum",
            "symbol",
            "timeframe",
            "spec_hash",
            "code_version",
            "settings_digest",
            "policy_id",
            "test_start_ts",
            "test_end_ts",
            "bootstrap_method",
            "created_at",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty str")
        for name in ("checksum", "evidence_checksum"):
            text = getattr(self, name)
            if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
                raise ValueError(f"{name} must be a lowercase sha256 hex digest")
        for name in (
            "spec_version",
            "complexity",
            "policy_version",
            "test_start_row",
            "test_end_row",
            "test_rows",
            "trades",
            "bootstrap_samples",
            "schema_version",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.trades < 1:
            raise ValueError("a locked holdout must report at least one trade")
        if self.policy_version < 1 or self.schema_version < 1:
            raise ValueError("policy_version and schema_version must be >= 1")
        if self.test_end_row <= self.test_start_row:
            raise ValueError("locked holdout test range must be non-empty")
        for name in (
            "net_profit",
            "profit_factor",
            "win_rate",
            "expectancy_r",
            "total_r",
            "max_drawdown",
            "expectancy_r_ci95_low",
            "expectancy_r_ci95_high",
            "total_r_ci95_low",
            "total_r_ci95_high",
            "bootstrap_confidence",
            "regime_coverage",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in ("win_rate", "regime_coverage"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a fraction in [0, 1]")
        if not 0.0 < float(self.bootstrap_confidence) < 1.0:
            raise ValueError("bootstrap_confidence must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Immutable evidence bundle for one candidate on one dataset build.

    ``checksum`` covers :meth:`to_dict` **without** ``checksum`` and
    ``created_at``, so the same inputs replay the same checksum while the
    creation stamp can differ. Identity fields make a stale bundle detectable:
    a different dataset, dataset checksum, spec, code version or execution
    settings must be rejected by :func:`verify_evidence` before any policy reads
    the numbers.
    """

    spec_hash: str
    spec_version: int
    complexity: int
    dataset_id: str
    dataset_checksum: str
    symbol: str
    timeframe: str
    code_version: str
    settings_digest: str
    seed: int
    trial_count: int
    search_budget: int
    splits: SplitBoundaries
    policy: EvidencePolicy
    folds: tuple[FoldEvidence, ...]
    aggregate: AggregateEvidence
    checksum: str
    created_at: str
    schema_version: int = EVIDENCE_SCHEMA_VERSION
    #: Optional frozen locked-final-holdout section. It is omitted from
    #: :meth:`to_dict` while ``None`` so every bundle built before this section
    #: existed keeps the same checksum; ``attach_holdout`` returns a new bundle
    #: with it filled in and the checksum re-derived.
    locked_oos: LockedOOSEvidence | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe nested payload; the checksum is computed over this minus stamps."""
        payload = asdict(self)
        if self.locked_oos is None:
            # Keep the historical payload byte-identical for bundles without a
            # holdout section. Adding a ``null`` key would change every checksum.
            payload.pop("locked_oos", None)
        return payload


# --- small helpers ------------------------------------------------------------


def _names_holdout(dataset_id: str) -> bool:
    lowered = dataset_id.lower()
    tokens = {token for token in _TOKEN_SPLIT.split(lowered) if token}
    tokens |= {token for token in _TOKEN_SPLIT.split(lowered.replace("-", "")) if token}
    return not tokens.isdisjoint(_HOLDOUT_TOKENS)


def settings_digest(settings: Settings) -> str:
    """Canonical sha256 of the contract, risk and cost blocks that price a run.

    Settings are not part of the dataset manifest, so without this digest two
    bundles built with different costs would share identity and could be
    compared as if they were the same experiment. The digest uses the same
    canonical JSON rules as candidate ids (sorted keys, no NaN).
    """
    if not isinstance(settings, Settings):
        raise TypeError(f"settings must be a Settings, got {type(settings).__name__}")
    return canonical_digest(asdict(settings))


def _finite(value: float, *, where: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceError("non_finite_evidence", f"{where} is not finite ({number!r})")
    return number


def _finite_pair(values: Sequence[float], *, where: str) -> tuple[float, float]:
    low, high = (_finite(value, where=where) for value in values)
    if low > high:
        raise EvidenceError("non_finite_evidence", f"{where} has low > high")
    return low, high


def _iso(stamp: np.datetime64) -> str:
    value = np.datetime64(stamp, "us").astype(datetime)
    return value.replace(tzinfo=UTC).isoformat()


def _block_edges(start: int, end: int, blocks: int) -> list[int]:
    """Contiguous half-open edges; the first blocks absorb the remainder."""
    total = end - start
    base, extra = divmod(total, blocks)
    sizes = [base + (1 if index < extra else 0) for index in range(blocks)]
    if any(size <= 0 for size in sizes):
        raise EvidenceError(
            "insufficient_folds",
            f"{total} discovery rows cannot fill {blocks} non-empty blocks",
        )
    edges = [start]
    for size in sizes:
        edges.append(edges[-1] + size)
    return edges


def _slice_frame(features: FeatureFrame, start: int, end: int) -> FeatureFrame:
    """Copy source rows ``[start, end)`` into a frame the simulator can run.

    ``valid_from`` resets to zero because the slice starts after the source
    warm-up. Only rows before the test split are ever requested.
    """
    if start < 0 or end > int(features.X.shape[0]) or end <= start:
        raise EvidenceError(
            "fold_geometry", f"slice [{start}, {end}) is outside a {features.X.shape[0]}-row frame"
        )
    return FeatureFrame(
        X=np.ascontiguousarray(features.X[start:end]),
        ts=np.asarray(features.ts[start:end]),
        ohlc=np.asarray(features.ohlc[start:end]),
        spread=np.asarray(features.spread[start:end]),
        atr=np.asarray(features.atr[start:end]),
        valid_from=0,
        columns=list(features.columns),
    )


def _regime_labels(frame: FeatureFrame) -> pl.DataFrame:
    """Rule-engine regime labels for one fold slice, aligned row for row.

    ``classify`` is causal, so classifying the fold slice alone never sees a
    later bar. The reconstructed bar frame only needs the OHLC path; the quote
    columns are filled from the frame's close and spread so the canonical bar
    schema is satisfied, and volume is irrelevant to the rule engine.
    """
    close = np.asarray(frame.ohlc[:, 3], dtype=np.float64)
    spread = np.asarray(frame.spread, dtype=np.float64)
    bars = pl.DataFrame(
        {
            "ts": pl.Series("ts", frame.ts),
            "open": np.asarray(frame.ohlc[:, 0], dtype=np.float64),
            "high": np.asarray(frame.ohlc[:, 1], dtype=np.float64),
            "low": np.asarray(frame.ohlc[:, 2], dtype=np.float64),
            "close": close,
            "bid_close": close,
            "ask_close": close + spread,
            "spread": spread,
            "volume": np.zeros(close.shape[0], dtype=np.int64),
        }
    )
    return classify(bars)


def _per_trade_regimes(
    trades: Sequence[Trade],
    labels: pl.DataFrame,
) -> pl.DataFrame:
    """Attach each trade's entry regime, with ``unknown`` for a missing label."""
    if not trades:
        return pl.DataFrame(
            schema={
                "entry_ts": pl.Datetime("us", "UTC"),
                "r_multiple": pl.Float64,
                "pnl": pl.Float64,
                "regime": pl.String,
            }
        )
    rows = pl.DataFrame(
        {
            "entry_ts": [trade.entry_ts for trade in trades],
            "r_multiple": [float(trade.r_multiple) for trade in trades],
            "pnl": [float(trade.pnl) for trade in trades],
        }
    ).sort("entry_ts")
    if labels.is_empty():
        return rows.with_columns(pl.lit(UNKNOWN_REGIME).alias("regime"))
    joined = rows.join_asof(
        labels.select(
            pl.col("ts").alias("_regime_ts"),
            pl.col("regime"),
        ).sort("_regime_ts"),
        left_on="entry_ts",
        right_on="_regime_ts",
        strategy="backward",
    ).drop("_regime_ts")
    return joined.with_columns(pl.col("regime").fill_null(UNKNOWN_REGIME).cast(pl.String))


def _regime_rows(per_trade: pl.DataFrame) -> tuple[RegimeEvidence, ...]:
    """One row per regime, sorted by label; ``unknown`` is kept, not hidden."""
    if per_trade.is_empty():
        return ()
    table = (
        per_trade.group_by("regime")
        .agg(
            pl.len().alias("trades"),
            (pl.col("r_multiple") > 0).sum().alias("wins"),
            pl.col("pnl").sum().alias("net_profit"),
            pl.col("r_multiple").mean().alias("expectancy_r"),
            pl.col("r_multiple").sum().alias("total_r"),
        )
        .sort("regime")
    )
    rows: list[RegimeEvidence] = []
    for record in table.iter_rows(named=True):
        trades = int(record["trades"])
        wins = int(record["wins"])
        rows.append(
            RegimeEvidence(
                regime=str(record["regime"]),
                trades=trades,
                net_profit=_finite(record["net_profit"], where=f"regime {record['regime']} pnl"),
                win_rate=_finite(wins / trades, where=f"regime {record['regime']} win_rate"),
                expectancy_r=_finite(
                    record["expectancy_r"], where=f"regime {record['regime']} expectancy_r"
                ),
                total_r=_finite(record["total_r"], where=f"regime {record['regime']} total_r"),
            )
        )
    return tuple(rows)


# --- shared fold helpers ------------------------------------------------------
#
# The locked-holdout evaluator needs the same regime attribution and slicing as a
# walk-forward fold. These public wrappers keep one implementation of the join
# and one definition of "a regime label at the entry bar" instead of forking the
# logic into a second module.


def slice_frame(features: FeatureFrame, start: int, end: int) -> FeatureFrame:
    """Copy source rows ``[start, end)`` into a simulatable frame."""
    return _slice_frame(features, start, end)


def regime_labels(frame: FeatureFrame) -> pl.DataFrame:
    """Causal rule-engine labels for one frame, aligned row for row."""
    return _regime_labels(frame)


def per_trade_regimes(trades: Sequence[Trade], labels: pl.DataFrame) -> pl.DataFrame:
    """Attach each trade's entry regime, with ``unknown`` for a missing label."""
    return _per_trade_regimes(trades, labels)


def regime_rows(per_trade: pl.DataFrame) -> tuple[RegimeEvidence, ...]:
    """One regime row per bucket, sorted by label; ``unknown`` is kept."""
    return _regime_rows(per_trade)


def _pooled_profit_factor(pnls: Sequence[float]) -> float:
    gross_win = sum(value for value in pnls if value > 0.0)
    gross_loss = -sum(value for value in pnls if value < 0.0)
    if gross_win <= 0.0 and gross_loss <= 0.0:
        return 0.0
    if gross_loss <= 0.0:
        return math.inf
    return gross_win / gross_loss


# --- bootstrap ----------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
    statistic: str = "mean",
) -> tuple[float, float]:
    """Percentile CI of ``mean`` or ``sum`` over an iid bootstrap of ``values``.

    Draws ``samples`` resamples with replacement using one
    :class:`numpy.random.Generator` seeded with ``seed``, so identical inputs
    produce identical bounds. ``values`` must be non-empty and finite. The
    interval is the ``[(1-c)/2, (1+c)/2]`` percentile pair of the resampled
    statistic. This is the same iid assumption documented in the module: it is
    not a block bootstrap and it makes no claim about serial dependence.
    """
    if statistic not in _BOOTSTRAP_STATISTICS:
        raise ValueError(f"statistic must be one of {sorted(_BOOTSTRAP_STATISTICS)}")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError(f"samples must be a positive int, got {samples!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    if not math.isfinite(float(confidence)) or not 0.0 < float(confidence) < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")

    data = np.asarray([float(value) for value in values], dtype=np.float64)
    if data.size == 0:
        raise ValueError("values must not be empty")
    if not bool(np.isfinite(data).all()):
        raise ValueError("values must be finite")

    rng = np.random.default_rng(seed)
    outcomes = np.empty(samples, dtype=np.float64)
    reducer = np.mean if statistic == "mean" else np.sum
    # One draw at a time keeps peak memory at O(n) even with many samples.
    for index in range(samples):
        sample = data[rng.integers(0, data.size, data.size)]
        outcomes[index] = float(reducer(sample))
    tail = (1.0 - float(confidence)) / 2.0 * 100.0
    low, high = np.percentile(outcomes, [tail, 100.0 - tail])
    return _finite_pair((float(low), float(high)), where=f"bootstrap {statistic} CI")


# --- fold evaluation ----------------------------------------------------------


def _fold_metrics(metrics: Metrics, trades: Sequence[Trade], *, fold: int) -> dict[str, float]:
    """Finite scalar view of one fold; ``inf`` profit factor fails closed."""
    if int(metrics.trades) != len(trades):
        raise EvidenceError(
            "fold_geometry",
            f"fold {fold}: metrics report {metrics.trades} trades but {len(trades)} were returned",
        )
    return {
        "trades": float(metrics.trades),
        "net_profit": _finite(metrics.net_profit, where=f"fold {fold} net_profit"),
        "profit_factor": _finite(metrics.profit_factor, where=f"fold {fold} profit_factor"),
        "win_rate": _finite(metrics.win_rate, where=f"fold {fold} win_rate"),
        "expectancy_r": _finite(metrics.expectancy_r, where=f"fold {fold} expectancy_r"),
        "max_drawdown": _finite(metrics.max_drawdown, where=f"fold {fold} max_drawdown"),
    }


def _stress_rows(
    evaluator: DslEvaluator,
    spec: StrategySpec,
    multipliers: Sequence[float],
    *,
    fold: int,
    base: Metrics,
) -> tuple[StressEvidence, ...]:
    """One rerun per configured multiplier; the base run serves ``1.0``."""
    rows: list[StressEvidence] = []
    for multiplier in multipliers:
        if multiplier == 1.0:
            metrics = base
        else:
            try:
                metrics = evaluator.simulate(spec, multiplier=multiplier).metrics
            except DslValidationError as exc:  # pragma: no cover - spec validated first
                raise EvidenceError("invalid_spec", f"fold {fold}: {exc}") from exc
        rows.append(
            StressEvidence(
                multiplier=multiplier,
                trades=int(metrics.trades),
                net_profit=_finite(metrics.net_profit, where=f"fold {fold} stress pnl"),
                profit_factor=_finite(metrics.profit_factor, where=f"fold {fold} stress pf"),
                max_drawdown=_finite(metrics.max_drawdown, where=f"fold {fold} stress dd"),
                expectancy_r=_finite(metrics.expectancy_r, where=f"fold {fold} stress expectancy"),
            )
        )
    return tuple(rows)


def _evaluate_fold(
    *,
    fold: int,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    frame: FeatureFrame,
    spec: StrategySpec,
    settings: Settings,
    config: EvidenceConfig,
    dataset_id: str,
    purge_bars: int,
    embargo_bars: int,
    ts: np.ndarray,
) -> tuple[FoldEvidence, tuple[float, ...], tuple[pl.DataFrame, ...]]:
    """Simulate one fold and return its evidence plus data for the aggregate."""
    if test_end <= test_start or train_end <= train_start:
        raise EvidenceError("fold_geometry", f"fold {fold} has an empty train or test range")
    test_rows = test_end - test_start
    if test_rows <= config.window:
        raise EvidenceError(
            "fold_too_small",
            (
                f"fold {fold} test block has {test_rows} rows; a {config.window}-bar "
                "observation window needs more than that to make a decision"
            ),
        )

    source_start = test_start + int(frame.valid_from)
    source_end = test_end + int(frame.valid_from)
    fold_frame = _slice_frame(frame, source_start, source_end)
    fold_dataset_id = f"validation:{dataset_id}:fold{fold}"

    try:
        evaluator = DslEvaluator(
            fold_frame,
            settings,
            symbol=config.symbol,
            timeframe=config.timeframe,
            dataset_id=fold_dataset_id,
            window=config.window,
            stress=(),
            max_steps=len(fold_frame.ts),
        )
        simulation = evaluator.simulate(spec)
    except DslValidationError as exc:
        raise EvidenceError("invalid_spec", f"fold {fold}: {exc}") from exc
    except ValueError as exc:
        # The evaluator refuses holdout-like ids and unusable frames.
        raise EvidenceError("fold_geometry", f"fold {fold}: {exc}") from exc

    metrics = _fold_metrics(simulation.metrics, simulation.trades, fold=fold)
    if metrics["trades"] < config.min_trades_per_fold:
        raise EvidenceError(
            "fold_trades_below_minimum",
            (
                f"fold {fold} closed {int(metrics['trades'])} trades; "
                f"the policy requires {config.min_trades_per_fold}"
            ),
        )

    labels = _regime_labels(fold_frame)
    if labels.is_empty():
        raise EvidenceError(
            "missing_regime_evidence",
            f"fold {fold}: the rule engine produced no regime labels",
        )
    per_trade = _per_trade_regimes(simulation.trades, labels)
    regimes = _regime_rows(per_trade)
    known = sum(row.trades for row in regimes if row.regime != UNKNOWN_REGIME)
    coverage = known / int(metrics["trades"])
    if coverage < float(config.regime_coverage_min):
        raise EvidenceError(
            "regime_coverage_below_minimum",
            (
                f"fold {fold} attributed {known}/{int(metrics['trades'])} trades to a regime; "
                f"the policy requires {config.regime_coverage_min:g}"
            ),
        )

    stress = _stress_rows(
        evaluator,
        spec,
        config.cost_stress,
        fold=fold,
        base=simulation.metrics,
    )
    r_multiples = tuple(
        _finite(trade.r_multiple, where=f"fold {fold} r_multiple") for trade in simulation.trades
    )
    evidence = FoldEvidence(
        fold=fold,
        train_start_row=train_start,
        train_end_row=train_end,
        test_start_row=test_start,
        test_end_row=test_end,
        train_rows=train_end - train_start,
        test_rows=test_rows,
        train_start_ts=_iso(ts[train_start]),
        train_end_ts=_iso(ts[train_end - 1]),
        test_start_ts=_iso(ts[test_start]),
        test_end_ts=_iso(ts[test_end - 1]),
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        trades=int(metrics["trades"]),
        net_profit=metrics["net_profit"],
        profit_factor=metrics["profit_factor"],
        win_rate=metrics["win_rate"],
        expectancy_r=metrics["expectancy_r"],
        max_drawdown=metrics["max_drawdown"],
        regimes=regimes,
        stress=stress,
    )
    return evidence, r_multiples, (per_trade,)


def _aggregate(
    folds: Sequence[FoldEvidence],
    r_multiples: Sequence[float],
    per_trade_frames: Sequence[pl.DataFrame],
    config: EvidenceConfig,
) -> AggregateEvidence:
    if not folds:
        raise EvidenceError("insufficient_folds", "no folds were evaluated")
    if not r_multiples:
        raise EvidenceError("insufficient_folds", "the folds produced no trades")
    values = np.asarray(r_multiples, dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise EvidenceError("non_finite_evidence", "pooled R multiples contain a non-finite value")

    total_trades = len(r_multiples)
    total_r = _finite(float(values.sum()), where="pooled total_r")
    expectancy = _finite(float(values.mean()), where="pooled expectancy_r")
    pnls = [float(row) for frame in per_trade_frames for row in frame["pnl"].to_list()]
    profit_factor = _finite(_pooled_profit_factor(pnls), where="pooled profit_factor")
    max_drawdown = _finite(max(fold.max_drawdown for fold in folds), where="aggregate drawdown")

    positive = sum(1 for fold in folds if fold.net_profit > 0.0)
    expectancy_ci = bootstrap_ci(
        r_multiples,
        samples=config.bootstrap_samples,
        seed=config.seed,
        confidence=config.bootstrap_confidence,
        statistic="mean",
    )
    total_ci = bootstrap_ci(
        r_multiples,
        samples=config.bootstrap_samples,
        seed=config.seed,
        confidence=config.bootstrap_confidence,
        statistic="sum",
    )

    known = sum(
        row.trades for fold in folds for row in fold.regimes if row.regime != UNKNOWN_REGIME
    )
    regime_coverage = _finite(known / total_trades, where="regime coverage")

    pooled = pl.concat(per_trade_frames) if per_trade_frames else pl.DataFrame()
    regime_rows = _regime_rows(pooled)

    return AggregateEvidence(
        total_trades=total_trades,
        total_r=total_r,
        expectancy_r=expectancy,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        positive_folds=positive,
        positive_fold_fraction=_finite(positive / len(folds), where="positive fold fraction"),
        expectancy_r_ci95_low=expectancy_ci[0],
        expectancy_r_ci95_high=expectancy_ci[1],
        total_r_ci95_low=total_ci[0],
        total_r_ci95_high=total_ci[1],
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_confidence=config.bootstrap_confidence,
        bootstrap_method="iid-bootstrap",
        regime_coverage=regime_coverage,
        regime_trades=regime_rows,
    )


# --- checksum / verification --------------------------------------------------


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("checksum", None)
    body.pop("created_at", None)
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def evidence_checksum(evidence: CandidateEvidence) -> str:
    """Recompute the canonical checksum of ``evidence`` without trusting its stamp.

    The promotion policy uses this to re-derive the identity before reading a
    metric, so a bundle whose stored ``checksum`` was overwritten is caught
    instead of being believed. ``verify_evidence`` uses the same function.
    """
    if type(evidence) is not CandidateEvidence:
        raise TypeError(f"evidence must be a CandidateEvidence, got {type(evidence).__name__}")
    return _checksum_payload(evidence.to_dict())


def locked_oos_checksum(section: LockedOOSEvidence) -> str:
    """Canonical sha256 of a locked-OOS section, excluding ``checksum``.

    ``created_at`` is excluded too, matching the holdout ledger's
    ``_result_checksum``: the same measured section can be re-serialised with a
    new creation stamp without becoming a different result. Every other field,
    including the metrics, intervals, regimes, stress and identity, is covered.
    """
    if type(section) is not LockedOOSEvidence:
        raise TypeError(f"section must be a LockedOOSEvidence, got {type(section).__name__}")
    payload = asdict(section)
    payload.pop("checksum", None)
    payload.pop("created_at", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _nested(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise EvidenceError("missing_section", f"{key} must be a JSON object")
    return value


def _nested_list(mapping: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = mapping.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvidenceError("missing_section", f"{key} must be a JSON array")
    return value


def evidence_from_dict(payload: Mapping[str, Any]) -> CandidateEvidence:
    """Rebuild a typed bundle from :meth:`CandidateEvidence.to_dict` output.

    The archive stores the canonical JSON of the bundle that justified a
    promotion. A later champion comparison needs the same typed object, not a
    bag of dictionaries, so this is the one place that knows the nested shape.
    Every missing or mistyped section raises :class:`EvidenceError`; the caller
    must still re-derive the checksum before trusting the result.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a mapping, got {type(payload).__name__}")
    try:
        policy = EvidencePolicy(**_nested(payload, "policy"))
        splits = SplitBoundaries(**_nested(payload, "splits"))
        folds = tuple(
            FoldEvidence(
                fold=int(row["fold"]),
                train_start_row=int(row["train_start_row"]),
                train_end_row=int(row["train_end_row"]),
                test_start_row=int(row["test_start_row"]),
                test_end_row=int(row["test_end_row"]),
                train_rows=int(row["train_rows"]),
                test_rows=int(row["test_rows"]),
                train_start_ts=str(row["train_start_ts"]),
                train_end_ts=str(row["train_end_ts"]),
                test_start_ts=str(row["test_start_ts"]),
                test_end_ts=str(row["test_end_ts"]),
                purge_bars=int(row["purge_bars"]),
                embargo_bars=int(row["embargo_bars"]),
                trades=int(row["trades"]),
                net_profit=float(row["net_profit"]),
                profit_factor=float(row["profit_factor"]),
                win_rate=float(row["win_rate"]),
                expectancy_r=float(row["expectancy_r"]),
                max_drawdown=float(row["max_drawdown"]),
                regimes=tuple(
                    RegimeEvidence(
                        regime=str(regime["regime"]),
                        trades=int(regime["trades"]),
                        net_profit=float(regime["net_profit"]),
                        win_rate=float(regime["win_rate"]),
                        expectancy_r=float(regime["expectancy_r"]),
                        total_r=float(regime["total_r"]),
                    )
                    for regime in _nested_list(row, "regimes")
                ),
                stress=tuple(
                    StressEvidence(
                        multiplier=float(stress["multiplier"]),
                        trades=int(stress["trades"]),
                        net_profit=float(stress["net_profit"]),
                        profit_factor=float(stress["profit_factor"]),
                        max_drawdown=float(stress["max_drawdown"]),
                        expectancy_r=float(stress["expectancy_r"]),
                    )
                    for stress in _nested_list(row, "stress")
                ),
            )
            for row in _nested_list(payload, "folds")
        )
        aggregate = _nested(payload, "aggregate")
        summary = AggregateEvidence(
            total_trades=int(aggregate["total_trades"]),
            total_r=float(aggregate["total_r"]),
            expectancy_r=float(aggregate["expectancy_r"]),
            profit_factor=float(aggregate["profit_factor"]),
            max_drawdown=float(aggregate["max_drawdown"]),
            positive_folds=int(aggregate["positive_folds"]),
            positive_fold_fraction=float(aggregate["positive_fold_fraction"]),
            expectancy_r_ci95_low=float(aggregate["expectancy_r_ci95_low"]),
            expectancy_r_ci95_high=float(aggregate["expectancy_r_ci95_high"]),
            total_r_ci95_low=float(aggregate["total_r_ci95_low"]),
            total_r_ci95_high=float(aggregate["total_r_ci95_high"]),
            bootstrap_samples=int(aggregate["bootstrap_samples"]),
            bootstrap_confidence=float(aggregate["bootstrap_confidence"]),
            bootstrap_method=str(aggregate["bootstrap_method"]),
            regime_coverage=float(aggregate["regime_coverage"]),
            regime_trades=tuple(
                RegimeEvidence(
                    regime=str(regime["regime"]),
                    trades=int(regime["trades"]),
                    net_profit=float(regime["net_profit"]),
                    win_rate=float(regime["win_rate"]),
                    expectancy_r=float(regime["expectancy_r"]),
                    total_r=float(regime["total_r"]),
                )
                for regime in _nested_list(aggregate, "regime_trades")
            ),
        )
        locked_raw = payload.get("locked_oos")
        if locked_raw is None:
            locked = None
        else:
            if not isinstance(locked_raw, Mapping):
                raise EvidenceError("missing_section", "locked_oos must be a JSON object")
            locked = LockedOOSEvidence(
                checksum=str(locked_raw["checksum"]),
                evidence_checksum=str(locked_raw["evidence_checksum"]),
                dataset_id=str(locked_raw["dataset_id"]),
                dataset_checksum=str(locked_raw["dataset_checksum"]),
                symbol=str(locked_raw["symbol"]),
                timeframe=str(locked_raw["timeframe"]),
                spec_hash=str(locked_raw["spec_hash"]),
                spec_version=int(locked_raw["spec_version"]),
                complexity=int(locked_raw["complexity"]),
                code_version=str(locked_raw["code_version"]),
                settings_digest=str(locked_raw["settings_digest"]),
                policy_id=str(locked_raw["policy_id"]),
                policy_version=int(locked_raw["policy_version"]),
                test_start_row=int(locked_raw["test_start_row"]),
                test_end_row=int(locked_raw["test_end_row"]),
                test_rows=int(locked_raw["test_rows"]),
                test_start_ts=str(locked_raw["test_start_ts"]),
                test_end_ts=str(locked_raw["test_end_ts"]),
                trades=int(locked_raw["trades"]),
                net_profit=float(locked_raw["net_profit"]),
                profit_factor=float(locked_raw["profit_factor"]),
                win_rate=float(locked_raw["win_rate"]),
                expectancy_r=float(locked_raw["expectancy_r"]),
                total_r=float(locked_raw["total_r"]),
                max_drawdown=float(locked_raw["max_drawdown"]),
                expectancy_r_ci95_low=float(locked_raw["expectancy_r_ci95_low"]),
                expectancy_r_ci95_high=float(locked_raw["expectancy_r_ci95_high"]),
                total_r_ci95_low=float(locked_raw["total_r_ci95_low"]),
                total_r_ci95_high=float(locked_raw["total_r_ci95_high"]),
                bootstrap_samples=int(locked_raw["bootstrap_samples"]),
                bootstrap_confidence=float(locked_raw["bootstrap_confidence"]),
                bootstrap_method=str(locked_raw["bootstrap_method"]),
                regime_coverage=float(locked_raw["regime_coverage"]),
                regimes=tuple(
                    RegimeEvidence(
                        regime=str(regime["regime"]),
                        trades=int(regime["trades"]),
                        net_profit=float(regime["net_profit"]),
                        win_rate=float(regime["win_rate"]),
                        expectancy_r=float(regime["expectancy_r"]),
                        total_r=float(regime["total_r"]),
                    )
                    for regime in _nested_list(locked_raw, "regimes")
                ),
                stress=tuple(
                    StressEvidence(
                        multiplier=float(stress["multiplier"]),
                        trades=int(stress["trades"]),
                        net_profit=float(stress["net_profit"]),
                        profit_factor=float(stress["profit_factor"]),
                        max_drawdown=float(stress["max_drawdown"]),
                        expectancy_r=float(stress["expectancy_r"]),
                    )
                    for stress in _nested_list(locked_raw, "stress")
                ),
                created_at=str(locked_raw.get("created_at", "")),
                schema_version=int(locked_raw.get("schema_version", 1)),
            )
        return CandidateEvidence(
            spec_hash=str(payload["spec_hash"]),
            spec_version=int(payload["spec_version"]),
            complexity=int(payload["complexity"]),
            dataset_id=str(payload["dataset_id"]),
            dataset_checksum=str(payload["dataset_checksum"]),
            symbol=str(payload["symbol"]),
            timeframe=str(payload["timeframe"]),
            code_version=str(payload["code_version"]),
            settings_digest=str(payload["settings_digest"]),
            seed=int(payload["seed"]),
            trial_count=int(payload["trial_count"]),
            search_budget=int(payload["search_budget"]),
            splits=splits,
            policy=policy,
            folds=folds,
            aggregate=summary,
            checksum=str(payload["checksum"]),
            created_at=str(payload["created_at"]),
            schema_version=int(payload.get("schema_version", EVIDENCE_SCHEMA_VERSION)),
            locked_oos=locked,
        )
    except EvidenceError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError("missing_section", f"cannot rebuild evidence: {exc}") from exc


def verify_evidence(
    evidence: CandidateEvidence,
    *,
    dataset_id: str,
    dataset_checksum: str,
    spec_hash: str,
    code_version: str,
    settings_digest: str,
) -> None:
    """Reject a stale, mismatched or incomplete bundle before it is trusted.

    Checks identity first (dataset id and checksum, spec, code version, execution
    settings), then the schema version and required sections, then the checksum.
    Every failure is a typed :class:`EvidenceError` with a stable code; a caller
    must not catch and ignore one.
    """
    if type(evidence) is not CandidateEvidence:
        raise TypeError(f"evidence must be a CandidateEvidence, got {type(evidence).__name__}")
    if evidence.dataset_id != dataset_id:
        raise EvidenceError(
            "stale_dataset",
            f"bundle is for dataset {evidence.dataset_id!r}, not {dataset_id!r}",
        )
    if evidence.dataset_checksum != dataset_checksum:
        raise EvidenceError(
            "stale_dataset_checksum",
            f"bundle is for dataset checksum {evidence.dataset_checksum!r}, "
            f"not {dataset_checksum!r}",
        )
    if evidence.spec_hash != spec_hash:
        raise EvidenceError(
            "stale_spec",
            f"bundle is for spec {evidence.spec_hash!r}, not {spec_hash!r}",
        )
    if evidence.code_version != code_version:
        raise EvidenceError(
            "stale_code_version",
            f"bundle was built by {evidence.code_version!r}, not {code_version!r}",
        )
    if evidence.settings_digest != settings_digest:
        raise EvidenceError(
            "stale_settings",
            f"bundle was built with settings {evidence.settings_digest!r}, not {settings_digest!r}",
        )
    if evidence.schema_version != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceError(
            "schema_mismatch",
            f"bundle schema {evidence.schema_version} is not {EVIDENCE_SCHEMA_VERSION}",
        )

    for name, value in (
        ("dataset_id", evidence.dataset_id),
        ("dataset_checksum", evidence.dataset_checksum),
        ("spec_hash", evidence.spec_hash),
        ("symbol", evidence.symbol),
        ("timeframe", evidence.timeframe),
        ("code_version", evidence.code_version),
        ("settings_digest", evidence.settings_digest),
    ):
        if not isinstance(value, str) or not value.strip():
            raise EvidenceError("missing_section", f"identity field {name} is empty")
    if not evidence.folds:
        raise EvidenceError("missing_section", "the bundle has no folds")
    if evidence.aggregate is None:  # type: ignore[comparison-overlap]
        raise EvidenceError("missing_section", "the bundle has no aggregate section")
    if evidence.aggregate.bootstrap_samples < 1:
        raise EvidenceError("missing_section", "the bundle has no bootstrap section")

    expected = _checksum_payload(evidence.to_dict())
    if evidence.checksum != expected:
        raise EvidenceError(
            "checksum_mismatch",
            f"bundle checksum {evidence.checksum!r} does not match {expected!r}",
        )


# --- build --------------------------------------------------------------------


def build_evidence(
    dataset_bundle: DatasetBundle,
    spec: StrategySpec,
    config: EvidenceConfig,
    *,
    features: FeatureFrame,
    settings: Settings | None = None,
    created_at: datetime | None = None,
) -> CandidateEvidence:
    """Build one immutable evidence bundle from the discovery timeline.

    ``features`` is required because a :class:`DatasetBundle` carries labels and
    split rows, not the OHLC/spread/ATR arrays the simulator prices fills with.
    It must be the frame the bundle was built from: its timestamps are compared
    against the bundle's and a mismatch fails closed.

    Only rows before ``manifest.validation.end`` are ever sliced. The final test
    split is neither read from ``features`` nor named to a fold evaluator.
    """
    if not isinstance(dataset_bundle, DatasetBundle):
        raise TypeError(
            f"dataset_bundle must be a DatasetBundle, got {type(dataset_bundle).__name__}"
        )
    if not isinstance(spec, StrategySpec):
        raise TypeError(f"spec must be a StrategySpec, got {type(spec).__name__}")
    if not isinstance(config, EvidenceConfig):
        raise TypeError(f"config must be an EvidenceConfig, got {type(config).__name__}")
    if not isinstance(features, FeatureFrame):
        raise TypeError(f"features must be a FeatureFrame, got {type(features).__name__}")

    manifest = dataset_bundle.manifest
    if config.folds < MIN_EVIDENCE_FOLDS:
        raise EvidenceError(
            "insufficient_folds",
            f"{config.folds} fold(s) requested; promotion evidence needs {MIN_EVIDENCE_FOLDS}",
        )
    if not config.cost_stress or not any(value > 1.0 for value in config.cost_stress):
        raise EvidenceError(
            "missing_cost_stress",
            "at least one cost-stress multiplier must be greater than 1.0",
        )
    if config.trial_count > config.search_budget:
        raise EvidenceError(
            "invalid_trial_context",
            f"trial_count {config.trial_count} exceeds search_budget {config.search_budget}",
        )
    if _names_holdout(manifest.dataset_id):
        raise EvidenceError(
            "holdout_dataset",
            f"dataset id {manifest.dataset_id!r} names a test/holdout split",
        )
    if manifest.validation.rows <= 0:
        raise EvidenceError(
            "fold_geometry",
            "the manifest has no validation rows; discovery evidence cannot be built",
        )
    if manifest.test.rows <= 0:
        raise EvidenceError(
            "fold_geometry",
            "the manifest has no test rows; the holdout must exist even though evidence never reads it",
        )
    if config.symbol != manifest.symbol:
        raise EvidenceError(
            "symbol_mismatch",
            f"config symbol {config.symbol!r} does not match dataset {manifest.symbol!r}",
        )
    if config.timeframe != manifest.timeframe:
        raise EvidenceError(
            "timeframe_mismatch",
            f"config timeframe {config.timeframe!r} does not match dataset {manifest.timeframe!r}",
        )

    offset = int(features.valid_from)
    expected_rows = int(features.X.shape[0]) - offset - int(manifest.horizon)
    if expected_rows != int(dataset_bundle.ts.shape[0]):
        raise EvidenceError(
            "frame_mismatch",
            (
                f"features has {features.X.shape[0]} rows from valid_from={offset} and "
                f"horizon={manifest.horizon}; expected {expected_rows} bundle rows, "
                f"got {dataset_bundle.ts.shape[0]}"
            ),
        )
    if not np.array_equal(
        np.asarray(dataset_bundle.ts),
        np.asarray(features.ts[offset : offset + int(dataset_bundle.ts.shape[0])]),
    ):
        raise EvidenceError(
            "frame_mismatch", "the bundle timestamps do not match the feature frame"
        )

    try:
        spec_hash = spec.canonical_hash()
        spec_version = int(spec.spec_version)
        complexity = int(spec.complexity())
    except DslValidationError as exc:
        raise EvidenceError("invalid_spec", str(exc)) from exc
    if config.complexity is not None and config.complexity != complexity:
        raise EvidenceError(
            "complexity_mismatch",
            f"config complexity {config.complexity} does not match spec complexity {complexity}",
        )

    splits = SplitBoundaries(
        train_start_row=int(manifest.train.start) - offset,
        train_end_row=int(manifest.train.end) - offset,
        validation_start_row=int(manifest.validation.start) - offset,
        validation_end_row=int(manifest.validation.end) - offset,
        test_start_row=int(manifest.test.start) - offset,
        test_end_row=int(manifest.test.end) - offset,
    )
    rows = int(dataset_bundle.split.shape[0])
    train_start = splits.train_start_row
    discovery_end = splits.validation_end_row
    test_start = splits.test_start_row
    if (
        train_start < 0
        or splits.train_end_row < train_start
        or splits.validation_start_row < splits.train_end_row
        or splits.validation_end_row <= splits.validation_start_row
        or discovery_end > rows
        or splits.test_end_row < test_start
        or splits.test_end_row > rows
    ):
        raise EvidenceError(
            "fold_geometry",
            f"split boundaries {splits} are not ordered inside {rows} bundle rows",
        )
    if test_start < discovery_end:
        raise EvidenceError(
            "fold_geometry",
            "the validation split ends after the test split starts; refusing to cut folds",
        )
    if bool((dataset_bundle.split[:discovery_end] == 2).any()):
        raise EvidenceError("fold_geometry", "a test row appears inside the discovery timeline")

    edges = _block_edges(train_start, discovery_end, config.folds + 1)
    purge_bars = max(int(config.purge_bars), int(manifest.purge_bars))
    embargo_bars = max(int(config.embargo_bars), int(manifest.embargo_bars))
    gap = purge_bars + embargo_bars
    if gap < 0:  # pragma: no cover - both inputs are non-negative
        raise EvidenceError("fold_geometry", "negative purge/embargo gap")

    timestamps = np.asarray(dataset_bundle.ts)
    settings = Settings() if settings is None else settings
    try:
        settings_hash = settings_digest(settings)
    except (TypeError, ValueError) as exc:
        raise EvidenceError("invalid_settings", str(exc)) from exc
    fold_rows: list[FoldEvidence] = []
    all_r: list[float] = []
    per_trade_frames: list[pl.DataFrame] = []
    previous_end: int | None = None
    for fold in range(1, config.folds + 1):
        test_lo = edges[fold]
        test_hi = edges[fold + 1]
        train_end = test_lo - gap
        if train_end <= train_start:
            raise EvidenceError(
                "fold_geometry",
                (
                    f"fold {fold}: a {gap}-bar purge/embargo leaves no training rows "
                    f"before test row {test_lo}"
                ),
            )
        if previous_end is not None and test_lo != previous_end:
            raise EvidenceError(
                "fold_geometry",
                f"fold {fold} starts at row {test_lo}, previous test ended at {previous_end}",
            )
        evidence, r_multiples, frames = _evaluate_fold(
            fold=fold,
            train_start=train_start,
            train_end=train_end,
            test_start=test_lo,
            test_end=test_hi,
            frame=features,
            spec=spec,
            settings=settings,
            config=config,
            dataset_id=manifest.dataset_id,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            ts=timestamps,
        )
        fold_rows.append(evidence)
        all_r.extend(r_multiples)
        per_trade_frames.extend(frames)
        previous_end = test_hi

    aggregate = _aggregate(fold_rows, all_r, per_trade_frames, config)
    if aggregate.regime_coverage < float(config.regime_coverage_min):
        raise EvidenceError(
            "regime_coverage_below_minimum",
            (
                f"pooled regime coverage {aggregate.regime_coverage:g} is below "
                f"the policy minimum {config.regime_coverage_min:g}"
            ),
        )

    policy = EvidencePolicy(
        folds=config.folds,
        purge_bars=int(config.purge_bars),
        embargo_bars=int(config.embargo_bars),
        min_trades_per_fold=config.min_trades_per_fold,
        cost_stress=config.cost_stress,
        regime_coverage_min=float(config.regime_coverage_min),
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_confidence=float(config.bootstrap_confidence),
        window=config.window,
    )
    stamp = (datetime.now(UTC) if created_at is None else created_at).isoformat(timespec="seconds")
    bundle_evidence = CandidateEvidence(
        spec_hash=spec_hash,
        spec_version=spec_version,
        complexity=complexity,
        dataset_id=manifest.dataset_id,
        dataset_checksum=manifest.checksum,
        symbol=manifest.symbol,
        timeframe=manifest.timeframe,
        code_version=config.code_version,
        settings_digest=settings_hash,
        seed=config.seed,
        trial_count=config.trial_count,
        search_budget=config.search_budget,
        splits=splits,
        policy=policy,
        folds=tuple(fold_rows),
        aggregate=aggregate,
        checksum="",
        created_at=stamp,
    )
    bundle_evidence = replace(
        bundle_evidence, checksum=_checksum_payload(bundle_evidence.to_dict())
    )
    verify_evidence(
        bundle_evidence,
        dataset_id=manifest.dataset_id,
        dataset_checksum=manifest.checksum,
        spec_hash=spec_hash,
        code_version=config.code_version,
        settings_digest=settings_hash,
    )
    return bundle_evidence
