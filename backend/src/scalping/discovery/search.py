"""Bounded evolutionary search over an injected strategy representation.

The loop is deliberately small. It knows how to keep a population, apply the
factory's operators, respect a trial budget, deduplicate by canonical hash and
write evidence to the archive. It knows nothing about strategy semantics: the
injected :class:`~scalping.discovery.protocols.Factory` owns generation and
validation, and the injected :class:`~scalping.discovery.protocols.Evaluator`
owns simulation and costs.

What the loop guarantees:

* **Determinism.** One seed, one sequence. The generator is
  ``numpy.random.default_rng`` and every ranking tie breaks on
  ``candidate_id``, so a replay produces the same candidate order, the same
  trial seeds and the same best candidate.
* **An exact budget.** ``max_trials`` caps *attempts*, not just successful
  evaluations. A duplicate, an invalid AST or a candidate over the complexity
  cap still spends one attempt. ``trials + skipped == max_trials`` whenever the
  run stops on ``budget``.
* **One trial per candidate per dataset.** A candidate already evaluated in
  this run or in the archive on this dataset is skipped as ``duplicate``; the
  evaluator is never asked for the same point twice.
* **No test-slice access.** The loop passes exactly ``config.dataset_id`` to
  every evaluator call. It has no other dataset handle.
* **An honest count.** Every recorded trial is persisted with its dataset, seed
  and metrics, and the run reports the total. A single winner after many trials
  is a search result, not an edge; see ``docs/autonomous-discovery.md``.

Survival ranks every recorded candidate by the selection metric, whether or not
the evaluator accepted it. Acceptance is the evidence gate that a champion
comparison must honour; it is not a fitness function. A search that only breeds
accepted candidates stalls after the first rejection wave, and the rejection
reasons are already recorded per trial for the human and the gate to read.

The run's ``best`` is narrower than survival. With an empty (default-off) gate,
a candidate that never trades reports ``net_profit = 0`` and would outrank every
trading candidate that lost money: a flat policy would "win" because it did
nothing. ``min_best_trades`` therefore requires a candidate to publish at least
that many closed trades before it can become ``best``. ``0`` disables the rule
for compatibility tests and explicit flat-policy runs; it is not a
recommendation. Survival ranking is unchanged, so budget and deduplication
behave exactly as before.

Stopping is explicit. ``budget`` means the attempt cap was reached, ``patience``
means the configured generations without a new best ran out, ``stopped`` means
the injected ``should_stop`` said so, and ``exhausted`` means either the
generation limit was reached or a generation could not contribute the required
number of novel candidates.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scalping.discovery.archive import DiscoveryArchive
from scalping.discovery.protocols import Candidate, Evaluator, Factory, TrialResult

__all__ = ["STOP_REASONS", "SearchConfig", "SearchResult", "evolve"]

#: Why a run ended. Reported verbatim, never inferred from counts downstream.
STOP_REASONS: tuple[str, ...] = ("budget", "patience", "exhausted", "stopped")

_SELECTION_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]*")

#: Skip categories. They are stable keys in the rejection histogram, so a log
#: can be compared between runs; the exception text stays out of the key.
_DUPLICATE = "duplicate"
_INVALID = "invalid"
_COMPLEXITY = "complexity"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Budget, operators and stop rules for one search run.

    ``dataset_id`` and ``code_version`` are part of the config because they are
    part of every trial's provenance. The evaluator receives them; the loop does
    not choose a dataset on its own.
    """

    dataset_id: str
    code_version: str
    dataset_checksum: str | None = None
    settings_digest: str | None = None
    population: int = 32
    generations: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    max_trials: int = 200
    seed: int = 0
    patience: int | None = 5
    complexity_cap: int = 40
    novelty_requirement: int = 1
    selection_metric: str = "net_profit"
    minimize: bool = False
    min_best_trades: int = 1

    def __post_init__(self) -> None:
        for name in ("dataset_id", "code_version", "selection_metric"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a str, got {type(value).__name__}")
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if _SELECTION_PATTERN.fullmatch(self.selection_metric) is None:
            raise ValueError(
                f"selection_metric {self.selection_metric!r} must match "
                f"{_SELECTION_PATTERN.pattern}"
            )
        if self.population < 2:
            raise ValueError("population must be >= 2; one individual cannot breed")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.max_trials < 1:
            raise ValueError("max_trials must be >= 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        for name in ("mutation_rate", "crossover_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.mutation_rate + self.crossover_rate > 1.0:
            raise ValueError("mutation_rate + crossover_rate must be <= 1")
        if self.patience is not None:
            if isinstance(self.patience, bool) or not isinstance(self.patience, int):
                raise TypeError("patience must be an int or None")
            if self.patience < 1:
                raise ValueError("patience must be >= 1 when set")
        if isinstance(self.complexity_cap, bool) or not isinstance(self.complexity_cap, int):
            raise TypeError("complexity_cap must be an int")
        if self.complexity_cap < 1:
            raise ValueError("complexity_cap must be >= 1")
        if isinstance(self.novelty_requirement, bool) or not isinstance(
            self.novelty_requirement, int
        ):
            raise TypeError("novelty_requirement must be an int")
        if self.novelty_requirement < 0:
            raise ValueError("novelty_requirement must be >= 0")
        if not isinstance(self.minimize, bool):
            raise TypeError("minimize must be a bool")
        if isinstance(self.min_best_trades, bool) or not isinstance(self.min_best_trades, int):
            raise TypeError("min_best_trades must be an int")
        if self.min_best_trades < 0:
            raise ValueError("min_best_trades must be >= 0")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """What one run produced, and how much evidence it cost.

    ``trials`` counts recorded evaluations and is the number that matters for
    multiple-testing accounting. ``skipped`` counts attempts that never reached
    the evaluator (duplicates, invalid ASTs, complexity overflows). The two
    together are the attempts spent against ``max_trials``.

    ``best`` is the best candidate **of this run** by the selection metric, not
    the archive-wide best; comparing against a champion is a separate step.
    ``lineage_depth`` is the deepest ancestor chain among this run's recorded
    candidates.

    ``skipped_reasons`` splits ``skipped`` into its stable categories
    (``invalid``, ``complexity``, ``duplicate``). A rerun on a saturated archive
    reports its duplicates there, so a caller can say "deduped" with a number
    instead of inferring it from a zero trial count.

    ``tradable`` counts accepted trials whose metrics report at least
    ``min_best_trades`` closed trades. It is the pool ``best`` was chosen from:
    when it is zero and accepted trials exist, the run found only flat policies
    and the decision layer can say "no tradable candidate" instead of pretending
    the flat candidate won.
    """

    dataset_id: str
    code_version: str
    seed: int
    trials: int
    skipped: int
    accepted: int
    rejections: dict[str, int] = field(default_factory=dict)
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "exhausted"
    generations: int = 0
    best: Candidate | None = None
    best_id: str | None = None
    best_metrics: dict[str, float] | None = None
    lineage_depth: int = 0
    tradable: int = 0

    def __post_init__(self) -> None:
        if self.stop_reason not in STOP_REASONS:
            raise ValueError(
                f"stop_reason must be one of {list(STOP_REASONS)}, got {self.stop_reason!r}"
            )
        for name in ("trials", "skipped", "accepted", "generations", "lineage_depth", "tradable"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.accepted > self.trials:
            raise ValueError("accepted cannot exceed trials")
        if self.tradable > self.accepted:
            raise ValueError("tradable cannot exceed accepted")
        if self.best is None and (self.best_id is not None or self.best_metrics is not None):
            raise ValueError("best_id and best_metrics require a best candidate")
        if self.best is not None and self.best_id is None:
            raise ValueError("a best candidate requires its id")
        total_skips = 0
        for reason, count in self.skipped_reasons.items():
            if reason not in (_DUPLICATE, _INVALID, _COMPLEXITY):
                raise ValueError(f"unknown skip reason {reason!r}")
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"skip count for {reason!r} must be an int")
            if count < 0:
                raise ValueError(f"skip count for {reason!r} must be >= 0")
            total_skips += count
        if total_skips > self.skipped:
            raise ValueError("skipped_reasons cannot exceed the skipped total")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary; the candidate is rendered by its canonical dict."""
        return {
            "dataset_id": self.dataset_id,
            "code_version": self.code_version,
            "seed": self.seed,
            "trials": self.trials,
            "skipped": self.skipped,
            "accepted": self.accepted,
            "tradable": self.tradable,
            "rejections": dict(sorted(self.rejections.items())),
            "skipped_reasons": dict(sorted(self.skipped_reasons.items())),
            "stop_reason": self.stop_reason,
            "generations": self.generations,
            "lineage_depth": self.lineage_depth,
            "best": (
                None
                if self.best is None
                else {
                    "candidate_id": self.best_id,
                    "canonical": self.best.canonical_dict(),
                    "metrics": self.best_metrics,
                }
            ),
        }


def _score(result: TrialResult, metric: str) -> float | None:
    """The selection value, or ``None`` when the evaluator did not publish it."""
    value = result.metrics.get(metric)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):  # TrialResult already rejects this; keep the guard local
        raise ValueError(f"selection metric {metric!r} is not finite")
    return number


def _is_tradable(metrics: Mapping[str, float], min_trades: int) -> bool:
    """Whether an accepted trial may be the run's best candidate.

    ``min_trades == 0`` disables the rule (compatibility mode). Otherwise the
    evaluator must publish ``trades`` and report at least that many closed
    trades. A missing count is not evidence of trading, so it fails closed: an
    evaluator that does not report trade counts cannot hand the run a champion.
    """
    if min_trades == 0:
        return True
    trades = metrics.get("trades")
    if trades is None:
        return False
    count = float(trades)
    return math.isfinite(count) and count.is_integer() and count >= float(min_trades)


def _precedes(a: tuple[float, str], b: tuple[float, str], *, minimize: bool) -> bool:
    """Deterministic ranking: primary metric, then candidate id ascending."""
    if a[0] != b[0]:
        return a[0] < b[0] if minimize else a[0] > b[0]
    return a[1] < b[1]


def _propose(
    factory: Factory,
    parents: list[tuple[Candidate, str]],
    config: SearchConfig,
    rng: np.random.Generator,
) -> tuple[Candidate, str | None, str | None, str]:
    """One child plus its derivation: (candidate, parent_a, parent_b, operator)."""
    if not parents:
        return factory.random(rng), None, None, "random"

    draw = float(rng.random())
    if draw < config.crossover_rate and len(parents) >= 2:
        first, second = (int(index) for index in rng.choice(len(parents), size=2, replace=False))
        a, a_id = parents[first]
        b, b_id = parents[second]
        child = factory.crossover(a, b, rng)
        if child is not None:
            return child, a_id, b_id, "crossover"
        # A failed crossover is not fatal: the chosen parent is mutated instead.
        return factory.mutate(a, rng), a_id, None, "mutation"
    if draw < config.crossover_rate + config.mutation_rate:
        index = int(rng.integers(0, len(parents)))
        parent, parent_id = parents[index]
        return factory.mutate(parent, rng), parent_id, None, "mutation"
    return factory.random(rng), None, None, "random"


def _select(
    population: list[tuple[Candidate, str]],
    scores: dict[str, float],
    config: SearchConfig,
) -> list[tuple[Candidate, str]]:
    """Keep the best ``population`` individuals; ties break on candidate id."""
    ranked: list[tuple[float, str, Candidate]] = [
        (scores[candidate_id], candidate_id, candidate)
        for candidate, candidate_id in population
        if candidate_id in scores
    ]
    if config.minimize:
        ranked.sort(key=lambda item: (item[0], item[1]))
    else:
        ranked.sort(key=lambda item: (-item[0], item[1]))
    return [
        (candidate, candidate_id) for _value, candidate_id, candidate in ranked[: config.population]
    ]


def evolve(
    factory: Factory,
    evaluator: Evaluator,
    archive: DiscoveryArchive,
    config: SearchConfig,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> SearchResult:
    """Run a bounded search and persist every trial.

    ``should_stop`` is polled before each attempt; it is the cancellation hook a
    scheduler or a user-facing cancel button attaches to. The loop never blocks
    on it between attempts and never calls the evaluator after it returns true.
    """
    rng = np.random.default_rng(config.seed)
    attempts = 0
    recorded = 0
    accepted = 0
    tradable = 0
    skipped: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    scores: dict[str, float] = {}
    known: set[str] = set()
    best_value: float | None = None
    best_candidate: Candidate | None = None
    best_id: str | None = None
    best_metrics: dict[str, float] | None = None
    deepest = 0
    generations_done = 0
    stale = 0
    parents: list[tuple[Candidate, str]] = []
    stop_reason: str | None = None

    def budget_left() -> int:
        return config.max_trials - attempts

    for generation in range(config.generations):
        generation_recorded = 0
        generation_improved = False
        generation_survivors: list[tuple[Candidate, str]] = []

        for _slot in range(config.population):
            if should_stop is not None and should_stop():
                stop_reason = "stopped"
                break
            if budget_left() <= 0:
                stop_reason = "budget"
                break

            candidate, parent_a, parent_b, operator = _propose(factory, parents, config, rng)
            attempts += 1

            try:
                factory.validate(candidate)
            except Exception:  # noqa: BLE001 - validate() raises anything for a bad AST
                # The factory's gate owns AST validity. Any exception means the
                # candidate is not a strategy the evaluator may see.
                skipped[_INVALID] += 1
                continue

            complexity = candidate.complexity()
            if isinstance(complexity, bool) or not isinstance(complexity, int):
                raise TypeError(f"complexity() must return an int, got {type(complexity).__name__}")
            if complexity < 0:
                raise ValueError(f"complexity() must be >= 0, got {complexity}")
            if complexity > config.complexity_cap:
                skipped[_COMPLEXITY] += 1
                continue

            candidate_id = candidate.canonical_hash()
            if candidate_id in known or archive.has_trial(
                candidate_id, config.dataset_id, code_version=config.code_version
            ):
                skipped[_DUPLICATE] += 1
                continue

            trial_seed = int(rng.integers(0, 2**31 - 1))
            result = evaluator.evaluate(
                candidate,
                dataset_id=config.dataset_id,
                code_version=config.code_version,
                budget=budget_left() + 1,
            )
            if not isinstance(result, TrialResult):
                raise TypeError(
                    f"evaluate() must return a TrialResult, got {type(result).__name__}"
                )

            record = archive.add_candidate(
                candidate,
                code_version=config.code_version,
                parent_a=parent_a,
                parent_b=parent_b,
                mutation=operator,
            )
            archive.record_trial(
                record.candidate_id,
                result,
                dataset_id=config.dataset_id,
                code_version=config.code_version,
                seed=trial_seed,
                dataset_checksum=config.dataset_checksum,
                settings_digest=config.settings_digest,
            )
            known.add(record.candidate_id)
            recorded += 1
            generation_recorded += 1
            deepest = max(deepest, archive.depth(record.candidate_id))
            generation_survivors.append((candidate, record.candidate_id))

            for reason in result.rejections:
                rejections[reason] += 1
            value = _score(result, config.selection_metric)
            if value is not None:
                # Rejected candidates keep their score: they are search material
                # even when they fail the acceptance gate, so the population does
                # not stall after a wave of rejections.
                scores[record.candidate_id] = value
            if not result.accepted:
                continue

            accepted += 1
            if value is None:
                raise ValueError(
                    "evaluator did not publish the selection metric "
                    f"{config.selection_metric!r} for an accepted trial"
                )
            if not _is_tradable(result.metrics, config.min_best_trades):
                continue
            tradable += 1
            if best_value is None or (
                value < best_value if config.minimize else value > best_value
            ):
                best_value = value
                best_candidate = candidate
                best_id = record.candidate_id
                best_metrics = dict(result.metrics)
                generation_improved = True

        generations_done = generation + 1
        if stop_reason is not None:
            break
        if generation_recorded < config.novelty_requirement:
            stop_reason = "exhausted"
            break

        stale = 0 if generation_improved else stale + 1
        if config.patience is not None and stale >= config.patience:
            stop_reason = "patience"
            break

        # Rank this generation's recorded candidates together with the parents
        # they came from, so a good parent survives a bad child generation.
        pool = {candidate_id: (candidate, candidate_id) for candidate, candidate_id in parents}
        for candidate, candidate_id in generation_survivors:
            pool[candidate_id] = (candidate, candidate_id)
        # With nothing scored the pool is empty and the next generation starts
        # from fresh random candidates instead of stalling.
        parents = _select(list(pool.values()), scores, config)

    if stop_reason is None:
        stop_reason = "exhausted"

    summary = SearchResult(
        dataset_id=config.dataset_id,
        code_version=config.code_version,
        seed=config.seed,
        trials=recorded,
        skipped=sum(skipped.values()),
        accepted=accepted,
        rejections=dict(sorted(rejections.items())),
        skipped_reasons=dict(sorted(skipped.items())),
        stop_reason=stop_reason,
        generations=generations_done,
        best=best_candidate,
        best_id=best_id,
        best_metrics=best_metrics,
        lineage_depth=deepest,
        tradable=tradable,
    )
    if summary.trials + summary.skipped != attempts:  # pragma: no cover - bookkeeping guard
        raise RuntimeError("search bookkeeping does not add up")
    return summary
