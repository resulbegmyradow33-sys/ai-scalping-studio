"""Interfaces between the discovery loop, a strategy representation and an evaluator.

This module is structure, not behaviour. It defines what a candidate, a
candidate factory and an evaluator must look like so the search loop in
:mod:`scalping.discovery.search` can run without knowing the strategy language.
Nothing here compiles, interprets or executes a candidate; the archive stores
canonical JSON and the loop asks an injected evaluator for a result.

Safety rule for implementers: a DSL that satisfies :class:`Factory` must read its
own typed nodes and must never call ``eval``/``exec`` on generated text, ``compile``
it, or import a module named by candidate data. A strategy is data. If a
representation needs arithmetic, it gets an interpreter with a closed set of
node types, not a dynamic expression evaluator.

Reproducibility rule: ``canonical_hash()`` must equal
:func:`canonical_digest` of ``canonical_dict()``. The archive refuses a candidate
whose id does not match its own canonical JSON, because the id is the identity
the whole evidence chain hangs on.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Candidate",
    "Evaluator",
    "Factory",
    "TrialResult",
    "canonical_digest",
    "canonical_json",
]


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Canonical spelling of a candidate or trial payload.

    Sorted keys, no whitespace, ASCII escapes and ``allow_nan=False``: the same
    rules as the model registry, so a payload has exactly one digest. NaN and
    infinity fail here instead of reaching the database.
    """
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_digest(payload: Mapping[str, Any]) -> str:
    """Lowercase sha256 hex of :func:`canonical_json`.

    A DSL should implement ``Candidate.canonical_hash`` by returning this value,
    which is what the archive verifies on insert.
    """
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TrialResult:
    """One evaluation of one candidate on one dataset.

    ``metrics`` are numbers only: trade counts, net profit, profit factor,
    drawdown, cost assumptions and anything else the evaluator wants to rank on.
    Non-numeric context (a symbol, a split name) belongs in ``dataset_id`` or in
    the run configuration, because :meth:`DiscoveryArchive.best` ranks by a
    numeric key.

    ``accepted`` is the evaluator's evidence gate, not a judgement about the
    search. A rejected result must say why; an accepted one cannot carry
    rejection reasons, so the rejection histogram in
    :class:`~scalping.discovery.search.SearchResult` is never double-counted.
    """

    metrics: dict[str, float]
    accepted: bool
    rejections: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, dict):
            raise TypeError(f"metrics must be a dict, got {type(self.metrics).__name__}")
        cleaned: dict[str, float] = {}
        for key, value in self.metrics.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metric names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"metric {key!r} must be a number, got {type(value).__name__}")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"metric {key!r} must be finite, got {number!r}")
            cleaned[key] = number
        object.__setattr__(self, "metrics", cleaned)

        if not isinstance(self.accepted, bool):
            raise TypeError(f"accepted must be a bool, got {type(self.accepted).__name__}")
        if not isinstance(self.rejections, tuple):
            raise TypeError("rejections must be a tuple of strings")
        for reason in self.rejections:
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("rejection reasons must be non-empty strings")
        if self.accepted and self.rejections:
            raise ValueError("an accepted trial cannot carry rejection reasons")
        if not self.accepted and not self.rejections:
            raise ValueError("a rejected trial must carry at least one rejection reason")


@runtime_checkable
class Candidate(Protocol):
    """One strategy representation, identified by its canonical hash."""

    def canonical_dict(self) -> dict[str, Any]:
        """JSON-safe payload that defines the candidate exactly.

        Two candidates with the same canonical dict are the same strategy, in
        this run and in every later one. The dict must contain everything that
        changes behaviour: nodes, parameters, windows and a format version.
        """
        ...

    def canonical_hash(self) -> str:
        """sha256 of :func:`canonical_json` over ``canonical_dict()``."""
        ...

    def complexity(self) -> int:
        """Non-negative structural cost, used for the complexity cap."""
        ...


@runtime_checkable
class Factory(Protocol):
    """Generative operators over an injected candidate representation.

    ``rng`` is a :class:`numpy.random.Generator`. NumPy guarantees the stream
    for a given bit generator and seed, which is what makes a search replayable
    from ``SearchConfig.seed``; the stdlib ``random`` module does not make that
    promise across versions.
    """

    def random(self, rng: np.random.Generator) -> Candidate:
        """A fresh candidate drawn from the search space."""
        ...

    def mutate(self, candidate: Candidate, rng: np.random.Generator) -> Candidate:
        """One child derived from ``candidate``; the parent is left untouched."""
        ...

    def crossover(self, a: Candidate, b: Candidate, rng: np.random.Generator) -> Candidate | None:
        """A child of two parents, or ``None`` when the pair cannot combine."""
        ...

    def validate(self, candidate: Candidate) -> None:
        """Raise ``Exception`` when the AST is structurally invalid.

        This is the safety gate. The search loop treats any exception as an
        invalid candidate, counts it and never evaluates or archives it.
        """
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores a candidate on one dataset.

    The loop hands the evaluator a **validation** dataset id only. Evaluating on
    the untouched test slice is a separate, later gate; a search that sees the
    final holdout turns it into another training set.

    ``budget`` is the number of trials the caller may still spend on this
    dataset, including this one. An evaluator that is expensive per trial can
    use it to widen or narrow its folds; it must not interpret it as a time
    limit.
    """

    def evaluate(
        self,
        candidate: Candidate,
        *,
        dataset_id: str,
        code_version: str,
        budget: int,
    ) -> TrialResult:
        """Evaluate one candidate and return its metrics and verdict."""
        ...
