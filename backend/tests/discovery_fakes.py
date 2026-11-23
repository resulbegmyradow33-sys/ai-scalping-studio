"""Toy strategy representation and operators for the discovery tests.

This is not a DSL and not a model of a trading rule. It is the smallest object
that satisfies the discovery protocols: three integers, a canonical dict, a
sha256 id and a complexity that is just the node count. Mutation, crossover and
validation are real enough to produce lineage and duplicates, which is what the
archive and search tests need. The real DSL/AST is a separate slice.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from scalping.discovery.protocols import TrialResult, canonical_digest

#: Values a node may take in a valid toy program.
MAX_NODE_VALUE = 5


@dataclass(frozen=True, slots=True)
class ToyCandidate:
    """A tiny program: a tuple of 1-4 small integers."""

    nodes: tuple[int, ...]

    def canonical_dict(self) -> dict[str, Any]:
        return {"format": "toy-v1", "nodes": list(self.nodes)}

    def canonical_hash(self) -> str:
        return canonical_digest(self.canonical_dict())

    def complexity(self) -> int:
        return len(self.nodes)


class ToyFactory:
    """Random, mutate, crossover and a structural validation gate."""

    def __init__(self, *, values: int = 6, max_nodes: int = 3, seed_offset: int = 0) -> None:
        if values < 2:
            raise ValueError("values must be >= 2")
        if max_nodes < 1:
            raise ValueError("max_nodes must be >= 1")
        self.values = values
        self.max_nodes = max_nodes
        self.seed_offset = seed_offset

    def _draw(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, self.values)) + self.seed_offset

    def random(self, rng: np.random.Generator) -> ToyCandidate:
        size = int(rng.integers(1, self.max_nodes + 1))
        return ToyCandidate(tuple(self._draw(rng) for _ in range(size)))

    def mutate(self, candidate: ToyCandidate, rng: np.random.Generator) -> ToyCandidate:
        nodes = list(candidate.nodes)
        position = int(rng.integers(0, len(nodes)))
        nodes[position] = self._draw(rng)
        if float(rng.random()) < 0.25 and len(nodes) < self.max_nodes:
            nodes.append(self._draw(rng))
        return ToyCandidate(tuple(nodes))

    def crossover(
        self, a: ToyCandidate, b: ToyCandidate, rng: np.random.Generator
    ) -> ToyCandidate | None:
        cut_a = int(rng.integers(0, len(a.nodes) + 1))
        cut_b = int(rng.integers(0, len(b.nodes) + 1))
        child = (a.nodes[:cut_a] + b.nodes[cut_b:])[: self.max_nodes]
        return ToyCandidate(child) if child else None

    def validate(self, candidate: ToyCandidate) -> None:
        if not candidate.nodes:
            raise ValueError("empty program")
        for node in candidate.nodes:
            if node < 0 or node > MAX_NODE_VALUE:
                raise ValueError(f"node {node} is outside [0, {MAX_NODE_VALUE}]")


class FixedFactory:
    """Always proposes the same candidate; used for duplicate and deadline tests."""

    def __init__(self, candidate: ToyCandidate) -> None:
        self.candidate = candidate

    def random(self, rng: np.random.Generator) -> ToyCandidate:
        return self.candidate

    def mutate(self, candidate: ToyCandidate, rng: np.random.Generator) -> ToyCandidate:
        return self.candidate

    def crossover(
        self, a: ToyCandidate, b: ToyCandidate, rng: np.random.Generator
    ) -> ToyCandidate | None:
        return self.candidate

    def validate(self, candidate: ToyCandidate) -> None:
        return None


class ToyEvaluator:
    """Score = sum of nodes; accepted when the score clears a threshold.

    ``calls`` records every invocation so a test can prove the evaluator only
    ever saw the configured dataset.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.0,
        score: Callable[[ToyCandidate], float] | None = None,
        crash: bool = False,
    ) -> None:
        self.threshold = threshold
        self._score = score or (lambda candidate: float(sum(candidate.nodes)))
        self.crash = crash
        self.calls: list[tuple[str, str, str, int]] = []

    def evaluate(
        self,
        candidate: ToyCandidate,
        *,
        dataset_id: str,
        code_version: str,
        budget: int,
    ) -> TrialResult:
        self.calls.append((candidate.canonical_hash(), dataset_id, code_version, budget))
        if self.crash:
            raise RuntimeError("evaluator crash")
        score = self._score(candidate)
        if not math.isfinite(score):
            raise ValueError("toy score is not finite")
        accepted = score >= self.threshold
        return TrialResult(
            metrics={"net_profit": score, "trades": float(len(candidate.nodes))},
            accepted=accepted,
            rejections=() if accepted else ("below_threshold",),
        )
