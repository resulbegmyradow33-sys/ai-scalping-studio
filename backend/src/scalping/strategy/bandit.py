"""LinUCB: linear upper-confidence-bound arm selection over the market context.

Model
-----
Each arm ``i`` keeps a ridge regression of its reward on the context ``x``. With
``A_i = ridge * I + sum_t x_t x_t^T`` and ``b_i = sum_t r_t x_t``, the closed-form
weight is ``theta_i = A_i^-1 b_i`` and the score of an arm is::

    p_i(x) = theta_i . x + alpha * sqrt(x^T A_i^-1 x)

The first term is the predicted reward, the second the exploration bonus; ``A^-1``
shrinks as an arm is pulled, so an arm is tried until its uncertainty is paid
down. ``alpha = 0`` is pure exploitation.

Bias column
-----------
Every context is augmented with a trailing constant ``1.0`` before it reaches the
design matrix, so ``theta`` has an extra intercept component. Without it a
constant reward would be unrepresentable: the context columns are zero-mean
(a one-hot regime, an RSI z-score, a success rate), so the least-squares slope
fits noise instead of the mean and a reliably rewarded arm would look like an
explored one. The intercept is what makes "always rewarded" beat "always not".
The public ``dim`` is the caller's context width; the intercept is internal and
reported by :meth:`LinUCB.arm_stats`, not by :meth:`LinUCB.weights`.

Determinism
-----------
:meth:`LinUCB.select` is a pure argmax over the scores with no random tie-break,
so an exact tie always resolves to the lowest arm index. A fresh bandit scores
every arm identically and therefore picks arm ``0``; after that pull its bonus is
the smallest, so the remaining untouched arms are tried in index order. The same
inputs always produce the same arm.

Persistence
-----------
:meth:`LinUCB.save` writes :func:`numpy.savez` only (no joblib, no pickle), and
:meth:`LinUCB.load` refuses a file whose arrays disagree with its metadata or
with the bandit it is restored into, rather than silently reinterpreting the
weights.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from scalping.strategy.labeling import CONTEXT_DIM

__all__ = ["LinUCB"]

#: File suffix :func:`numpy.savez` appends when the caller omits it.
_SUFFIX: str = ".npz"

#: Keys a saved file must carry. ``A``/``b`` embed the bias column.
_PAYLOAD_KEYS: tuple[str, ...] = (
    "dim",
    "n_arms",
    "alpha",
    "ridge",
    "A",
    "b",
    "pulls",
    "reward_sum",
)


def _npz_path(path: str | Path) -> Path:
    """The file :func:`numpy.savez` would actually write for ``path``.

    ``np.savez("model")`` writes ``model.npz``, so save and load both normalise
    the suffix; without it a caller that omitted ``.npz`` could not load what it
    just saved.
    """
    target = Path(path)
    if target.suffix != _SUFFIX:
        target = target.with_name(target.name + _SUFFIX)
    return target


class LinUCB:
    """A LinUCB bandit whose arms are positions in a frozen universe.

    ``n_arms`` and ``dim`` describe the arm and context counts; ``alpha`` is the
    exploration width and ``ridge`` the prior precision that keeps ``A`` positive
    definite before any pull. The instance starts with every arm at the prior:
    ``theta_i = 0`` and a bonus of ``alpha * sqrt(x^T (ridge I)^-1 x)``, which is
    what lets a zero-pull arm be tried.
    """

    def __init__(
        self,
        n_arms: int = 5,
        dim: int = CONTEXT_DIM,
        alpha: float = 0.5,
        ridge: float = 1.0,
    ) -> None:
        n_arms = int(n_arms)
        dim = int(dim)
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        alpha = float(alpha)
        ridge = float(ridge)
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError(f"alpha must be finite and >= 0, got {alpha}")
        if not isfinite(ridge) or ridge <= 0.0:
            raise ValueError(f"ridge must be finite and > 0, got {ridge}")

        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha
        self.ridge = ridge

        # Design width: the caller's context plus the internal bias column.
        self._width = dim + 1
        self._A = np.array([ridge * np.eye(self._width) for _ in range(n_arms)], dtype=np.float64)
        self._b = np.zeros((n_arms, self._width), dtype=np.float64)
        self._pulls = np.zeros(n_arms, dtype=np.int64)
        self._reward_sum = np.zeros(n_arms, dtype=np.float64)

    # -- validation ---------------------------------------------------------

    def _context(self, x: Any) -> np.ndarray:
        """``x`` as a finite float64 vector of width ``dim``, or ``ValueError``."""
        vector = np.asarray(x, dtype=np.float64)
        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(
                f"context must be a ({self.dim},) vector for dim={self.dim}, "
                f"got shape {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("context contains a non-finite value")
        return vector

    def _arm(self, arm: Any) -> int:
        """``arm`` as an in-range index; ``TypeError`` for a non-integer."""
        if isinstance(arm, bool) or not isinstance(arm, (int, np.integer)):
            raise TypeError(f"arm must be an integer, got {arm!r}")
        index = int(arm)
        if not 0 <= index < self.n_arms:
            raise ValueError(f"arm {index} outside 0..{self.n_arms - 1}")
        return index

    def _design(self, vector: np.ndarray) -> np.ndarray:
        """``x`` with the trailing bias column LinUCB's intercept lives in."""
        return np.append(vector, 1.0)

    # -- math ---------------------------------------------------------------

    def _solve(self, arm: int, right: np.ndarray) -> np.ndarray:
        """``A_i^-1 right``.

        ``ridge > 0`` makes ``A`` positive definite, so ``solve`` is the normal
        path. A pathological ``ridge`` (or a corrupted state) can still make the
        factorisation fail; the pseudo-inverse keeps the result finite instead of
        raising from inside a trading decision.
        """
        A = self._A[arm]
        try:
            return np.linalg.solve(A, right)
        except np.linalg.LinAlgError:  # pragma: no cover - ridge guards this
            return np.linalg.pinv(A) @ right

    def _theta(self, arm: int) -> np.ndarray:
        """Closed-form weight ``A_i^-1 b_i``, intercept included."""
        return self._solve(arm, self._b[arm])

    def _scores(self, design: np.ndarray) -> np.ndarray:
        """``p_i`` for every arm at one design vector, in arm order."""
        scores = np.empty(self.n_arms, dtype=np.float64)
        for arm in range(self.n_arms):
            solved = self._solve(arm, design)
            mean = float(self._b[arm] @ solved)
            variance = float(design @ solved)
            # A symmetric PD matrix gives a non-negative quadratic form; clipping
            # only absorbs round-off below zero before sqrt.
            scores[arm] = mean + self.alpha * np.sqrt(max(variance, 0.0))
        if not np.isfinite(scores).all():  # pragma: no cover - guarded upstream
            raise ValueError("non-finite bandit score")
        return scores

    # -- public API ---------------------------------------------------------

    def select(self, x: Any) -> int:
        """The highest-scoring arm for context ``x``, lowest index on a tie."""
        scores = self._scores(self._design(self._context(x)))
        # np.argmax returns the first maximum, i.e. the lowest arm index.
        return int(np.argmax(scores))

    def update(self, arm: int, x: Any, reward: float) -> None:
        """Fold one observed ``(context, reward)`` pair into ``arm``."""
        index = self._arm(arm)
        design = self._design(self._context(x))
        value = float(reward)
        if not isfinite(value):
            raise ValueError(f"reward must be finite, got {value}")
        self._A[index] += np.outer(design, design)
        self._b[index] += value * design
        self._pulls[index] += 1
        self._reward_sum[index] += value

    def fit_offline(self, X: Any, arms: Any, rewards: Any) -> LinUCB:
        """Replace the state with the closed-form ridge fit of a batch.

        ``X`` is ``(n, dim)``, ``arms`` an ``(n,)`` integer array of arm indices
        and ``rewards`` an ``(n,)`` array of R multiples. Each arm's ``A``/``b``
        is accumulated in one pass, so the result is exactly the online state the
        same events would produce, and it is deterministic: the fit starts from
        the ridge prior, it does not stack on whatever the instance held before.

        An arm with no pull keeps ``theta = 0`` and its prior ``A = ridge I``, so
        its score is the exploration bonus alone and it is still tried. An empty
        batch is a valid no-op that leaves the prior in place; no mean is ever
        divided by a zero pull count.
        """
        contexts = np.asarray(X, dtype=np.float64)
        if contexts.ndim != 2 or contexts.shape[1] != self.dim:
            raise ValueError(
                f"X must have shape (n, {self.dim}) for dim={self.dim}, got {contexts.shape}"
            )
        if not np.isfinite(contexts).all():
            raise ValueError("X contains a non-finite value")

        indices = np.asarray(arms)
        values = np.asarray(rewards, dtype=np.float64)
        count = contexts.shape[0]
        if indices.shape != (count,) or values.shape != (count,):
            raise ValueError(
                f"arms and rewards must have shape ({count},), got "
                f"{indices.shape} and {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("rewards contain a non-finite value")
        if count and not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"arms must be integers, got dtype {indices.dtype}")
        if count:
            low, high = int(indices.min()), int(indices.max())
            if low < 0 or high >= self.n_arms:
                raise ValueError(
                    f"arm {high if high >= self.n_arms else low} outside 0..{self.n_arms - 1}"
                )

        # Back to the prior, then one accumulation pass over the batch.
        self._A[:] = self.ridge * np.eye(self._width)
        self._b[:] = 0.0
        self._pulls[:] = 0
        self._reward_sum[:] = 0.0
        if count == 0:
            return self

        design = np.column_stack([contexts, np.ones(count, dtype=np.float64)])
        indices = indices.astype(np.int64)
        for arm in range(self.n_arms):
            mask = indices == arm
            if not mask.any():
                continue
            rows = design[mask]
            self._A[arm] += rows.T @ rows
            self._b[arm] += rows.T @ values[mask]
            self._pulls[arm] = int(mask.sum())
            self._reward_sum[arm] = float(values[mask].sum())
        return self

    def weights(self) -> np.ndarray:
        """``(n_arms, dim)`` closed-form slopes, the intercept column dropped.

        The dropped intercept is the bias term the context is augmented with; it
        is per-arm and available from :meth:`arm_stats`.
        """
        theta = np.empty((self.n_arms, self._width), dtype=np.float64)
        for arm in range(self.n_arms):
            theta[arm] = self._theta(arm)
        return theta[:, : self.dim].copy()

    def arm_stats(self) -> list[dict[str, float | int]]:
        """Per arm: pulls, mean reward (``0.0`` when unpulled) and intercept."""
        means = np.zeros(self.n_arms, dtype=np.float64)
        np.divide(
            self._reward_sum,
            self._pulls,
            out=means,
            where=self._pulls > 0,
        )
        theta = np.empty((self.n_arms, self._width), dtype=np.float64)
        for arm in range(self.n_arms):
            theta[arm] = self._theta(arm)
        return [
            {
                "arm": arm,
                "pulls": int(self._pulls[arm]),
                "mean_reward": float(means[arm]),
                "intercept": float(theta[arm, -1]),
            }
            for arm in range(self.n_arms)
        ]

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the state with :func:`numpy.savez`; returns the file written."""
        target = _npz_path(path)
        np.savez(
            target,
            dim=np.array(self.dim),
            n_arms=np.array(self.n_arms),
            alpha=np.array(self.alpha),
            ridge=np.array(self.ridge),
            A=self._A,
            b=self._b,
            pulls=self._pulls,
            reward_sum=self._reward_sum,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> LinUCB:
        """Rebuild a bandit from a :meth:`save` file.

        The stored metadata has to agree with the stored arrays (``A`` must be
        ``(n_arms, dim + 1, dim + 1)``), and every number has to be finite; a
        mismatch raises ``ValueError`` instead of silently loading weights into
        the wrong shape.
        """
        target = _npz_path(path)
        with np.load(target, allow_pickle=False) as data:
            missing = [key for key in _PAYLOAD_KEYS if key not in data.files]
            if missing:
                raise ValueError(f"saved bandit is missing arrays: {missing}")

            dim = int(data["dim"])
            n_arms = int(data["n_arms"])
            bandit = cls(
                n_arms=n_arms,
                dim=dim,
                alpha=float(data["alpha"]),
                ridge=float(data["ridge"]),
            )

            expected = (n_arms, dim + 1, dim + 1)
            A = np.array(data["A"], dtype=np.float64)
            if A.shape != expected:
                raise ValueError(
                    f"saved A has shape {A.shape}, expected {expected} for "
                    f"dim={dim} and n_arms={n_arms}"
                )
            b = np.array(data["b"], dtype=np.float64)
            if b.shape != (n_arms, dim + 1):
                raise ValueError(f"saved b has shape {b.shape}, expected {(n_arms, dim + 1)}")
            pulls = np.array(data["pulls"], dtype=np.int64)
            reward_sum = np.array(data["reward_sum"], dtype=np.float64)
            if pulls.shape != (n_arms,) or reward_sum.shape != (n_arms,):
                raise ValueError(
                    f"saved pulls/reward_sum have shapes {pulls.shape} and "
                    f"{reward_sum.shape}, expected {(n_arms,)}"
                )
            if not (
                np.isfinite(A).all() and np.isfinite(b).all() and np.isfinite(reward_sum).all()
            ):
                raise ValueError("saved bandit contains a non-finite value")

        bandit._A = A
        bandit._b = b
        bandit._pulls = pulls
        bandit._reward_sum = reward_sum
        return bandit

    def load_into(self, path: str | Path) -> LinUCB:
        """Restore a :meth:`save` file into this instance, if the shapes match.

        This is the "load over a live model" path: a file trained for a different
        ``dim`` or ``n_arms`` raises ``ValueError`` rather than filling the
        matrices with the wrong width. The hyperparameters stored in the file are
        adopted with the state.
        """
        other = type(self).load(path)
        if other.dim != self.dim:
            raise ValueError(f"cannot load a dim={other.dim} bandit into a dim={self.dim} bandit")
        if other.n_arms != self.n_arms:
            raise ValueError(
                f"cannot load an n_arms={other.n_arms} bandit into an n_arms={self.n_arms} bandit"
            )
        self.alpha = other.alpha
        self.ridge = other.ridge
        self._A = other._A
        self._b = other._b
        self._pulls = other._pulls
        self._reward_sum = other._reward_sum
        return self

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(n_arms={self.n_arms}, dim={self.dim}, "
            f"alpha={self.alpha}, ridge={self.ridge}, pulls={self._pulls.tolist()})"
        )
