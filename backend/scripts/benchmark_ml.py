"""Benchmark classical ML classifiers at a fixed synthetic shape.

Measures fit time, prediction latency and in-sample holdout accuracy for the
CPU baselines. Optional LightGBM/XGBoost rows report their import/runtime error
instead of disappearing. This is a local measurement, not a model comparison:
the numbers depend on this machine, shape and seed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scalping.ml.models import MODEL_NAMES, build_model

DEFAULT_MODELS = ("logistic", "random_forest", "extra_trees", "hist_gradient")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark classical ML baselines.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--rows", type=int, default=4_000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    return parser


def _data(rows: int, features: int, classes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, features))
    signal = X[:, 0] - 0.5 * X[:, 1]
    y = np.digitize(signal, np.quantile(signal, np.linspace(0, 1, classes + 1)[1:-1]))
    return X, y.astype(np.int64)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rows < 30 or args.features < 1 or args.classes < 2:
        raise SystemExit("rows >= 30, features >= 1 and classes >= 2 are required")
    X, y = _data(args.rows, args.features, args.classes, args.seed)
    split = int(args.rows * 0.7)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    requested = [part.strip() for part in args.models.split(",") if part.strip()]
    unknown = [name for name in requested if name not in MODEL_NAMES]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; choose from {MODEL_NAMES}")

    rows: list[dict[str, object]] = []
    for name in requested:
        try:
            model = build_model(name, seed=args.seed)
            started = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            probabilities = model.predict_proba(X_test)
            predict_seconds = time.perf_counter() - started
            labels = np.asarray(model.classes_)[probabilities.argmax(axis=1)]
            accuracy = float((labels == y_test).mean())
            rows.append(
                {
                    "model": name,
                    "ok": True,
                    "fit_seconds": fit_seconds,
                    "predict_ms": predict_seconds * 1000.0,
                    "accuracy": accuracy,
                    "train_rows": int(X_train.shape[0]),
                    "test_rows": int(X_test.shape[0]),
                    "features": args.features,
                }
            )
        except (ImportError, OSError, RuntimeError) as exc:
            rows.append({"model": name, "ok": False, "error": str(exc)})

    payload = {"rows": rows}
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the subprocess
    raise SystemExit(main())
