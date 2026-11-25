"""Bounded honest model comparison on one real chronological split.

Fixed question: on the same XAUUSD H1 rows, does logistic regression, LightGBM
or a small MLP beat the `ret_5` rule and the train-prior baseline on prediction
metrics, and do any of them trade profitably after costs?

Everything here is fixed a priori: the split, feature groups, horizon,
purge/embargo, model hyperparameters, probability threshold and cost stress.
Nothing is tuned on validation or test, and the test split is read once for all
arms in the same run and never used to select a model.

The output is descriptive evidence, not a promotion decision.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Keep the run single-threaded and reproducible on the 8 GB M2 target.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.datasets.builder import build_dataset
from scalping.discovery.evidence import settings_digest
from scalping.discovery.run import code_version
from scalping.features.build import build_features
from scalping.features.registry import DEFAULT_GROUPS
from scalping.ml.calibration import calibration_summary
from scalping.ml.evaluate import evaluate_model
from scalping.ml.train import MLResult, TrainSpec, train_classifier
from scalping.nn.evaluate import momentum_rule_labels

ROOT = Path(
    os.environ.get(
        "MODEL_COMPARE_STORE",
        Path(__file__).resolve().parents[4] / "data-ticks5-store",
    )
)
SYMBOL = "XAUUSD"
TIMEFRAME = "H1"
HORIZON = 5
SEED = 0
PURGE = 5
EMBARGO = 5
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
THRESHOLD = 0.55
EXIT_THRESHOLD = 0.55
WINDOW = 60
STRESS = (2.0, 3.0)
SPLITS = ("validation", "test")
SPLIT_INDEX = {"train": 0, "validation": 1, "test": 2}
OUT = Path(__file__).resolve().parents[2] / "docs/evidence/model-comparison-2026-11-25.json"


class RuleDirectionModel:
    """`ret_5 > 0` as a degenerate probability model.

    ``evaluate_model`` expects ``classes_`` and ``predict_proba``; with
    ``p_up = 1`` on a positive 5-bar return and both probabilities 0 otherwise,
    the 0.55 entry threshold reproduces the discovery `ret_5` long-only policy
    exactly (BUY when momentum is positive, never SELL, never an explicit
    CLOSE). Only the trading path is read from this object; prediction metrics
    for the rule come from :func:`momentum_rule_labels`.
    """

    classes_ = np.array([-1, 1])

    def __init__(self, ret5_index: int) -> None:
        self.ret5_index = ret5_index

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)[:, self.ret5_index]
        up = (values > 0.0).astype(np.float64)
        down = (values < 0.0).astype(np.float64)
        return np.column_stack([down, up])


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _prior_baseline(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    classes: tuple[int, ...],
) -> dict[str, Any]:
    labels, counts = np.unique(y_train, return_counts=True)
    majority = int(labels[int(np.argmax(counts))])
    prior = np.zeros(len(classes), dtype=np.float64)
    for index, label in enumerate(classes):
        prior[index] = float(np.sum(y_train == label)) / max(1, y_train.size)
    prior = np.clip(prior, 1e-9, None)
    prior /= prior.sum()
    return {
        "majority_class": majority,
        "accuracy": float(np.mean(y_eval == majority)),
        "log_loss": float(log_loss(y_eval, np.tile(prior, (y_eval.size, 1)), labels=list(classes))),
        "class_prior": {str(label): float(prior[index]) for index, label in enumerate(classes)},
    }


def _prediction_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[int, ...],
) -> dict[str, Any]:
    predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    one_hot = np.zeros_like(probabilities)
    for index, label in enumerate(classes):
        one_hot[y_true == label, index] = 1.0
    calibration = calibration_summary(y_true, probabilities, classes)
    return {
        "rows": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(classes))),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": float(calibration["ece"]),
        "mce": float(calibration["mce"]),
        "ece_positive_class": 1,
        "predicted_counts": {str(label): int((predicted == label).sum()) for label in classes},
    }


def _rule_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    return {
        "rows": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "log_loss": None,
        "brier": None,
        "ece": None,
        "mce": None,
        "probability_metrics_reason": "label-only predictor: no probabilities to score",
        "predicted_counts": {
            "-1": int((predicted == -1).sum()),
            "0": int((predicted == 0).sum()),
            "1": int((predicted == 1).sum()),
        },
    }


def _equity_floor(metrics: dict[str, Any]) -> dict[str, Any]:
    floor = -0.2 * 10_000.0
    return {
        "initial_balance": 10_000.0,
        "episode_termination_equity": 8_000.0,
        "net_at_or_below_termination_floor": bool(metrics["net_profit"] <= floor + 1e-6),
    }


def _trading(evaluation: Any) -> dict[str, Any]:
    return {
        "split": evaluation.split,
        "threshold": evaluation.threshold,
        "exit_threshold": evaluation.exit_threshold,
        "base": evaluation.metrics,
        "base_floor_check": _equity_floor(evaluation.metrics),
        "hold": evaluation.hold,
        "stress": evaluation.stress,
    }


def _mlp_result(
    bundle: Any,
    columns: tuple[str, ...],
    dataset_id: str,
    split_counts: dict[str, int],
    baseline_validation: dict[str, Any],
) -> tuple[MLResult, float]:
    keep = bundle.y_direction != 0
    train_mask = (bundle.split == 0) & keep
    classes = tuple(int(value) for value in np.unique(bundle.y_direction[train_mask]))
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32,),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=60,
                    shuffle=True,
                    random_state=SEED,
                    early_stopping=False,
                ),
            ),
        ]
    )
    started = time.monotonic()
    model.fit(bundle.X[train_mask], bundle.y_direction[train_mask])
    duration = time.monotonic() - started

    probabilities = model.predict_proba(bundle.X[(bundle.split == 1) & keep])
    validation = _prediction_metrics(
        bundle.y_direction[(bundle.split == 1) & keep], probabilities, classes
    )
    test_keep = (bundle.split == 2) & keep
    test = _prediction_metrics(
        bundle.y_direction[test_keep], model.predict_proba(bundle.X[test_keep]), classes
    )
    spec = TrainSpec(
        model="mlp",
        horizon=HORIZON,
        groups=DEFAULT_GROUPS,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        purge_bars=PURGE,
        embargo_bars=EMBARGO,
        seed=SEED,
        evaluate_test=True,
        include_flat=False,
    )
    return (
        MLResult(
            model=model,
            spec=spec,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            dataset_id=dataset_id,
            columns=columns,
            validation=validation,
            test=test,
            baseline=baseline_validation,
            split_counts=split_counts,
        ),
        duration,
    )


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(item) for item in value]
    return value


def main() -> int:
    started = time.monotonic()
    if not ROOT.is_dir():
        print(f"store not found: {ROOT}", file=sys.stderr)
        return 2
    meta_path = ROOT / "_meta" / SYMBOL / f"{TIMEFRAME}.json"
    store_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    settings = Settings()
    store = ParquetStore(str(ROOT))
    frame = store.read(SYMBOL, TIMEFRAME)
    if frame.is_empty():
        print(f"no {SYMBOL} {TIMEFRAME} bars under {ROOT}", file=sys.stderr)
        return 2
    features = build_features(frame, settings.costs)

    spec = TrainSpec(
        model="logistic",
        horizon=HORIZON,
        groups=DEFAULT_GROUPS,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        purge_bars=PURGE,
        embargo_bars=EMBARGO,
        seed=SEED,
        evaluate_test=True,
        include_flat=False,
    )
    bundle = build_dataset(
        features,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        groups=spec.groups,
        horizon=spec.horizon,
        train_fraction=spec.train_fraction,
        validation_fraction=spec.validation_fraction,
        purge_bars=max(spec.purge_bars, spec.horizon),
        embargo_bars=spec.embargo_bars,
    )
    keep = bundle.y_direction != 0
    classes = tuple(
        int(value) for value in np.unique(bundle.y_direction[(bundle.split == 0) & keep])
    )
    y_train = bundle.y_direction[(bundle.split == 0) & keep]
    split_masks = {split: (bundle.split == SPLIT_INDEX[split]) & keep for split in SPLITS}
    baselines = {
        split: _prior_baseline(y_train, bundle.y_direction[mask], classes)
        for split, mask in split_masks.items()
    }
    split_counts = {
        "train": int(((bundle.split == 0) & keep).sum()),
        "validation": int(split_masks["validation"].sum()),
        "test": int(split_masks["test"].sum()),
        "purged": int((bundle.split == -1).sum()),
        "flat_dropped": int((~keep).sum()),
    }

    arms: list[dict[str, Any]] = []
    for name in ("logistic", "lightgbm"):
        arm_spec = TrainSpec(
            model=name,
            horizon=HORIZON,
            groups=DEFAULT_GROUPS,
            train_fraction=TRAIN_FRACTION,
            validation_fraction=VALIDATION_FRACTION,
            purge_bars=PURGE,
            embargo_bars=EMBARGO,
            seed=SEED,
            evaluate_test=True,
            include_flat=False,
        )
        fit_started = time.monotonic()
        result = train_classifier(features, symbol=SYMBOL, timeframe=TIMEFRAME, spec=arm_spec)
        fit_duration = time.monotonic() - fit_started
        predictions = {
            split: _prediction_metrics(
                bundle.y_direction[split_masks[split]],
                result.model.predict_proba(bundle.X[split_masks[split]]),
                classes,
            )
            for split in SPLITS
        }
        trading = {}
        for split in SPLITS:
            evaluation = evaluate_model(
                result,
                features,
                settings,
                split=split,
                threshold=THRESHOLD,
                exit_threshold=EXIT_THRESHOLD,
                window=WINDOW,
                stress=STRESS,
                compute_regime=False,
            )
            trading[split] = _trading(evaluation)
        arms.append(
            {
                "name": name,
                "kind": "probabilistic classifier",
                "spec": asdict(arm_spec),
                "fit_seconds": round(fit_duration, 2),
                "prediction": {
                    split: {
                        "metrics": predictions[split],
                        "prior_baseline": baselines[split],
                    }
                    for split in SPLITS
                },
                "trading": trading,
            }
        )
        print(
            f"{name}: fit {fit_duration:.1f}s; "
            f"val acc {predictions['validation']['accuracy']:.4f} "
            f"logloss {predictions['validation']['log_loss']:.4f}; "
            f"test acc {predictions['test']['accuracy']:.4f} "
            f"logloss {predictions['test']['log_loss']:.4f}",
            file=sys.stderr,
        )

    # Small MLP on the same rows, with a scaler fitted on train only.
    try:
        mlp_started = time.monotonic()
        mlp_result, mlp_fit = _mlp_result(
            bundle,
            tuple(features.columns),
            bundle.manifest.dataset_id,
            split_counts,
            baselines["validation"],
        )
        mlp_total = time.monotonic() - mlp_started
        trading = {}
        for split in SPLITS:
            evaluation = evaluate_model(
                mlp_result,
                features,
                settings,
                split=split,
                threshold=THRESHOLD,
                exit_threshold=EXIT_THRESHOLD,
                window=WINDOW,
                stress=STRESS,
                compute_regime=False,
            )
            trading[split] = _trading(evaluation)
        arms.append(
            {
                "name": "mlp",
                "kind": "small MLP (32 hidden, scaler fitted on train)",
                "spec": asdict(mlp_result.spec),
                "fit_seconds": round(mlp_fit, 2),
                "fit_plus_scoring_seconds": round(mlp_total, 2),
                "prediction": {
                    "validation": {
                        "metrics": mlp_result.validation,
                        "prior_baseline": baselines["validation"],
                    },
                    "test": {
                        "metrics": mlp_result.test,
                        "prior_baseline": baselines["test"],
                    },
                },
                "trading": trading,
            }
        )
        print(
            f"mlp: fit {mlp_fit:.1f}s; "
            f"val acc {mlp_result.validation['accuracy']:.4f} "
            f"logloss {mlp_result.validation['log_loss']:.4f}; "
            f"test acc {mlp_result.test['accuracy']:.4f} "
            f"logloss {mlp_result.test['log_loss']:.4f}",
            file=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001 - a bounded arm records its own failure
        arms.append({"name": "mlp", "kind": "small MLP", "error": f"{type(exc).__name__}: {exc}"})
        print(f"mlp: stopped ({type(exc).__name__}: {exc})", file=sys.stderr)

    # The rule: same rows, label-only prediction, same simulator/backtest.
    ret5_index = list(features.columns).index("ret_5")
    rule_predictions = {}
    for split in SPLITS:
        predicted = momentum_rule_labels(bundle, bundle.ts[split_masks[split]])
        rule_predictions[split] = _rule_metrics(bundle.y_direction[split_masks[split]], predicted)
    rule_result = MLResult(
        model=RuleDirectionModel(ret5_index),
        spec=TrainSpec(
            model="ret_5",
            horizon=HORIZON,
            groups=DEFAULT_GROUPS,
            train_fraction=TRAIN_FRACTION,
            validation_fraction=VALIDATION_FRACTION,
            purge_bars=PURGE,
            embargo_bars=EMBARGO,
            seed=SEED,
        ),
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        dataset_id=bundle.manifest.dataset_id,
        columns=tuple(features.columns),
        validation={},
        test=None,
        baseline=baselines["validation"],
        split_counts=split_counts,
    )
    rule_trading = {}
    for split in SPLITS:
        evaluation = evaluate_model(
            rule_result,
            features,
            settings,
            split=split,
            threshold=THRESHOLD,
            exit_threshold=EXIT_THRESHOLD,
            window=WINDOW,
            stress=STRESS,
            compute_regime=False,
        )
        rule_trading[split] = _trading(evaluation)
    arms.append(
        {
            "name": "ret_5",
            "kind": "rule baseline: sign of 5-bar return, long-only",
            "spec": {"model": "ret_5", "horizon": HORIZON, "threshold": THRESHOLD},
            "prediction": {
                split: {
                    "metrics": rule_predictions[split],
                    "prior_baseline": baselines[split],
                }
                for split in SPLITS
            },
            "trading": rule_trading,
        }
    )
    print(
        "ret_5: val acc "
        f"{rule_predictions['validation']['accuracy']:.4f}; "
        f"test acc {rule_predictions['test']['accuracy']:.4f}",
        file=sys.stderr,
    )

    verdicts: dict[str, Any] = {}
    for split in SPLITS:
        baseline = baselines[split]
        rows = []
        for arm in arms:
            if "error" in arm:
                continue
            metrics = arm["prediction"][split]["metrics"]
            rows.append(
                {
                    "name": arm["name"],
                    "accuracy": metrics["accuracy"],
                    "accuracy_delta_vs_prior": metrics["accuracy"] - baseline["accuracy"],
                    "log_loss": metrics["log_loss"],
                    "log_loss_delta_vs_prior": (
                        None
                        if metrics["log_loss"] is None
                        else metrics["log_loss"] - baseline["log_loss"]
                    ),
                }
            )
        best_predictive = min(
            (row for row in rows if row["log_loss"] is not None),
            key=lambda row: row["log_loss"],
        )
        trading_rows = []
        for arm in arms:
            if "error" in arm:
                continue
            base = arm["trading"][split]["base"]
            trading_rows.append(
                {
                    "name": arm["name"],
                    "trades": base["trades"],
                    "net_profit": base["net_profit"],
                    "profit_factor": base["profit_factor"],
                    "expectancy_r": base["expectancy_r"],
                    "total_costs": base["total_costs"],
                    "hit_equity_floor": arm["trading"][split]["base_floor_check"][
                        "net_at_or_below_termination_floor"
                    ],
                }
            )
        verdicts[split] = {
            "prior_baseline": baseline,
            "arms": rows,
            "best_by_log_loss": best_predictive["name"],
            "any_model_beats_prior_log_loss": any(
                (row["log_loss_delta_vs_prior"] is not None)
                and row["log_loss_delta_vs_prior"] < 0.0
                for row in rows
            ),
            "trading": trading_rows,
            "any_arm_positive_at_base_cost": any(row["net_profit"] > 0.0 for row in trading_rows),
            "any_arm_positive_under_stress": any(
                value["net_profit"] > 0.0
                for arm in arms
                if "error" not in arm
                for value in arm["trading"][split]["stress"].values()
            ),
        }

    elapsed = time.monotonic() - started
    payload = {
        "report": "model-comparison-2026-11-25",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "descriptive prediction/trading comparison; NOT validated, NOT promoted",
        "question": (
            "On one fixed chronological XAUUSD H1 split, do logistic, LightGBM or a "
            "small MLP beat the ret_5 rule and the train prior on prediction metrics, "
            "and do any survive simulator costs?"
        ),
        "provenance": {
            "git_commit": _git_commit(),
            "code_version": code_version(),
            "settings_digest": settings_digest(settings),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "sklearn": __import__("sklearn").__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "numpy": np.__version__,
            "libomp_available": (Path("/opt/homebrew/opt/libomp/lib/libomp.dylib").is_file()),
            "store_root": str(ROOT),
            "store_checksum": store_meta["checksum"],
            "store_source": store_meta["source"],
            "store_synthetic": store_meta["synthetic"],
            "store_first_ts": store_meta["first_ts"],
            "store_last_ts": store_meta["last_ts"],
            "store_rows": store_meta["rows"],
            "dataset_id": bundle.manifest.dataset_id,
            "dataset_checksum": bundle.manifest.checksum,
            "dataset_columns": len(bundle.manifest.columns),
            "dataset_rows": bundle.manifest.rows,
            "split_counts": split_counts,
        },
        "protocol": {
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "horizon_bars": HORIZON,
            "feature_groups": list(DEFAULT_GROUPS),
            "feature_columns": len(features.columns),
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "initial_balance": 10_000.0,
            "risk_daily_loss_pct": settings.risk.daily_loss_pct,
            "risk_per_trade_pct": settings.risk.risk_pct,
            "episode_termination": (
                "simulator ends the episode at equity <= 80% of initial (-$2,000); "
                "losing runs therefore report a net near that floor, not their "
                "unbounded path loss"
            ),
            "costs": asdict(settings.costs),
            "purge_bars": max(PURGE, HORIZON),
            "embargo_bars": EMBARGO,
            "flat_rows": "dropped (exact-zero direction labels excluded)",
            "seed": SEED,
            "entry_threshold": THRESHOLD,
            "exit_threshold": EXIT_THRESHOLD,
            "simulator_window": WINDOW,
            "cost_stress_multipliers": list(STRESS),
            "splits_scored": list(SPLITS),
            "tuning": "none: fixed hyperparameters, threshold and split",
            "test_split_policy": (
                "read once for every pre-registered arm in the same run; never used "
                "to select, tune or promote a model"
            ),
            "promotion": "none; this file is not promotion evidence",
        },
        "arms": arms,
        "verdicts": verdicts,
        "limits": [
            "Prediction accuracy is not a trading edge; the simulator and cost stress are the gate.",
            "One store, one symbol, one timeframe, one chronological split.",
            "No hyperparameter search, no walk-forward, no locked holdout.",
            "The test split was read once for all arms at once; it was not used to choose a model.",
            "The MLP uses an in-train StandardScaler and a fixed 60-epoch budget.",
            "The ret_5 rule has no probabilities, so log loss, Brier and ECE are not defined for it.",
        ],
        "resource": {
            "elapsed_seconds": round(elapsed, 2),
            "max_rss_bytes": int(
                __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss
            ),
            "threads": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "POLARS_MAX_THREADS": os.environ.get("POLARS_MAX_THREADS"),
            },
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_round(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT} in {elapsed:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
