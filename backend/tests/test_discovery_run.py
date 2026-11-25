"""End-to-end discovery: stored bars → validation slice → search → decision.

Every test uses a tiny synthetic store under ``tmp_path``; nothing reads the
real data directory. The decision paths are pinned by construction, not by
luck: constant-price bars make any trade lose costs, so a run either finds a
flat candidate (``no_trade``), a losing one (``no_change``) or a gate that
rejects everything (``no_trade``). One test mutates every bar after the
validation split to prove the test rows cannot influence a score.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from scalping import cli
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.datasets.builder import build_dataset
from scalping.discovery import (
    DECISIONS,
    DiscoveryArchive,
    DiscoveryRunConfig,
    DiscoveryRunError,
    decide,
    run_discovery,
)
from scalping.discovery.run import BASELINE_LABELS
from scalping.env.simulator import WINDOW
from scalping.features.build import FEATURE_COLUMNS, build_features
from scalping.registry.gate import GateCriteria

SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
START = datetime(2026, 1, 5, tzinfo=UTC)
ROWS = 1200


def constant_bars(rows: int = ROWS, price: float = 1000.0) -> pl.DataFrame:
    """A flat M1 tape with a 0.20 spread: any round turn loses its costs."""
    stamps = [START + timedelta(minutes=index) for index in range(rows)]
    return pl.DataFrame(
        {
            "ts": stamps,
            "open": [price] * rows,
            "high": [price + 0.001] * rows,
            "low": [price - 0.001] * rows,
            "close": [price] * rows,
            "bid_close": [price] * rows,
            "ask_close": [price + 0.2] * rows,
            "spread": [0.2] * rows,
            "volume": [10] * rows,
        }
    )


def synthetic_bars() -> pl.DataFrame:
    provider = SyntheticProvider(seed=3)
    return provider.fetch(SYMBOL, START, START + timedelta(days=1))


def store(tmp_path: Path, bars: pl.DataFrame) -> Path:
    root = tmp_path / "store"
    ParquetStore(root).write(SYMBOL, TIMEFRAME, bars)
    return root


def config(tmp_path: Path, **overrides: Any) -> DiscoveryRunConfig:
    values: dict[str, Any] = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "max_trials": 3,
        "population": 3,
        "generations": 1,
        "patience": None,
        "seed": 0,
        "archive": tmp_path / "discovery.sqlite",
        "stress": (1.5, 2.0),
        "use_cache": False,
    }
    values.update(overrides)
    return DiscoveryRunConfig(**values)


def trial_rows(archive_path: str | Path) -> list[tuple[str, str, int, str]]:
    with DiscoveryArchive(archive_path) as archive:
        rows = archive.connection.execute(
            "SELECT candidate_id, metrics_json, seed, dataset_id FROM trials ORDER BY id"
        ).fetchall()
    return [(str(row[0]), str(row[1]), int(row[2]), str(row[3])) for row in rows]


def decision_rows(archive_path: str | Path) -> list[tuple[str, str, str]]:
    with DiscoveryArchive(archive_path) as archive:
        return [
            (record.candidate_id, record.decision, record.reason) for record in archive.decisions()
        ]


def decision_details(archive_path: str | Path) -> list[dict[str, Any] | None]:
    with DiscoveryArchive(archive_path) as archive:
        return [record.detail for record in archive.decisions()]


def champion_count(archive_path: str | Path) -> int:
    with DiscoveryArchive(archive_path) as archive:
        row = archive.connection.execute("SELECT COUNT(*) FROM champions").fetchone()
    return 0 if row is None else int(row[0])


def payload_of(result: Any) -> dict[str, Any]:
    """Round-trip through JSON with ``allow_nan=False``: the CLI's contract."""
    return json.loads(json.dumps(result.to_dict(), allow_nan=False))


# --- end-to-end ---------------------------------------------------------------


def test_run_scores_a_validation_slice_and_records_its_decision(tmp_path: Path) -> None:
    root = store(tmp_path, synthetic_bars())
    cfg = config(tmp_path, max_trials=3, population=3, generations=2, stress=(2.0,), seed=1)

    result = run_discovery(cfg, root=root)

    assert result.dataset_id.startswith("validation:XAUUSD:M1:h1:")
    assert result.validation["rows"] > cfg.window
    assert result.trials + result.skipped == cfg.max_trials
    assert result.stop_reason == "budget"
    assert result.decision in DECISIONS
    assert result.decision != "promoted"
    assert result.tradable > 0
    assert result.best_id is not None
    assert result.best_canonical is not None
    assert result.best_metrics is not None
    assert result.best_metrics["trades"] > 0, "a flat policy cannot be the run's best"

    payload = payload_of(result)
    assert payload["best"]["candidate_id"] == result.best_id
    assert payload["best"]["canonical"] == result.best_canonical
    assert payload["decision"] == result.decision
    assert payload["archive"] == str(cfg.archive)

    recorded = decision_rows(cfg.archive)
    assert len(recorded) == 1
    assert recorded[0][1] == result.decision
    assert recorded[0][2] == result.reason
    assert recorded[0][0] == result.decision_candidate


def test_the_test_split_cannot_affect_a_run(tmp_path: Path) -> None:
    bars = constant_bars()
    root = store(tmp_path, bars)

    first = run_discovery(config(tmp_path, archive=tmp_path / "first.sqlite"), root=root)
    span = first.validation

    # Rebuild the manifest to locate the untouched split, then rewrite every bar
    # at and after the validation end. A run that leaked test rows would move.
    features = build_features(bars)
    bundle = build_dataset(features, symbol=SYMBOL, timeframe=TIMEFRAME)
    assert bundle.manifest.validation.to_dict() == {
        "start": span["start"],
        "end": span["end"],
        "rows": span["rows"],
    }
    assert span["end"] <= bundle.manifest.test.start

    mutated = bars.with_columns(
        pl.when(pl.int_range(0, pl.len()) >= span["end"])
        .then(pl.lit(2000.0))
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(pl.int_range(0, pl.len()) >= span["end"])
        .then(pl.lit(2000.001))
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when(pl.int_range(0, pl.len()) >= span["end"])
        .then(pl.lit(1999.999))
        .otherwise(pl.col("low"))
        .alias("low"),
        pl.when(pl.int_range(0, pl.len()) >= span["end"])
        .then(pl.lit(5.0))
        .otherwise(pl.col("spread"))
        .alias("spread"),
    )
    ParquetStore(tmp_path / "mutated").write(SYMBOL, TIMEFRAME, mutated)
    second = run_discovery(
        config(tmp_path, archive=tmp_path / "second.sqlite"),
        root=tmp_path / "mutated",
    )

    assert second.validation == span
    first_rows = trial_rows(tmp_path / "first.sqlite")
    second_rows = trial_rows(tmp_path / "second.sqlite")
    assert [row[0] for row in first_rows] == [row[0] for row in second_rows]
    assert [row[1] for row in first_rows] == [row[1] for row in second_rows]
    assert all(row[3].startswith("validation:") for row in first_rows)


def test_the_same_inputs_reproduce_the_same_trials(tmp_path: Path) -> None:
    root = store(tmp_path, constant_bars())
    first = run_discovery(config(tmp_path, archive=tmp_path / "a.sqlite", seed=4), root=root)
    second = run_discovery(config(tmp_path, archive=tmp_path / "b.sqlite", seed=4), root=root)

    assert first.dataset_id == second.dataset_id
    assert first.code_version == second.code_version
    assert first.best_id == second.best_id
    assert first.reason == second.reason
    assert trial_rows(tmp_path / "a.sqlite") == trial_rows(tmp_path / "b.sqlite")


def test_a_second_run_on_the_same_archive_is_deduped(tmp_path: Path) -> None:
    root = store(tmp_path, constant_bars())
    cfg = config(tmp_path)
    first = run_discovery(cfg, root=root)
    assert first.trials == cfg.max_trials

    before = {(candidate, dataset) for candidate, _m, _s, dataset in trial_rows(cfg.archive)}
    second = run_discovery(cfg, root=root)

    rows = trial_rows(cfg.archive)
    pairs = [(candidate, dataset) for candidate, _m, _s, dataset in rows]
    assert len(pairs) == len(set(pairs)), "a candidate was trialled twice on one dataset"
    new = set(pairs) - before
    assert len(rows) == len(before) + len(new)
    assert second.trials == len(new)
    assert second.deduped >= 1
    assert second.trials + second.skipped <= cfg.max_trials
    assert second.decision in DECISIONS
    assert len(decision_rows(cfg.archive)) == 2


# --- baselines ----------------------------------------------------------------


def test_baseline_table_is_scored_on_the_same_slice(tmp_path: Path) -> None:
    root = store(tmp_path, synthetic_bars())
    result = run_discovery(config(tmp_path, seed=1), root=root)

    names = [row["name"] for row in result.baselines]
    assert names == ["hold", "ret_5"]
    for row in result.baselines:
        assert row["policy"] == BASELINE_LABELS[row["name"]]
        assert {"net_profit", "trades"} <= set(row["metrics"])
        assert isinstance(row["selection"], float)

    hold = result.baselines[0]
    rule = result.baselines[1]
    assert hold["metrics"]["trades"] == 0
    assert hold["selection"] == 0.0
    assert rule["metrics"]["trades"] > 0, "the ret_5 baseline must actually trade"
    assert result.comparison["metric"] == "net_profit"
    assert result.comparison["best_baseline"] in {"hold", "ret_5"}
    assert result.comparison["best_baseline_value"] == max(
        row["selection"] for row in result.baselines
    )
    assert result.comparison["best_candidate"] == (
        None if result.best_metrics is None else result.best_metrics["net_profit"]
    )


def test_ret5_baseline_matches_the_cli_index_at_the_default_window() -> None:
    policy = cli._rule_baseline  # the hard-coded WINDOW version
    from scalping.discovery.run import _ret5_policy

    window = WINDOW
    assert (window - 1) * len(FEATURE_COLUMNS) + FEATURE_COLUMNS.index("ret_5") == (cli._RET5_INDEX)
    short = _ret5_policy(window)
    size = window * len(FEATURE_COLUMNS) + 5
    for value in (-1.0, 0.0, 1.0):
        obs = np.zeros(size, dtype=np.float32)
        obs[(window - 1) * len(FEATURE_COLUMNS) + FEATURE_COLUMNS.index("ret_5")] = value
        assert short(obs) == policy(obs)


# --- decision paths -----------------------------------------------------------


def test_no_trade_when_the_gate_rejects_everything(tmp_path: Path) -> None:
    root = store(tmp_path, constant_bars())
    cfg = config(tmp_path, gate=GateCriteria(min_trades=10_000))

    result = run_discovery(cfg, root=root)

    assert result.trials > 0
    assert result.accepted == 0
    assert result.decision == "no_trade"
    assert "gate rejected all" in result.reason
    assert any(key.startswith("min_trades") for key in result.rejections)
    assert decision_rows(cfg.archive)[0][1] == "no_trade"


def test_no_tradable_candidate_when_every_accepted_policy_is_flat(tmp_path: Path) -> None:
    """Seed 0 produces accepted candidates with zero closed trades only."""
    root = store(tmp_path, constant_bars())
    cfg = config(tmp_path, max_trials=3, population=3, generations=2, seed=0)

    result = run_discovery(cfg, root=root)

    assert result.trials == 3
    assert result.accepted == 3
    assert result.tradable == 0
    assert result.best_id is None, "a zero-trade policy must not win by net_profit 0"
    assert result.decision == "no_trade"
    assert "no tradable candidate" in result.reason
    assert "flat by construction" not in result.reason
    assert champion_count(cfg.archive) == 0

    details = decision_details(cfg.archive)
    assert len(details) == 1 and details[0] is not None
    assert details[0]["reason_code"] == "no_tradable_candidate"
    assert details[0]["tradable"] == 0
    assert details[0]["accepted"] == 3
    assert details[0]["min_best_trades"] == 1


def test_a_trading_candidate_wins_when_the_gate_is_empty_and_others_are_flat(
    tmp_path: Path,
) -> None:
    """Seed 0 mixes one trading candidate with flat ones; the trader is best."""
    root = store(tmp_path, constant_bars())
    cfg = config(tmp_path, max_trials=4, population=4, generations=2, seed=0)

    result = run_discovery(cfg, root=root)

    assert result.accepted > 0
    assert result.tradable == 1
    assert result.best_id is not None
    assert result.best_metrics is not None
    assert result.best_metrics["trades"] > 0
    assert result.best_metrics["net_profit"] < 0.0
    assert result.decision == "no_change"
    assert "does not beat the best baseline" in result.reason
    assert champion_count(cfg.archive) == 0
    details = decision_details(cfg.archive)
    assert details[0] is not None and details[0]["reason_code"] == "no_change"


def test_min_best_trades_zero_restores_the_flat_best_path(tmp_path: Path) -> None:
    """Compatibility only: 0 disables the rule and the old decision returns."""
    root = store(tmp_path, constant_bars())
    cfg = config(
        tmp_path,
        max_trials=4,
        population=4,
        generations=1,
        seed=0,
        min_best_trades=0,
    )

    result = run_discovery(cfg, root=root)

    assert result.best_metrics is not None
    assert result.best_metrics["trades"] == 0.0
    assert result.decision == "no_trade"
    assert "flat by construction" in result.reason
    details = decision_details(cfg.archive)
    assert details[0] is not None and details[0]["reason_code"] == "flat_candidate"


def test_no_change_when_nothing_beats_the_baseline(tmp_path: Path) -> None:
    root = store(tmp_path, constant_bars())
    cfg = config(
        tmp_path,
        seed=2,
        max_trials=3,
        population=3,
        generations=1,
        gate=GateCriteria(min_trades=1),
    )

    result = run_discovery(cfg, root=root)

    assert result.accepted > 0
    assert result.best_metrics is not None
    assert result.best_metrics["trades"] > 0
    assert result.best_metrics["net_profit"] < 0.0
    assert result.decision == "no_change"
    assert "does not beat the best baseline" in result.reason
    assert result.comparison["beats_baseline"] is False


def test_cost_stress_can_reject_everything(tmp_path: Path) -> None:
    root = store(tmp_path, constant_bars())
    cfg = config(
        tmp_path,
        gate=GateCriteria(require_positive_cost_stress=(2.0,)),
        stress=(1.5, 2.0),
    )

    result = run_discovery(cfg, root=root)

    assert result.accepted == 0
    assert result.decision == "no_trade"
    assert any(key.startswith("cost_stress") for key in result.rejections)


def test_decide_covers_every_outcome_without_promoting() -> None:
    baselines = [{"name": "hold", "selection": 0.0}]

    assert (
        decide(
            metric="net_profit",
            trials=3,
            accepted=0,
            best_id=None,
            best_metrics=None,
            baselines=baselines,
        )[0]
        == "no_trade"
    )

    no_tradable = decide(
        metric="net_profit",
        trials=3,
        accepted=3,
        best_id=None,
        best_metrics=None,
        baselines=baselines,
        tradable=0,
    )
    assert no_tradable[0] == "no_trade"
    assert "no tradable candidate" in no_tradable[1]

    flat = decide(
        metric="net_profit",
        trials=3,
        accepted=1,
        best_id="a" * 64,
        best_metrics={"net_profit": 0.0, "trades": 0.0},
        baselines=baselines,
        tradable=0,
    )
    assert flat[0] == "no_trade" and "no tradable candidate" in flat[1]

    flat_compat = decide(
        metric="net_profit",
        trials=3,
        accepted=1,
        best_id="a" * 64,
        best_metrics={"net_profit": 0.0, "trades": 0.0},
        baselines=baselines,
        min_best_trades=0,
    )
    assert flat_compat[0] == "no_trade" and "flat by construction" in flat_compat[1]

    unchanged = decide(
        metric="net_profit",
        trials=3,
        accepted=1,
        best_id="b" * 64,
        best_metrics={"net_profit": -12.5, "trades": 3.0},
        baselines=baselines,
    )
    assert unchanged[0] == "no_change"

    improved = decide(
        metric="net_profit",
        trials=3,
        accepted=1,
        best_id="c" * 64,
        best_metrics={"net_profit": 12.5, "trades": 3.0},
        baselines=baselines,
    )
    assert improved[0] == "rejected"
    assert "registry gate" in improved[1]

    unmeasured = decide(
        metric="net_profit",
        trials=3,
        accepted=1,
        best_id="d" * 64,
        best_metrics={"net_profit": 12.5, "trades": 3.0},
        baselines=[],
    )
    assert unmeasured[0] == "no_change"

    # The public decision function must not trust a caller-supplied best id
    # when the evaluator omitted the trade count: no trade evidence, no edge.
    missing_trade_count = decide(
        metric="net_profit",
        trials=3,
        accepted=1,
        best_id="e" * 64,
        best_metrics={"net_profit": 50.0},
        baselines=baselines,
    )
    assert missing_trade_count[0] == "no_trade"
    assert "trade-count evidence" in missing_trade_count[1]

    invalid_trade_count = decide(
        metric="net_profit",
        trials=3,
        accepted=1,
        best_id="f" * 64,
        best_metrics={"net_profit": 50.0, "trades": 1.5},
        baselines=baselines,
    )
    assert invalid_trade_count[0] == "no_trade"
    assert "trade-count evidence" in invalid_trade_count[1]

    for decision, _reason in (
        no_tradable,
        flat,
        flat_compat,
        unchanged,
        improved,
        unmeasured,
    ):
        assert decision in DECISIONS
        assert decision != "promoted"


# --- errors and config --------------------------------------------------------


def test_a_store_without_bars_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DiscoveryRunError, match="ingest bars first"):
        run_discovery(config(tmp_path), root=tmp_path / "empty")


def test_a_dataset_without_validation_rows_is_an_error(tmp_path: Path) -> None:
    root = store(tmp_path, constant_bars())
    with pytest.raises(DiscoveryRunError, match="no validation rows"):
        run_discovery(config(tmp_path, validation_fraction=0.0), root=root)


def test_a_validation_split_shorter_than_the_window_is_an_error(tmp_path: Path) -> None:
    root = store(tmp_path, constant_bars())
    with pytest.raises(DiscoveryRunError, match="observation window"):
        run_discovery(config(tmp_path, window=ROWS), root=root)


def test_config_rejects_bad_budgets_and_baselines() -> None:
    with pytest.raises(ValueError, match="max_trials"):
        DiscoveryRunConfig(max_trials=0)
    with pytest.raises(ValueError, match="mutation_rate \\+ crossover_rate"):
        DiscoveryRunConfig(mutation_rate=0.8, crossover_rate=0.5)
    with pytest.raises(ValueError, match="baseline"):
        DiscoveryRunConfig(baselines=("champion",))
    with pytest.raises(ValueError, match="stress"):
        DiscoveryRunConfig(stress=(0.0,))
    with pytest.raises(ValueError, match="distinct"):
        DiscoveryRunConfig(baselines=("hold", "hold"))
    with pytest.raises(ValueError, match="min_best_trades"):
        DiscoveryRunConfig(min_best_trades=-1)
    with pytest.raises(TypeError, match="min_best_trades"):
        DiscoveryRunConfig(min_best_trades=True)


def test_result_refuses_promoted() -> None:
    from scalping.discovery.run import DiscoveryRunResult

    with pytest.raises(ValueError, match="registry gate owns promotion"):
        DiscoveryRunResult(
            dataset_id="validation:XAUUSD:M1:h1:abcd",
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            validation={"start": 0, "end": 1, "rows": 1},
            seed=0,
            code_version="test",
            trials=1,
            skipped=0,
            deduped=0,
            accepted=1,
            stop_reason="budget",
            rejections={},
            best_id="a" * 64,
            best_canonical={"kind": "strategy"},
            best_metrics={"net_profit": 1.0},
            baselines=(),
            comparison={},
            decision="promoted",
            reason="should not happen",
            decision_id=1,
            decision_candidate="a" * 64,
            archive="archive.sqlite",
        )


# --- CLI ----------------------------------------------------------------------


def test_cli_prints_one_json_line_and_writes_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = store(tmp_path, constant_bars())
    out = tmp_path / "report.json"

    code = cli.main(
        [
            "discovery",
            "run",
            "--root",
            str(root),
            "--max-trials",
            "2",
            "--population",
            "2",
            "--generations",
            "1",
            "--archive",
            str(tmp_path / "cli.sqlite"),
            "--stress",
            "",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    printed = capsys.readouterr()
    assert printed.err == ""
    assert printed.out.count("\n") == 1
    payload = json.loads(printed.out)
    assert payload["decision"] in {"no_trade", "no_change", "rejected"}
    assert payload["out"] == str(out)
    assert json.loads(out.read_text())["decision"] == payload["decision"]
    assert payload["trials"] + payload["skipped"] == 2
    if payload["best"] is not None:
        assert not any(key.startswith("stress_") for key in payload["best"]["metrics"])


def test_cli_reports_a_data_error_as_json_and_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "discovery",
            "run",
            "--root",
            str(tmp_path / "empty"),
            "--archive",
            str(tmp_path / "cli.sqlite"),
        ]
    )

    assert code == 2
    printed = capsys.readouterr()
    assert printed.out == ""
    assert json.loads(printed.err)["error"].startswith("no XAUUSD M1 bars")
