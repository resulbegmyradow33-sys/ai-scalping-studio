"""Strategy × regime table: join, coverage and aggregation.

The tests use explicit entry timestamps around label boundaries, because the
only thing that can go wrong here is attaching the wrong regime (usually the
next one). Costs and fills are the simulator's job; this module only groups the
closed trades it is given.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from scalping.regime import regime_metrics, strategy_regime_table, trades_with_regime
from scalping.types import Side, Trade

T0 = datetime(2026, 1, 5, tzinfo=UTC)


def labels_map(boundaries: dict[datetime, str]) -> pl.DataFrame:
    stamps = sorted(boundaries)
    return pl.DataFrame(
        {"ts": stamps, "regime": [boundaries[stamp] for stamp in stamps]}
    ).cast({"ts": pl.Datetime("us", "UTC")})


def trade(
    entry: datetime,
    r_multiple: float,
    *,
    pnl: float | None = None,
    strategy: str = "rule",
    side: Side = Side.LONG,
) -> Trade:
    return Trade(
        entry_ts=entry,
        exit_ts=entry + timedelta(minutes=5),
        side=side,
        lot=0.1,
        entry=100.0,
        exit=101.0,
        sl=99.0,
        tp=102.0,
        pnl=r_multiple * 100.0 if pnl is None else pnl,
        costs=1.0,
        r_multiple=r_multiple,
        reason="tp",
        strategy=strategy,
    )


def test_regime_is_attached_at_the_entry_bar() -> None:
    labels = labels_map(
        {
            T0: "a",
            T0 + timedelta(minutes=10): "b",
            T0 + timedelta(minutes=20): "c",
        }
    )
    trades = [
        trade(T0 + timedelta(minutes=5), 1.0),
        trade(T0 + timedelta(minutes=10), 1.0),
        trade(T0 + timedelta(minutes=19), 1.0),
        trade(T0 + timedelta(minutes=20), 1.0),
        trade(T0 - timedelta(minutes=1), 1.0),
    ]

    joined = trades_with_regime(trades, labels)

    assert joined["regime"].to_list() == ["unknown", "a", "b", "b", "c"]
    assert joined["strategy"].to_list() == ["rule"] * 5


def test_table_aggregates_per_strategy_and_regime() -> None:
    labels = labels_map({T0: "a", T0 + timedelta(minutes=10): "b"})
    trades = [
        trade(T0 + timedelta(minutes=1), 1.0, pnl=100.0, strategy="rule"),
        trade(T0 + timedelta(minutes=2), -0.5, pnl=-50.0, strategy="rule"),
        trade(T0 + timedelta(minutes=11), 0.5, pnl=50.0, strategy="rule"),
        trade(T0 + timedelta(minutes=3), -1.0, pnl=-100.0, strategy="other"),
    ]

    table = strategy_regime_table(trades, labels)
    rows = {
        (row["strategy"], row["regime"]): row
        for row in table.to_dicts()
    }

    rule_a = rows[("rule", "a")]
    assert rule_a["trades"] == 2
    assert rule_a["wins"] == 1
    assert rule_a["win_rate"] == pytest.approx(0.5)
    assert rule_a["expectancy_r"] == pytest.approx(0.25)
    assert rule_a["total_r"] == pytest.approx(0.5)
    assert rule_a["total_pnl"] == pytest.approx(50.0)
    assert rule_a["profit_factor"] == pytest.approx(2.0)

    rule_b = rows[("rule", "b")]
    assert rule_b["trades"] == 1
    assert rule_b["profit_factor"] is None  # wins with no losses: no finite ratio

    other_a = rows[("other", "a")]
    assert other_a["wins"] == 0
    assert other_a["profit_factor"] == pytest.approx(0.0)


def test_empty_trades_keep_the_schema() -> None:
    labels = labels_map({T0: "a"})

    joined = trades_with_regime([], labels)
    assert joined.height == 0
    assert joined.columns == [
        "entry_ts",
        "exit_ts",
        "strategy",
        "side",
        "r_multiple",
        "pnl",
        "regime",
    ]

    table = strategy_regime_table([], labels)
    assert table.height == 0
    assert table.columns == [
        "strategy",
        "regime",
        "trades",
        "wins",
        "win_rate",
        "expectancy_r",
        "total_r",
        "total_pnl",
        "profit_factor",
    ]


def test_missing_or_duplicate_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        trades_with_regime([trade(T0, 1.0)], pl.DataFrame({"ts": [T0]}))

    duplicate = pl.DataFrame(
        {"ts": [T0, T0], "regime": ["a", "b"]}
    ).cast({"ts": pl.Datetime("us", "UTC")})
    with pytest.raises(ValueError, match="duplicate"):
        trades_with_regime([trade(T0, 1.0)], duplicate)


def test_regime_metrics_describe_one_run_per_regime() -> None:
    labels = labels_map({T0: "a", T0 + timedelta(minutes=10): "b"})
    trades = [
        trade(T0 + timedelta(minutes=1), 1.0, pnl=100.0),
        trade(T0 + timedelta(minutes=2), -0.5, pnl=-50.0),
        trade(T0 + timedelta(minutes=11), 0.5, pnl=50.0),
    ]

    stats = {row["regime"]: row for row in regime_metrics(trades, labels).to_dicts()}

    first = stats["a"]
    assert first["trades"] == 2
    assert first["wins"] == 1
    assert first["win_rate"] == pytest.approx(0.5)
    assert first["expectancy_r"] == pytest.approx(0.25)
    assert first["total_r"] == pytest.approx(0.5)
    assert first["total_pnl"] == pytest.approx(50.0)
    assert first["profit_factor"] == pytest.approx(2.0)
    assert first["avg_hold_minutes"] == pytest.approx(5.0)

    second = stats["b"]
    assert second["trades"] == 1
    assert second["profit_factor"] is None


def test_regime_metrics_keep_unknown_and_empty_contracts() -> None:
    labels = labels_map({T0: "a"})
    before = trade(T0 - timedelta(minutes=1), 1.0, pnl=100.0)
    rows = regime_metrics([before], labels).to_dicts()
    assert [row["regime"] for row in rows] == ["unknown"]

    empty = regime_metrics([], labels)
    assert empty.height == 0
    assert empty.columns == [
        "regime",
        "trades",
        "wins",
        "win_rate",
        "expectancy_r",
        "total_r",
        "total_pnl",
        "profit_factor",
        "avg_hold_minutes",
    ]
