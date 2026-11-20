from datetime import UTC, datetime

import pytest

from scalping.config import Settings
from scalping.types import Bar, ContractSpec, RiskConfig, Side


def test_contract_defaults_match_spec():
    c = ContractSpec()
    assert (c.size, c.point, c.min_lot, c.commission_per_lot, c.leverage) == (
        100.0,
        0.001,
        0.01,
        7.0,
        20.0,
    )


def test_risk_defaults_match_spec():
    r = RiskConfig()
    assert (r.risk_pct, r.daily_loss_pct, r.max_notional_x, r.max_bars_in_trade) == (
        0.005,
        0.03,
        20.0,
        120,
    )


def test_bar_rejects_low_above_high():
    with pytest.raises(ValueError, match="low"):
        Bar(ts=datetime(2026, 1, 5, tzinfo=UTC), open=1, high=1, low=2,
            close=1, bid_close=1, ask_close=1, spread=0, volume=1)


def test_side_arithmetic():
    assert Side.LONG * -1 is Side.SHORT
    assert Side.FLAT * -1 is Side.FLAT


def test_settings_load_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("contract:\n  initial_balance: 5000\nrisk:\n  risk_pct: 0.01\n", encoding="utf-8")
    s = Settings.load(p)
    assert s.contract.initial_balance == 5000
    assert s.risk.risk_pct == 0.01
    assert s.risk.daily_loss_pct == 0.03  # default preserved
