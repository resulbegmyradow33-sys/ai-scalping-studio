"""Symbol and timeframe registry."""

from __future__ import annotations

import pytest

from scalping.data.registry import (
    TIMEFRAMES,
    contract_for,
    get_symbol,
    symbols,
    timeframe_rule,
)
from scalping.types import ContractSpec

REQUIRED = {
    "XAUUSD",
    "XAGUSD",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "NAS100",
    "US30",
    "USOIL",
    "BTCUSD",
}


def test_all_required_symbols_are_registered() -> None:
    assert REQUIRED <= set(symbols())


@pytest.mark.parametrize("symbol", sorted(REQUIRED))
def test_specs_map_to_a_valid_contract(symbol: str) -> None:
    spec = get_symbol(symbol)
    contract = contract_for(symbol)
    assert isinstance(contract, ContractSpec)
    assert contract.size == spec.contract_size
    assert contract.point == spec.point
    assert contract.min_lot == spec.min_lot
    assert spec.max_lot >= spec.min_lot
    assert spec.price_precision >= 0


def test_lookup_is_case_insensitive_and_rejects_unknown() -> None:
    assert get_symbol("xauusd").symbol == "XAUUSD"
    with pytest.raises(ValueError, match="unknown symbol"):
        get_symbol("NOTREAL")


def test_timeframes_include_the_required_ladder() -> None:
    assert {"M1", "M5", "M15", "M30", "H1", "H4", "D1"} <= set(TIMEFRAMES)
    assert timeframe_rule("m30") == "30m"
    assert timeframe_rule("H4") == "4h"
    with pytest.raises(ValueError, match="unsupported timeframe"):
        timeframe_rule("W1")
