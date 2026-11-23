"""The MT5 adapter is a skeleton: disabled by default and refused everywhere.

The tests never need a terminal. A stub stands in for the terminal package so
the enabled path can be exercised, and the source is scanned so the import stays
lazy and no network client or secret creeps in.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scalping.paper import mt5
from scalping.paper.mt5 import ORDER_METHODS, AdapterDisabled, Mt5DemoAdapter


def _stub_package(monkeypatch) -> SimpleNamespace:
    """Replace the module's importlib handle, not the process-wide one."""
    package = SimpleNamespace(name="MetaTrader5")
    monkeypatch.setattr(mt5, "importlib", SimpleNamespace(import_module=lambda name: package))
    return package


def _missing_package(monkeypatch) -> None:
    def fail(name: str):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(mt5, "importlib", SimpleNamespace(import_module=fail))


def test_adapter_is_disabled_by_default():
    with pytest.raises(AdapterDisabled) as excinfo:
        Mt5DemoAdapter()
    assert "disabled by default" in str(excinfo.value)


def test_enable_without_the_package_still_refuses(monkeypatch):
    _missing_package(monkeypatch)
    with pytest.raises(AdapterDisabled) as excinfo:
        Mt5DemoAdapter(enabled=True)
    message = str(excinfo.value)
    assert "not installed" in message
    assert "real-money" in message


def test_enable_only_loads_the_package(monkeypatch):
    _stub_package(monkeypatch)
    adapter = Mt5DemoAdapter(enabled=True)
    assert adapter.enabled is True


@pytest.mark.parametrize("name", ORDER_METHODS)
def test_every_operation_refuses(monkeypatch, name):
    _stub_package(monkeypatch)
    adapter = Mt5DemoAdapter(enabled=True)
    method = getattr(adapter, name)
    with pytest.raises(NotImplementedError) as excinfo:
        method()
    message = str(excinfo.value)
    assert "real-money execution" in message
    assert "out of scope" in message


def test_importing_the_module_does_not_load_the_terminal_package():
    """The terminal package must never appear in ``sys.modules`` from our import."""
    was_loaded = "MetaTrader5" in sys.modules
    module = importlib.reload(mt5)
    assert ("MetaTrader5" in sys.modules) is was_loaded
    assert not hasattr(module, "MetaTrader5")


def test_source_has_no_module_level_terminal_import():
    source = Path(mt5.__file__).read_text(encoding="utf-8")
    assert "import MetaTrader5" not in source
    assert "from MetaTrader5" not in source
    for line in source.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("import MetaTrader5")
        assert not stripped.startswith("from MetaTrader5")


def test_source_has_no_network_client_or_credentials():
    source = Path(mt5.__file__).read_text(encoding="utf-8").lower()
    for network_module in ("socket", "requests", "httpx", "urllib", "http.client", "aiohttp"):
        assert f"import {network_module}" not in source
        assert f"from {network_module}" not in source
    for marker in ("password", "passwd", "api_key", "apikey", "access_token", "secret="):
        assert marker not in source


def test_a_caller_cannot_unlock_an_order_path(monkeypatch):
    """There is no flag or attribute that turns a refusal into a call."""
    _stub_package(monkeypatch)
    adapter = Mt5DemoAdapter(enabled=True)
    for attribute in ("live", "real", "trade_allowed", "allow_trading", "demo"):
        monkeypatch.setattr(adapter, attribute, True, raising=False)
    with pytest.raises(NotImplementedError):
        adapter.place_order(side="long", lot=0.1)
