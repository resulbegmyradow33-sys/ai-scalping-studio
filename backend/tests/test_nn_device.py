"""Device choice: MPS self-test, auto fallback and strict explicit request."""

from __future__ import annotations

import pytest
import torch

import scalping.nn.device as device_module
from scalping.nn import train as trainer
from scalping.nn.device import choose_device, mps_self_test
from scalping.nn.train import TrainError


def test_cpu_is_always_cpu() -> None:
    choice = choose_device("cpu")

    assert choice.device.type == "cpu"
    assert choice.fallback_reason is None
    assert choice.to_dict() == {"device": "cpu", "requested": "cpu", "fallback_reason": None}


def test_auto_picks_a_working_device(monkeypatch) -> None:
    monkeypatch.setattr(device_module, "mps_self_test", lambda: (False, "mps_unavailable"))
    cpu = choose_device("auto")
    assert cpu.device.type == "cpu"
    assert cpu.fallback_reason == "mps_unavailable"

    monkeypatch.setattr(device_module, "mps_self_test", lambda: (True, None))
    mps = choose_device("auto")
    assert mps.device.type == "mps"
    assert mps.fallback_reason is None


def test_explicit_mps_is_strict(monkeypatch) -> None:
    monkeypatch.setattr(
        device_module, "mps_self_test", lambda: (False, "mps_error:RuntimeError")
    )

    with pytest.raises(TrainError, match="self-test failed"):
        choose_device("mps")


def test_self_test_reports_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    ok, reason = mps_self_test()

    assert ok is False
    assert reason == "mps_unavailable"


def test_self_test_swallows_backend_errors(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    def explode(*args, **kwargs):
        raise RuntimeError("driver exploded")

    monkeypatch.setattr(torch, "ones", explode)
    ok, reason = mps_self_test()

    assert ok is False
    assert reason == "mps_error:RuntimeError"


def test_trainer_select_device_uses_the_same_policy(monkeypatch) -> None:
    monkeypatch.setattr(device_module, "mps_self_test", lambda: (False, "mps_unavailable"))

    assert trainer.select_device("auto").type == "cpu"
    assert trainer.select_device("cpu").type == "cpu"
