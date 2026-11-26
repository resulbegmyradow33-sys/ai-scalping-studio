"""Device choice with a real MPS self-test and an explicit CPU fallback.

``torch.backends.mps.is_available()`` can be true while the backend still fails
on the first kernel (driver/OS mismatch). For a training run that is a crash
halfway through the first epoch, so ``auto`` now runs a tiny matmul on MPS
before committing; a failure falls back to CPU and records why.

An explicit ``mps`` request stays strict: silently running on CPU would make a
benchmark or a report lie. It raises instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from scalping.nn.train import TrainError

__all__ = ["DeviceRequest", "DeviceSelection", "choose_device", "mps_self_test"]

DeviceRequest = Literal["auto", "cpu", "mps"]

_SELF_TEST_SIZE = 8


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    """The chosen device plus the reason a fallback happened, if any."""

    device: torch.device
    requested: DeviceRequest
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "device": self.device.type,
            "requested": self.requested,
            "fallback_reason": self.fallback_reason,
        }


def mps_self_test() -> tuple[bool, str | None]:
    """Run a tiny kernel on MPS; return ``(ok, reason)`` without raising."""
    if not torch.backends.mps.is_available():
        return False, "mps_unavailable"
    try:
        left = torch.ones((_SELF_TEST_SIZE, _SELF_TEST_SIZE), device="mps")
        value = (left @ left).sum().item()
        if value != float(_SELF_TEST_SIZE**3):
            return False, "mps_wrong_result"
        return True, None
    except Exception as exc:  # noqa: BLE001 - any backend failure means fallback
        return False, f"mps_error:{type(exc).__name__}"


def choose_device(request: DeviceRequest = "auto") -> DeviceSelection:
    """Resolve ``auto``/``cpu``/``mps`` with the MPS self-test and fallback."""
    if request == "cpu":
        return DeviceSelection(torch.device("cpu"), request)
    if request == "mps":
        ok, reason = mps_self_test()
        if not ok:
            raise TrainError(
                f"MPS was requested but the self-test failed ({reason}); "
                "use device='cpu' or 'auto'"
            )
        return DeviceSelection(torch.device("mps"), request)
    ok, reason = mps_self_test()
    if ok:
        return DeviceSelection(torch.device("mps"), request)
    return DeviceSelection(torch.device("cpu"), request, fallback_reason=reason)
