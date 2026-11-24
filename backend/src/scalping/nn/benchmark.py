"""Small, honest benchmark for the sequence-model zoo.

What is measured: parameter count, forward latency (median of repeated runs),
throughput and the process high-water RSS. What is *not* claimed: the RSS number
is process-wide and monotonic, so it is a smoke signal, not isolated per-model
memory. Run one model per process for an isolated figure.

Latency is CPU-first (the default); ``device="mps"`` synchronises around each
measurement. Training time, gradient memory and data loading are out of scope.
"""

from __future__ import annotations

import resource
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Final

import torch

from scalping.nn.models import MODEL_NAMES, ModelName, build_model
from scalping.nn.train import DeviceRequest, select_device

__all__ = ["BenchmarkRow", "benchmark_models", "process_rss_mb"]

_TORCH_THREADS: Final[int] = 1


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    """One model's resource footprint at a fixed input shape."""

    model: str
    parameters: int
    parameter_mb: float
    latency_ms_median: float
    throughput_windows_per_s: float
    rss_high_water_mb: float
    device: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "model": self.model,
            "parameters": self.parameters,
            "parameter_mb": self.parameter_mb,
            "latency_ms_median": self.latency_ms_median,
            "throughput_windows_per_s": self.throughput_windows_per_s,
            "rss_high_water_mb": self.rss_high_water_mb,
            "device": self.device,
        }


def process_rss_mb() -> float:
    """High-water RSS in MiB (``ru_maxrss`` is bytes on macOS, KiB on Linux)."""
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage / (1024.0 * 1024.0) if sys.platform == "darwin" else usage / 1024.0


def _synchronise(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def benchmark_models(
    models: list[ModelName] | tuple[ModelName, ...] | None = None,
    *,
    n_features: int = 59,
    window: int = 60,
    batch: int = 8,
    repeats: int = 5,
    warmup: int = 2,
    hidden: int = 64,
    n_classes: int = 3,
    dropout: float = 0.1,
    device: DeviceRequest = "cpu",
    seed: int = 0,
) -> list[BenchmarkRow]:
    """Benchmark each requested model in order; defaults to the whole zoo."""
    names = list(MODEL_NAMES if models is None else models)
    unknown = [name for name in names if name not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"unknown models: {unknown}; choose from {MODEL_NAMES}")
    if n_features < 1 or window < 1 or batch < 1:
        raise ValueError("n_features, window and batch must be >= 1")
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup >= 0")

    target = select_device(device)
    torch.manual_seed(seed)
    torch.set_num_threads(_TORCH_THREADS)
    x = torch.randn(batch, window, n_features, device=target)

    rows: list[BenchmarkRow] = []
    for name in names:
        model = build_model(
            name,
            n_features=n_features,
            window=window,
            n_classes=n_classes,
            hidden=hidden,
            dropout=dropout,
            seed=seed,
        ).to(target)
        model.eval()
        parameters = sum(value.numel() for value in model.parameters())

        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            _synchronise(target)
            timings: list[float] = []
            for _ in range(repeats):
                start = time.perf_counter()
                model(x)
                _synchronise(target)
                timings.append(time.perf_counter() - start)

        median = statistics.median(timings)
        rows.append(
            BenchmarkRow(
                model=name,
                parameters=parameters,
                parameter_mb=parameters * 4 / (1024.0 * 1024.0),
                latency_ms_median=median * 1000.0,
                throughput_windows_per_s=(batch / median) if median > 0 else 0.0,
                rss_high_water_mb=process_rss_mb(),
                device=target.type,
            )
        )
    return rows
