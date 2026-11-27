"""Profile one training step of a sequence model on CPU or MPS.

The output is a local profile: parameter count, the top operators by CPU time
and by self CPU memory, and a textual table. It is a diagnostic, not a
benchmark ranking; shapes and seeds are fixed so runs are comparable on the
same machine.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from scalping.nn.models import MODEL_NAMES, build_model
from scalping.nn.train import select_device


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile one sequence-model step.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="gru")
    parser.add_argument("--n-features", type=int, default=59)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch < 1 or args.window < 1 or args.n_features < 1 or args.steps < 1:
        raise SystemExit("batch, window, n-features and steps must be >= 1")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    model = build_model(
        args.model,
        n_features=args.n_features,
        window=args.window,
        n_classes=3,
        hidden=args.hidden,
        seed=args.seed,
    ).to(device)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.randn(args.batch, args.window, args.n_features, device=device)
    y = torch.randint(0, 3, (args.batch,), device=device)

    activities = [ProfilerActivity.CPU]
    if device.type == "mps":
        activities.append(ProfilerActivity.MPS)
    with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
        for _ in range(args.steps):
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()

    averages = prof.key_averages()
    top_cpu = [
        {
            "op": event.key,
            "cpu_time_ms": event.cpu_time_total / 1000.0,
            "self_cpu_memory_mb": event.self_cpu_memory_usage / (1024.0 * 1024.0),
            "calls": int(event.count),
        }
        for event in sorted(
            averages, key=lambda item: item.cpu_time_total, reverse=True
        )[:10]
    ]
    top_memory = [
        {
            "op": event.key,
            "self_cpu_memory_mb": event.self_cpu_memory_usage / (1024.0 * 1024.0),
        }
        for event in sorted(
            averages, key=lambda item: item.self_cpu_memory_usage, reverse=True
        )[:10]
    ]
    payload = {
        "model": args.model,
        "device": device.type,
        "parameters": sum(value.numel() for value in model.parameters()),
        "shape": {"batch": args.batch, "window": args.window, "features": args.n_features},
        "steps": args.steps,
        "top_cpu": top_cpu,
        "top_memory": top_memory,
        "table": averages.table(sort_by="self_cpu_memory_usage", row_limit=10),
    }
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the subprocess
    raise SystemExit(main())
