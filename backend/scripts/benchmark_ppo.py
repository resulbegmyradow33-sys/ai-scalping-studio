"""Benchmark a tiny PPO run: training throughput and inference latency.

The point is a repeatable local number, not a policy quality claim. A short
training run on synthetic bars measures steps/second; `predict` latency is
measured on observations from the same env. Results depend on this machine,
thread count and the chosen device.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch
from stable_baselines3 import PPO

from scalping.config import Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.env.trading_env import TradingEnv
from scalping.features.build import build_features


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a small PPO run.")
    parser.add_argument("--bars", type=int, default=3_000)
    parser.add_argument("--timesteps", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--predictions", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bars < 500 or args.timesteps < 1 or args.predictions < 1:
        raise SystemExit("bars >= 500, timesteps >= 1 and predictions >= 1 are required")
    if args.timesteps < args.n_steps:
        raise SystemExit("timesteps must be >= n_steps")

    start = datetime(2026, 1, 5, tzinfo=UTC)
    frame = SyntheticProvider(seed=args.seed).fetch(
        "XAUUSD", start, start + timedelta(minutes=args.bars)
    )
    env = TradingEnv(build_features(frame), Settings(), seed=args.seed)
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available")
        device = "mps"
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(1)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=args.n_steps,
        batch_size=min(args.batch_size, args.n_steps),
        seed=args.seed,
        device=device,
        verbose=0,
    )
    started = time.perf_counter()
    model.learn(total_timesteps=args.timesteps)
    train_seconds = time.perf_counter() - started

    observation, _ = env.reset(seed=args.seed)
    timings: list[float] = []
    for _ in range(args.predictions):
        moment = time.perf_counter()
        model.predict(observation, deterministic=True)
        timings.append(time.perf_counter() - moment)
    median = statistics.median(timings)

    payload = {
        "rows": [
            {
                "model": "ppo",
                "ok": True,
                "device": device,
                "timesteps": args.timesteps,
                "train_seconds": train_seconds,
                "steps_per_second": args.timesteps / train_seconds if train_seconds > 0 else 0.0,
                "predict_ms_median": median * 1000.0,
                "predictions": args.predictions,
                "bars": args.bars,
            }
        ]
    }
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the subprocess
    raise SystemExit(main())
