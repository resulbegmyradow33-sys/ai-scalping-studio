"""Risk-limit ablation: what changes when each enabled limit is removed.

The baseline is the policy under the caller's settings. Each variant disables
**one** currently enabled risk limit and reruns the same policy over the same
frame. The report is descriptive: it shows which limit bound the run, not which
setting is "better".

Cost stress already covers spread/slippage/commission multipliers; this module
is the risk-side counterpart and deliberately does not mutate ``Settings``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, Final

from scalping.backtest.engine import run_backtest
from scalping.config import RiskConfig, Settings
from scalping.features.build import FeatureFrame

__all__ = ["ABLATABLE_LIMITS", "risk_ablation"]

#: Risk fields that can be disabled. Values are their "off" settings.
ABLATABLE_LIMITS: Final[tuple[tuple[str, float | int | bool], ...]] = (
    ("daily_loss_pct", 1.0),
    ("weekly_loss_pct", 0.0),
    ("max_drawdown_pct", 0.0),
    ("consecutive_loss_pause", 0),
    ("max_spread_points", 0.0),
    ("min_equity_frac", 0.0),
    ("max_slippage_points", 0.0),
    ("data_quality_block", False),
)


def _metrics(result, baseline_profit: float) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "trades": int(metrics.trades),
        "net_profit": float(metrics.net_profit),
        "expectancy_r": float(metrics.expectancy_r),
        "profit_factor": float(metrics.profit_factor),
        "max_drawdown": float(metrics.max_drawdown),
        "win_rate": float(metrics.win_rate),
        "delta_net_profit": float(metrics.net_profit) - baseline_profit,
    }


def risk_ablation(
    policy: Callable[[Any], int],
    features: FeatureFrame,
    settings: Settings,
    *,
    policy_name: str = "ablation",
) -> list[dict[str, Any]]:
    """Run the baseline and one variant per enabled risk limit.

    A limit that is already off is not emitted as a no-op variant. Rows are
    ordered: ``baseline`` first, then :data:`ABLATABLE_LIMITS` order.
    """
    if not callable(policy):
        raise TypeError(f"policy must be callable, got {type(policy).__name__}")

    def run(current: Settings, name: str):
        return run_backtest(
            policy,
            features,
            current,
            policy_name=f"{policy_name}:{name}",
            max_steps=len(features.ts) + 1,
        )

    baseline = run(settings, "baseline")
    baseline_profit = float(baseline.metrics.net_profit)
    rows: list[dict[str, Any]] = [
        {
            "variant": "baseline",
            "field": None,
            "base_value": None,
            "settings": _settings_payload(settings.risk),
            **_metrics(baseline, baseline_profit),
        }
    ]
    for field, off_value in ABLATABLE_LIMITS:
        base_value = getattr(settings.risk, field)
        if base_value in (0, 0.0, False) or base_value == off_value:
            continue
        if field == "daily_loss_pct" and float(base_value) >= 1.0:
            continue
        variant = replace(settings.risk, **{field: off_value})  # type: ignore[arg-type]
        result = run(replace(settings, risk=variant), f"disable:{field}")
        rows.append(
            {
                "variant": f"disable:{field}",
                "field": field,
                "base_value": base_value,
                "settings": _settings_payload(variant),
                **_metrics(result, baseline_profit),
            }
        )
    return rows


def _settings_payload(risk: RiskConfig) -> dict[str, Any]:
    return {
        field: getattr(risk, field)
        for field, _ in ABLATABLE_LIMITS
    }
