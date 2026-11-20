"""Runtime settings: static defaults plus optional YAML overrides.

``Settings.load`` merges a partial YAML document over the dataclass defaults,
so a file only needs the keys that differ from the spec.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]  # dev stub types-PyYAML not installed

from scalping.types import ContractSpec, CostConfig, RiskConfig

__all__ = [
    "ContractSpec",
    "CostConfig",
    "RiskConfig",
    "Settings",
]

_BLOCKS = ("contract", "risk", "costs")


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``override`` on a copy of ``base``."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _merge(current, value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True, slots=True)
class Settings:
    contract: ContractSpec = field(default_factory=ContractSpec)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)

    @classmethod
    def load(cls, path: str | Path) -> Settings:
        """Read ``path`` and return settings with defaults filled in."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise TypeError("settings file must contain a YAML mapping")
        unknown = set(raw) - set(_BLOCKS)
        if unknown:
            raise ValueError(f"unknown settings blocks: {sorted(unknown)}")
        merged = _merge(asdict(cls()), raw)
        for block in _BLOCKS:
            if not isinstance(merged[block], Mapping):
                raise TypeError(f"settings block '{block}' must be a mapping")
        return cls(
            contract=ContractSpec(**merged["contract"]),
            risk=RiskConfig(**merged["risk"]),
            costs=CostConfig(**merged["costs"]),
        )
