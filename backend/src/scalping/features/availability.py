"""Feature availability rules.

A feature matrix is only meaningful if the source could actually supply the
inputs behind each column. Two failure modes matter:

* a volume feature trained on a file whose volume field is empty (all zeros);
* a signed-flow feature trained on bars that never carried tick volumes.

The rules are an allow-list contract, not a repair: selecting an unavailable
column raises with the missing capability named, so a dataset manifest can never
quietly describe features the data could not support.

Spread columns deserve a note. The feature preparer can model a spread when a
broker export has no quotes, and the project deliberately allows that. Such
columns are available when ``modeled_spread`` is true; requiring ``real_quotes``
would reject every CSV-only dataset for no reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from scalping.features.registry import FEATURE_GROUPS, columns_for

__all__ = [
    "COLUMN_REQUIREMENTS",
    "FEATURE_AVAILABILITY",
    "Capabilities",
    "FeatureUnavailableError",
    "capabilities_from_frame",
    "ensure_columns_available",
    "unavailable_columns",
]


class FeatureUnavailableError(ValueError):
    """A selected column needs a capability the source does not have."""


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What the source can actually provide.

    Defaults describe the canonical Parquet store: OHLC + volume, no real
    quotes (spread may be modelled), no tick-flow layer and an M1 source from
    which higher timeframes can be derived.
    """

    bar_volume: bool = True
    real_quotes: bool = False
    modeled_spread: bool = True
    tick_flow: bool = False
    m1_source: bool = True

    @property
    def spread(self) -> bool:
        return self.real_quotes or self.modeled_spread


#: Requirement names used by :data:`COLUMN_REQUIREMENTS`.
FEATURE_AVAILABILITY: Final[dict[str, str]] = {
    "bar_volume": "a non-zero bar volume field",
    "real_quotes": "real bid/ask quotes",
    "spread": "real quotes or an allowed spread model",
    "tick_flow": "signed tick volumes (bid/ask volume per tick)",
    "m1_source": "an M1 source that higher timeframes can be derived from",
}

#: Column → required capabilities. Unlisted columns have no requirement.
COLUMN_REQUIREMENTS: Final[dict[str, frozenset[str]]] = {
    "volume_z20": frozenset({"bar_volume"}),
    "volume_ratio_20": frozenset({"bar_volume"}),
    "spread_norm": frozenset({"spread"}),
    "spread_z20": frozenset({"spread"}),
    "m5_ret_1": frozenset({"m1_source"}),
    "m5_rsi_14": frozenset({"m1_source"}),
    "m5_atr_norm": frozenset({"m1_source"}),
    "m5_ema_gap_21": frozenset({"m1_source"}),
    "m5_pos_range_20": frozenset({"m1_source"}),
    "m15_ret_1": frozenset({"m1_source"}),
    "m15_rsi_14": frozenset({"m1_source"}),
    "m15_atr_norm": frozenset({"m1_source"}),
    "m15_ema_gap_21": frozenset({"m1_source"}),
    "m15_pos_range_20": frozenset({"m1_source"}),
    "htf_bos": frozenset({"m1_source"}),
    "htf_fvg_dist": frozenset({"m1_source"}),
}


def _has(capabilities: Capabilities, requirement: str) -> bool:
    if requirement == "spread":
        return capabilities.spread
    return bool(getattr(capabilities, requirement))


def unavailable_columns(
    columns: list[str] | tuple[str, ...],
    capabilities: Capabilities,
) -> dict[str, tuple[str, ...]]:
    """Map each unavailable column to the capabilities it is missing.

    A column that is not in :data:`COLUMN_REQUIREMENTS` is always available.
    ``flow_*`` names are reserved for the tick-flow layer and require
    ``tick_flow`` even before such columns are registered.
    """
    missing: dict[str, tuple[str, ...]] = {}
    for column in columns:
        required = COLUMN_REQUIREMENTS.get(column)
        if required is None and column.startswith("flow_"):
            required = frozenset({"tick_flow"})
        if not required:
            continue
        absent = tuple(sorted(name for name in required if not _has(capabilities, name)))
        if absent:
            missing[column] = absent
    return missing


def ensure_columns_available(
    columns: list[str] | tuple[str, ...],
    capabilities: Capabilities,
) -> None:
    """Raise :class:`FeatureUnavailableError` naming every missing input."""
    missing = unavailable_columns(columns, capabilities)
    if not missing:
        return
    details = "; ".join(
        f"{column} needs {', '.join(FEATURE_AVAILABILITY[name] for name in absent)}"
        for column, absent in sorted(missing.items())
    )
    raise FeatureUnavailableError(f"selected features are unavailable: {details}")


def capabilities_from_frame(df: pl.DataFrame) -> Capabilities:
    """Infer the honest capabilities of a raw provider frame.

    ``tick_flow`` is never inferred from bars: only an explicit tick pipeline
    can set it. Quotes count as real when every row carries a non-null
    ``ask_close``/``spread`` pair.
    """
    required = ("bid_close", "ask_close", "spread")
    has_quotes = all(name in df.columns for name in required) and df.select(
        [pl.col(name).null_count().alias(name) for name in required]
    ).sum_horizontal().item() == 0
    volume = (
        int(df.select((pl.col("volume") > 0).sum()).item())
        if "volume" in df.columns
        else 0
    )
    return Capabilities(
        bar_volume=volume > 0,
        real_quotes=bool(has_quotes),
        modeled_spread=True,
        tick_flow=False,
        m1_source=True,
    )


def group_columns(groups: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Columns for ``groups`` (``None`` = every registered group), in order."""
    if groups is None:
        groups = tuple(FEATURE_GROUPS)
    return columns_for(groups)
