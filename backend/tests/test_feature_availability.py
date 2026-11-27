"""Feature availability: honest capability checks before a dataset build."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from scalping.data.csv_provider import CsvProvider
from scalping.data.synthetic import SyntheticProvider
from scalping.datasets.builder import build_dataset
from scalping.features.availability import (
    Capabilities,
    FeatureUnavailableError,
    capabilities_from_frame,
    ensure_columns_available,
    group_columns,
    unavailable_columns,
)
from scalping.features.build import build_features

FIX = Path(__file__).parent / "fixtures" / "xauusd_gmt3.csv"


def synthetic_frame() -> pl.DataFrame:
    return SyntheticProvider(seed=7).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 6, tzinfo=UTC)
    )


def test_defaults_allow_modelled_spread_but_not_tick_flow() -> None:
    caps = Capabilities()

    assert caps.spread is True
    assert unavailable_columns(["spread_norm", "volume_z20"], caps) == {}
    assert unavailable_columns(["flow_signed"], caps) == {"flow_signed": ("tick_flow",)}


def test_missing_bar_volume_blocks_volume_columns() -> None:
    caps = Capabilities(bar_volume=False)

    assert unavailable_columns(["volume_z20", "ret_1"], caps) == {
        "volume_z20": ("bar_volume",)
    }
    with pytest.raises(FeatureUnavailableError, match="non-zero bar volume"):
        ensure_columns_available(["volume_z20", "ret_1"], caps)


def test_spread_needs_either_real_quotes_or_a_model() -> None:
    caps = Capabilities(real_quotes=False, modeled_spread=False)

    assert unavailable_columns(["spread_norm"], caps) == {"spread_norm": ("spread",)}
    assert unavailable_columns(["spread_norm"], Capabilities(real_quotes=True)) == {}


def test_every_missing_column_is_named() -> None:
    caps = Capabilities(bar_volume=False, modeled_spread=False, real_quotes=False)

    with pytest.raises(FeatureUnavailableError) as info:
        ensure_columns_available(["volume_ratio_20", "spread_z20"], caps)

    message = str(info.value)
    assert "volume_ratio_20" in message and "spread_z20" in message
    assert "bar volume" in message and "spread" in message


def test_tick_flow_is_never_inferred_from_bars() -> None:
    caps = capabilities_from_frame(synthetic_frame())

    assert caps.real_quotes is True
    assert caps.bar_volume is True
    assert caps.tick_flow is False


def test_capabilities_detect_a_missing_quote_file() -> None:
    frame = CsvProvider(FIX, tz_offset_hours=3).fetch("XAUUSD", None, None)
    caps = capabilities_from_frame(frame)

    assert caps.real_quotes is False
    assert caps.spread is True  # modelling is still allowed


def test_zero_volume_disables_the_volume_capability() -> None:
    frame = synthetic_frame().with_columns(pl.lit(0, dtype=pl.Int64).alias("volume"))

    assert capabilities_from_frame(frame).bar_volume is False


def test_build_dataset_enforces_the_contract_only_when_asked() -> None:
    features = build_features(synthetic_frame())

    build_dataset(features, groups=("price",), capabilities=Capabilities(bar_volume=False))
    build_dataset(features)  # no capabilities: historical behaviour

    with pytest.raises(FeatureUnavailableError, match="bar volume"):
        build_dataset(features, capabilities=Capabilities(bar_volume=False))


def test_group_columns_preserves_the_registry_order() -> None:
    from scalping.features.registry import columns_for

    columns = group_columns(("price", "volume"))

    assert columns == columns_for(("price", "volume"))
    assert columns[0] == "ret_1"
    assert "volume_ratio_20" in columns
    with pytest.raises(ValueError, match="unknown feature groups"):
        group_columns(("price", "magic"))


def test_empty_frame_has_no_capabilities() -> None:
    empty = synthetic_frame().head(0)

    caps = capabilities_from_frame(empty)

    assert caps.bar_volume is False
    assert caps.real_quotes is False
    assert caps.modeled_spread is False
    assert caps.tick_flow is False
    assert caps.m1_source is False
    assert caps.spread is False


def test_every_prefixed_feature_has_a_capability_requirement() -> None:
    from scalping.features.availability import COLUMN_REQUIREMENTS, group_columns

    rules = {
        "volume_": "bar_volume",
        "spread_": "spread",
        "m5_": "m1_source",
        "m15_": "m1_source",
        "m30_": "m1_source",
        "h4_": "m1_source",
    }
    for column in group_columns(None):
        for prefix, capability in rules.items():
            if column.startswith(prefix):
                required = COLUMN_REQUIREMENTS.get(column, frozenset())
                assert capability in required, f"{column} must require {capability}"
