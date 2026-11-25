"""Optional M30/H4 (``htf``) feature group: selection, causality and caching.

The 59-column matrix must not change when the group is off. When it is on, the
new columns sit after the defaults and obey the same rule as every other
higher-timeframe feature: a bucket stamped ``s`` is visible only at ``s + width``
(or later), never while it is still forming. These tests pin that rule at exact
boundaries and prove it by mutating future bars.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scalping.config import Settings
from scalping.data.resample import resample
from scalping.data.synthetic import SyntheticProvider
from scalping.datasets.builder import build_dataset
from scalping.features import cache as cache_module
from scalping.features.build import FEATURE_COLUMNS, build_features
from scalping.features.cache import FeatureCache, cached_build_features, frame_meta
from scalping.features.context import MTF_CONTEXT_COLUMNS, htf_context
from scalping.features.indicators import atr
from scalping.features.registry import (
    ALL_FEATURE_COLUMNS,
    DEFAULT_GROUPS,
    FEATURE_GROUPS,
    OPTIONAL_FEATURE_GROUPS,
    build_columns,
    columns_for,
    indices_for,
    optional_groups,
)

M30 = timedelta(minutes=30)
H4 = timedelta(hours=4)
SYMBOL = "XAUUSD"


@pytest.fixture(scope="module")
def m1() -> pl.DataFrame:
    return SyntheticProvider(seed=3).fetch(
        SYMBOL, datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 7, 12, tzinfo=UTC)
    )


# --- registry and default matrix -------------------------------------------


def test_default_matrix_and_columns_are_unchanged() -> None:
    assert len(FEATURE_COLUMNS) == 59
    assert DEFAULT_GROUPS == tuple(FEATURE_GROUPS)
    assert columns_for(None) == FEATURE_COLUMNS
    assert build_columns(None) == FEATURE_COLUMNS
    assert optional_groups(None) == ()
    # Optional names never leak into the default column tuple.
    assert not set(MTF_CONTEXT_COLUMNS) & set(FEATURE_COLUMNS)


def test_htf_group_is_optional_and_canonically_ordered() -> None:
    assert OPTIONAL_FEATURE_GROUPS == {"htf": MTF_CONTEXT_COLUMNS}
    assert columns_for(("htf",)) == MTF_CONTEXT_COLUMNS
    assert build_columns(("htf",)) == FEATURE_COLUMNS + MTF_CONTEXT_COLUMNS
    assert ALL_FEATURE_COLUMNS == FEATURE_COLUMNS + MTF_CONTEXT_COLUMNS
    assert indices_for(("htf",)) == list(range(59, 64))
    # Naming a base group alongside htf does not reorder or duplicate columns.
    assert columns_for(("price", "htf")) == FEATURE_GROUPS["price"] + MTF_CONTEXT_COLUMNS
    assert optional_groups(("price",)) == ()
    assert optional_groups(("price", "htf")) == ("htf",)


def test_unknown_group_still_raises() -> None:
    with pytest.raises(ValueError, match="unknown feature groups"):
        columns_for(("price", "astrology"))


# --- build_features selection ----------------------------------------------


def test_build_without_the_group_is_the_historical_matrix(m1: pl.DataFrame) -> None:
    default = build_features(m1)
    base_only = build_features(m1, groups=DEFAULT_GROUPS)
    assert default.columns == list(FEATURE_COLUMNS)
    assert base_only.columns == list(FEATURE_COLUMNS)
    assert base_only.X.shape == default.X.shape
    assert default.X.tobytes() == base_only.X.tobytes()


def test_htf_group_appends_columns_without_touching_the_base_matrix(m1: pl.DataFrame) -> None:
    default = build_features(m1)
    extended = build_features(m1, groups=("htf",))

    assert extended.columns == list(FEATURE_COLUMNS) + list(MTF_CONTEXT_COLUMNS)
    assert extended.X.shape == (default.X.shape[0], 64)
    # Every base column is bit-identical; the new context is appended only.
    assert extended.X[:, :59].tobytes() == default.X.tobytes()
    assert np.isfinite(extended.X[extended.valid_from :]).all()


def test_empty_frame_keeps_the_selected_columns(m1: pl.DataFrame) -> None:
    empty = build_features(m1.head(0), groups=("htf",))
    assert empty.X.shape == (0, 64)
    assert empty.columns == list(FEATURE_COLUMNS) + list(MTF_CONTEXT_COLUMNS)


# --- closed-bucket rule at exact boundaries --------------------------------


def test_m30_and_h4_values_switch_only_when_a_bucket_closes(m1: pl.DataFrame) -> None:
    m5 = resample(m1, "5m")
    m15 = resample(m1, "15m")
    m30 = resample(m1, "30m")
    h4 = resample(m1, "4h")
    joined = htf_context(m1, m5, m15, m30=m30, h4=h4)

    m30_ret = (m30["close"] / m30["close"].shift(1) - 1.0).to_numpy()
    m30_labels = m30["ts"].to_list()
    h4_close = h4["close"].to_numpy()
    h4_atr = atr(h4, 14).to_numpy()
    h4_labels = h4["ts"].to_list()

    def value(moment: datetime, column: str) -> float:
        row = joined.filter(pl.col("ts") == moment).row(0, named=True)
        return float(row[column])

    def last_closed(labels: list[datetime], width: timedelta, moment: datetime) -> int:
        closed = [index for index, label in enumerate(labels) if label + width <= moment]
        assert closed, "the fixture has at least one closed bucket"
        return closed[-1]

    def expected_m30_ret(moment: datetime) -> float:
        return float(m30_ret[last_closed(m30_labels, M30, moment)])

    def expected_h4_atr_frac(moment: datetime) -> float:
        index = last_closed(h4_labels, H4, moment)
        return float(h4_atr[index] / h4_close[index])

    # 10:29 is still inside the 10:00-10:29 M30 bucket.
    ten_29 = datetime(2026, 1, 5, 10, 29, tzinfo=UTC)
    assert value(ten_29, "m30_ret") == pytest.approx(expected_m30_ret(ten_29))
    assert value(ten_29, "m30_ret") == pytest.approx(
        m30_ret[m30_labels.index(datetime(2026, 1, 5, 9, 30, tzinfo=UTC))]
    )
    # At 10:30 the 10:00 bucket has closed and becomes the reference.
    ten_30 = datetime(2026, 1, 5, 10, 30, tzinfo=UTC)
    assert value(ten_30, "m30_ret") == pytest.approx(expected_m30_ret(ten_30))
    assert value(ten_30, "m30_ret") == pytest.approx(
        m30_ret[m30_labels.index(datetime(2026, 1, 5, 10, 0, tzinfo=UTC))]
    )
    assert value(ten_30, "m30_ret") != value(ten_29, "m30_ret")

    # The H4 08:00 bucket closes at 12:00, not before.
    eleven_59 = datetime(2026, 1, 5, 11, 59, tzinfo=UTC)
    twelve_00 = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
    assert value(eleven_59, "h4_atr_frac") == pytest.approx(expected_h4_atr_frac(eleven_59))
    assert value(eleven_59, "h4_atr_frac") == pytest.approx(
        float(
            h4_atr[h4_labels.index(datetime(2026, 1, 5, 4, 0, tzinfo=UTC))]
            / h4_close[h4_labels.index(datetime(2026, 1, 5, 4, 0, tzinfo=UTC))]
        )
    )
    assert value(twelve_00, "h4_atr_frac") == pytest.approx(expected_h4_atr_frac(twelve_00))
    assert value(twelve_00, "h4_atr_frac") == pytest.approx(
        float(
            h4_atr[h4_labels.index(datetime(2026, 1, 5, 8, 0, tzinfo=UTC))]
            / h4_close[h4_labels.index(datetime(2026, 1, 5, 8, 0, tzinfo=UTC))]
        )
    )


# --- no look-ahead ---------------------------------------------------------


def test_mutating_future_bars_cannot_change_earlier_htf_values(m1: pl.DataFrame) -> None:
    cut = 900  # 15:00 UTC on the first day
    full = build_features(m1, groups=("htf",))

    def distort(column: str) -> pl.Expr:
        return (
            pl.when(pl.int_range(pl.len()) > cut)
            .then(pl.col(column) * 1.5)
            .otherwise(pl.col(column))
        )

    mutated_frame = m1.with_columns(
        [distort(name).alias(name) for name in ("open", "high", "low", "close")]
        + [
            pl.when(pl.int_range(pl.len()) > cut)
            .then(pl.col("volume") * 3)
            .otherwise(pl.col("volume"))
            .alias("volume")
        ]
    )
    mutated = build_features(mutated_frame, groups=("htf",))

    # Rows up to and including the cut cannot see any of the distorted future.
    np.testing.assert_array_equal(
        full.X[: cut + 1],
        mutated.X[: cut + 1],
        err_msg="a future bar changed an earlier feature row",
    )
    # And the mutation is not vacuous: later HTF values do change.
    assert not np.array_equal(full.X[cut + 240 :], mutated.X[cut + 240 :])


def test_truncating_the_source_cannot_change_a_computed_htf_row(m1: pl.DataFrame) -> None:
    full = build_features(m1, groups=("htf",))
    for t in (89, 120, 137, 300, 599, 900):
        part = build_features(m1.head(t + 1), groups=("htf",))
        np.testing.assert_allclose(
            full.X[t, 59:],
            part.X[-1, 59:],
            rtol=0,
            atol=1e-6,
            err_msg=f"M30/H4 look-ahead at row {t}",
        )


# --- dataset selection -----------------------------------------------------


def test_dataset_can_select_the_htf_group(m1: pl.DataFrame) -> None:
    tiny = m1.head(600)
    extended = build_features(tiny, groups=("htf",))
    bundle = build_dataset(
        extended, symbol=SYMBOL, groups=("price", "htf"), horizon=1, purge_bars=1
    )

    assert bundle.manifest.columns == columns_for(("price", "htf"))
    assert bundle.manifest.groups == ("price", "htf")
    assert bundle.X.shape[1] == len(bundle.manifest.columns)
    assert bundle.X.shape[1] > len(FEATURE_GROUPS["price"])


def test_dataset_rejects_htf_when_the_matrix_was_not_built_with_it(m1: pl.DataFrame) -> None:
    default = build_features(m1.head(600))
    with pytest.raises(ValueError, match="does not provide"):
        build_dataset(default, groups=("htf",), horizon=1, purge_bars=1)


# --- cache key and round trip ---------------------------------------------


class GroupCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...] | None] = []

    def __call__(
        self,
        frame: pl.DataFrame,
        costs=None,
        *,
        groups: tuple[str, ...] | None = None,
    ):
        self.calls.append(groups)
        real = build_features
        if groups is None:
            return real(frame, costs)
        return real(frame, costs, groups=groups)


def test_cache_key_and_entry_carry_the_optional_group(
    tmp_path: Path, m1: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FeatureCache(tmp_path / "cache")
    frame = m1.head(600)
    costs = Settings().costs
    meta = frame_meta(frame)
    base_key = cache.key_for(SYMBOL, "M1", meta, costs)
    htf_key = cache.key_for(SYMBOL, "M1", meta, costs, groups=("htf",))
    # Base-only selections share the default key; htf must not.
    assert cache.key_for(SYMBOL, "M1", meta, costs, groups=("price",)) == base_key
    assert htf_key != base_key

    counter = GroupCounter()
    monkeypatch.setattr(cache_module, "build_features", counter)

    base = cached_build_features(
        cache, symbol=SYMBOL, timeframe="M1", frame=frame, settings=Settings()
    )
    extended = cached_build_features(
        cache,
        symbol=SYMBOL,
        timeframe="M1",
        frame=frame,
        settings=Settings(),
        groups=("htf",),
    )
    assert counter.calls == [None, ("htf",)]
    assert base.columns == list(FEATURE_COLUMNS)
    assert extended.columns == list(FEATURE_COLUMNS) + list(MTF_CONTEXT_COLUMNS)

    # Both entries are cached under different keys and reload bit-identically.
    again_base = cached_build_features(
        cache, symbol=SYMBOL, timeframe="M1", frame=frame, settings=Settings()
    )
    again_htf = cached_build_features(
        cache,
        symbol=SYMBOL,
        timeframe="M1",
        frame=frame,
        settings=Settings(),
        groups=("htf",),
    )
    assert len(counter.calls) == 2, "a second request must hit the cache"
    assert again_base.X.tobytes() == base.X.tobytes()
    assert again_htf.X.tobytes() == extended.X.tobytes()
    assert cache.stats()["entries"] == 2

    stored = cache.load(htf_key)
    assert stored is not None
    assert stored.columns == list(FEATURE_COLUMNS) + list(MTF_CONTEXT_COLUMNS)
    manifest = json.loads(cache._manifest_path(htf_key).read_text(encoding="utf-8"))
    assert manifest["groups"] == ["htf"]


def test_cache_rejects_a_manifest_whose_groups_disagree_with_columns(
    tmp_path: Path, m1: pl.DataFrame
) -> None:
    cache = FeatureCache(tmp_path / "cache")
    frame = m1.head(600)
    key = cache.key_for(SYMBOL, "M1", frame_meta(frame), Settings().costs, groups=("htf",))
    features = build_features(frame, groups=("htf",))
    cache.save(key, features, groups=("htf",))

    manifest_path = cache._manifest_path(key)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["groups"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert cache.load(key) is None
    assert cache.stats()["corrupt"] == 1
    assert not cache._manifest_path(key).exists()
