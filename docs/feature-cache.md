# Feature cache

Disk cache for the 59-column feature matrix. `build_features` is the expensive
step shared by training, walk-forward evaluation and candidate search; this
cache materialises one `FeatureFrame` per input combination and returns the same
arrays afterwards. The cache is an optimisation, never a correctness proof: the
bit-identity tests are.

Status: Phase 14 enabler (issue #69, item 1 of
[language-decision.md](language-decision.md)).

## Where it lives

```
<root>/.feature-cache/<key[:2]>/<key>.npz
<root>/.feature-cache/<key[:2]>/<key>.json
```

* API: `<runs_root>/.feature-cache`, created by `MLRunner` on first write.
* CLI: `<--root>/.feature-cache` for `dataset build`, `ml train`, `ml cv`,
  `ml evaluate` and the `backtest`/`montecarlo`/`sensitivity` simulation paths.

Both roots are already gitignored. One entry is an uncompressed `.npz` with the
five arrays (`X`, `ts`, `ohlc`, `spread`, `atr`) and a JSON manifest with the
shapes, dtypes, columns, `valid_from`, row count and a BLAKE2b checksum. Arrays
are loaded with `allow_pickle=False`, so an object array cannot be read back.

## Key

`FeatureCache.key_for(symbol, timeframe, frame_meta, settings)` is a BLAKE2b-256
digest of a canonical JSON payload (`sort_keys`, no whitespace, no NaN):

| Field | Content | Why |
|---|---|---|
| `version` | `CACHE_VERSION` | Bumped when a feature **formula** changes without a column change |
| `symbol`, `timeframe` | the request's identifiers | Different instruments never share a matrix |
| `registry` | sha256 of `WARMUP` + the canonical `FEATURE_COLUMNS` tuple | A column added, removed or reordered misses instead of mixing matrices |
| `data.rows`, `data.first_ts`, `data.last_ts` | row count and bounds | Cheap sanity identity, visible in the manifest |
| `data.fingerprint` | BLAKE2b-160 of `ts`, OHLC, quotes, `spread`, `volume` (name, dtype, length and bytes per column, rows sorted by `ts`) | Detects a value change that leaves row count and bounds untouched |
| `costs` | the full `CostConfig` block | `spread_floor`/`spread_vol_mult` feed `prepare_bars`; the whole block is keyed conservatively |

`provider` is deliberately excluded: the source label does not change a single
feature value. Keying the full cost block (including slippage and commission,
which the matrix never reads) trades a few unnecessary misses for the guarantee
that a changed cost config is never a silent hit.

The fingerprint costs one pass over the raw columns: **0.34 ms** for 4,321 bars
and **2.8 ms** for 36,001 bars, against the build itself. Key computation is a
pure function of its arguments plus the registry digest, so two processes agree;
a test asserts that in a fresh interpreter.

## Hit, miss and corruption

* **Hit**: the manifest, key, column list, array names, shapes, dtypes,
  `valid_from`, finiteness (after warm-up) and checksum all validate, and the
  arrays are returned exactly as saved.
* **Miss**: no files, or any validation failure. A miss rebuilds with
  `build_features` and swaps the entry in.
* **Corrupt/partial**: logged with the reason, the pair is renamed
  `<name>.<stamp>.bad` (quarantined) and the load is a miss. A later build
  writes a clean entry under the same key; the `.bad` file is never read.
  Quarantine failures (read-only directory) are logged and the miss stands.
* **Write failure** (full disk, permissions) is logged and ignored by
  `cached_build_features`: the freshly built frame is returned, so a cache
  problem never fails a training run.

Writes go to a sibling `tempfile.mkstemp` file and land with `os.replace`, for
the `.npz` and the manifest separately. A reader therefore never opens a
truncated file. The manifest is written after the `.npz`; if a crash lands
between the two, the pair fails the key/checksum checks and is quarantined.

Two writers of the *same* key produce the same arrays by construction, so a
reader that pairs a newer `.npz` with an older manifest still sees a matching
checksum. Different keys use different files and cannot collide.

## Invalidation

| Change | Key changes? |
|---|---|
| New/removed rows, any price, quote, spread or volume value | Yes (fingerprint, rows, bounds) |
| Rows reordered, `provider` label changed | No (sorted, excluded) |
| Any `CostConfig` field | Yes (conservative; only the two spread fields alter the frame) |
| `FEATURE_COLUMNS` or `WARMUP` | Yes (registry digest) |
| Feature formula changed in place | **No** — bump `CACHE_VERSION` in `features/cache.py` |
| New cache format / validation rule | Bump `CACHE_VERSION` |

`FeatureCache.invalidate(key)` removes one live entry, `clear()` removes every
cache file under the root (including quarantined and temp files) and returns the
count, and `stats()` reports hits, misses, saves, corrupt entries, unexpected
read errors, accumulated load/save seconds and the live entry count. There is no
LRU eviction: keys are deterministic, so a stale entry only costs disk. Clear the
directory when disk use matters.

## Measured timings

Synthetic XAUUSD M1, Apple M2, single run medians of five, empty cache:

| Rows | `build_features` | `frame_meta` | `save` | `load` | Entry size |
|---:|---:|---:|---:|---:|---:|
| 4,321 (3 days) | 28.1 ms | 0.34 ms | 2.3 ms | 1.9 ms | 1.3 MB |
| 36,001 (35 days) | 148.3 ms | 2.8 ms | 14.3 ms | 12.6 ms | 10.5 MB |

The hit path is roughly 10–15x faster than the build on these small frames; the
gap grows with rows because the build is superlinear in the `fvg` scan. For the
1.77M-row M1 store the recorded build is 1.93 s
([language-decision.md](language-decision.md)); the same save/load throughput
extrapolates to roughly 0.6–0.7 s and ~516 MB (`1.77M × 59 × 4` bytes for `X`
plus the other four arrays). That real-store save/load pair was **not measured
here** — this worktree has no production store.

## Honest limitations

* A cache is not a correctness proof. The proof is the test suite: a hit is
  compared byte for byte (dtype, shape, NaN included) against `build_features`,
  and every downstream id (`dataset_id`, `data_hash`) hashes those bytes.
* A formula change that does not change the column list is invisible to the key
  unless `CACHE_VERSION` is bumped. Reviewers must treat formula edits as cache
  format edits.
* The full cost block is keyed, so tuning slippage or commission in a backtest
  also invalidates the feature entry. Correct, slightly wasteful.
* Uncompressed `.npz` trades disk for load speed: the full M1 matrix is about
  516 MB on disk (418 MB of `X` plus the aligned arrays), and a quarantined
  `.bad` copy doubles that until it is cleared.
* No automatic eviction; `clear()` is manual.
* Two concurrent *first* builds of the same key both pay the build cost and then
  overwrite; the cache does not deduplicate in-flight work. The last writer wins
  with identical bytes.
* The cache lives per root. Two checkouts with separate `data/` or `runs/`
  directories do not share entries.
