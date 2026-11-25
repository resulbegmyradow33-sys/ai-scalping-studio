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

The fingerprint costs one pass over the raw columns: **0.34 ms** for 4,321 bars,
**2.8 ms** for 36,001 bars and **23–32 ms** for a 200,000-row M1 slice (three
real-store runs on 2026-11-24, see below). Key computation is a pure function of
its arguments plus the registry digest, so two processes agree; a test asserts
that in a fresh interpreter.

### Key cost on real frames

The key is not the bottleneck. On the measured slices `frame_meta` is
**0.8–1.2 %** of build + key time in the same run, and **1.6–3.2 %** against the
idle merged-harness build numbers; `key_for` itself is 15–44 µs. The full M1
store (1,773,331 rows) was not rebuilt here, but the fingerprint is O(rows) over
the same columns as the build: the 200k measurement (23–32 ms) puts it at tens
of milliseconds, against the recorded 1.93 s build.

A store-provided checksum was considered and rejected as a replacement:

* `data/provenance.py` writes a `_meta/<symbol>/<timeframe>.json` checksum, but
  it covers `ts, open, high, low, close, spread, volume` only. `frame_meta` also
  hashes `bid_close`/`ask_close`; those are conservative (the matrix only reads
  quotes when the spread is missing, and then `prepare_bars` overwrites them),
  but dropping them changes the identity contract, not just its cost.
* It identifies the **whole dataset at ingest**. `cached_build_features`
  receives any `pl.DataFrame`, including `frame.head(rows)` slices, and a slice
  has a different identity from the dataset it came from.
* It is a sidecar, not a property of the bytes handed to the builder. A parquet
  replaced or edited without re-running provenance would keep a stale checksum,
  which is exactly the silent hit the fingerprint exists to prevent.

So the key stays `frame_meta`. The measured cost does not justify trading
correctness for a faster identity.

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

A version bump changes the key (the version is in the payload) and the loader
also rejects an old manifest by its `version` field; tests cover both halves
(`test_a_version_bump_rejects_an_existing_entry`). The end-to-end invalidation
tests are `test_a_settings_change_rebuilds_and_is_not_a_hit` and
`test_a_data_change_rebuilds_and_is_not_a_hit`: each asserts a rebuild and two
live entries instead of a silent hit.

`FeatureCache.invalidate(key)` removes one live entry, `clear()` removes every
cache file under the root (including quarantined and temp files) and returns the
count, and `stats()` reports hits, misses, saves, corrupt entries, unexpected
read errors, accumulated load/save seconds and the live entry count. There is no
LRU eviction: keys are deterministic, so a stale entry only costs disk. Clear the
directory when disk use matters.

## Small synthetic frames

Synthetic XAUUSD M1, Apple M2, single run medians of five, empty cache:

| Rows | `build_features` | `frame_meta` | `save` | `load` | Entry size |
|---:|---:|---:|---:|---:|---:|
| 4,321 (3 days) | 28.1 ms | 0.34 ms | 2.3 ms | 1.9 ms | 1.3 MB |
| 36,001 (35 days) | 148.3 ms | 2.8 ms | 14.3 ms | 12.6 ms | 10.5 MB |

The hit path is roughly 10–15x faster than the build on these small frames.

## Real-store measurement (2026-11-24)

`backend/scripts/bench_feature_cache.py` measures bounded slices of a real
parquet store with the same `measure_stage` aggregator as
`scripts/bench_pipeline.py`; the full report is
[evidence/cache-2026-11-24.json](evidence/cache-2026-11-24.json). Reproduce with:

```
cd backend && uv run python scripts/bench_feature_cache.py \
  --case m1_20k=../../../data-ticks5-store:M1:20000 \
  --case m1_200k=../../../data-ticks5-store:M1:200000 \
  --case h1_full=../../../data-real-store:H1:0 \
  --repeats 5 --warmup 1 --out ../docs/evidence/cache-2026-11-24.json
```

Host: Mac14,3 (Apple M2, 8 GB), macOS 27.2.1. Load average was **36.6** at
start with five other worktree jobs running, so **wall times are upper
bounds**; CPU medians and same-run ratios are the stable part. Each stage is the
median of five repeats after one discarded warm-up. The same run was repeated
three times as load rose from 22 to 44; wall seconds moved by about 2x between
runs while the load-to-build ratio stayed in the same band.

| Dataset | Rows | Build miss (wall / CPU) | Cache load (wall / CPU) | Load ÷ build | Save (wall) | Entry |
|---|---:|---:|---:|---:|---:|---:|
| M1 `data-ticks5-store` | 20,000 | 345 ms / 190 ms | 14.4 ms / 10.6 ms | 4.2 % wall, 5.6 % CPU | 11.6 ms | 5.84 MB |
| M1 `data-ticks5-store` | 200,000 | 2,942 ms / 1,700 ms | 232 ms / 106 ms | 7.9 % wall, 6.2 % CPU | 211.3 ms | 58.4 MB |
| H1 `data-real-store` | 100,083 (all) | 1,601 ms / 1,056 ms | 55.7 ms / 45.7 ms | 3.5 % wall, 4.3 % CPU | 92.1 ms | 29.2 MB |

So a hit costs **4–8 % of a rebuild**, i.e. 13–29x faster on wall and 16–23x
faster on CPU. The entry is essentially the raw arrays plus a 1.8 KB manifest:
`npz` is 5,841,214 / 58,401,214 / 29,225,450 bytes and `X` alone is 4.72 /
47.2 / 23.6 MB. `load` includes the full checksum re-validation, not just the
disk read; `save` includes the same hash before writing.

`cached_build_features` through a warm cache, 10 calls: first call cold,
then **9/10 hits** (0.9), warm median **14.2 / 223.7 / 123.2 ms**. On an entry
that already exists all calls hit; the 0.9 is the deliberate cold-start sample.

Corruption and invalidation were checked on every slice, all true: a truncated
`.npz` returns `None`, increments `corrupt` by one, produces one miss and
quarantines both files (`2` `.bad`, `0` live entries); a changed `spread_floor`
and a single edited `close` each produce a different key and a rebuild (three
live entries); a bumped `CACHE_VERSION` changes the key and the old entry is
rejected by the manifest `version` check.

### Full M1 estimate

The full M1 matrix is ~518 MB (`1.77M × 59 × 4` bytes for `X` plus the four
aligned arrays). The measured 200k entry is 58.4 MB and loads in 106 ms of CPU
time: at the same hashing/read throughput (~550 MB/s) the full entry extrapolates
to **~0.95 s CPU**, and to **~2.0 s wall** at this run's contended 252 MB/s.
Against the recorded 1.93 s build
([language-decision.md](language-decision.md)) the margin at full size is
therefore much narrower than at 200k, and could disappear on a slower disk. This
is an extrapolation, not a measurement. Compression or a memory-mapped/streaming
format would be the next lever if full-store cache latency matters.

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
  518 MB on disk (418 MB of `X` plus the aligned arrays), and a quarantined
  `.bad` copy doubles that until it is cleared.
* The 2026-11-24 real-store numbers were taken on a contended host (load
  average 22–44 with up to five concurrent jobs). Treat them as same-run
  ratios; the absolute wall seconds would be lower on an idle machine.
* Every hit pays `frame_meta` before the lookup and a full BLAKE2b checksum over
  all five arrays after the read. Both are deliberate — the first prevents a
  stale hit, the second prevents returning a corrupt file — but they make the
  hit cost O(bytes), so the cache's advantage shrinks as the matrix grows.
* No automatic eviction; `clear()` is manual.
* Two concurrent *first* builds of the same key both pay the build cost and then
  overwrite; the cache does not deduplicate in-flight work. The last writer wins
  with identical bytes.
* The cache lives per root. Two checkouts with separate `data/` or `runs/`
  directories do not share entries.
