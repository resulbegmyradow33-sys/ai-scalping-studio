# Research smoke — discovery → evidence → reproduce (2026-11-24)

Bounded smoke run of the discovery pipeline on the **real local XAUUSD H1 store**.
The point is to show what the pipeline actually does with real bars, not to
validate a strategy. No promotion was attempted, no champion changed.

Machine-readable summary: [`docs/evidence/discovery-smoke-2026-11-24.json`](evidence/discovery-smoke-2026-11-24.json).
Raw artifacts (archive, bundles, scratch driver) stay in `runs/` and are gitignored.

## Data

| Field | Value |
| --- | --- |
| Store | `data-ticks5-store` (`.worktrees/v1/`) |
| Store manifest checksum | `0930bedd3c64554621c599ad6ee20214` |
| Symbol / timeframe | XAUUSD / H1 |
| Rows | 29,586 (2021-05-24 → 2026-05-22) |
| Source | `ticks:data-ticks5`, `synthetic: false` |
| Dataset id | `XAUUSD:H1:h1:439ad79658c7` |
| Dataset checksum | `439ad79658c774371ceb32c7e605d184` |
| Split rows (train / val / test) | 20,667 / 4,428 / 4,428 |
| Validation slice | 2024-11-20T10:00Z → 2025-08-21T16:00Z |

The dataset checksum was rebuilt independently from the same bars and matched the
checksum recorded in the archive (`439ad79…`). The test split was never sliced
or scored.

## Commands

```bash
# discovery (fresh archive, seeded)
PYTHONPATH=$PWD/backend/src <venv>/python -m scalping.cli discovery run \
  --root /Users/resul2004/Documents/ai-scalping-studio/.worktrees/v1/data-ticks5-store \
  --symbol XAUUSD --timeframe H1 \
  --max-trials 25 --population 8 --generations 4 --seed 7 \
  --archive runs/discovery/xauusd-h1-seed7.sqlite \
  --out runs/discovery/xauusd-h1-seed7.run.json --no-cache

# evidence bundle (scratch driver, no repo code changed)
POLARS_MAX_THREADS=1 PYTHONPATH=$PWD/backend/src <venv>/python runs/discovery/smoke_evidence.py

# reproduce (pinned thread pool)
POLARS_MAX_THREADS=1 PYTHONPATH=$PWD/backend/src <venv>/python -m scalping.cli discovery reproduce \
  --bundle runs/discovery/evidence/most-active.evidence.json \
  --store-root runs/discovery/store \
  --out runs/discovery/evidence/most-active.reproduce.threads1.json
```

Reproduce needs `<store-root>/datasets/<checksum>/artifact.json`. A scratch store
root under `runs/discovery/store` holds the dataset artifact and a symlink to the
store's `XAUUSD/H1` partitions; the shared store was not written to.

## Discovery result

25 attempts: **22 evaluated, 3 skipped before the evaluator, 0 deduped, 22
accepted, 0 gate rejections**, stop reason `budget`. The gate defaults are all
zero/empty, so "accepted" carries no quality meaning here.

| Baseline | Trades | Net profit | PF | Sharpe | Max DD | Expectancy R | Costs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hold | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| ret_5 | 500 | 226.42 | 1.014 | 2.476 | 0.101 | 0.0113 | 2,270.04 |

| Best candidate (validation) | Value |
| --- | --- |
| Candidate id | `ebb1d79d3a06…` (complexity 27) |
| Trades | **0** |
| Net profit / PF / expectancy | 0.00 / 0.00 / 0.00 |
| Comparison vs best baseline | 0.00 vs 226.42 → does not beat `ret_5` |
| **Decision** | **`no_trade`** — "flat by construction (0 closed trades)" |

The search's best-by-`net_profit` candidate is a degenerate no-trade tree
(1-bar holding limit, 0 closed trades on the slice). With an empty gate and
`net_profit` selection, flat strategies score exactly 0 and win whenever every
trading candidate loses, which is what happened.

## Evidence bundle

`build_evidence` was called with 3 folds, stress `1.5/2.0`,
`min_trades_per_fold=5`, seed 7, `trial_count=22`, `search_budget=25`, window 60.
There is no persist module for `CandidateEvidence` in this tree, so the bundle is
JSON plus a spec sidecar under `runs/discovery/evidence/`.

**Run winner — refused.** `fold_trades_below_minimum`: fold 1 closed 0 trades,
minimum 5. That is the designed fail-closed path for a flat candidate.

**Most-active bundled candidate** (highest-trade accepted candidate, chosen only
to exercise the plumbing; it is **not** the run winner and not a promotion
candidate):

| Fold | Test rows | Trades | Net profit | Expectancy R | Stress x1.5 (trades / net) | Stress x2.0 (trades / net) |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 6,274 | 119 | −2,020.88 | −0.3928 | 117 / −2,028.91 | 86 / −2,006.88 |
| 2 | 6,274 | 191 | −2,007.17 | −0.2373 | 114 / −2,007.67 | 94 / −2,009.94 |
| 3 | 6,274 | 1,217 | −2,035.20 | −0.0406 | 569 / −2,033.09 | 327 / −2,018.60 |

| Aggregate (3 folds) | Value |
| --- | --- |
| Trades / total R | 1,527 / −141.41 |
| Expectancy R (95% iid CI) | −0.0926 (−0.1572, −0.0247) |
| Profit factor | 0.861 |
| Worst-fold max DD | 0.263 |
| Positive folds | 0 / 3 |

On its validation trial the same candidate did 724 trades for −2,006.07 net. It
is a known loser; the bundle shows costs and the simulator behaving, not an edge.

## Reproduce

**Pinned threads (`POLARS_MAX_THREADS=1`): reproduced, exit 0.**
Bundle and rebuilt evidence both hash to `09f97bf5…`.

**Default thread pool: mismatch, exit 1.** Two runs of the identical command
returned two different checksums (`4326974d…` with folds+aggregate differing,
then `d37f1043…` with aggregate differing). Differences are last-bit float drift
in regime sums, e.g. `folds[2].regimes[0].net_profit -300.7723024042524` vs
`-300.77230240425234` (~1e-16 relative). The reproduce path reported the
mismatch instead of silently passing, which is correct behaviour, but the
checksum is not stable across unpinned thread counts.

Search-level reproducibility is fine: a second seeded run on an empty archive
returned the same candidate id, the same metrics, the same baseline table and the
same `no_trade` decision.

## Resources

| Step | Wall time | Peak RSS |
| --- | ---: | ---: |
| discovery run | 14.35 s | 229 MiB |
| discovery repeat | 15.35 s | 266 MiB |
| evidence build (default threads) | 4.91 s | 215 MiB |
| reproduce (default threads) | 4.43 s | 254 MiB |
| reproduce (pinned threads) | 3.62 s | 257 MiB |

Worst step is ~0.27 GB against the 6 GB budget; the whole smoke is ~43 s against
the 10 minute budget. No limit was approached.

## Limitations

1. **22 evaluations on 4,428 bars is a wiring smoke, not a search.** The
   multiple-testing denominator is 22; a winner after 22 trials would not be
   evidence of an edge, and here there was no winner.
2. **The gate is empty by default** (all thresholds zero, no required stress), so
   all 22 evaluated candidates were accepted. Acceptance is not quality.
3. **`net_profit` with an empty gate ranks a flat policy first** when every
   trading candidate loses. The recorded "best" is exactly that degenerate tree;
   the decision rule catches it and records `no_trade`.
4. **Validation slice only.** No test access, no walk-forward, no locked OOS, no
   regime-stability proof. Discovery cannot promote, and none was attempted.
5. **The bundled candidate is a known validation loser**, included to exercise
   folds/stress/reproduce. Its negative expectancy CI is consistent with costs
   dominating, not with a hidden edge.
6. **Thread-pool nondeterminism** in regime sums (see Reproduce) means an
   evidence checksum is only reproducible with the thread count pinned. That is
   an environment gap, not a candidate property.
7. **One store, one symbol, one timeframe.** Nothing generalises beyond
   `data-ticks5-store` XAUUSD H1.
8. **iid bootstrap CI** ignores trade clustering, so the interval can look
   tighter than the trade sequence supports.
9. No evidence persist module exists, so bundles are local `runs/` JSON only.

## Conclusion

The pipeline behaves as designed on the real store: seeded search, 22 recorded
evaluations, a `no_trade` decision for a flat best candidate, a fail-closed
refusal when evidence is asked of that flat candidate, a built 3-fold bundle with
1.5/2.0 cost stress for a trading candidate, and a reproduce pass with threads
pinned. Nothing here is validation, nothing is profitable, and no promotion was
attempted. A validation-slice winner after N trials would still be a candidate
and a multiple-testing warning; in this run the winner was flat.
