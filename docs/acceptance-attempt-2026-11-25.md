# Real-data acceptance attempt — XAUUSD H1 — 2026-11-25

**Status: stopped at promotion/paper. NO TRADE. Not validated, not profitable, not paper-traded.**
**Locked holdout: not sealed, not spent. Production promotion: none.**

This is a bounded attempt to run the real chain on the local
`data-ticks5-store` XAUUSD H1 slice and to record exactly where it stops. The
machine-readable record is
[`docs/evidence/acceptance-attempt-2026-11-25.json`](evidence/acceptance-attempt-2026-11-25.json).

## Inputs

| Item | Value |
|---|---|
| Branch / base | `feat/realaccept` / `50fbca6` |
| Source store | `.worktrees/v1/data-ticks5-store` (read-only) |
| Symbol / timeframe | `XAUUSD` / `H1` |
| Rows / span | 29,586 / 2021-05-24 00:00 → 2026-05-22 20:00 UTC |
| Synthetic | no (`source: ticks:data-ticks5`) |
| Store manifest checksum | `0930bedd3c64554621c599ad6ee20214` |
| Dataset id / checksum | `XAUUSD:H1:h1:439ad79658c7` / `439ad79658c774371ceb32c7e605d184` |
| Code version | `05c600129db13136` |
| Settings digest | `2226685b488a17e799a58f8d7e1da7db74be94352663575c0447516c0951c5ed` |

The evidence writer persists arrays under `<store_root>/datasets/<checksum>`.
Because the source store is read-only, it was copied byte-for-byte to
`backend/runs/realaccept/store` (61 MB) and every persisting stage ran against
the copy. The dataset checksum is identical, so this is the same data identity.

## Chain, in order, with exact outcomes

All commands run from `backend/`. Exit codes are the real ones.

### 1. Validate — exit 0

```bash
uv run python -m scalping.cli data validate \
  --root ../../../data-ticks5-store --symbol XAUUSD --timeframe H1
```

`ok: true`, 29,586 rows. Two warnings:

- `time_gaps` 29,585 — H1 bars are 60 minutes apart; the 10-minute threshold is
  M1-oriented. Not an H1 defect.
- `price_outliers` 159 — retrospective robust-z warnings. Nothing was repaired
  or removed.

### 2. Quality — exit 0

`scalping.data.quality.detect_bar_outliers(OutlierConfig())` on the same slice:
159 price outliers, 0 volume outliers. Reconciliation was skipped because no
second source exists. No data was changed.

### 3. Dataset — exit 0

```bash
uv run python -m scalping.cli dataset build \
  --root ../../../data-ticks5-store --symbol XAUUSD --timeframe H1 \
  --horizon 1 --out runs/realaccept/dataset-manifest.json
```

29,525 rows, 59 columns; train 20,667 / validation 4,428 / test 4,428; 2 purged.
Dataset id `XAUUSD:H1:h1:439ad79658c7`, checksum
`439ad79658c774371ceb32c7e605d184`.

### 4. Discovery — exit 0, decision `no_trade`

```bash
POLARS_MAX_THREADS=1 uv run python -m scalping.cli discovery run \
  --root ../../../data-ticks5-store --symbol XAUUSD --timeframe H1 \
  --max-trials 64 --population 8 --generations 6 --seed 7 \
  --archive runs/realaccept/discovery.sqlite \
  --out runs/realaccept/03-discovery.json --no-cache \
  --min-trades 30 --min-expectancy-r 0.0 --min-profit-factor 1.05 \
  --require-stress 1.5 --min-best-trades 10
```

Real gate, not an empty one: at least 30 closed trades, profit factor ≥ 1.05,
and **strictly positive net profit at 1.5× costs**. `min-best-trades 10` prevents
a flat policy from winning by default.

| Counter | Value |
|---|---|
| Attempt cap | 64 |
| Trials evaluated | 22 |
| Skipped before evaluator | 2 |
| Accepted | 0 |
| Tradable | 0 |
| Stop reason | `patience` |
| Decision | `no_trade` |

Rejection events: `cost_stress` 22, `min_profit_factor` 22, `min_trades` 11.
Every evaluated candidate either lost money at 1.5× costs or traded too little.
The recorded decision candidate (`15bdae8a1e15…`) is flat: 0 validation trades.
There is no best candidate. Baselines: `hold` 0 trades / 0 net; `ret_5` 500
trades / +226.42 net / PF 1.014 / +0.0113R.

This is the real search result. 22 evaluations over a 4,428-bar slice is not a
search for an edge, and none was found.

### 5. Evidence — split outcome

**Decision candidate: refused (exit 2).**

```bash
POLARS_MAX_THREADS=1 uv run python -m scalping.cli discovery evidence \
  --root runs/realaccept/store --symbol XAUUSD --timeframe H1 \
  --spec runs/realaccept/specs/15bdae8a1e15….json \
  --out runs/realaccept/evidence/decision-candidate.evidence.json \
  --folds 3 --window 60 --horizon 1 --min-trades-per-fold 5 \
  --search-budget 64 --trials 22 --seed 7 --no-cache
```

Refusal: **`fold_trades_below_minimum`** — fold 1 closed 0 trades; the policy
requires 5. The flat decision candidate cannot produce promotion evidence. This
is the correct natural stop for the discovery winner.

**Least-bad gate-rejected trading candidate: persisted (exit 0).**

Candidate `70fe1963823c…` (gate-rejected: 34 validation trades, −51.09 net,
PF 0.916). It was chosen only to exercise persist → reproduce → promotion
plumbing. It is **not a winner and not a promotion candidate**.

Bundle checksum `7819f9d9a8b4c7d4adad821efc19157fdae41b8b5cb0215ac198947a7e0e7c02`,
schema v3, 3 folds, stress 1.5× / 2.0×, 1,000-sample iid bootstrap at 95%.
Dataset artifact persisted at
`runs/realaccept/store/datasets/439ad79658c774371ceb32c7e605d184`.

Aggregate (200 trades):

| Metric | Value |
|---|---|
| Expectancy | **−0.0771R** |
| Expectancy 95% CI | **[−0.1844, +0.0347]** (includes zero) |
| Total R / CI | −15.41R / [−36.87, +6.94] |
| Profit factor | 0.764 |
| Max drawdown | 5.36% |
| Positive folds | 0 / 3 |
| Regime coverage | 1.00 |

Per-fold net at base / 1.5× / 2.0×: fold 1 −80.97 / −289.45 / −484.40; fold 2
−453.73 / −659.74 / −863.85; fold 3 −219.44 / −354.93 / −519.07. Every fold is
negative before and after cost stress.

`locked_oos` is `null`. No holdout was attached.

### 6. Reproduce — exit 0

```bash
POLARS_MAX_THREADS=1 uv run python -m scalping.cli discovery reproduce \
  --bundle runs/realaccept/evidence/least-bad.evidence.json \
  --store-root runs/realaccept/store \
  --out runs/realaccept/05-reproduce.json
```

`reproduced: true`, checksum `7819f9d9…` matched, no mismatches. Threads were
pinned to 1 because the 2026-11-24 smoke recorded ~1e-16 regime-sum drift under
the default polars pool.

### 7. Promotion policy evaluation — exit 0, both policies reject

The bundle was evaluated with a **named test-only policy**
(`acceptance-test-policy` v1, `require_locked_oos=false`, digest
`1927e267bd7a…`) and, for the record, with production `PROMOTION_V1`. Neither
spends a holdout; the production policy simply sees no `locked_oos` section.

| Policy | Accepted | Rejection codes |
|---|---|---|
| `acceptance-test-policy` (test) | no | `min_positive_fold_fraction`, `min_expectancy_r`, `min_profit_factor`, `cost_stress`, `stress_degradation` |
| `promotion-v1` (production) | no | the above plus `min_ci_lower_r`, `missing_locked_oos` |

Champion creation failed closed in both directions:

- Test policy → **`non_production_policy`**: only a registered production policy
  may promote; the test policy digest is not registered.
- Production policy → **`missing_locked_oos`**: the bundle has no locked
  out-of-sample section.

### 8. Paper advance — not attempted

The paper path only accepts an evidence-mode champion whose promotion policy is
in the production allow-list. No such champion exists:

```
scope=XAUUSD:H1 champion_present=false
refusal=NoChampionError: no champion in scope 'XAUUSD:H1'; promote a validated candidate first
```

Stopping here is the required behaviour, not a workaround. Using the test policy
to mint a paper champion is explicitly forbidden by the promotion module and by
the champion reader.

## The remaining gate

**A real locked-holdout spend.** `PROMOTION_V1` requires a `locked_oos` section,
and only a production promotion can create a paper-tradable champion. That spend
is one-shot and irreversible, so it was not made. Spending it on this candidate
would be indefensible: expectancy is negative, the interval includes zero, no
fold is positive, and every cost-stress rerun loses. The next step needs the
owner's explicit approval and a candidate that is at least plausibly positive
before the holdout is unlocked.

## Limitations

- 22 evaluations on one validation slice; not a search for an edge.
- The evidence bundle belongs to a gate-rejected candidate, used for plumbing.
- Negative expectancy, zero positive folds, negative stress at both multipliers.
- iid bootstrap ignores trade clustering; the interval may be too tight.
- The test split was not read. No walk-forward beyond the three folds, no
  locked OOS, no regime-stability proof.
- One store, one symbol, one timeframe. Nothing generalises.
- `dataset build` (without `--no-cache`) wrote a `.feature-cache/` directory
  under the source store. It was removed; Parquet data and the manifest checksum
  were unchanged. Later stages used `--no-cache` or the writable copy.

## What this does and does not show

**Does show:** the stages connect on real bars; the gate rejects losing
candidates; a flat winner is refused evidence; a bundle persists and reproduces
exactly under pinned threads; promotion fails closed for both a non-production
policy and a missing locked OOS; paper refuses to run without a governed champion.

**Does not show:** validation, profitability, paper-trading performance, or a
promotable strategy. It shows the system saying no, with reasons and artifacts.
