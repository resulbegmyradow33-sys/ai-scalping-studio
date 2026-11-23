# Local learning engine

Phase 10 turns ad-hoc training runs into a registry-backed loop: a candidate is
trained, measured, gated, and only then promoted; the previous champion stays
recoverable and the samples are compared for drift. This page covers slice 2 —
the candidate → champion gate (`scalping.registry.gate`) and drift diagnostics
(`scalping.registry.drift`). The SQLite registry itself is described by its
module docstring and `docs/data-pipeline.md`; the scheduler is a later slice.

## Candidate → champion gate

`GateCriteria` is a frozen dataclass of thresholds. **Every threshold is
disabled at `0`**, and an empty cost-stress tuple or missing evidence disables
that check too; this follows the `RiskConfig` convention, so a default
`GateCriteria()` accepts every candidate and changes no existing run. Passing
values is the caller's choice, made explicit at the call site.

| Criterion | Default | Enabled behaviour | What rejects |
|---|---|---|---|
| `min_trades` | `0` (off) | pass when `trades >= value` | fewer closed trades than required |
| `min_expectancy_r` | `0.0` (off) | pass when `expectancy_r >= value` | negative or too small expectancy |
| `min_profit_factor` | `0.0` (off) | pass when `profit_factor >= value` | profit factor below the floor |
| `max_drawdown` | `0.0` (off) | pass when `max_drawdown <= value`; fraction in `[0, 1]` | deeper peak-to-equity drop than allowed |
| `max_ece` | `0.0` (off) | pass when `ece <= value` | calibration worse than allowed |
| `require_positive_cost_stress` | `()` (off) | each multiplier must exist in the stress evidence and have `net_profit > 0` | a missing stress run, or one that loses money |
| `min_regime_coverage` | `0.0` (off) | pass when labelled trades / headline trades `>= value` | too many trades whose entry regime is unknown |

Boundary rules are deliberate and tested at the exact value: the three `min_*`
criteria and `min_regime_coverage` use `>=`, `max_drawdown` and `max_ece` use
`<=`, and cost stress is strictly positive — a break-even stressed run is a
rejection, because paying more costs without losing money is not evidence that
the edge survives them.

A recommended starting policy is stricter than the defaults:

```python
GateCriteria(
    min_trades=100,
    min_expectancy_r=0.05,
    min_profit_factor=1.3,
    max_drawdown=0.25,
    max_ece=0.10,
    require_positive_cost_stress=(1.5, 2.0),
    min_regime_coverage=0.85,
)
```

The values are policy, not statistics. They encode how much evidence this
project demands before a model may trade; they cannot prove the model is good.
A candidate that clears them has passed checks a human chose, nothing more.

### Missing evidence rejects

An enabled criterion whose input is absent produces a rejection, never a silent
pass. The codes are:

| Code | Meaning |
|---|---|
| `missing_metric` | a metric the criterion needs (`trades`, `expectancy_r`, `profit_factor`, `max_drawdown`) is absent or not a number |
| `missing_calibration` | `max_ece` is enabled but no ECE was supplied |
| `missing_cost_stress` | a required multiplier has no `net_profit` entry |
| `missing_regime` | `min_regime_coverage` is enabled but the regime table is absent or empty, or coverage cannot be measured |
| `min_trades` / `min_expectancy_r` / `min_profit_factor` / `max_drawdown` / `max_ece` / `cost_stress` / `min_regime_coverage` | the evidence is present and fails the policy |

`GateDecision.rejections` holds one `"code: message"` string per failed check;
`GateDecision.codes` returns the codes alone. `to_dict()` is JSON-safe and keeps
the criteria, the measured values and the summary, so a log or dashboard can
explain a rejection without re-running the evaluation.

### Promotion and forced overrides

`promote_candidate(registry, model_id, decision, scope=..., reason=...)` is the
only gate-aware promotion path; `ModelRegistry.promote` keeps its signature and
stays the primitive.

- An accepted decision promotes and writes the gate summary into the promotion
  reason (`"<human reason>; gate accepted: trades=…, expectancy_r=…"`).
- A rejected decision raises `PromotionRefused` (a `RegistryError` subclass)
  and leaves the champion untouched.
- `force=True` is the escape hatch. It requires a non-empty human `reason`, and
  the stored reason begins with `forced: <reason>; gate rejected (…)`, so the
  audit trail records that the policy was overridden instead of hiding it.
- The promotion chain still records `previous_model_id`, and the registry's
  `rollback` restores an earlier champion with its own reason.

Comparison against the current champion's *measured* metrics is not part of
this slice: the caller decides which candidates reach the gate, and the gate
only judges the evidence it is handed.

## Drift diagnostics

Drift answers "did the new sample move?", not "is the model broken?". All
functions are pure over 1-D arrays; non-finite values are dropped before any
statistic, and `drift_from_datasets` counts the dropped rows per feature so a
mostly-missing column cannot look stable.

| Metric | What it compares | Default | Notes |
|---|---|---|---|
| `population_stability_index(reference, current, bins=10)` | feature or score histogram | bins = 10 reference quantiles | directional (the reference defines the bins); empty-bin share floored at `1e-6`; identical samples return exactly `0.0`; a constant reference is compared as "equal" vs "anything else" |
| `ks_statistic(reference, current)` | empirical CDFs | — | returns `(D, p)`; stdlib `math` only, no scipy dependency; `p` is the asymptotic tail probability with effective sample size, not an exact permutation test |
| `drift_from_datasets(reference_bundle, current_bundle, ...)` | every feature column of a `FeatureFrame` or a mapping of arrays | PSI threshold `0.2` | `DriftReport` holds per-feature PSI/KS, `drifted`, notes and a human summary |
| `prediction_drift(reference_probabilities, current_probabilities, ...)` | per-row maximum class probability, plus mean ECE when outcomes are supplied | PSI threshold `0.2` | confidence PSI/KS; `drifted` is set by PSI alone, ECE is reported as `ece_delta` for context |

`drifted` uses a strict `psi > threshold`; a PSI exactly at `0.2` is not drift.
The conventional reading is PSI below `0.1` = no meaningful shift, `0.1–0.2` =
worth watching, above `0.2` = a significant shift — a rule of thumb, not a
significance test.

Mean ECE averages the per-class ECE over the classes that actually occur in the
outcome vector. A class the model never sees is excluded rather than scored,
and missing outcomes leave the ECE fields `None` with a note instead of inventing
a value. A feature whose PSI cannot be computed appears with `psi=None` and a
note, never as "stable".

Drift detection is descriptive. PSI has no null distribution, KS's p-value is
asymptotic and uncorrected for multiple features, and a market that simply got
quieter can move every number without the model being wrong. These metrics are
a reason to look at fresh evidence; they do not promote, reject or retrain
anything by themselves.

## Not in this slice

- Scheduler and job cancellation, one-heavy-job enforcement, RAM budget.
- Champion-vs-candidate metric comparison inside the gate.
- Drift-triggered retraining: the report exists, the automation that reads it
  does not.
- Regime-level promotion rules; `min_regime_coverage` only checks that labels
  exist, not that performance is stable inside each regime.
