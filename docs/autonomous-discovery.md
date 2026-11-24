# Autonomous strategy discovery

Phase 16. The objective is strategy-independent discovery: the system
must generate and test candidates without the owner supplying entry rules, exit
rules or named strategies. Slice A built the candidate contract, the archive
that holds the evidence and a bounded search loop. Slice B added the
DSL/AST and its interpreter (`docs/strategy-dsl.md`). Slice C wires that DSL to
the simulator in `scalping.discovery.evaluate.DslEvaluator`, so a candidate now
produces real net-of-cost trial metrics. Slice D adds
`scalping.discovery.run`: stored bars go in, a validation-only search runs, and
a no-change/no-trade/rejected decision is written to the archive. Phase 19 slice
B adds strategy champion governance: a discovered spec can be promoted to the
paper champion, compared, replaced or rolled back with an audit trail, and
`ChampionProvider` hands the stored spec to the paper runner. The protocol seam
is unchanged: the loop still knows nothing about strategy semantics.

Nothing in the discovery package executes generated code. A candidate is data:
canonical JSON, a sha256 id and a structural cost. The evaluator interprets
typed nodes with a closed set of operations, and the only dataset the loop ever
names is the validation one it is configured with.

## Candidate contract

`scalping.discovery.protocols` defines three structural types.

| Type | Methods | Owner |
|---|---|---|
| `Candidate` | `canonical_dict()`, `canonical_hash()`, `complexity()` | the DSL |
| `Factory` | `random(rng)`, `mutate(c, rng)`, `crossover(a, b, rng)`, `validate(c)` | the DSL |
| `Evaluator` | `evaluate(c, *, dataset_id, code_version, budget)` | the simulator wiring |

`canonical_hash()` must be sha256 over the canonical JSON of
`canonical_dict()`; `canonical_digest()` computes it and the archive refuses a
candidate whose id disagrees with its payload. Two spellings of the same
strategy therefore collapse to one identity, one lineage and one trial history.

`validate()` is the safety gate. The loop calls it before it touches complexity,
the id or the evaluator, and treats any exception as an invalid candidate. A DSL
implementation must interpret typed nodes with a closed set of operations. It
must not call `eval`/`exec`/`compile` on generated text or import a module named
by candidate data.

`TrialResult` carries numeric metrics, an `accepted` verdict and rejection
reasons. An accepted result cannot carry rejections and a rejected one must, so
the rejection histogram cannot double-count. Metrics are numbers only; a
candidate's symbol, costs and split belong in the `dataset_id` or in the run
configuration, because ranking needs a numeric key.

## Archive

`DiscoveryArchive` is one SQLite file with WAL, `busy_timeout`, connection-scoped
foreign keys and a versioned schema, following `scalping.registry.db`. Schema
version 1 created four domain tables plus `schema_version`; version 2 adds
strategy champion governance (`champion_promotions`, `champions`), the
`rolled_back` outcome and a JSON `detail_json` on `decisions`; version 3 adds the
promotion `mode` (`evidence`, `exploratory`, `manual`), an evidence pointer on
the champion rows and an immutable `evidence` table keyed by the bundle checksum.
The migration is idempotent and keeps every existing candidate, trial, lineage
edge and decision; existing promotions default to `exploratory`, which is what
they were.

`candidates` — one row per canonical hash.

| Column | Meaning |
|---|---|
| `candidate_id` | sha256 of the canonical JSON, primary key |
| `canonical_json` | the representation itself, verified as a JSON object |
| `complexity` | structural cost at registration |
| `created_at` | UTC ISO-8601, TEXT ordering is chronological |
| `parent_a`, `parent_b` | direct parents, foreign keys to `candidates` |
| `mutation` | operator that produced the row: `random`, `mutation` or `crossover` |
| `code_version` | code that registered it |
| `trials` | cached trial counter, maintained with every insert |

`trials` — one row per evaluation.

| Column | Meaning |
|---|---|
| `id` | append-only identity, ordering is the run order |
| `candidate_id` | foreign key to `candidates` |
| `dataset_id` | provenance key, expected to carry the dataset hash |
| `code_version` | evaluator code version |
| `seed` | the per-trial seed needed to replay this evaluation |
| `metrics_json` | numeric metrics, canonical JSON |
| `accepted` | 0/1 evidence verdict |
| `rejections_json` | rejection reasons, canonical JSON array |
| `created_at` | UTC ISO-8601 |

`lineage` — `(candidate_id, parent_id, relation)`, relation `mutation` or
`crossover`. Edges are written on first registration only, so a later
re-proposal cannot rewrite the derivation that actually happened.

`decisions` — `promoted`, `rejected`, `no_change`, `no_trade` or `rolled_back`,
with a reason, an optional champion id, a timestamp and an optional JSON
`detail_json` holding the numbers the decision used. `no_change` and `no_trade`
are first-class outcomes: rejecting every candidate is a valid result, and a
search that finds nothing better than the champion must be able to say so.

`champion_promotions` — the append-only history of one scope: `kind`
(`promoted` or `rolled_back`), the candidate, the predecessor, the metric and the
two values that were compared, the stored reason and a timestamp.

`champions` — exactly one row per scope (`PRIMARY KEY (scope)`): the current
candidate, the promotion id that put it there, the timestamp and the reason. The
row points into the history, so the live champion cannot drift from the audit
trail; both are written in the same transaction.

Deletes are restricted, not cascading. Deleting a candidate with children or
trials is refused because a trial cannot be reconstructed from a path the way a
model checkpoint can.

`add_candidate` deduplicates by hash and returns the existing row. It does not
raise: a search proposes the same strategy more than once, and that is not a
second identity. The stored payload and the incoming payload must agree, or the
insert is refused as an id collision. `best(metric, minimum_trials)` ranks
candidates by the best qualifying trial value and breaks ties on the id, so two
archives built from the same run agree.

## Search loop

`evolve(factory, evaluator, archive, config, should_stop=...)` is a generational
evolutionary loop with an exact attempt budget.

1. Generation 0 proposes `population` random candidates. Later generations draw
   an operator per slot: crossover with probability `crossover_rate`, otherwise
   mutation with `mutation_rate`, otherwise a fresh random candidate. A
   crossover that returns `None` falls back to mutating one parent.
2. Each proposal costs one attempt. `max_trials` caps attempts, not only
   successful evaluations, so a run has a hard deadline even when the factory
   keeps returning duplicates or invalid trees.
3. `validate()` runs first; an exception is counted as `invalid`.
4. A candidate above `complexity_cap` is counted as `complexity`.
5. A candidate already evaluated in this run, or already in the archive for the
   same dataset and code version, is counted as `duplicate`. A different code
   version is a new experiment and is allowed to run.
6. Everything else is evaluated and persisted with a derived per-trial seed.
   The evaluator receives the dataset id, the code version and the remaining
   trial budget, including the current trial.
7. Survivors are ranked by the selection metric. Rejected candidates keep their
   score: acceptance is the evidence gate a champion comparison must honour, not
   a fitness function, and a search that breeds only accepted candidates stalls
   after the first wave of rejections. The rejection reasons stay in the archive
   for the gate and the human.
8. A generation that records fewer than `novelty_requirement` trials ends the
   run as `exhausted`. That is how a saturated factory stops instead of burning
   the rest of the budget on duplicates.

`SearchConfig` defaults:

| Field | Default | Meaning |
|---|---|---|
| `dataset_id` | required | the only dataset the evaluator is offered |
| `code_version` | required | recorded on every candidate and trial |
| `population` | `32` | proposals per generation |
| `generations` | `10` | maximum rounds |
| `mutation_rate` | `0.3` | per-slot mutation probability |
| `crossover_rate` | `0.2` | per-slot crossover probability |
| `max_trials` | `200` | attempt cap, duplicates and invalid trees included |
| `seed` | `0` | the RNG seed, also the replay key |
| `patience` | `5` | generations without a new best before stopping |
| `complexity_cap` | `40` | maximum `complexity()` a candidate may declare |
| `novelty_requirement` | `1` | minimum new trials a generation must contribute |
| `selection_metric` | `net_profit` | metric the evaluator must publish |
| `minimize` | `False` | rank lowest instead of highest |

The loop stops for exactly one recorded reason.

| `stop_reason` | Meaning |
|---|---|
| `budget` | the attempt cap was reached; `trials + skipped == max_trials` |
| `patience` | `patience` generations passed without a new best |
| `exhausted` | the generation limit was reached, or a generation produced fewer than `novelty_requirement` new trials |
| `stopped` | the injected `should_stop` callback returned true before an attempt |

`SearchResult` reports the best candidate of this run (not the archive-wide
best), its metrics and id, the number of recorded trials and accepted trials,
the skip and rejection histograms, the deepest lineage chain recorded, the
generations completed and the stop reason. `trials + skipped` is the number of
attempts spent.

Determinism is a promise with a narrow scope: same `numpy.random.Generator`
stream (PCG64), same seed, same factory and evaluator, same archive state, same
sequence. Ties in ranking always break on the candidate id, never on dictionary
iteration. The trial seed is derived from the search RNG, so a single evaluation
can be replayed from its row even without re-running the search.

## Run contract: `discovery run`

`scalping.discovery.run.run_discovery(config, root=..., should_stop=...)` is the
end-to-end entry point. It reads the Parquet store at `root`, builds the causal
feature matrix through `<root>/.feature-cache` (bypassed with
`use_cache=False`), cuts a chronological dataset, keeps only the validation
split, scores the baselines on that slice, runs the evolutionary search and
records one decision.

```
store.read(symbol, timeframe)
  → cached_build_features(...)
  → build_dataset(...)                  # purge/embargo geometry only
  → frame = features[validation.start:validation.end]
  → DslEvaluator(frame, gate, window, stress)
  → run_backtest(hold), run_backtest(ret_5)
  → evolve(StrategyFactory, evaluator, DiscoveryArchive)
  → decision
```

**Split discipline.** The test rows are never copied, never scored and never
named to an evaluator. The dataset id is `validation:` plus the manifest id, so
`DslEvaluator` accepts it and any other id is refused. A build whose validation
split is empty is an error ("a manifest whose only rows are in test" cannot
drive discovery), and a validation split no longer than the observation window
is an error too, because the simulator would have no decision bar. The tests
mutate every bar after `validation.end` and assert bit-identical candidate
metrics: if a run leaked test rows, that test moves.

**Cost parity.** Candidates and both baselines run on the same slice, with the
same `Settings`, the same observation `window` and the same weekly session gate.
`hold` never trades; `ret_5` buys while the last bar's 5-bar momentum is
positive and carries the CLI's `rule:ret_5-baseline` label. A strategy that
"beats hold" only because the baseline paid different costs has beaten nothing.

`DiscoveryRunConfig` defaults (CLI names in parentheses):

| Field | Default | Meaning |
|---|---|---|
| `symbol` (`--symbol`) | `XAUUSD` | stored symbol |
| `timeframe` (`--timeframe`) | `M1` | stored timeframe |
| `max_trials` (`--max-trials`) | `100` (`24` in the CLI) | attempt cap, duplicates and invalid trees included |
| `population` (`--population`) | `16` (`8`) | proposals per generation |
| `generations` (`--generations`) | `5` (`4`) | generation cap |
| `seed` (`--seed`) | `0` | search RNG seed, also the replay key |
| `mutation_rate` (`--mutation-rate`) | `0.3` | per-slot mutation probability |
| `crossover_rate` (`--crossover-rate`) | `0.2` | per-slot crossover probability |
| `complexity_cap` (`--complexity-cap`) | `40` | maximum structural cost |
| `archive` (`--archive`) | `runs/discovery/discovery.sqlite` | SQLite evidence file |
| `use_cache` (`--no-cache`) | `True` | feature cache on |
| `stress` (`--stress`) | `(1.5, 2.0)` | simulator cost multipliers |
| `gate` (`--min-trades`, `--require-stress`, ...) | empty | acceptance criteria |
| `baselines` (`--baseline`) | `hold,ret_5` | ordered comparison table |
| `horizon`, `--train-fraction`, `--validation-fraction`, `--purge`, `--embargo` | `1`, `0.70`, `0.15`, `0`, `0` | split geometry |
| `window` (`--window`) | `60` | observation warm-up |
| `patience` (`--patience`) | `5` (`3`) | generations without a new best |
| `max_lookback` / `max_depth` | `20` / `3` | DSL generation bounds |
| `code_version` | source digest | recorded on every candidate and trial |

```bash
cd backend
uv run python -m scalping.cli discovery run \
  --root data --symbol XAUUSD --timeframe M1 \
  --max-trials 24 --population 8 --generations 4 --seed 0 \
  --archive runs/discovery/discovery.sqlite --out runs/discovery/last.json
```

The default CLI budget is a smoke budget: it proves the wiring in seconds.
A real search needs thousands of trials, and the trial count is the honest
denominator for every later claim (see
[Trial accounting and multiple testing](#trial-accounting-and-multiple-testing)).
`--stress ""` disables cost stress; the gate only rejects for a multiplier it
was given and that was actually run (`missing_cost_stress` otherwise).

`DiscoveryRunResult.to_dict()` is JSON-safe with no NaN or Infinity and carries
the dataset id, the validation range (`start`, `end`, `rows`, first/last
timestamp), the seed, the code version, the trial/skip/accepted counts and stop
reason, the best accepted candidate's hash, canonical spec and metrics, the
baseline table, the comparison numbers the decision used, the decision with its
reason, the archive path and the decision row id.

### Decision semantics

`run_discovery` never writes `promoted`: a run ends at a recorded hypothesis.
Promotion is a separate, explicit `promote_with_evidence` call after review
(see [Champion governance](#champion-governance)); the registry gate continues to
own *model* promotion. The run's vocabulary is:

| Decision | When | Meaning |
|---|---|---|
| `no_trade` | the gate rejected every trial, or the best accepted candidate is flat (0 closed trades) | standing aside is the result, not a failure |
| `no_change` | no accepted candidate beat the best baseline | the champion (or the baseline) stands |
| `rejected` | an accepted candidate strictly beat the best baseline | an improving candidate exists; the run hands it to the champion governor instead of promoting it |

The order is part of the contract and is implemented by the pure function
`decide(...)`. A baseline that did not publish the selection metric is not
"beaten"; a candidate that did not publish it cannot be compared. Every
decision is persisted with `record_decision` under the best candidate of the
run (or the archive's best when a rerun added no trials), and the reason string
carries the numbers: the selection metric, the candidate value, the best
baseline's name and value, and the trial count.

**A validation winner after N trials is not evidence of a durable edge.** The
decision only says what happened on one validation slice at a chosen cost
level; it does not say the candidate will survive the untouched test split, a
forward period, another symbol or another cost regime. `rejected` is a handoff
to the champion governor, not a promotion, and `no_change`/`no_trade` are
complete, honest outcomes.

### Reproducibility and deduplication

The dataset id comes from the manifest digest, the trial seeds from the search
RNG and the code version from a digest of the modules that change arithmetic.
Same store slice, seed, code version and an empty archive therefore reproduce
the same candidate hashes and metrics. A second run on the same archive never
re-tries a `(candidate, dataset, code version)` triple: repeats are counted as
`duplicate` and reported as `deduped`. Because a skipped proposal does not draw
a trial seed, the RNG stream diverges after a repeat, so a rerun explores new
candidates instead of replaying the first run; `trials + skipped <= max_trials`
still holds. The four `stop_reason` values are the search loop's
(budget, patience, exhausted, stopped), reported verbatim.

## Champion governance

A discovered candidate becomes the paper-trading champion through the archive,
not through a hand-written strategy file. `promote_with_evidence` (policy
decision), `promote_exploratory` (research only), `rollback` and the read side
(`champion`, `champion_history`, `compare_to_champion`) own the lifecycle. `run_discovery` stops at a recorded decision; a
`rejected` row hands the candidate over, and promoting it is this explicit call.

### Scope

A scope is a free-text key, conventionally `<symbol>:<timeframe>` —
`scalping.discovery.champion.scope_for("XAUUSD", "M5")` builds it. Each scope has
its own champion, so promoting a strategy for gold never replaces the one for
euro. A trial belongs to a scope when every alphanumeric token of the scope
appears in its `dataset_id`: `XAUUSD:M5` matches
`validation:XAUUSD:M5:abcd1234` and not `validation:XAUUSD:M15:abcd1234`. This is
a naming convention, not a guarantee — a dataset id that omits the timeframe will
not qualify.

### Promotion

Production promotion requires a policy decision over a walk-forward evidence
bundle; it is not a metric comparison:

```python
from scalping.discovery.promotion import (
    PROMOTION_V1,
    EvidenceIdentity,
    promote_with_evidence,
)

champion = promote_with_evidence(
    archive,
    candidate_id,
    evidence,                     # CandidateEvidence
    PROMOTION_V1,                 # named, versioned, fail-closed
    scope="XAUUSD:M5",
    reason="passed promotion-v1 on identical folds",
    expected_identity=EvidenceIdentity(
        code_version=current_code_version,
        dataset_checksum=current_dataset_checksum,
        settings_digest=current_settings_digest,
    ),
)
```

The policy is evaluated **inside** the call; there is no decision object to
forge and no public archive promotion method. The current champion is read from
the archive and its stored evidence is the baseline. The function writes the
policy id/version/digest, evidence checksum, dataset id/checksum, settings digest
and code version into the audit detail. The history row stores no metric
(`metric`/`value` are NULL) and `mode`/`detail_json.mode` is `"evidence"`.
`PROMOTION_V1.require_locked_oos` is `True`, so nothing promotes until the
holdout slice attaches a locked-OOS section. The policy, its thresholds and
every rejection code are documented in `docs/local-learning-engine.md`.

The older one-metric path remains for research only:

```python
archive.promote_exploratory(
    candidate_id,
    scope="XAUUSD:M5",
    reason="exploratory ablation, never the paper champion",
    metric="expectancy_r",   # default
    force=False,             # default
)
```

`promote_exploratory` runs in `BEGIN IMMEDIATE` and refuses, writing nothing:

| Refusal | Exception | Forceable? |
|---|---|---|
| unknown candidate id | `UnknownCandidateError` | no |
| no trial whose `dataset_id` names every scope token | `ArchiveError` | no |
| no **accepted** trial on the scope's dataset (gate rejected) | `ArchiveError` | no |
| accepted trial publishes no value for `metric` | `ArchiveError` | yes |
| current champion has no value for `metric` | `ArchiveError` | yes |
| `metric` does not strictly beat the champion's | `ArchiveError` | yes |
| empty reason | `ValueError` | — |

`force` overrides the comparison only; the evidence gate is not forceable, so a
rejected candidate cannot be promoted by any flag. The stored reason is
`"<human reason>; <metric> <value> vs <previous value> of <candidate prefix>"`,
and a forced promotion prefixes it with `forced: `. The previous champion stays
in `champion_promotions` and in `candidates`; nothing is deleted.

The comparison reads each side's newest accepted trial on the scope's dataset
that publishes the metric. Missing numbers are not wins: an unmeasured candidate,
or an unmeasured champion to compare against, is refused rather than treated as
zero. Only one metric is compared. A `promoted` decision is appended with
`detail_json` holding `scope`, `metric`, `value`, `previous_candidate_id`,
`previous_value`, `forced` and `promotion_id`.

### Reading the champion

`archive.champion(scope)` returns `None` before the first promotion, else the
candidate row, its canonical spec payload, the newest accepted metrics on the
scope's dataset, and the promotion id, timestamp and reason.
`champion_history(scope)` returns every promotion and rollback oldest first.
`compare_to_champion(candidate_id, scope, metric)` is the read-only preview the
gate and a dashboard use: candidate value, champion value, delta, and
`better=True` only for a strict win with both values measured.

### Rollback

```python
archive.rollback("XAUUSD:M5", to_candidate_id=previous_id, reason="decay detected")
```

Rollback restores a candidate that was **champion of that scope before** and not
the current one. A candidate that never won is promoted, not rolled back to, and
mixing the two would make the audit trail lie. It appends a `rolled_back` row to
`champion_promotions` and a `rolled_back` decision, and it works on a scope with
no champion only as an error: there is nothing to undo.

### ChampionProvider

`scalping.discovery.champion.ChampionProvider` turns the stored row into a
validated `StrategySpec` for the paper path:

```python
provider = ChampionProvider(archive, scope="XAUUSD:M5")
spec = provider.spec()            # StrategySpec, validated before it is returned
run = provider.champion()         # scope, spec, candidate id, promotion, metrics
```

It raises `NoChampionError` when nothing was promoted, `NoEvidenceChampion` when
the champion came from the exploratory or manual path, and
`InvalidChampionSpec` when the payload does not load as a strategy or its
recomputed hash differs from the archived id. There is no fallback to a hand-written strategy: a silent
default would mean the autonomous run stops testing what the search produced.
The module imports the archive and the DSL only; `scalping.paper` is never
imported from `scalping.discovery`, so the dependency stays one-way.

### Honest limits

- **Promotion is policy, not proof.** It applies one comparison rule to
  validation evidence. Passing it says the candidate beat the incumbent on one
  metric on one dataset; it does not say the candidate will make money.
- **The metrics come from validation trials.** No out-of-sample or walk-forward
  result is produced here, and the evaluator refuses a holdout dataset id. A
  promotion built on a validation split that the search already touched inherits
  that data-snooping risk.
- **Drift and decay monitoring is not automatic.** Nothing recomputes a
  champion's metrics on a schedule, queues a re-test or rolls back on its own.
  `compare_to_champion` and `rollback` are the tools; the trigger is a human or a
  later orchestrator slice.
- **One metric, higher is better, in the exploratory path.** A lower-is-better
  metric such as drawdown must not be passed to `promote_exploratory`; it would
  invert the rule. Production promotion reads the policy bundle instead.
- **`force` hides nothing but it does override policy.** The audit trail records
  `forced:` and the numbers that were rejected, which is exactly why the flag
  exists; it is not evidence that the candidate is good.
- **Restart safety is the database's.** The pointer and history are committed in
  one transaction with WAL, so a crash leaves either the old champion or the new
  one, never both. An OS-level power loss can still drop the newest committed
  transaction.

## Trial accounting and multiple testing

The archive records every evaluation, and `SearchResult.trials` reports how many
the run spent. That number is the honest denominator for a discovery result.

A single winning backtest after N trials is not evidence of a durable edge. If
1 000 candidates are tested and one has a good equity curve, the search has
demonstrated that a good-looking curve exists in this space on this data; it has
not demonstrated that the edge survives costs, another period or another symbol.
Discovery is a hypothesis generator. The gate that can promote anything is
separate: walk-forward validation, untouched out-of-sample data, cost stress,
regime and symbol coverage, and a champion comparison. The archive makes those
steps auditable because the trial count, the seeds and the rejection reasons are
persisted rather than reconstructed from memory.

`trials` also grows across runs on the same dataset. `best(metric,
minimum_trials=...)` exists so a caller must ask for a minimum amount of
evidence instead of taking whichever row happens to be first.

## Dataset discipline

The loop has one dataset handle and passes it to every evaluation. It is meant
to be a **validation** dataset. The untouched test slice is not reachable from
here: there is no split switch, no path to the store and no test dataset id.
Evaluating on the final holdout during search turns the holdout into another
training set, and no later gate can undo that.

Purge and embargo geometry is not reimplemented. The caller builds the
validation frame — reusing `scalping.ml.cv.fold_geometry` and the dataset
builder's purge/embargo rules — and hands that one frame to the evaluator. The
evaluator does not split, fold or embargo on its own. `run_discovery` is the
caller that does this today: it slices the validation split out of the
manifest and never touches the test slice.

`scalping.discovery.evidence` is the caller that cuts several chronological
folds across the **train + validation** timeline and records promotion-grade
evidence: per-fold metrics, cost stress, regime attribution, bootstrap
intervals and trial context. It still never reads the test split and it is not
a promotion. The schema, fold geometry, fail-closed codes and remaining gaps
are documented in
[`backtesting-methodology.md`](backtesting-methodology.md#candidate-evidence-bundle-discovery).

## DSL evaluator: `DslEvaluator`

`scalping.discovery.evaluate.DslEvaluator` is the production `Evaluator`. It
interprets one `StrategySpec` bar by bar and drives the same `Simulator` as
`scalping.backtest.engine.run_backtest`: `reset()`, then one `step(action)` per
bar, recording the marked equity and every closed `Trade`. Metrics come from
`scalping.backtest.metrics.compute`, so spread, slippage and commission are
already inside the fills and are never subtracted twice.

```python
DslEvaluator(
    features, settings,
    *, symbol, timeframe, dataset_id,
    gate=GateCriteria(), window=60, stress=(), max_steps=None,
)
```

`evaluate(candidate, *, dataset_id, code_version, budget) -> TrialResult`.

`features` is a **prebuilt** `FeatureFrame`. It is validated once at
construction (2-D `X`, column names, `ts`/`ohlc`/`spread`/`atr` shapes,
`valid_from`, finiteness after warm-up) and reused for every candidate. Nothing
is rebuilt per proposal and the frame is never mutated. `code_version` and
`budget` are accepted for the protocol and do not change the arithmetic: one
candidate costs one frame pass per requested stress multiplier.

### Bar order and the action mapping

A decision at bar `i` reads row `i` plus the policy's history of rows before
`i`. History is seeded with at most `max_lookback` rows that already occurred
before the first decision bar, so a lookback is available from the first
decision without reading a future row. Rows before `valid_from` carry warm-up
`NaN`; the interpreter treats them as unknown, not as zero.

| DSL action | Simulator action | Note |
|---|---|---|
| `BUY` | `1` | entry, subject to the risk engine and session gate |
| `SELL` | `2` | entry, same gates |
| `CLOSE` | `3` | rejected as `no_position` when flat |
| `HOLD` | `0` | |
| `NO_TRADE` | `0` | suppresses the entry; it does **not** flatten a position |

`NO_TRADE → HOLD` is deliberate. The interpreter evaluates `no_trade` before
exits, so on a true no-trade bar it has already suppressed the exit decision;
flattening at the execution layer would contradict the DSL's own precedence. An
open position is closed by an exit rule or the holding limit on a later bar.

Exits are also evaluated before entries, and a `CLOSE` on a flat account is
rejected as `no_position`. A strategy whose exit condition is true while flat
therefore never enters; those candidates appear as zero-trade trials, which is
the interpreter's precedence, not an evaluator bug.

`window` is the simulator's observation warm-up and therefore sets the first
decision bar (`max(valid_from + window - 1, window - 1)`), exactly as the
backtest engine does. `max_steps` overrides the simulator's 20 000-step default;
pass the frame length when a long validation window must run end to end.

### Dataset guard

The evaluator is bound to one `dataset_id` at construction.

- Any other id passed to `evaluate` raises `ValueError` before a bar is read.
- An id whose tokens name the untouched split — `test`, `tests`, `holdout`,
  `final`, `locked`, including `final-test` and `hold-out` spellings — is
  refused at construction. The match is token-based, so `latest` is allowed.
- The evaluator has no store handle and no path to the final split. This is a
  guard, not a substitute for split discipline: the caller still owns the
  chronological fold geometry and must hand it a validation frame.

### Cost stress

`stress=(2.0, 3.0, ...)` reruns the same candidate once per multiplier with
`FeatureFrame.spread * m` (a shallow copy; `X` stays as observed) and
`spread_floor`, `slippage_atr_frac` and `commission_per_lot` scaled by `m`.
The policy's decisions are identical because the features are; the fills and
the account path move, so account-level gates (daily loss, kill switches) can
block different later entries. A multiplier of `1` reuses the base run.

Each run's finite metrics are flattened into `TrialResult.metrics` as
`stress_<label>_<metric>` (`stress_x2_net_profit`, `stress_x2_bars`, ...), and
the gate receives the `{"x2": {"net_profit": ...}}` shape it expects. A gate
that requires a multiplier the caller did not request rejects with
`missing_cost_stress` rather than inventing evidence.

### Acceptance

When a gate threshold is enabled, `accepted` and `rejections` come verbatim
from `scalping.registry.gate.evaluate_candidate`. Rejections keep the
`code: message` spelling; group them on the text before the first `:`.
Calibration and regime evidence are never produced by this evaluator, so
`max_ece` and `min_regime_coverage` reject with `missing_calibration` and
`missing_regime` instead of passing silently. A default `GateCriteria` is empty
and accepts every candidate — that is search-fitness mode, not promotion.

Non-finite metrics are omitted rather than clamped: `profit_factor` is `inf`
for wins without losses, and a gate that requires it then rejects with
`missing_metric` instead of reading a fabricated number.

## What is deliberately absent

- **Nested validation inside the evaluator.** It runs one candidate on one
  frame. Chronological folds, purge/embargo geometry and the locked final test
  are a caller/orchestrator concern; the evaluator only refuses a holdout id.
  `scalping.discovery.evidence` supplies the multi-fold builder and
  `scalping.discovery.promotion` judges it; wiring both into the orchestrator is
  a later slice.
- **Out-of-sample and forward testing.** No test-slice access and no paper
  period. Those are later gates.
- **Automatic champion monitoring.** `champion`, `compare_to_champion`,
  `promote_with_evidence` and `rollback` are explicit calls with an audit
  trail. Nothing
  schedules a comparison, watches the champion's live paper results or rolls
  back because a metric moved. `run_discovery` itself compares only against
  `hold` and `ret_5` and never promotes; the incumbent comparison is the
  governor's job, and its trigger belongs to the orchestrator.
- **Symbol, timeframe, regime and cost sweeps.** One run evaluates one dataset
  id. Cross-condition stability needs a caller that runs several and compares.
- **Scheduler cancellation.** `should_stop` is the hook; connecting it to a job
  record or a UI button is orchestrator work.
- **Parameter-sensitivity analysis.** The archive makes it possible; no
  neighbourhood scan is implemented here.
- **Sizing from DSL nodes.** `FixedFraction`/`AtrMultiple` are part of the
  candidate identity but do not reach the broker: the simulator's `RiskEngine`
  owns position size from its frozen config. Two specs differing only in sizing
  hash differently but simulate identically.

## Honest note

This slice proves a search process wired to a cost-aware simulator and a
recorded decision, not a trading edge. It shows that candidates can be generated
without owner-written rules, identified by content, deduplicated, simulated with
real fills and costs, gate-checked, lineage-tracked, baselined against `hold`
and `ret_5`, archived reproducibly within a budget, and ended with an explicit
`no_trade`, `no_change` or `rejected`. Whether any candidate makes money after
spread, commission, slippage and drawdown on data it did not see is a question
for the walk-forward and promotion gates, and the answer may be no trade. A
large trial count is a warning about data-snooping risk, not a qualification.
One candidate with a positive validation backtest after N trials, alone, is not
evidence of a durable edge.
