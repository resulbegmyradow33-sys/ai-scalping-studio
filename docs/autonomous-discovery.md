# Autonomous strategy discovery

Phase 16, slice A. The objective is strategy-independent discovery: the system
must generate and test candidates without the owner supplying entry rules, exit
rules or named strategies. This slice builds the parts that do not depend on the
strategy language — the candidate contract, the archive that holds the evidence,
and a bounded search loop. The DSL/AST and its interpreter are a separate slice;
they plug in through the protocols here without changing the loop.

Nothing in this slice executes generated code. A candidate is data: canonical
JSON, a sha256 id and a structural cost. The evaluator that scores it is
injected, and the only dataset the loop ever names is the validation one it is
configured with.

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
version 1 creates four domain tables plus `schema_version`.

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

`decisions` — `promoted`, `rejected`, `no_change` or `no_trade`, with a reason,
an optional champion id and a timestamp. `no_change` and `no_trade` are
first-class outcomes: rejecting every candidate is a valid result, and a search
that finds nothing better than the champion must be able to say so.

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

The loop has one dataset handle and passes it to every evaluation. In this slice
it is meant to be a **validation** dataset. The untouched test slice is not
reachable from here: there is no split switch, no path to the store and no test
dataset id. Evaluating on the final holdout during search turns the holdout into
another training set, and no later gate can undo that.

Purge and embargo geometry is not reimplemented. The evaluator owns its splits
and can reuse `scalping.ml.cv.fold_geometry`, which is the same arithmetic
`cross_validate` and `fit_oof` use.

## What is deliberately absent

- **The DSL/AST.** No grammar, no node types, no interpreter, no parser. The
  protocols are the seam another slice plugs into.
- **A real evaluator.** Nothing here wires the simulator, costs or walk-forward
  validation. `TrialResult` is a contract; producing honest metrics is the
  evaluator's job.
- **Out-of-sample and forward testing.** No test-slice access and no paper
  period. Those are later gates.
- **Champion comparison and promotion.** `decisions` records an outcome; the
  policy that decides it, compares against the champion and rolls back lives
  with the registry and the gate.
- **Symbol, timeframe, regime and cost sweeps.** One run evaluates one dataset
  id. Cross-condition stability needs a caller that runs several and compares.
- **Scheduler cancellation.** `should_stop` is the hook; connecting it to a job
  record or a UI button is orchestrator work.
- **Parameter-sensitivity analysis.** The archive makes it possible; no
  neighbourhood scan is implemented here.

## Honest note

This slice proves a search process, not a trading edge. It shows that candidates
can be generated without owner-written rules, identified by content, deduplicated,
evaluated, lineage-tracked and archived reproducibly within a budget. Whether
any candidate makes money after spread, commission, slippage and drawdown is a
question for the evaluator and the validation gates, and the answer may be no
trade. A large trial count is a warning about data-snooping risk, not a
qualification.
