# Strategy DSL and AST

`scalping.strategy.dsl` is the safe, versioned representation the autonomous
discovery engine searches over. A strategy is a tree of frozen dataclasses. It
is never Python source and no generated text is executed.

The DSL defines a **bounded search space**, not a trading edge. A valid spec is
a hypothesis that can be evaluated; it says nothing about profitability until an
independent, cost-aware, out-of-sample evaluation proves it on data the search
did not see.

## Layers

| Module | Responsibility |
| --- | --- |
| `nodes.py` | Frozen node dataclasses, payload serialisation, `DSL_SPEC_VERSION = 1`. |
| `validate.py` | Structural/type gate, caps, unknown node/field rejection, future-reference rules. |
| `canonical.py` | Canonical JSON, sha256 identity, structural `complexity()`. |
| `compile.py` | `compile_policy(spec)` → `Policy.decide(row, state)` / `evaluate_series(frame)`. |
| `mutate.py` | Deterministic `random_spec`, `mutate`, `crossover`, `StrategyFactory`. |

The root object implements the discovery protocol's `Candidate` surface:
`canonical_dict()`, `canonical_hash()` and `complexity()`. `StrategyFactory`
implements `Factory` with `random`, `mutate`, `crossover` and `validate`. No
adapter is needed when the discovery slice is wired in; this slice does not
import or call the simulator or search loop.

## Grammar

```
spec        := StrategySpec(
                 spec_version = 1,
                 entries      = [entry, ...],        # >= 1
                 exits        = [exit, ...],         # >= 1
                 holding      = holding_limit | none,
                 no_trade     = no_trade | none,
                 sizing       = fixed_fraction | atr_multiple | none,
                 max_lookback = int in [1, 500]
               )

entry       := EntryRule(side = "long" | "short", condition = condition)
exit        := ExitRule(condition = condition)
holding     := HoldingLimit(bars = int in [1, 500])
no_trade    := NoTrade(condition = condition)

condition   := compare | and | or | not | between | cross | rise | fall
compare     := Compare(number, op, number)
and         := And(condition, condition)
or          := Or(condition, condition)
not         := Not(condition)
between     := Between(number, number, number)       # low <= value <= high
cross       := CrossAbove(number, number, window) | CrossBelow(number, number, window)
rise        := Rise(number, window)
fall        := Fall(number, window)

number      := Feature(name)                          # one of 59 causal columns
             | Const(finite float)
             | Change(number, window)                 # x[t] - x[t-window]

op          := "<" | "<=" | ">" | ">=" | "==" | "!="
window      := int in [1, max_lookback]
side        := "long" | "short"
```

`Feature` accepts exactly the causal columns in
`scalping.features.build.FEATURE_COLUMNS`: the seven registry groups (price,
momentum, volatility, volume, structure, session, multi-timeframe), 59 names in
total. Unknown names fail at construction and again during validation.

## Node table

| Node | Fields | Kind | Meaning |
| --- | --- | --- | --- |
| `Feature` | `name` | number | Value of a registered feature on the bar. |
| `Const` | `value` | number | Finite constant, normalised to float. |
| `Change` | `source`, `window` | number | `source[t] - source[t-window]`. |
| `Compare` | `left`, `op`, `right` | condition | Numeric comparison. |
| `And` / `Or` | `left`, `right` | condition | Three-valued conjunction/disjunction. |
| `Not` | `operand` | condition | Three-valued negation. |
| `Between` | `value`, `low`, `high` | condition | Inclusive range. |
| `CrossAbove` | `left`, `right`, `window` | condition | `left > right` now and `left <= right` `window` bars ago. |
| `CrossBelow` | `left`, `right`, `window` | condition | `left < right` now and `left >= right` `window` bars ago. |
| `Rise` | `source`, `window` | condition | `source[t] > source[t-window]`. |
| `Fall` | `source`, `window` | condition | `source[t] < source[t-window]`. |
| `EntryRule` | `side`, `condition` | rule | Request a long or short entry. |
| `ExitRule` | `condition` | rule | Request a close. |
| `HoldingLimit` | `bars` | rule | Close after at most `bars` held bars. |
| `NoTrade` | `condition` | rule | Suppress entries while true. |
| `FixedFraction` | `fraction` | sizing | Fraction of equity in `(0, 1]`. |
| `AtrMultiple` | `multiple` | sizing | ATR risk multiple `> 0`. |
| `StrategySpec` | see grammar | root | Versioned strategy. |

There is no arithmetic tree, no function call, no attribute access, no variable,
no loop, no import and no signed lag. Adding one of those would enlarge the
attack and look-ahead surface; `Change` is the only arithmetic and its offset is
strictly backward.

## Validation rules

`validate(spec)` raises `DslValidationError` (a `ValueError`) when any rule
fails:

1. Root is a `StrategySpec`; `spec_version == 1`; `max_lookback` is an int in
   `[1, 500]`.
2. `entries` and `exits` are tuples, non-empty, and contain only the matching
   rule classes. At least one long or short entry and one exit are required.
3. Every condition is structurally a condition; every numeric operand is a
   numeric node. A condition cannot be compared as a number and a number cannot
   stand where a condition is required.
4. Comparison operators are restricted to the six-name whitelist. Entry sides
   are `long`/`short`.
5. All constants, fractions and multiples are finite. `FixedFraction` is in
   `(0, 1]`; `AtrMultiple` is `> 0`.
6. Every backward window is an int `>= 1` and `<= max_lookback`. The holding
   limit is a forward dwell, not history, and is bounded separately in
   `[1, 500]`. **No node can reference a bar after the decision bar**: there is
   no zero/negative offset and no future node kind. A payload with an unknown
   `kind` such as `future` is rejected as an unknown node.
7. Expression depth is capped at 16 and total nodes (root, rules and
   expressions) at 64.
8. Unknown object types, arbitrary mappings and callables are rejected. Payload
   objects are loaded strictly: unknown *and* missing fields both fail. There is
   no `eval`, `exec`, dynamic import or source compilation anywhere in the
   package; a source-scan test enforces this.

`StrategySpec.from_dict(payload)` only accepts the canonical payload shape and
runs `validate` before returning. `to_dict()` is a lossless JSON-safe view.

## Canonical hash

`canonical_dict()` produces the payload with one spelling per strategy:

- keys sorted, compact separators, ASCII escapes, `allow_nan=False`;
- floats normalised (`1` and `1.0` are the same `Const`, `-0.0` becomes `0.0`);
- structural field order and dict insertion order cannot change the bytes;
- the spec version and every behavioural field are included.

`canonical_hash()` is the lowercase sha256 hex of `canonical_json(canonical_dict())`.
It is exactly the identity the discovery archive verifies, so two candidates with
the same hash are the same strategy across runs. Any threshold, feature, window,
rule or sizing change produces a different hash.

## Compile semantics

`compile_policy(spec)` validates and returns an immutable `Policy`.

`Policy.decide(row, state)` reads the current row plus `state.history` (previous
rows, oldest first). The per-bar precedence is fixed:

1. `NO_TRADE` — the `no_trade` condition is true. This wins over exits and
   entries; the execution layer decides whether to flatten.
2. `CLOSE` — an open position has reached `HoldingLimit.bars`.
3. `CLOSE` — any exit condition is true.
4. `NO_TRADE` — a long and a short entry both fired (`conflicting_entry`);
   neither is chosen arbitrarily.
5. `BUY` / `SELL` — a long / short entry fired.
6. `HOLD`.

`Decision` carries `action`, a stable `reason`, `version` (the DSL spec version)
and the `holding_limit`. `Policy.version` exposes the same version.

Missing features, non-finite values and insufficient history are **unknown**,
not false. Unknown propagates through `And`/`Or`/`Not` with three-valued logic
and never counts as a true condition, so `Not(Compare(missing, ...))` does not
fire on absent data. At the top of a rule only `True` triggers.

`evaluate_series(frame)` returns one action string per bar and is deliberately
**position-free**: it does not open or close positions, and therefore does not
apply the holding limit. `frame` is a `FeatureFrame`-like object (`X` plus
`columns`) or a mapping of column name to a 1-D sequence. Holding limits are
enforced by `decide` when the caller supplies `PolicyState.position` and
`bars_held`; the execution/backtest layer owns position accounting.

The function is pure: no I/O, no randomness, no module-level mutable state. The
same rows in the same order produce the same decisions.

## Generation and mutation

All operators take `numpy.random.Generator` for replayability:

- `random_spec(rng, max_lookback=20, max_depth=3, feature_pool=None)` draws a
  valid spec from the grammar.
- `mutate(spec, rng, ...)` rewrites a constant, feature, subcondition, window,
  rule list, holding limit or sizing. Failed structural attempts are retried;
  the valid parent is returned rather than an invalid child. The parent object is
  never modified.
- `crossover(a, b, rng)` combines rules per slot, uses the larger parent
  `max_lookback` so inherited windows stay valid, and returns `None` when the
  combined tree would exceed the structural caps.
- `StrategyFactory` exposes the same operations under the discovery protocol
  method names.

Seed determinism is guaranteed within a fixed NumPy version and code version;
the experiment record must store both, exactly as the discovery archive does.

## Complexity

```
nodes   = 1 (root) + sum(1 + expression nodes per rule) + holding/sizing nodes
depth   = deepest expression depth
penalty = ceil(max_lookback / 10)   # holding is not history and is not counted
complexity = nodes + depth + penalty
```

The value is a non-negative int and deterministic. It is a structural cost for
the discovery cap, not a quality score. A large search that finds one good
backtest is still a search result, not evidence of durability; the evaluation
slice must apply walk-forward splits, untouched out-of-sample data, transaction
cost stress and multiple-testing accounting.

## Safety boundary

- A strategy cannot execute code, import a module, access the filesystem,
  inspect process state, call a function, or name an arbitrary attribute.
- Only the exact node dataclasses above are accepted. Payload loaders reject
  unknown node kinds and unknown/missing fields.
- No node can read past the decision bar. Every offset is a positive backward
  window checked against `max_lookback`.
- Missing/NaN inputs cannot silently satisfy a rule.
- Node, depth and lookback caps bound memory and runtime per candidate.
- The package has no I/O and no global mutable state in validation or
  interpretation, so a candidate cannot change the environment for the next one.

## Known limitations

- This slice does not evaluate profitability, costs or risk. It provides the
  representation and interpreter only; simulator and discovery wiring come
  later.
- The DSL cannot express arithmetic combinations, regime state machines or
  multi-symbol conditions beyond the feature columns. That is intentional until
  an experiment justifies expanding the grammar and its caps.
- `evaluate_series` is a signal series, not a portfolio simulation. Holding
  limits and position accounting require the caller's `PolicyState`.
- Cross detection compares the two endpoints (`now` and `window` bars ago); it
  does not test every intermediate bar.
- Equality on floats (`==`, `!=`) is exact, as written. Feature values are the
  normalised matrix values, not display-rounded prices.
- Determinism is tied to the NumPy generator and the code version. A generator
  upgrade or a grammar change requires a new `spec_version`/lineage record.
