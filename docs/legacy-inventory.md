# Legacy strategy path inventory (master-prompt §37)

Status: analysis report. No production code was changed.
Branch: `feat/legacyaudit` (base `main` @ `6113197`).
Date: 2026-11-24.

This document inventories every consumer of the five legacy families named in
§37 and `AGENTS.md` §Legacy classification, verifies their runtime role in code,
and proposes the migration order. Every claim carries a `file:line` citation.
Items that could not be verified from the repository are marked **[UNVERIFIED]**
in §7.

---

## 1. Verdicts

| Question | Verdict | Evidence |
|---|---|---|
| What is the legacy flow path's current runtime role? | The panel/API simulator replays a promoted `PolicyCard` + `FlowMomentumRule` on every `/api/sim/start`; without a card it replays a five-bar M1 momentum rule. It is not imported by discovery, the DSL, the orchestrator or the autonomous paper path. | `backend/src/scalping/api/runner.py:429-453`, `:655-674`; `backend/src/scalping/api/app.py:128`; boundary proof in §4.4 |
| Can the legacy path auto-promote itself? | **Yes**, through two independent write paths: the operator-run script (cron/launchd per `docs/live-strategy.md:44-56`) and an HTTP endpoint that the panel button calls with `dry_run=false`. Promotion is gated only by holdout trade count, positive expectancy and beating the incumbent. No human approval, no DSL evidence; the script appends `runs/strategy_history.jsonl` and the HTTP path leaves no promotion record beyond the card file itself. | `backend/src/scalping/strategy/calibrate.py:303-314`; `backend/scripts/nightly_calibrate.py:60-64`; `backend/src/scalping/api/routes.py:501-505`; `frontend/src/hooks/use-strategy-card.ts:28-33` |
| Does a missing DSL champion silently fall back to legacy? | **Autonomous paper path: no** — it records auditable `NO TRADE`. **UI/API panel: yes** — it silently replays the legacy M1 momentum rule, and it also silently drops a card it cannot build while the card endpoint still reports `active: true`. | Fail-closed: `backend/src/scalping/orchestrator/handlers.py:937-964`; fallback: `backend/src/scalping/api/runner.py:434-444`; misreport: `backend/src/scalping/api/runner.py:379-389` |
| Is any legacy family a dependency of the DSL/discovery engine? | **No.** A ripgrep for the five legacy modules under `discovery/`, `orchestrator/`, `paper/`, `strategy/dsl/`, `registry/` and `knowledge/` returns zero import edges. | §4.4, reproduction command in §8 |

The gap that matters for §37 is therefore not "legacy is load-bearing for the
new engine" — it is not — but "the user-facing panel is still a legacy runtime,
it can promote itself with one click, and it hides that fact."

---

## 2. Classification contract

The master prompt defines the target treatment:

- `flow_momentum` / `FlowMomentumRule` / `PolicyCard` / `strategy_card.json` /
  `nightly_calibrate.py`: legacy incumbent and historical benchmark. Preserve
  for reproducibility; do not seed or constrain DSL search; retire the
  default/fallback promotion path only after the DSL champion passes acceptance
  and migration tests (`docs/requirements/master-prompt.md:654`).
- `strategy/signals.py` five families + `strategy/bandit.py` LinUCB: legacy
  finite strategy-menu selector; baseline/ablation only, do not expand.
  (`docs/requirements/master-prompt.md:655`).
- PPO/behaviour cloning and `nightly_rl.py`: experimental learning baseline; not
  the autonomous strategy engine; cannot promote itself without the same
  evidence gates (`docs/requirements/master-prompt.md:656`).
- `hold`, `ret_5` and the incumbent flow rule: baseline suite only
  (`docs/requirements/master-prompt.md:657`).
- No deletion merely because it is old: inventory imports, API/UI paths, tests
  and stored artifacts first, then migrate behind a versioned interface, prove
  parity, and remove only in a separate reviewed change
  (`docs/requirements/master-prompt.md:659-664`).

`AGENTS.md:97-98` restates the same rule ("Keep legacy code until all consumers
are inventoried and migration/parity tests pass"), and `AGENTS.md:100-110` is the
short classification table.

---

## 3. Family inventory

### 3.1 `flow_momentum` / `FlowMomentumRule` / `strategy/flow.py`

Definition:

- `backend/src/scalping/strategy/rule.py:31` — `FlowMomentumRule` dataclass.
- `backend/src/scalping/strategy/rule.py:87-108` — `build(frame, flow, threshold, signal)`.
- `backend/src/scalping/strategy/rule.py:57-65` — `act()`; `:67-85` — `blocked_reason()` (`no_flow`).
- `backend/src/scalping/strategy/flow.py:15-34` — `flow_series()`.
- `backend/src/scalping/strategy/flow.py:37-45` — `flow_thresholds()` (causal trailing quantile, shifted one bar).
- `backend/src/scalping/strategy/card.py:40` — `STRATEGY = "flow_momentum"`.

Production consumers:

| Location | Role |
|---|---|
| `backend/src/scalping/api/runner.py:65` | imports `FlowMomentumRule` |
| `backend/src/scalping/api/runner.py:64` | imports `flow_series`, `flow_thresholds` |
| `backend/src/scalping/api/runner.py:207` | holds `self._rule: FlowMomentumRule \| None` |
| `backend/src/scalping/api/runner.py:422-426` | builds flow series, threshold, `sig_momentum`, then the rule |
| `backend/src/scalping/api/runner.py:466-467`, `:530-531` | resets the rule on start/reset |
| `backend/src/scalping/api/runner.py:628-629` | attaches `blocked_reason` to the decision message |
| `backend/src/scalping/api/runner.py:655-657` | rule is the action source when present |
| `backend/src/scalping/api/routes.py:70`, `:484` | calibration route builds the flow series |
| `backend/src/scalping/strategy/calibrate.py:39`, `:205`, `:266` | threshold construction for calibration |
| `backend/src/scalping/strategy/calibrate.py:164-169` | momentum labels joined with `sig_momentum` |

Scripts:

| Location | Role |
|---|---|
| `backend/scripts/nightly_calibrate.py:36`, `:46` | reads the flow side table |
| `backend/scripts/train_flow.py:30`, `:84-85` | flow series/threshold for the RL experiment |
| `backend/scripts/train_flow.py:32`, `:91-94` | `FlowMomentumRule` is the behaviour-cloning teacher |
| `backend/scripts/forward_test.py:34-35`, `:105-113` | rule replay for the forward test |
| `backend/scripts/forward_test.py:175-176` | threshold + `sig_momentum` |
| `backend/scripts/nightly_rl.py:40-41`, `:60-66` | rule replay as the comparison arm |
| `backend/scripts/nightly_rl.py:166`, `:181-182` | flow/threshold/signal for that replay |

API routes (only indirect):

- `GET /api/strategy/card` — `backend/src/scalping/api/routes.py:457-460`.
- `POST /api/strategy/calibrate` — `backend/src/scalping/api/routes.py:463-506`.
- `POST /api/sim/start` → `SimulationRunner.start` → `_load_features` (card/rule) —
  `backend/src/scalping/api/routes.py:429-447`; `backend/src/scalping/api/runner.py:455-488`.

UI:

- Home simulator (`/`): `frontend/src/app/page.tsx:65` (`useStrategyCard`),
  `:79-88` (card timeframe drives the chart), `:239-243` ("Seçili strateji" stat
  prints `sim.lastDecision.strategy`).
- `StrategyCardPanel`: `frontend/src/components/strategy-card.tsx:70`, `:80`,
  `:146`.
- `GET /api/strategy/card` client: `frontend/src/lib/api.ts:195-197`;
  `frontend/src/hooks/use-strategy-card.ts:20-23`.

Tests:

| Location | Asserts |
|---|---|
| `backend/tests/test_strategy_card.py:23` | imports `FlowMomentumRule` |
| `backend/tests/test_strategy_card.py:48-58` | no extreme flow ⇒ hold + `no_flow` |
| `backend/tests/test_strategy_card.py:60-71` | entry follows the signal sign |
| `backend/tests/test_strategy_card.py:201-234` | rule replays in the simulator and trades are labelled `flow_momentum` |
| `backend/tests/test_api_strategy.py:39-52` | builds a `PolicyCard` for M1 |
| `backend/tests/test_api_strategy.py:101-115` | runner starts on the card timeframe |
| `backend/tests/test_api_strategy.py:118-127` | **runner falls back to M1 replay** when the card frame is missing |

Config / stored artifacts:

| Artifact | Location | Notes |
|---|---|---|
| Tracked seed card | `backend/config/strategy_card.json:21` (`"strategy": "flow_momentum"`) | copied to `runs/` per `README.md:112` |
| Historical card | `docs/cards/strategy_card_2026-11-22.json:21` | same payload as the seed |
| Runtime card | `runs/strategy_card.json` (default: `backend/src/scalping/api/app.py:128`; script default: `backend/scripts/nightly_calibrate.py:87`) | gitignored (`.gitignore:11-12`) |
| Calibration history | `runs/strategy_history.jsonl` (`backend/scripts/nightly_calibrate.py:88`) | gitignored |
| Snapshot history | `docs/cards/strategy_history_2026-11-22.jsonl` | 2 lines, tracked |
| Flow side table | `<flow_root>/<SYMBOL>/M1/*.parquet` (`backend/src/scalping/strategy/flow.py:20-23`) | gitignored data |
| Forward history | `runs/forward_history.jsonl` (`backend/src/scalping/api/runner.py:356`) | read by the card endpoint |
| Ledger label | `paper.sqlite` `sessions.strategy` (`backend/src/scalping/paper/ledger.py:79`) | stores the string, no classification |

Targets: `Makefile:26-28` (`serve-live`), `Makefile:30-32` (`train-bc`),
`Makefile:42-44` (`forward-test`), `Makefile:46-48` (`rl-lab`),
`Makefile:50-52` (`calibrate`).

### 3.2 `PolicyCard` / `strategy_card.json` / `calibrate.py` / `nightly_calibrate.py`

Definition:

- `backend/src/scalping/strategy/card.py:37-40` — `CARD_VERSION = 1`, `STRATEGY`.
- `backend/src/scalping/strategy/card.py:45-66` — `RuleParams` (validates timeframe, quantile, window, SL/TP, hold).
- `backend/src/scalping/strategy/card.py:69-92` — `RuleMetrics`.
- `backend/src/scalping/strategy/card.py:96-163` — `PolicyCard` (immutable; `from_dict` rejects unknown version/strategy at `:125-128`).
- `backend/src/scalping/strategy/card.py:166-189` — `frame_digest`.
- `backend/src/scalping/strategy/card.py:192-198` — atomic `save_card`; `:201-215` — `load_card` (returns `None` on absence, raises on malformed).
- `backend/src/scalping/strategy/calibrate.py:222-323` — `calibrate()`.
- `backend/src/scalping/strategy/calibrate.py:53-63` — fixed grid: 3 risk profiles × 3 quantiles × 3 windows = 27.
- `backend/src/scalping/strategy/calibrate.py:303-314` — promotion gate.
- `backend/src/scalping/strategy/calibrate.py:326-342` — `card_from_report()`.

Consumers:

| Location | Role |
|---|---|
| `backend/src/scalping/api/runner.py:63` | imports `PolicyCard`, `load_card` |
| `backend/src/scalping/api/runner.py:206` | holds `self._card` |
| `backend/src/scalping/api/runner.py:246-249` | `card_params()` reloads from disk |
| `backend/src/scalping/api/runner.py:379-389` | `strategy_card()` status payload |
| `backend/src/scalping/api/runner.py:391-427` | applies card risk settings, reads its timeframe, builds the rule |
| `backend/src/scalping/api/runner.py:430-439` | reloads the card on every start |
| `backend/src/scalping/api/routes.py:486-506` | calibration endpoint; writes the card when promoted and not dry-run |
| `backend/src/scalping/api/app.py:128` | injects `runs/strategy_card.json` into the runner |
| `backend/scripts/nightly_calibrate.py:41-78` | full nightly cycle |
| `backend/scripts/nightly_calibrate.py:60-64` | `save_card` when promoted |
| `backend/scripts/forward_test.py:33`, `:156` | reads the card for the forward test |
| `backend/scripts/nightly_rl.py:39`, `:167-170` | reads the card as the comparison baseline; refuses to run without one |

API routes: `GET /api/strategy/card` (`routes.py:457-460`), `POST
/api/strategy/calibrate` (`routes.py:463-506`).

UI: `frontend/src/components/strategy-card.tsx:63-157` renders the card, holdout,
forward result and the "Yeniden kalibre et" button; typed by
`frontend/src/lib/types.ts:279-325`.

Tests:

| Location | Asserts |
|---|---|
| `backend/tests/test_strategy_card.py:73-86` | card round-trip |
| `backend/tests/test_strategy_card.py:88-103` | unknown version rejected |
| `backend/tests/test_strategy_card.py:105-125` | digest tracks data |
| `backend/tests/test_strategy_card.py:128-150` | positive holdout promotes |
| `backend/tests/test_strategy_card.py:152-173` | negative holdout rejected |
| `backend/tests/test_strategy_card.py:175-198` | better incumbent rejected |
| `backend/tests/test_api_strategy.py:55-59` | no card ⇒ `active: false` |
| `backend/tests/test_api_strategy.py:61-71` | card read from disk |
| `backend/tests/test_api_strategy.py:73-77` | calibrate needs a flow store (409) |
| `backend/tests/test_api_strategy.py:79-99` | `dry_run` writes nothing |
| `backend/tests/test_api_strategy.py:130-150` | forward result attached to the card |
| `backend/tests/test_paper_ledger.py:98`, `:152` | ledger stores a `flow_momentum` strategy label |
| `backend/tests/test_api_paper.py:22` | paper session body uses `"strategy": "flow_momentum"` |

Artifacts: see §3.1 table. The only writers are
`backend/scripts/nightly_calibrate.py:61` and
`backend/src/scalping/api/routes.py:504` (confirmed by grepping `save_card`).

### 3.3 `strategy/signals.py` five fixed families

Definition:

- `backend/src/scalping/strategy/signals.py:70-78` — `Strategy` enum (`momentum`, `reversal`, `sweep`, `fvg`, `trend`).
- `backend/src/scalping/strategy/signals.py:80-88` — frozen `ALL_STRATEGIES` order; the bandit arm index is a position in this tuple.
- `backend/src/scalping/strategy/signals.py:250-256` — rule map; `:148-247` are the five rule builders.
- `backend/src/scalping/strategy/signals.py:338-359` — `strategy_signals`.
- `backend/src/scalping/strategy/signals.py:362-378` — `active_strategy`.
- `backend/src/scalping/strategy/__init__.py:10-27` — package re-exports.

Production consumers:

| Location | Which family / role |
|---|---|
| `backend/src/scalping/api/runner.py:66`, `:424` | `sig_momentum` only — the legacy rule gate |
| `backend/src/scalping/strategy/calibrate.py:41`, `:168` | `sig_momentum` only — calibration gate |
| `backend/src/scalping/strategy/labeling.py:60` | all five — counterfactual labels |
| `backend/src/scalping/strategy/labeling.py:201-218` | per-arm signal/outcome tables |
| `backend/src/scalping/strategy/labeling.py:518-553` | `strength_matrix` |
| `backend/src/scalping/strategy/labeling.py:555-578` | `signal_matrix` |
| `backend/src/scalping/strategy/labeling.py:580-643` | `context_matrix` |
| `backend/src/scalping/env/simulator.py:78`, `:765-769` | validates an arm index and labels the position with the family name |
| `backend/src/scalping/env/simulator.py:101` | `_STRATEGY_CONTEXT = len(ALL_STRATEGIES) + 1` |
| `backend/src/scalping/env/trading_env.py:29-40` | optional +6 strategy context in the observation |
| `backend/src/scalping/api/train_runner.py:67`, `:252-258` | maps bandit weights back to family names |

Scripts: `backend/scripts/train_flow.py:33`, `:93`; `backend/scripts/nightly_rl.py:42`,
`:182`; `backend/scripts/forward_test.py:36`, `:176` (all `sig_momentum` only).

API/UI: the five-family menu surfaces only as bandit weights —
`GET /api/runs/{run_id}` (`routes.py:368-381`), schema
`backend/src/scalping/api/schemas.py:345-363`, UI
`frontend/src/app/training/page.tsx:241-263`, arm order
`frontend/src/lib/runs.ts:18`.

Tests:

| Location | Covers |
|---|---|
| `backend/tests/test_signals.py:74-98` | frozen order, enum suffixes, column pairs, value range |
| `backend/tests/test_signals.py:117-137` | causality (truncated prefix identical) |
| `backend/tests/test_signals.py:139-182` | reversal, fvg, trend, abstain ⇒ zero strength |
| `backend/tests/test_signals.py:184-214` | empty schema, missing columns |
| `backend/tests/test_signals.py:217-315` | NaN/Inf/leading-NaN handling |
| `backend/tests/test_labeling.py:72-107` | context shape/finiteness/neutral success |
| `backend/tests/test_labeling.py:375-409` | `bandit_training_data` pairing |
| `backend/tests/test_env_strategy.py:56-175` | +6 obs, one-hot, strength, per-bar selector call |
| `backend/tests/test_env_strategy.py:228-283` | gate blocks/allows by arm signal |
| `backend/tests/test_train_smoke.py:39`, `:397`, `:504` | obs width includes `len(ALL_STRATEGIES)` |

### 3.4 `strategy/bandit.py` LinUCB

Definition:

- `backend/src/scalping/strategy/bandit.py:96-136` — `LinUCB` (ridge + bias column).
- `backend/src/scalping/strategy/bandit.py:201-208` — `select`/`scores`.
- `backend/src/scalping/strategy/bandit.py:222-230` — `freeze()` (`alpha = 0`).
- `backend/src/scalping/strategy/bandit.py:259-324` — `fit_offline`.
- `backend/src/scalping/strategy/bandit.py:364-470` — save/load/load_into (atomic, shape-checked).
- Context dimension: `backend/src/scalping/strategy/labeling.py:131`.

Production consumers:

| Location | Role |
|---|---|
| `backend/src/scalping/api/train_runner.py:65-66` | imports; API training fits a bandit |
| `backend/src/scalping/api/train_runner.py:352-358` | fits + freezes + (later) saves `bandit.npz` |
| `backend/src/scalping/api/train_runner.py:410` | records `use_strategy` in run config |
| `backend/src/scalping/api/train_runner.py:235-258` | reads per-arm weights for `GET /api/runs/{id}` |
| `backend/src/scalping/api/train_runner.py:606-608` | attaches bandit weights to run detail |
| `backend/src/scalping/api/train_runner.py:619-628` | reloads the frozen bandit for backtest/evaluate |
| `backend/src/scalping/rl/train.py:73`, `:204`, `:547` | optional selector in `TrainConfig`/`train` |
| `backend/src/scalping/rl/train.py:643` | records whether a bandit was used |
| `backend/src/scalping/backtest/engine.py:163-168` | `strategy_selector` adds the +6 context |
| `backend/src/scalping/backtest/evaluate.py:223`, `:269` | walk-forward forwards the selector |
| `backend/src/scalping/env/simulator.py:232-234`, `:755-769` | calls `select` and validates the arm |
| `backend/src/scalping/env/trading_env.py:31-32` | accepts `LinUCB` by duck type |

Default-on in the API: `backend/src/scalping/api/schemas.py:309`
(`use_strategy: bool = True`) and `backend/src/scalping/api/train_runner.py:288`.
That is, every `/api/train` request that does not opt out fits a LinUCB and feeds
its arm context to PPO.

Script: `backend/scripts/train_flow.py:29`, `:96-98`.

API/UI: `POST /api/train` (`routes.py:330-360`), `GET /api/runs/{run_id}`
(`routes.py:368-381`), `POST /api/backtest` (`routes.py:399-411`), `POST
/api/evaluate` (`routes.py:413-427`); UI table
`frontend/src/app/training/page.tsx:241-263`.

Tests:

| Location | Covers |
|---|---|
| `backend/tests/test_bandit.py:23-142` | selection, learning, offline fit, determinism, shapes |
| `backend/tests/test_bandit.py:146-232` | persistence, suffix, validation, hyperparameters |
| `backend/tests/test_bandit.py:251-330` | restore, "weights are not the model", atomic save |
| `backend/tests/test_bandit.py:355-370` | freeze ⇒ argmax of predictions; digest |
| `backend/tests/test_train_smoke.py:331-366` | run id covers the selector; stale `bandit.npz` removed |
| `backend/tests/test_train_smoke.py:377-406` | frozen bandit saved; **PPO must not train it** |
| `backend/tests/test_train_smoke.py:488-521` | selector adds the context and saves the bandit |
| `backend/tests/test_api_train.py:446-460` | detail returns weights in strategy order |
| `backend/tests/test_api_train.py:474-491` | API training fits a bandit by default |
| `backend/tests/test_env_strategy.py:56-175` | env contract |
| `backend/tests/test_labeling.py:375-409` | offline batch shape |

Stored artifact: `runs/<run_id>/bandit.npz`
(`backend/src/scalping/rl/registry.py:50`), gitignored.

### 3.5 PPO / behaviour cloning / `nightly_rl.py`

Definition:

- `backend/src/scalping/rl/train.py:44` — `from stable_baselines3 import PPO`.
- `backend/src/scalping/rl/train.py:97-160` — `TrainConfig` (`bc_steps`, `bc_epochs`, `bc_only`, …).
- `backend/src/scalping/rl/train.py:402-485` — `_behaviour_clone`.
- `backend/src/scalping/rl/train.py:543-820` — `train()`.
- `backend/src/scalping/rl/policy.py:1-40`, `:128-132` — loads `model.zip` + `vecnormalize.pkl`.
- `backend/src/scalping/rl/registry.py:44-50` — artifact names.
- `backend/src/scalping/rl/callbacks.py:287` — checkpoint writes `model.zip`.
- `backend/src/scalping/env/masks.py:5-7` — masked PPO note (training remains plain PPO per `docs/task-board.md:381-382`).
- `backend/scripts/nightly_rl.py:80-134` — BC+PPO subprocess per split.
- `backend/scripts/nightly_rl.py:190-210` — `beats_rule` and history write.

Consumers:

| Location | Role |
|---|---|
| `backend/src/scalping/api/train_runner.py:46`, `:689`, `:782` | HTTP training + backtest/evaluate load the policy |
| `backend/src/scalping/api/routes.py:330-427` | `/api/train`, `/api/backtest`, `/api/evaluate` |
| `backend/scripts/train_flow.py:27`, `:132` | CLI experiment loads the trained policy |
| `backend/scripts/nightly_rl.py:82-110` | invokes `scripts.train_flow` |
| `backend/src/scalping/backtest/evaluate.py:5` | evaluates the frozen bandit/PPO pair |

`nightly_rl.py` never writes a card: it only calls `load_card`
(`backend/scripts/nightly_rl.py:167`) and appends `runs/rl_history.jsonl`
(`:207-209`). Promotion is explicitly manual (`:11-13`).

API/UI: `/api/train`, `/api/runs`, `/api/backtest`, `/api/evaluate`; UI
`frontend/src/app/training/page.tsx` (PPO curves + bandit table).

Tests:

| Location | Covers |
|---|---|
| `backend/tests/test_train_smoke.py:112-135` | training produces a loadable model |
| `backend/tests/test_train_smoke.py:193-257` | `load_policy` determinism, fresh process, VecNormalize required |
| `backend/tests/test_train_smoke.py:258-330` | metrics/config/manifest keys |
| `backend/tests/test_train_smoke.py:408-487` | stop/pause interruptions leave a loadable run |
| `backend/tests/test_train_smoke.py:604-660` | BC config validation, cloning recorded, BC-only clone |
| `backend/tests/test_api_train.py:78-210` | `/api/train` lifecycle and control |
| `backend/tests/test_api_train.py:252-327` | `/api/backtest` uses the stored policy, rejects mismatches |
| `backend/tests/test_ml_nn_comparison.py` and `test_nn_*` | unrelated neural baselines, not PPO |

Artifacts: `runs/<run_id>/model.zip`, `vecnormalize.pkl`, `config.json`,
`metrics.json`, `metrics.jsonl`, `manifest.json`
(`backend/src/scalping/rl/registry.py:44-50`); `runs/rl_history.jsonl`
(`backend/scripts/nightly_rl.py:220`). All gitignored.

Makefile: `Makefile:30-32` (`train-bc`), `Makefile:46-48` (`rl-lab`).

### 3.6 Baselines `hold` / `ret_5` (for completeness)

- `backend/src/scalping/discovery/run.py:89-96` — `BASELINES = ("hold", "ret_5")`, label `rule:ret_5-baseline`.
- `backend/src/scalping/discovery/run.py:406-427` — baseline policies.
- `backend/src/scalping/cli.py:88-95`, `:773`, `:1326`, `:1344`, `:1380` — CLI policies.
- `backend/src/scalping/nn/evaluate.py:266-268` — `ret_5` rule comparison.
- `backend/src/scalping/bench/suite.py:610-628` — `ret_5` walk-forward timing.
- Test: `backend/tests/test_discovery_run.py:234` (`names == ["hold", "ret_5"]`).

These are already labelled as baselines in the run report, so they need no new
"legacy" wording beyond §6.

---

## 4. Runtime role, auto-promotion, fallback

### 4.1 The legacy flow path is the panel runtime

`SimulationRunner.start` calls `_load_features` on every start
(`backend/src/scalping/api/runner.py:465`). `_load_features` does the following,
in order:

1. `self._card = load_card(self._card_path)` (`:430`).
2. If a card exists, try `_load_card_features` (`:431-433`). That reads the
   card's timeframe (`:406`), requires ≥ `MIN_BARS` rows (`:416-418`), requires a
   configured flow root (`:419-421`), and builds `FlowMomentumRule` (`:422-426`).
3. If the card cannot be built (`OSError`, `ValueError`, `KeyError`), it logs a
   warning and falls through (`:434-436`).
4. Fallback: `self._rule = None`, read M1 bars, and use the five-bar momentum
   rule in `_action` (`:440-453`, `:655-674`).
5. Decision messages carry `sim.strategy` (`:632-635`); the strategy label
   constant is `STRATEGY = "momentum"` (`:92`, comment at `:35-38`).

The card is reloaded, not cached, precisely so a nightly promotion lands without
a restart (`:205-206`; module docstring `backend/src/scalping/strategy/card.py:5-6`).

Nothing in the autonomous loop calls this runner. `paper.advance` uses
`AutonomousPaperRunner(ledger, champion.spec, ...)`
(`backend/src/scalping/orchestrator/handlers.py:972-984`), and the discovery
engine writes DSL candidates only.

### 4.2 Auto-promotion: yes, twice, with no human gate

The gate itself (`backend/src/scalping/strategy/calibrate.py:301-314`):

- reject if `best.holdout.trades < min_holdout_trades` (default 30);
- reject if `best.holdout.expectancy_r <= 0`;
- reject if an incumbent exists and the candidate does not beat it on the same
  holdout;
- otherwise `promoted = True` with reason "holdout positive and at least as good
  as the incumbent".

There is no walk-forward, cost-stress, regime, stability, provenance or
promotion-record requirement, and the holdout is reused across runs
(`backend/src/scalping/strategy/calibrate.py:16-20`; the card note says so at
`backend/scripts/nightly_calibrate.py:38`).

Write path 1 — script:

- `backend/scripts/nightly_calibrate.py:49-59` calibrates against the incumbent
  card params.
- `:60-64` writes the card when `report.promoted`.
- `:99-119` loops forever with `--loop`.
- Default card path is `runs/strategy_card.json` (`:87`).

Write path 2 — HTTP, reachable from the panel with one click:

- `frontend/src/components/strategy-card.tsx:84-93` renders the button; the
  handler comes from `frontend/src/hooks/use-strategy-card.ts:28-40`.
- `frontend/src/hooks/use-strategy-card.ts:32` calls `api.calibrateStrategy({})`.
- `frontend/src/lib/api.ts:199-202` posts to `/api/strategy/calibrate`.
- `backend/src/scalping/api/schemas.py:550` — `dry_run: bool = False`.
- `backend/src/scalping/api/routes.py:501-505` — writes the card when promoted
  and not dry-run.

So a single button press can replace the incumbent card, and the next
`/api/sim/start` replays it. `dry_run` is never sent by the panel, so the panel
always takes the write path.

The autonomous path does not read this card: `paper.advance` reads the discovery
archive's champion only (`backend/src/scalping/orchestrator/handlers.py:937-964`).

### 4.3 Fallback matrix

| Situation | Autonomous `paper.advance` | Panel `/api/sim/start` | `GET /api/strategy/card` |
|---|---|---|---|
| No discovery archive | `NO TRADE`, reason_class `no_champion` (`handlers.py:925-935`) | not consulted | not consulted |
| Archive, no champion | `NO TRADE` (`handlers.py:937-946`) | not consulted | not consulted |
| Champion without evidence policy | `NO TRADE`, `no_evidence_champion` (`handlers.py:947-955`) | not consulted | not consulted |
| Invalid/tampered champion spec | `NO TRADE`, `invalid_champion` (`handlers.py:956-964`) | not consulted | not consulted |
| No `strategy_card.json` | n/a | **legacy 5-bar M1 momentum** (`runner.py:440-453`, `:655-674`) | `{"active": false, "strategy": "momentum", ...}` (`runner.py:381-388`) |
| Card exists, frame/flow missing | n/a | **legacy 5-bar M1 momentum** (`runner.py:434-444`) | **`{"active": true, ...}`** (`runner.py:389`) — misreport |
| Card exists and builds | n/a | flow-momentum rule (`runner.py:437-439`) | `active: true` |

The last row's `active: true` is correct. The second-to-last row is the
truthfulness defect: the runtime silently switched back to legacy but the status
endpoint still reports the card as active because it only checks file existence.
`backend/tests/test_api_strategy.py:118-127` locks that behaviour in.

The autonomous path is consistent with the master prompt:
`docs/requirements/master-prompt.md:597`
("No champion means NO TRADE, not a silent legacy-strategy fallback") and
`docs/autonomous-system-audit-2026-11-24.md:258-259`
("no silent flow_momentum fallback"), implemented by
`backend/src/scalping/discovery/champion.py:1-30`
("It does not fall back") and `backend/src/scalping/orchestrator/handlers.py:878-906`.

### 4.4 Import boundary proof

A ripgrep for `strategy.signals|strategy.rule|strategy.card|strategy.flow|strategy.bandit|strategy.calibrate`
under `backend/src/scalping/{discovery,orchestrator,paper,strategy/dsl,registry,knowledge}`
returns no matches. Reproduce with §8. The only comment-level mention is
`backend/src/scalping/paper/continuous.py:39` ("No legacy strategy fallback"),
which is documentation, not an import.

---

## 5. Migration matrix

### 5.1 Decisions

| Legacy component | Decision | Target / interface | Blocks | Parity / regression tests |
|---|---|---|---|---|
| `strategy/rule.py`, `strategy/flow.py` (`FlowMomentumRule`) | **Wrap + keep-with-label** | Freeze as a benchmark adapter behind a versioned `LegacyRulePolicy`; never a default. Reuse the existing `policy` callable seam (`backend/src/scalping/paper/runner.py:63`) and `run_backtest(policy_name=...)` (`backend/src/scalping/backtest/engine.py:139`). | Removing the panel default | `test_strategy_card.py:48,60,201`; `test_api_strategy.py:101` |
| `PolicyCard` / `strategy_card.json` | **Keep-with-label, then read-only** | Treat as historical artifact + benchmark record. Add an explicit `classification: "legacy_baseline"` field to the status payload (`runner.py:379-389`) and the frontend type (`types.ts:311-325`). | Panel champion swap | `test_strategy_card.py:73,88`; `test_api_strategy.py:55,61,130`; `strategy-card.test.tsx:54,63` |
| `strategy/calibrate.py` + `nightly_calibrate.py` + `POST /api/strategy/calibrate` | **Retire auto-promotion** (separate PR) | Keep as a benchmark harness; make `dry_run=True` the default and require an explicit `--promote`/`dry_run=false` with a reasoned record, mirroring the discovery promotion audit. Do not call it the learning loop. | Panel card writes | `test_strategy_card.py:128,152,175`; `test_api_strategy.py:73,79` |
| `api/runner.py` card/rule path | **Replace** | `ChampionProvider` + DSL `AutonomousPaperRunner`; no champion ⇒ NO TRADE with a reason. Ship behind a config flag first, then flip. | §5.2 steps 2-4 | `test_discovery_champion.py:424,435,469`; `test_orchestrator_handlers.py:375,395,457`; new panel tests |
| `strategy/signals.py` five families | **Keep-with-label** | Baseline/ablation only. Do not add arms (`signals.py:80-88` is frozen). Label the UI table. | none | `test_signals.py:74,117,126`; `test_env_strategy.py:56` |
| `strategy/bandit.py` LinUCB | **Keep-with-label** | Frozen experimental selector; no new arms; keep `freeze()` semantics. Label the training page. | none | `test_bandit.py:355`; `test_train_smoke.py:377,488`; `test_api_train.py:474` |
| PPO/BC, `nightly_rl.py` | **Keep-with-label, isolated** | Experimental baseline; no privileged promotion; must not write the card (already true). | none | `test_train_smoke.py:604,610,632` |
| `hold` / `ret_5` | **Keep-with-label** | Already labelled `rule:ret_5-baseline` (`discovery/run.py:96`). | none | `test_discovery_run.py:234` |

### 5.2 Dependency order

1. **Freeze the contract (this document).** No code change.
2. **Truthful labelling, no behaviour change.** Add `classification` to
   `GET /api/strategy/card`, badge the panel, tag the bandit table, fix the
   `active: true` misreport when the card frame cannot be built
   (`runner.py:379-389`). Parity risk: none (labels only); UI snapshot tests
   change.
3. **Parity harness for the legacy rule.** Record a golden
   `(frame, flow, threshold, signal) -> actions` fixture and
   `run_backtest` metrics. This is the baseline that any future DSL champion is
   compared against. Depends on step 2 only for stable labels.
4. **Introduce a champion seam in the panel behind a flag.** `api/runner.py`
   asks `ChampionProvider` first; a missing/invalid champion produces a
   NO TRADE-style idle stream with a reason, not the M1 fallback. Keep the
   legacy path selectable by explicit configuration only. Depends on step 3 for
   comparison data.
5. **Flip the panel default** after steps 3-4 pass and the panel regression
   suite (start/pause/reset/episode_end + card/build failures) is green. This
   is the change that turns "missing DSL champion silently falls back" into
   "missing DSL champion is NO TRADE" for users.
6. **Retire automatic promotion.** Make `nightly_calibrate` and
   `POST /api/strategy/calibrate` non-writing by default; promotion becomes an
   explicit, audited action. Depends on step 5: once the panel no longer reads
   the card, the write path has no runtime consumer.
7. **Separate reviewed deletion.** After a retention window, delete only what
   step 3's benchmark adapter does not need. §37 forbids deleting in the same
   change as the migration.

### 5.3 Tests that need parity checks

Rule / card (must prove the legacy baseline is bit-stable before it is demoted):

- `backend/tests/test_strategy_card.py:48`, `:60`, `:201` — rule semantics and
  simulator parity.
- `backend/tests/test_strategy_card.py:73`, `:88`, `:105` — card schema and
  digest.
- `backend/tests/test_strategy_card.py:128`, `:152`, `:175` — calibration gate.
- `backend/tests/test_api_strategy.py:55`, `:61`, `:79`, `:101`, `:118`, `:130`
  — the endpoint and the fallback that step 4/5 replaces.

Signals / labels / bandit (must stay byte-identical if they remain baselines):

- `backend/tests/test_signals.py:74`, `:82`, `:117`, `:126` — frozen order,
  column contract, causality.
- `backend/tests/test_labeling.py:72`, `:79`, `:181`, `:375`, `:401` — context
  causality and batch pairing.
- `backend/tests/test_bandit.py:51`, `:103`, `:110`, `:355`, `:366` and the
  persistence tests `:64`, `:146`, `:157`, `:310`.
- `backend/tests/test_env_strategy.py:56`, `:75`, `:86`, `:155`.
- `backend/tests/test_train_smoke.py:331`, `:377`, `:394`, `:488`, `:512`,
  `:542`.
- `backend/tests/test_api_train.py:446`, `:474`.

Fail-closed invariants that the panel must inherit:

- `backend/tests/test_discovery_champion.py:424`, `:435`, `:469`.
- `backend/tests/test_orchestrator_handlers.py:375`, `:395`, `:457`.
- `backend/tests/test_paper_continuous.py:442` (continuous module contains no
  `flow_momentum`).

UI labels that will change in step 2:

- `frontend/src/components/strategy-card.test.tsx:54`, `:63` (asserts
  `"kart yok · M1 momentum"` and `"flow + momentum"`).
- `frontend/src/app/page.test.tsx:330-345` (card timeframe chart label).

---

## 6. Places where the UI/docs should say "legacy baseline" but do not

Positive controls first: `README.md:16` and `README.md:51` call it a legacy
incumbent, `STATUS.md:36` says "Legacy flow_momentum card", and
`docs/orchestrator.md:654` says "never falls back to the legacy `flow_momentum`
rule". Those are honest.

Gaps:

| # | Location | Current wording / behaviour | Problem | Suggested |
|---|---|---|---|---|
| 1 | `frontend/src/components/strategy-card.tsx:70` | heading `Canlı strateji` ("Live strategy") | presents the legacy card as the live production strategy | `Legacy baseline (flow + momentum)` |
| 2 | `frontend/src/components/strategy-card.tsx:80` | badge `flow + momentum` | no classification; looks like the intended champion | add `legacy baseline` to the badge |
| 3 | `frontend/src/components/strategy-card.tsx:146` | "panel doğrulanmamış M1 momentumunu oynatıyor" (plays an unvalidated M1 momentum) | says unvalidated but not that it is the legacy fallback path | `legacy fallback · M1 momentum (no DSL champion)` |
| 4 | `frontend/src/app/page.tsx:239-243` | stat label `Seçili strateji`, value is raw `momentum` / `flow_momentum` | no legacy qualifier on the main screen | `Seçili strateji (legacy)` when the value is a legacy family |
| 5 | `frontend/src/app/training/page.tsx:242` and `:255` | `Bandit ağırlıkları` / `Bandit strateji ağırlıkları` | LinUCB baseline shown as a neutral feature | `Legacy LinUCB baseline ağırlıkları` |
| 6 | `frontend/src/app/paper/page.tsx:578`, `:616`, `:826` | column/fact `Strateji` prints the raw stored string | `flow_momentum` sessions look like current strategies | render a `legacy baseline` chip for known legacy labels |
| 7 | `backend/src/scalping/api/runner.py:379-389` | status payload has no classification field, and reports `active: true` even after falling back | consumer cannot tell a card run from a fallback run | add `classification` and a `fallback`/`active_rule` field |
| 8 | `backend/src/scalping/api/runner.py:92` | `STRATEGY = "momentum"` is emitted on every decision message without qualifier | stream consumers see a bare strategy name | emit `legacy_baseline:momentum` or a separate `classification` field |
| 9 | `backend/src/scalping/api/schemas.py:569` | `PaperSessionCreate.strategy` accepts any string, default `manual` | no way to mark a session as legacy/DSL/baseline | validate against a known set, or add `strategy_kind` |
| 10 | `frontend/src/lib/types.ts:279-325` | `RuleParams` doc says "the promoted flow-momentum rule"; `StrategyCard` has no classification | the type itself reasserts "promoted" | add `classification?: "legacy_baseline" \| "dsl_champion"` |
| 11 | `backend/scripts/forward_test.py:113` | `policy_name="flow_momentum_forward"` | report label reads like a current policy | `legacy_baseline:flow_momentum_forward` |
| 12 | `backend/scripts/nightly_rl.py:66` | `policy_name="flow_momentum"` on the comparison arm | comparison baseline is unlabelled in `rl_history.jsonl` | `legacy_baseline:flow_momentum` |
| 13 | `docs/live-strategy.md:1-7` | title "Canlı strateji ve gece kalibrasyonu" and "canlı sistem kuralı çalıştırır" | reads as the recommended live system | mark it "legacy baseline and historical calibration" in the title/intro |
| 14 | `docs/architecture-audit.md:99` | "Live card \| flow_momentum, H1, ..." | same | "Legacy live card (baseline)" |
| 15 | `docs/paper-trading.md:151` | code example `strategy="flow_momentum"` | teaches the legacy label as normal usage | annotate it as a legacy baseline example, or use a DSL spec hash |

The docs table in `docs/autonomous-system-audit-2026-11-24.md:185-192` already
labels five of these correctly; the panel and the two API payloads are the
remaining user-facing gaps.

### 6.1 Status after issue #139

The user-facing gaps 1–10 and 13 are implemented on `fix/legacylabels`; the
analysis above is kept as the before-state.

| Gap | Now |
|---|---|
| 1, 2 | `frontend/src/components/strategy-card.tsx` heading is "Strateji kartı" with a visible `legacy baseline` badge (`legacy-badge.tsx`). The state chip says `legacy fallback` and maps `legacy_fallback: false` to `NO TRADE`. |
| 3 | The no-card hint names the legacy fallback and the NO TRADE case; it never calls the path validated. |
| 4 | `frontend/src/app/page.tsx` passes `hint="legacy baseline"` to the strategy stat when `isLegacyStrategy` matches (`lib/strategy-labels.ts`). |
| 5 | `frontend/src/app/training/page.tsx` badges the bandit heading and the table caption. |
| 6 | `frontend/src/app/paper/page.tsx` renders the badge for legacy session labels in both the table and the detail fact. |
| 7 | `SimulationRunner.strategy_card` (`backend/src/scalping/api/runner.py`) now returns `classification` (`legacy`/`fallback`), `validation: "legacy-heuristic"`, `active_rule`, `legacy_fallback` and a safe `fallback_reason`; a card that cannot be built reports `active: false` with `strategy_card_not_buildable`. |
| 8 | Decision messages carry `classification`; the disabled fallback adds `no_trade: true` and `reason: "legacy_fallback_disabled"`. |
| 9 | Not changed in #139: `PaperSessionCreate.strategy` still accepts any string. The UI labels known legacy values instead of validating them. |
| 10 | `frontend/src/lib/types.ts` documents the legacy rule and adds the classification/fallback fields to `StrategyCard`. |
| 13 | `docs/live-strategy.md` is retitled "Legacy baseline: …" and documents the classification contract, the `--no-legacy-fallback` flag and the audit log. |

Also in #139:

- `backend/src/scalping/strategy/card.py` writes `classification: "legacy"` into
  every saved card, so a stored artifact labels itself.
- `POST /api/strategy/calibrate` appends `runs/strategy_calibration_audit.jsonl`
  (`at`, `actor`, `dry_run`, `result`, `card_written`, `classification`) through
  `strategy.calibrate.record_calibration`; `backend/scripts/nightly_calibrate.py`
  calls the same helper.
- The 5-bar M1 fallback sits behind `SimulationRunner(allow_legacy_fallback=...)`
  / `serve --legacy-fallback|--no-legacy-fallback`, default on. Legacy modules
  are untouched; the flag only decides whether the fallback may trade.

Still open from this table: row 9 (session `strategy` validation), row 11/12
(script report labels), 14/15 (older architecture/paper docs). They are outside
the #139 scope.

---

## 7. Unverified items

- **Host scheduling.** The repository contains no cron unit, launchd plist or
  systemd timer for `nightly_calibrate.py`; only `Makefile:51-52` and
  `docs/live-strategy.md:44-56` tell a human to run it. Whether the target host
  actually runs it on a schedule is **[UNVERIFIED]**.
- **Runtime card contents.** `runs/strategy_card.json` is gitignored
  (`.gitignore:11-12`), so whether the live host currently has a card (and which
  vintage) is **[UNVERIFIED]**. The tracked seed is
  `backend/config/strategy_card.json` (digest `b970c5aa345d6952`).
- **Host data availability.** Whether the host store has H1 bars and a flow
  side table — the two conditions that decide whether the panel uses the card
  or the fallback (`runner.py:406-421`) — is **[UNVERIFIED]**.
- **Forward history.** `runs/forward_history.jsonl` is produced by
  `backend/scripts/update_forward.py` and read by `runner.py:352-377`; no copy is
  tracked. The tracked snapshot is
  `docs/cards/strategy_history_2026-11-22.jsonl` (2 lines), and its digest match
  with the card was not independently re-derived here. **[UNVERIFIED]**
- **Historical performance claims.** Numeric expectancies in the tracked card
  (`backend/config/strategy_card.json`) and README table are historical
  evidence, not revalidated in this audit. Any forward/paper performance claim
  remains **[UNVERIFIED]** for current data.
- **Browser-level label coverage.** `frontend/e2e/simulator.spec.ts` contains no
  strategy/card assertions (grep for `strategy|momentum|kart` returned nothing),
  so no end-to-end test would catch a mislabel. Confirmed absent, not unverified.

---

## 8. Reproduction

Run from the repository root:

```bash
# 1. Definition and every direct consumer of the legacy modules
rg -n "strategy\.(rule|flow|card|calibrate|signals|bandit)" backend/src backend/scripts backend/tests

# 2. Card writers (should be exactly two)
rg -n "save_card|card_from_report" backend/src backend/scripts

# 3. Import boundary: no legacy edge into discovery/orchestrator/paper/DSL
rg -n "strategy\.(signals|rule|card|flow|bandit|calibrate)" \
  backend/src/scalping/discovery backend/src/scalping/orchestrator \
  backend/src/scalping/paper backend/src/scalping/strategy/dsl \
  backend/src/scalping/registry backend/src/scalping/knowledge
# expected: no matches

# 4. Fallback path
sed -n '379,453p' backend/src/scalping/api/runner.py
sed -n '919,1035p' backend/src/scalping/orchestrator/handlers.py

# 5. Promotion gate
sed -n '297,323p' backend/src/scalping/strategy/calibrate.py
sed -n '480,510p' backend/src/scalping/api/routes.py

# 6. Scheduled writers (repository-level only)
rg -n "nightly_calibrate|nightly_rl" Makefile docs backend/scripts

# 7. Frontend labels
rg -n "Canlı strateji|flow \+ momentum|Bandit|Seçili strateji" frontend/src
```

---

## 9. Summary for the §37 review

- The legacy path is a **self-contained panel runtime + a self-promoting
  calibration loop**, not a dependency of the autonomous engine.
- The autonomous paper path already satisfies §28's fail-closed rule
  (`handlers.py:937-964`); the panel does not, and it can promote itself with a
  single button (`routes.py:501-505`).
- Migration can be incremental and low-risk: labels first, then a parity
  harness, then a champion seam behind a flag, then retire card writes, then a
  separate deletion PR.
- The highest-value single fix is the `active: true` misreport at
  `backend/src/scalping/api/runner.py:379-389`: it makes a silent legacy
  fallback look like a governed card run.
