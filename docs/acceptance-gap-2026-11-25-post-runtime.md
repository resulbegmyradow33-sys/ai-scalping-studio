# §36 acceptance gap — re-assessment after runtime promotion, provenance and paper auto-enqueue — 2026-11-25

Read-only re-assessment of the master-prompt §36 criteria at `main` base
`54eef5e` (`feat/accept3`, no commits ahead of base). This report changes no
code. It re-runs the acceptance evidence now that three runtime links are on
main:

- #220 `discovery promote` — runtime promotion with a one-shot owner-approved
  holdout;
- #216 provenance-aware evidence and fail-closed promotion;
- #218 `paper.advance` auto-enqueue for an evidence champion.

It is a diff against
[`acceptance-gap-2026-11-25.md`](acceptance-gap-2026-11-25.md) (base
`50fbca6`, 4 met / 7 partial / 2 not met).

**Verdict: not accepted. 4 met / 8 partial / 1 not met.** One criterion moved
status (#9 not met → partial). The three new runtime links exist, are tested and
were executed here, but the real-data happy path is still broken by two concrete
wiring defects and by the H1 gap policy. No real evidence champion exists, the
real holdout was not spent, and no governed real paper session ran.

## Scorecard diff

| # | §36 criterion | Before (`50fbca6`) | Now (`54eef5e`) | What moved | What is still missing for acceptance |
|---|---|---|---|---|---|
| 1 | Unseen data import, provenance/quality, leakage-safe dataset | partial | **partial** | `source`/`synthetic` now exist in `DatasetManifest`, `CandidateEvidence` and `LockedOOSEvidence` (schema v4), and `PROMOTION_V1` rejects an unknown `synthetic` flag or `synthetic=true` | `discovery evidence` writes `source:null, synthetic:null` for a real store with a valid `_meta` sidecar; no recorded import of a previously unseen dataset |
| 2 | Generate candidates without owner rules | met | **met** | — | — |
| 3 | Reproduce from data hash, code version, config, seed | met | **met** | Re-verified on real H1: `discovery reproduce` exit 0, checksum `c7e6c6ba…` | — |
| 4 | Reject overfit candidates on unseen chronological data + cost stress | partial | **partial** | Runtime holdout evaluator is now reachable through `discovery promote`; internal/synthetic holdout rejections recorded | No locked-OOS evaluation on real data; the real candidate is rejected before any spend and no deliberately overfit candidate experiment is recorded |
| 5 | NO TRADE / no candidate beats the champion | partial | **partial** | Champion comparison now runs in the shipped command (observed `champion_same_spec`, `champion_improvement` on a second attempt) | No real champion; champion comparison has only run against a fixture-tape promotion |
| 6 | PDF + dataset + source refs + traceable hypotheses | partial | **partial** | — | PDF extraction remains unsupported (`knowledge/ingest.py`), no recorded PDF run |
| 7 | Detect new data / matured outcomes, auto-queue research | partial | **partial** | New bars now auto-enqueue `paper.advance` for an evidence champion (this is execution, not research) | A matured batch still yields a proposal with `executed:false`; `ProposalQueueTriggerSource` is not wired into the daemon CLI; a data trigger still enqueues only `dataset.build` |
| 8 | Train, compare champion, promote/reject/rollback, audit | partial | **partial** | `discovery promote` is now the runtime call site; accept, reject, refusal, one-shot spend and audit rows were executed; rollback tests still pass | No promotion record from real market data; the accepted record here is a protocol fixture |
| 9 | Paper trade autonomously without a human selecting | not met | **partial** | Governed champion reader + one-shot promotion + periodic `paper.advance` enqueue all exist and were executed together on a synthetic store | No real evidence champion (blocked by findings F1/F2); no scheduler gap policy for H1 (F3); no recorded real paper session whose `model_version` is the champion hash |
| 10 | Risk controls cannot be bypassed | met | **met** | — | — |
| 11 | Recovery after restart / interruption / cancellation | met | **met** | Targeted suites pass | — |
| 12 | Operate within measured M2 limits | partial | **partial** | New single-job peaks: 363 MiB (`discovery run`) and 385 MiB (`discovery evidence`) | Full pipeline, walk-forward and real paper stage still unmeasured; bare `Orchestrator` still defaults to `AllowAllGuard` |
| 13 | Demonstrate the complete loop | not met | **not met** | Every loop link now has runtime code, but there is still no real artifact | No real import→discover→locked OOS→promote→paper→feedback run; the acceptance harness still bypasses the promotion decision in its paper stage and uses its own synthetic holdout ledger |

Counts: **4 met** (#2, #3, #10, #11), **8 partial** (#1, #4, #5, #6, #7, #8,
#9, #12), **1 not met** (#13).

## 1. Verification by execution on a synthetic store

All runs used the shipped CLI against a temporary store built from the same
profitable-trend protocol fixture as `tests/test_discovery_promote.py`
(`source=test:fixture`, `synthetic=false`). This proves the mechanism, not an
edge. Code version was the live `f28d820e9371f995`, settings digest
`2226685b488a…`.

| Case | Command shape | Exit | `status` / `codes` | Archive / ledger effect |
|---|---|---|---|---|
| A | `discovery promote --dry-run` on a plain bundle | **1** | `rejected` / `missing_locked_oos` | none written |
| B | `--dry-run` on a bundle with an attached fixture holdout | **0** | `accepted` / `[]` | none written |
| C | `--dry-run` on `synthetic=true` | **1** | `rejected` / `synthetic_data`, `missing_locked_oos` | none written |
| D | `--dry-run` on `source=null, synthetic=null` | **1** | `rejected` / `provenance_unknown`, `missing_locked_oos` | none written |
| E | real promotion, no `--spend-holdout` | **2** | `refused` / `holdout_not_spent` | holdout untouched |
| F | `--spend-holdout`, no token | **2** | `refused` / `approval_missing` | holdout untouched |
| G | `--spend-holdout` + token, candidate not in archive | **2** | `refused` / `unknown_candidate` | `holdout.sqlite` not created |
| H1 | `--spend-holdout` + token, seeded archive | **0** | `accepted` / `[]`; holdout `spent:true`, 120 trades, expectancy 0.7797R | champion promoted (`promotion_id 1`, mode `evidence`, policy `promotion-v1` v1) |
| H2 | repeat H1 | **2** | `refused` / `holdout_spent`; decision also `champion_same_spec`, `champion_improvement` | champion unchanged |
| I | `--policy holdout-eligibility` | **2** | `refused` / `non_production_policy` | nothing written |

Additional checks:

- The token string was absent from `discovery.sqlite` and `holdout.sqlite`
  (binary grep), and no response or decision record echoed it.
- The accepted archive stored the attached evidence checksum
  `6c6b577650b1…`; the holdout ledger shows `status=spent`, `source=test:fixture`,
  `synthetic=0`, `policy_id=promotion-v1`.
- The accepted promotion's evidence was the fixture's hand-built bundle, not a
  simulator result. It must not be cited as market validation.

This confirms the #220 contract: holdout spend needs both the explicit flag and
a non-empty owner token; a synthetic/unknown bundle is refused; a second spend
is refused; a structural failure is refused before the ledger is touched.

## 2. `paper.advance` scheduler: evidence champion only

Two orchestrator roots were built with the same stored bars. One held the
evidence champion from case H1; the other held the same candidate promoted
through `promote_exploratory(force=True)`.

| Case | Command | Exit | Result |
|---|---|---|---|
| Evidence champion | `orchestrator run --once --schedule-paper --max-jobs-per-tick 0` | **0** | exactly **1** queued job `paper.advance`, dedupe `paper.advance:XAUUSD:M1:2026-01-06T02:38:00+00:00`, payload `{symbol,timeframe,bars:1480}` (no `gap_policy`) |
| Same root, second scan | same | **0** | 0 queued, 0 duplicates; job count stays 1 |
| Exploratory champion | same | **0** | 0 queued, 0 duplicates; audit event `paper.advance_refused`, reason `no_evidence_champion` |

So the scheduler is evidence-only and idempotent. The exploratory path never
falls back to a legacy rule.

## 3. Real XAUUSD H1, read-only

Store: `.worktrees/v1/data-real-store`, H1 sidecar
`checksum=e48cd48d6aa1cdfb8071a511f0a2464a`, `rows=100083`,
`source=csv:data-real-csv`, `synthetic=false`. The bars and sidecar were not
modified. The CLI wrote a derived `datasets/b189…/` artifact under the store
root as it is designed to; it was removed after the run, and the store root is
back to `XAUUSD/` and `_meta/` only.

### 3.1 Search

```text
discovery run --max-trials 30 --population 8 --generations 4 --seed 7 --no-cache
exit 0; 27 attempts, 13 tradable, decision "rejected" (the archive gate never promotes)
best a8ded01d9613… net_profit +54.7314; stress x1.5 -200.981, x2.0 -436.985
peak RSS 380,878,848 B (363.2 MiB), 29.79 s wall, POLARS_MAX_THREADS=1
```

### 3.2 Evidence

```text
discovery evidence --folds 3 --stress 1.5,2.0 --trials 27 --search-budget 30 --no-cache
exit 0
dataset_id         XAUUSD:H1:h1:b189a7fdbad6
dataset_checksum   b189a7fdbad6cea1ec0c4a33c7111de1
evidence_checksum  c7e6c6ba2c413b3b4edb3a33ba9c576789a159cdc3ea12084b1cbd825ac44b1d
source=null, synthetic=null, locked_oos=null
folds: 109 / 74 / 81 trades; expectancy -0.0353 / -0.1920 / +0.0069R
aggregate: expectancy -0.0663R, PF 0.8509, positive folds 0/3
peak RSS 403,308,544 B (384.6 MiB), 8.14 s
```

### 3.3 Promotion dry run (no spend)

```text
discovery promote --dry-run --archive .../discovery.sqlite \
  --candidate a8ded01d9613… --evidence .../evidence.json --data-root data-real-store
exit 1
codes: provenance_unknown, min_positive_fold_fraction, min_expectancy_r,
       min_profit_factor, min_ci_lower_r, cost_stress, stress_degradation,
       missing_locked_oos
```

### 3.4 Structural refusal before any spend

The same command with `--spend-holdout` and a token file exits **2** with
`no_accepted_trial`:

```text
candidate 'a8ded01d9613…' has no accepted trial on 'XAUUSD:H1:h1:b189a7fdbad6'
under code version 'f28d820e9371f995'
```

No `holdout.sqlite` was created. This is the correct fail-closed behaviour, but
it also exposes finding F2: the archive trial recorded by the search is keyed
`validation:XAUUSD:H1:h1:b189a7fdbad6`, while the evidence bundle is keyed
`XAUUSD:H1:h1:b189a7fdbad6`. No shipped command records a trial under the
evidence id.

### 3.5 Reproduction

`discovery reproduce` exit **0**, `{"reproduced": true, "checksum":
"c7e6c6ba…", "mismatches": []}`. The store artifact was removed afterwards.

## 4. Criterion 9: what is code-complete and what is not

Code-complete and executed:

1. The evidence-mode champion is the only paper source:
   `ChampionProvider` rejects exploratory/manual champions; the scheduler emits
   no job and an audit reason.
2. The one-shot promotion path exists: `discovery promote` evaluates
   `PROMOTION_V1`, spends the holdout only with flag + token, attaches the
   locked section, re-evaluates and writes the champion, holdout ledger and
   archive evidence (exit 0 above; second attempt exit 2 `holdout_spent`).
3. Periodic selection is no longer a human step: `orchestrator run
   --schedule-paper` turns a new decidable bar into exactly one
   `paper.advance` job, deduped across scans/restarts.
4. The handler refuses missing/exploratory/invalid champions with an auditable
   NO TRADE and no legacy fallback.

Not code-complete / not demonstrated:

1. **No real evidence champion.** `discovery evidence` does not read the store
   sidecar, so a real bundle declares unknown provenance and `PROMOTION_V1`
   refuses `provenance_unknown` before any metric is read (F1).
2. **No matching archive trial.** Search trials use a `validation:` dataset id;
   evidence uses the manifest id, so `_structural_precheck` refuses
   `no_accepted_trial` before the holdout (F2). Fixing F1 alone would not make
   promotion succeed.
3. **No gap policy in the scheduler payload.** The trigger deliberately sends
   only `symbol`, `timeframe`, `bars`; `paper.advance` therefore runs `strict`.
   The real H1 store has 3,602 gaps larger than one hour (1,100 larger than two
   hours, max 77 h) across 100,083 bars, so the strict runner refuses. The
   handler accepts `gap_policy` values `strict|session_aware|skip_with_report`
   and `session_aware` was proven on this store in
   `paper-gap-acceptance-2026-11-25.md`, but the scheduler cannot select it
   (F3).
4. **No recorded real paper session** whose ledger `model_version` equals the
   promoted spec hash and whose job result carries the `promotion_id`.

Status **partial** is therefore the honest reading: the autonomous selector and
clock exist and work together on a fixture, but a real champion and a real
governed paper advance are still absent.

## 5. Criterion 13: what is code-complete and what is not

Loop links that now have runtime code: import/validate (`data`), dataset build,
search, evidence, holdout evaluation, promote, schedule paper, paper advance,
matured-outcome proposal.

Still missing for the criterion:

1. **A real end-to-end artifact.** No real import→…→feedback run exists. The
   real H1 attempt stops at the promotion gate as documented above.
2. **The feedback edge is not wired into the daemon.** `paper.outcomes` stops at
   a proposal with `queued:false, executed:false`. `ProposalQueueTriggerSource`
   can turn an approved proposal into a `discovery.run` job, but no CLI flag
   registers it; the shipped daemon cannot complete the cycle even after a
   human approves.
3. **The acceptance harness is not the runtime loop.** Its holdout stage spends
   a separate synthetic ledger (`acceptance/run.py:730-780`), and its paper
   stage runs `AutonomousPaperRunner` on `self.spec` (`:808`) — the discovery
   best spec, which may be the candidate the promotion stage just rejected —
   rather than the governed champion. Its report still claims
   `validated=false`, `paper_traded=false`, which is honest, but it does not
   exercise #220/#218.

Status stays **not met**.

## 6. New findings (runtime integration)

| ID | Finding | Evidence | Effect |
|---|---|---|---|
| F1 | `discovery evidence` never loads store provenance | `cli.py:_cmd_discovery_evidence` calls `persist_evidence_run` without `source`/`synthetic`; `load_provenance` is used by paper/API only. Real H1 bundle recorded `null/null`. | Every real bundle is refused `provenance_unknown`; no real promotion is possible. |
| F2 | Trial and evidence dataset ids differ for the same checksum | `discovery/run.py:765` records `validation:<manifest_dataset_id>`; `persist_evidence_run` records `<manifest_dataset_id>`. Verified in the real archive and the `no_accepted_trial` refusal. | Promotion refuses before the holdout even when evidence passes metrics. |
| F3 | Scheduler cannot choose a gap policy | `paper_scheduler.py` payload is exactly symbol/timeframe/bars; handler default is `strict`; H1 has 3,602 >1h gaps. | Real H1 paper advance refuses; #9 cannot complete on the real store. |
| F4 | Provenance is a declared label, not a verified ingest chain | `_check_provenance` only inspects `synthetic`; `source` is recorded but never validated or joined to an ingest record. The fixture `test:fixture` passed as “real”. | Fail-closed on missing/synthetic works; fail-closed on a mislabelled tape or an empty source does not. |
| F5 | The proposal bridge is library-only | `ProposalQueueTriggerSource` has tests but no CLI wiring. | The learning loop cannot close unattended. |

## 7. Exact next owner-approved command sequence

Steps A1–A3 are code changes and require the owner's explicit approval; they
were **not** performed in this assessment. Do not start with the holdout: the
current real candidate fails six metric gates, so spending would burn the
one-shot evaluation for nothing.

### Phase A — close the runtime defects (reviewed PR, no gate relaxation)

1. **A1 provenance wiring.** In `_cmd_discovery_evidence`, call
   `load_provenance(root, symbol, timeframe)` and pass `source`/`synthetic` into
   `persist_evidence_run`. A missing sidecar stays unknown; never default to
   real. Acceptance: a bundle built from `data-real-store` records
   `csv:data-real-csv` / `false`; a store without a sidecar records
   `null`/`null` and is still refused.
2. **A2 identity alignment.** Record an accepted trial for the evidence bundle
   under the manifest dataset id (for example extend `discovery evidence` with
   `--archive` to write the evidence trial after the bundle is built), or make
   `_structural_precheck` resolve the `validation:<id>` trial explicitly with
   the same checksum/settings/code checks. Do not relax the identity check.
   Acceptance: `discovery promote --dry-run` on a real bundle no longer refuses
   `no_accepted_trial`, and a stale trial still refuses.
3. **A3 paper gap policy.** Add a reviewed, per-scope `gap_policy` to the
   scheduler payload (`session_aware` for XAUUSD H1, as measured in
   `paper-gap-acceptance-2026-11-25.md`), or run paper only on a genuinely
   gap-free slice. Acceptance: the scheduler job carries the policy and the
   handler result records `gap_policy` and `gaps_skipped`.

### Phase B — real evidence and the one-shot holdout

Run from `backend` in the main worktree; the store is `../data-real-store`.

```bash
STORE=../data-real-store
RUNS=../runs/acceptance-post-runtime
mkdir -p "$RUNS/discovery"

# 1. Search a larger budget than the 30-trial wiring run.
POLARS_MAX_THREADS=1 uv run python -m scalping.cli discovery run \
  --root "$STORE" --symbol XAUUSD --timeframe H1 \
  --max-trials 200 --population 16 --generations 8 --seed 7 \
  --archive "$RUNS/discovery/discovery.sqlite" \
  --out "$RUNS/run.json" --no-cache

# 2. Export the best canonical tree as spec.json (the run report carries it).
python - "$RUNS/run.json" "$RUNS/spec.json" <<'PY'
import json, sys
run = json.load(open(sys.argv[1]))
best = run.get("best")
if not best:
    raise SystemExit("no best candidate; record NO TRADE and stop")
open(sys.argv[2], "w").write(json.dumps(best["canonical"], indent=2) + "\n")
print(best["candidate_id"])
PY

# 3. Evidence (writes <store>/datasets/<checksum>/ by design).
POLARS_MAX_THREADS=1 uv run python -m scalping.cli discovery evidence \
  --root "$STORE" --symbol XAUUSD --timeframe H1 \
  --spec "$RUNS/spec.json" --folds 3 --stress 1.5,2.0 \
  --trials 200 --search-budget 200 \
  --out "$RUNS/evidence.json" --no-cache

# 4. Dry run: the gate to the holdout.
uv run python -m scalping.cli discovery promote \
  --archive "$RUNS/discovery/discovery.sqlite" \
  --candidate "<candidate_id printed in step 2>" \
  --evidence "$RUNS/evidence.json" --data-root "$STORE" \
  --reason "post-runtime dry run" --dry-run \
  --out "$RUNS/promotion.dryrun.json"

# 5. STOP unless the dry run printed exactly codes ["missing_locked_oos"].
#    Any other code means the candidate is not eligible; do not spend.
```

Only after step 5 is clean, and with the owner's explicit approval:

```bash
# 6. One-shot holdout spend. Token file: one line, chmod 600, never committed.
uv run python -m scalping.cli discovery promote \
  --archive "$RUNS/discovery/discovery.sqlite" \
  --candidate "<candidate_id>" \
  --evidence "$RUNS/evidence.json" --data-root "$STORE" \
  --scope XAUUSD:H1 \
  --reason "owner approved the one-shot holdout spend" \
  --spend-holdout \
  --approval-token-file ~/.secrets/xauusd-approval.token \
  --out "$RUNS/promotion.json"
# exit 0 accepted -> champion written; exit 1 rejected -> holdout is burned,
# record it and do not retry; exit 2 refused -> nothing was spent.
```

### Phase C — governed paper advance

```bash
# 7. Queue only (does not execute) on the next decidable bar.
uv run python -m scalping.cli orchestrator run --once --profile none \
  --db "$RUNS/orchestrator.sqlite3" --root "$STORE" \
  --runs-root "$RUNS" --archive-root "$RUNS" \
  --max-jobs-per-tick 0 --schedule-paper

# 8. Verify exactly one paper.advance job and its gap policy, then execute one.
uv run python -m scalping.cli orchestrator run --once --profile none \
  --db "$RUNS/orchestrator.sqlite3" --root "$STORE" \
  --runs-root "$RUNS" --archive-root "$RUNS" \
  --max-jobs-per-tick 1 --schedule-paper

# 9. Verify the session is the champion, not a fixture.
sqlite3 "$RUNS/paper.sqlite" \
  "select session_id, model_version, config_json from sessions order by rowid desc limit 1;"
# model_version must equal the promoted spec hash; config_json must record the
# gap policy, data.dataset_id and provenance. The paper.advance job result in
# "$RUNS/orchestrator.sqlite3" must carry the champion's promotion_id.
```

### Phase D — feedback edge

```bash
# 10. After trades mature, the outcome trigger writes a proposal only.
uv run python -m scalping.cli orchestrator run --once --profile none \
  --db "$RUNS/orchestrator.sqlite3" --root "$STORE" \
  --runs-root "$RUNS" --archive-root "$RUNS" \
  --watch-paper-ledger "$RUNS/paper.sqlite" --max-jobs-per-tick 1

# 11. Only after the daemon can register ProposalQueueTriggerSource (the F5
#     wiring): approve the proposal in ProposalStore, then let the daemon
#     enqueue discovery.run and verify the job dedupe key names the proposal id.
```

Until the F5 bridge is reviewed, step 11 is a library call, not an autonomous
runtime step. That is the remaining difference between “code exists” and §36 #13.

## 8. Verification log

Environment: `uv sync` created `.venv` with Python 3.12.13. Commands run from
`backend/`. Exit codes are actual.

| Command | Result |
|---|---|
| `uv run pytest tests/test_discovery_promote.py tests/test_paper_scheduler.py tests/test_discovery_provenance.py tests/test_locked_holdout.py -q` | **exit 0**, 67 passed |
| `uv run pytest tests/test_promotion_policy.py tests/test_evidence_adversarial.py tests/test_paper_outcome_triggers.py tests/test_orchestrator_handlers.py -q` | **exit 0**, 243 passed |
| `uv run pytest -q` (full suite) | **exit 0**, 2,443 tests in 121 files |
| synthetic `discovery promote` cases A–I | exit codes **1, 0, 1, 1, 2, 2, 2, 0, 2, 2** as in §1 |
| synthetic `orchestrator run --once --schedule-paper` (evidence champion) | **exit 0**, 1 job; second scan 0 new |
| synthetic `orchestrator run --once --schedule-paper` (exploratory champion) | **exit 0**, 0 jobs, `no_evidence_champion` |
| real `discovery run` (30-trial budget) | **exit 0**, 27 attempts, 13 tradable, `rejected` |
| real `discovery evidence` | **exit 0**, bundle `c7e6c6ba…`, provenance `null` |
| real `discovery promote --dry-run` | **exit 1**, 8 codes in §3.3 |
| real `discovery promote --spend-holdout --approval-token-file` (structural precheck) | **exit 2**, `no_accepted_trial`; no ledger file |
| real `discovery reproduce` | **exit 0**, checksum reproduced |
| H1 gap census (read-only) | 3,602 gaps >1h, 1,100 >2h, max 77h |

Full-suite note: `uv run pytest -q` on this worktree exited 0; this repo's
pytest config suppresses the summary line, so the count comes from
`uv run pytest --collect-only -q` (2,443 tests, 121 files). The run was local,
not GitHub CI. A green suite is not acceptance evidence and changes no status
above.

## 9. What this report does not claim

- It does not claim a real champion, a real holdout spend or a governed real
  paper session. The only accepted promotion in this report is a protocol
  fixture on a synthetic tape.
- It does not treat the 30-trial H1 search as a real discovery budget or the
  candidate as tradeable. That run exists to exercise the refusal path.
- It does not claim the system is close to profitability. No profitability
  artifact exists, and the real candidate's cost-stress rows are negative.
- It does not claim the new links are worthless: #216, #218 and #220 close the
  “no runtime call site” gaps honestly. The remaining blockers are narrow,
  named and fixable, but they must be fixed before a holdout is spent.
