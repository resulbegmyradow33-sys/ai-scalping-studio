# Data provenance audit — 2026-11-24

Audit branch `feat/provaudit`, base `main` at `626409f`.

Scope: trace `source`, `synthetic`, `dataset_id` and checksums from the market
data provider to the research trigger, and record every point where the chain is
dropped, re-derived into a different namespace, or defaulted. This is a code
audit. No production behaviour was changed. The only code added is regression
coverage for invariants the documentation already promises (section 6).

Method: read the producer and consumer of each hop, cite the exact lines, and
distinguish "carried", "re-derived" and "absent". A field that is absent is
reported as unknown, never as `False`.

## 1. Verdict

Provenance is solid inside two islands and broken between them.

- **Island A, data ingest.** The store sidecar is written atomically with
  `source`, `synthetic`, a content checksum and a `dataset_id`
  (`backend/src/scalping/data/provenance.py:50-69`,
  `:141-151`). Every normal ingest path writes it
  (`backend/src/scalping/cli.py:227-234`,
  `backend/src/scalping/api/routes.py:477-486`,
  `backend/scripts/aggregate_ticks_v2.py:92-100`,
  `backend/scripts/fetch_dukascopy_forward.py:185-206`,
  `backend/scripts/ingest_2026_ticks.py:145-165`). The one exception is
  `data derive`, which defaults every derived timeframe to `synthetic=False`
  (`backend/src/scalping/cli.py:298-304`).
- **Island B, paper ledger to trigger.** `ContinuousPaperRunner` reads the
  sidecar at session creation and refuses a later known change
  (`backend/src/scalping/paper/continuous.py:377-379`, `:423-440`,
  `:477-499`); the session stores the pair; `matured_outcomes` copies it to every
  row; `outcome_fingerprint` hashes it; the trigger payload forwards it; and both
  the trigger source and the handler refuse `synthetic=True` by default
  (`backend/src/scalping/orchestrator/paper_triggers.py:391-399`,
  `:428-438`, `:648-677`).
- **The bridge is missing.** Features, the dataset manifest, the dataset
  artifact, the evidence bundle, promotion, the holdout ledger and every ML/NN
  run carry no `source`/`synthetic` field at all. `AutonomousPaperRunner`, which
  is what the acceptance harness and the CLI-sized paper path use, writes no
  provenance and no dataset id. So a synthetic store can be built into a
  dataset, evidence bundle and champion, paper-traded through the autonomous
  runner, and emitted as a matured outcome whose `synthetic` is `null`; the
  trigger only refuses `True`, so `null` passes. `AutonomousPaperRunner` is the
  runner the acceptance harness uses and the one the package exports for direct
  callers; the orchestrator's continuous path does record provenance.

The master prompt forbids exactly this: promotion must fail closed on
provenance evidence (`docs/requirements/master-prompt.md:678`), and real must be
distinguished from synthetic (`:72-73`). Today that rule is enforced only where
the flag happens to reach the trigger.

## 2. Hop matrix

Every hop names its producer/consumer and what actually travels. `dataset_id`
means the field of that name, not "some identifier".

| # | Hop | Code point | `source` | `synthetic` | `dataset_id` | checksum | Status |
|---|---|---|---|---|---|---|---|
| 1 | Provider → bars | `data/synthetic.py:33,239`; `data/csv_provider.py:331-356`; `data/json_provider.py:111`; `data/dukascopy.py:379` | provider label only, in the `provider` column | inferred by caller, not in the frame | — | — | partial: `provider` ≠ `source` spelling |
| 2 | Ingest → sidecar | `cli.py:215-237`; `cli.py:285-306`; `api/routes.py:475-487`; scripts above | yes (`source`) | yes (`synthetic`) | yes, `symbol:tf:checksum[:12]` (`provenance.py:64-66`) | yes, blake2b over ts/OHLC/spread/volume (`provenance.py:31-46`) | written, except `data derive` |
| 3 | Store write | `data/store.py:107-135` | not touched | not touched | not touched | not touched | sidecar can go stale; see G9 |
| 4 | Bars → feature frame | `features/build.py:150-167` | absent | absent | absent | `frame_meta` hash (`features/cache.py:128-155`) | dropped |
| 5 | Feature cache | `features/cache.py:84-96`, `:220-235` | excluded from cache key by design | absent | `data.fingerprint` only | blake2b-160 of bars | dropped |
| 6 | Dataset manifest | `datasets/builder.py:56-95`, `:127-229` | absent | absent | `symbol:tf:h{horizon}:{digest[:12]}` (`:206`) | array digest (`:118-124`) | dropped |
| 7 | Dataset artifact | `datasets/artifact.py:244-250` | absent | absent | copies manifest id | per-file digests (`:237-242`) | dropped |
| 8 | Evidence bundle | `discovery/evidence.py:482-513`, `:1511-1530` | absent | absent | copies manifest id | evidence checksum (`:987-1003`) | dropped |
| 9 | Evidence verify | `discovery/evidence.py:1220-1281` | not checked | not checked | id/checksum compared | recomputed | cannot fail closed on provenance |
| 10 | Holdout / promotion | `discovery/holdout.py:908-999`; `discovery/promotion.py:1652` | absent | absent | id/checksum only | result checksum | dropped |
| 11 | Champion → paper runner | `discovery/champion.py:77-92`, `:200-212`; `orchestrator/handlers.py:1004-1013` | absent | absent | absent | evidence checksum exists but is dropped | dropped |
| 12 | Paper session config | `paper/continuous.py:365-399` | yes, from sidecar | yes, from sidecar | `data.dataset_id` from a *new* frame hash (`:401-414`) | frame fingerprint | carried, different id namespace |
| 13 | Session write/read | `paper/ledger.py:77-88`, `:426-454`, `:457-472`, `:762-812` | via `config_json` | via `config_json` | via `config_json` | none | carried; unknown stays `None` |
| 14 | Autonomous paper session | `paper/autonomous.py:238-261` | **absent** | **absent** | **absent** | none | dropped |
| 15 | Matured outcomes | `paper/continuous.py:607-648`, `:718-772` | copied per row | copied per row | copied | none | carried when 12 ran |
| 16 | Outcome fingerprint | `paper/continuous.py:775-804`, `:906-919` | hashed | hashed | hashed | sha256 | carried |
| 17 | Trigger payload | `orchestrator/paper_triggers.py:127-144`, `:428-438` | forwarded | forwarded | forwarded | `fingerprint` | carried |
| 18 | Trigger source gate | `orchestrator/paper_triggers.py:386-417` | logged | refuses only `True` | refuses missing | — | partial: `None` passes |
| 19 | Trigger handler | `orchestrator/paper_triggers.py:731-752`, `:648-677` | forwarded | refuses only `True` | forwarded | — | partial: `None` passes |
| 20 | Orchestrator analysis | `orchestrator/handlers.py:1111-1272` | absent | absent | absent | re-derived bar hash (`:1088-1108`) | dropped |
| 21 | Orchestrator dataset build | `orchestrator/handlers.py:773-829` | absent | absent | manifest id | array digest | dropped |
| 22 | Registry dataset record | `registry/db.py:166-196`, `:570-623` | absent | default `False` | id | digest | invented default, no production caller |
| 23 | ML / NN runs | `ml/train.py:198`; `nn/data.py:186` | absent | absent | dataset id only | checkpoint metadata | dropped |
| 24 | Acceptance harness | `acceptance/run.py:797-856`, `:930-950` | report only | report only (`synthetic=True`) | stage counts | ledger/manifest digests | ledger disagrees with report |

## 3. Three different `dataset_id` namespaces

There is no mapping function between these, so a ledger row cannot be joined to
the dataset or sidecar that produced it:

| Namespace | Formula | Written at | Checksum covers |
|---|---|---|---|
| Store sidecar | `symbol:timeframe:{provenance.checksum[:12]}` | `data/provenance.py:64-66` | ts, OHLC, spread, volume |
| Dataset manifest | `symbol:timeframe:h{horizon}:{array_digest[:12]}` | `datasets/builder.py:206` | X, labels, ts, split |
| Paper session | `symbol:timeframe:{frame_meta.fingerprint[:16]}` | `paper/continuous.py:409-414` | ts, OHLC, bid/ask, spread, volume |

Even the payloads disagree: the provenance checksum excludes `bid_close` /
`ask_close` and the `provider` column (`provenance.py:36`), while `frame_meta`
includes the quotes but excludes `provider` (`features/cache.py:84-96`). The same
bars therefore produce three unrelated hex strings. Nothing anywhere verifies
one against another.

## 4. Gaps, by severity

Severity is about evidence integrity, not code quality: **S1** can let a
synthetic or unverifiable strategy reach a promotion or the research loop;
**S2** breaks traceability or can invent/downgrade a flag; **S3** is a
documented limitation or a latent pattern with no current consumer.

### S1 — provenance cannot gate promotion or autonomous paper

**G1. Dataset manifest and artifact are provenance-blind.**
`DatasetManifest` (`datasets/builder.py:56-95`) has no `source`/`synthetic`; the
artifact payload copies only `dataset_id` and the manifest
(`datasets/artifact.py:244-250`). A dataset built from a synthetic M1 store is
indistinguishable from a real one except by tracing the store by hand.
*Fix:* add `source: str | None`, `synthetic: bool | None`, and a
`provenance_checksum` field to `DatasetManifest` and `build_dataset`; persist
them in `artifact.py::save_dataset_arrays`. Decode missing fields as `None`
(unknown) in `discovery/reproduce.py::_manifest`, never as `False`. Include the
fields in `to_dict()` so the manifest checksum and the evidence identity cover
them.

**G2. Evidence, holdout and promotion have no provenance field.**
`CandidateEvidence` (`discovery/evidence.py:482-513`) and `LockedOOSEvidence`
(`:362-407`) carry `dataset_id`/`dataset_checksum` only; `verify_evidence`
(`:1220-1281`) compares identity and checksum, and `promotion.py:1652` compares
`dataset_id`. A synthetic dataset can therefore produce a fully "verified"
evidence bundle. This is the specific failure the master prompt says must fail
closed.
*Fix:* add `source`/`synthetic` to `EvidenceConfig`, `CandidateEvidence` and
`LockedOOSEvidence`; bump `EVIDENCE_SCHEMA_VERSION` to 4;
`verify_evidence(..., source=..., synthetic=...)` must reject a mismatch and a
production promotion policy must reject `synthetic is not False` (or take an
explicit, audited demo policy). Wire the values from G1.

**G3. `AutonomousPaperRunner` records no provenance and no dataset id.**
Its session config is `origin` + `dsl` only
(`paper/autonomous.py:238-261`). This is the runner used by the acceptance
harness (`acceptance/run.py:797-813`). Any trade it closes yields
`OutcomeBatch.synthetic is None`; `PaperOutcomeTriggerSource` refuses only
`True` (`paper_triggers.py:391-399`), so an unknown-provenance batch is proposed
for research.
*Fix:* give `AutonomousPaperRunner` the same provenance step as
`ContinuousPaperRunner` (load the sidecar, write `provenance` and
`data.dataset_id` into the config). Factor
`ContinuousPaperRunner._dataset_provenance` / `_data_block`
(`paper/continuous.py:401-440`) into a shared helper rather than duplicating it.

**G4. `data derive` defaults derived timeframes to real.**
`_cmd_data_derive` calls `build_provenance(..., source=f"derived:{args.source}")`
without `synthetic` (`cli.py:298-304`), and `build_provenance` defaults
`synthetic=False` (`data/provenance.py:87-95`). Running `data derive` over a
synthetic M1 store marks M5–D1 as non-synthetic. `test_cli.py:609-641` runs
exactly that path and never checks the derived flag, so the bug is unpinned.
*Fix:* in `_cmd_data_derive`, load the source timeframe's sidecar and pass
`source` through; refuse to derive when the source sidecar is missing, or write
unknown. Do the same in the script loops that derive M5–D1
(`fetch_dukascopy_forward.py:197-206`, `ingest_2026_ticks.py:157-165`), which
happen to be correct today only because their source is real.

### S2 — traceability and flag integrity

**G5. No join between sidecar, dataset manifest and paper session ids.**
See section 3. A paper outcome records the frame-hash id, while the champion's
evidence records the manifest id. Nothing links them.
*Fix:* make the sidecar `dataset_id` canonical. `build_dataset` (G1) should take
it plus the provenance checksum; `paper/continuous.py::_data_block` should read
the sidecar and store its `dataset_id` instead of minting a third one. Add a
one-to-one test asserting the ledger id equals the sidecar id for one store.

**G6. Champion evidence identity is dropped at paper advance.**
`ChampionSpec` carries `promotion_id`, `evidence_checksum`, `policy_id`,
`policy_version`, `policy_digest` (`discovery/champion.py:77-92`, populated at
`:200-212`), but `handlers.py:1004-1013` passes only spec + scope to
`ContinuousPaperRunner`, whose config records only `dsl`/`scope`/`provenance`/
`data` (`paper/continuous.py:365-381`). A ledger row cannot say which evidence
bundle authorised the trade.
*Fix:* add an `evidence` parameter to `ContinuousPaperRunner.__init__`, persist
`promotion_id`, `evidence_checksum`, `policy_digest` under `config["dsl"]` or a
new `config["evidence"]` block, and pass `champion` from
`_paper_advance_handler`.

**G7. `paper.analyse` mixes synthetic and unknown sessions into one total.**
The handler reads every in-scope session, sums P&L and reports
`data_fingerprint`, but has no `source`/`synthetic` field and no exclusion
(`orchestrator/handlers.py:1111-1272`, rows at `:1198-1222`, summary at
`:1239-1257`). A demo ledger contaminates the research summary silently.
*Fix:* call `session_provenance(session.config)` per session in
`_paper_analyse_handler`, add `source`/`synthetic` to each row, and report
`synthetic_sessions` plus separate totals (or exclude `synthetic is not False`
by default with an `include_synthetic` flag).

**G8. Sidecar decoding coerces instead of refusing, while session decoding
refuses.**
`DatasetProvenance.from_dict` does `synthetic=bool(raw["synthetic"])` and
`source=str(raw["source"])` (`data/provenance.py:72-84`). A hand-edited or
corrupt sidecar with `"synthetic": 0` becomes `False` (fails open), and a string
becomes an invented boolean. The ledger's `session_provenance` deliberately
refuses mistyped values (`paper/ledger.py:426-454`), so the same concept has two
strictness levels.
*Fix:* type-check in `from_dict` — require `isinstance(raw["synthetic"], bool)`
and a non-empty string `source`, raise `TypeError`/`ValueError`. Match
`session_provenance`'s semantics.

**G9. The sidecar can go stale after a store write, and nothing notices.**
`ParquetStore.write` merges rows without touching or validating the sidecar
(`data/store.py:107-135`). The docs call this intentional ("replaced only by an
explicit write", `docs/data-pipeline.md:337`), but there is no check that the
recorded `rows`/`checksum` still describe the slice. `_dataset_provenance`
(`paper/continuous.py:423-440`) reads the sidecar without comparing it to the
bars it is about to trade.
*Fix:* at session creation, recompute `provenance.checksum(bars)` and refuse
when it differs from the sidecar (or record the mismatch explicitly). Provide a
`data provenance` refresh command so the fix is operable rather than a dead end.

**G10. Orchestrator dataset build ignores the sidecar.**
`_dataset_handler` reads bars and builds a manifest without reading
`load_provenance` (`orchestrator/handlers.py:773-829`), so the orchestrator path
produces exactly the provenance-blind dataset of G1.
*Fix:* read the sidecar in `_dataset_handler`, pass it to `build_dataset` once
G1 exists, and refuse (or record unknown) when it is missing.

**G11. Unknown provenance is never refused; only `True` is.**
Both the source and handler gate on `synthetic is True`
(`paper_triggers.py:391`, `:648`). `None` means "unknown" and passes. The docs
say downstream *can* refuse `synthetic is not false`
(`docs/orchestrator.md:958-960`), but no downstream component does.
*Fix:* decide the policy explicitly. Either the autonomous path requires known
provenance (`require_known_provenance=True`, default for production) or the
promotion policy rejects `None`. Document the chosen default; do not silently
treat unknown as real.

**G12. The acceptance harness labels the report, not the ledger.**
`AcceptanceResult.synthetic=True` is set on the report
(`acceptance/run.py:326-330`, `:930-950`), but the paper stage uses
`AutonomousPaperRunner` with no provenance (`:797-813`), so the ledger and the
outcomes file from the same run say `None`.
*Fix:* after G3, pass an explicit `source="acceptance:synthetic"`,
`synthetic=True` block from `_stage_paper`, and assert the outcomes batch agrees
with `AcceptanceResult.synthetic`.

**G13. `register_dataset` invents `synthetic=False`.**
`registry/db.py:570-592` defaults `synthetic: bool = False` and stores it. No
production caller exists today (only tests), but the API invites an invented
real flag.
*Fix:* make it `synthetic: bool | None = None` and store `NULL`; reject `NULL`
at any production promotion gate.

### S3 — documented limits and latent patterns

**G14. Features and the feature cache carry no provenance.**
`FeatureFrame` has arrays and `valid_from` only (`features/build.py:150-167`);
the cache key and manifest exclude `provider` on purpose
(`features/cache.py:84-96`, `:220-235`). This is defensible (features are a pure
function of the bars), but it means provenance cannot be reconstructed from a
cached frame. No fix required; keep the sidecar as the single source of truth.

**G15. ML/NN artifacts record a dataset id but no flag.**
`ml/train.py:198`, `nn/data.py:186` copy `dataset_id` only. They inherit G1.
*Fix:* once G1 lands, copy `source`/`synthetic` into the run metadata and
checkpoint report.

**G16. Proposal `dataset_id` is an unverified free-text hint.**
`proposal_handlers.py:445-463` accepts a string and only rejects holdout-like
names; the queue event records it (`:486-507`) with no provenance.
*Fix:* when a proposal names a dataset, resolve it against the registry and
carry its provenance; otherwise keep it a hint and say so in the event.

**G17. `provider` and `source` use different vocabularies.**
Bars carry `csv` / `csv:no-spread` / `synthetic` / `dukascopy`
(`base.py:38-40`, `csv_provider.py:331-338`), while sidecars carry
`provider:synthetic`, `csv:<file>`, `dukascopy:tick-crawl`, `derived:M1`
(`cli.py:251`, `:267`, scripts). There is no normalisation table, so equality
tests on `source` are path-spelling-sensitive.
*Fix:* define one enum/prefix scheme in `data/provenance.py` and have every
writer use it; keep the free-text detail in `notes`.

**G18. Sidecars are content hashes, not attestations.**
Already stated in `docs/data-pipeline.md:469-470`. Anyone who can write the
store can write a "real" sidecar. This audit does not change that; if producer
attestation is needed it is a separate design (keyed digest / signature).

## 5. The no-synthetic-as-real-evidence rule

The rule is not new; it is a repository invariant:

- `AGENTS.md` non-negotiable: "No ... silent use of synthetic data as real
  market evidence."
- `docs/requirements/master-prompt.md:72-73`: "Distinguish real from synthetic
  data."
- `docs/requirements/master-prompt.md:678`: production/paper promotion must
  fail closed on "regime/stability, champion-comparison or provenance
  requirements."
- `docs/paper-trading.md:384-407`: unknown is `None`, never an invented
  `False`; a demo run must be distinguishable from real evidence.

Where it is enforced today:

| Layer | Enforcement | Code |
|---|---|---|
| Ingest | explicit flag per provider; `synthetic` is never inferred from the symbol | `cli.py:251`; `api/routes.py:484`; scripts |
| Store | quality report flags a missing sidecar as `warn/provenance_missing` | `api/routes.py:161-181`, `:296-306` |
| Continuous paper | flag read at creation; unreadable sidecar refuses; a later known change refuses | `paper/continuous.py:377-379`, `:431-437`, `:477-499` |
| Ledger | mistyped provenance refuses at write and at read; unknown stays `None` | `paper/ledger.py:426-454`, `:781-786` |
| Outcomes | flag copied to every row and batch; a row/batch mismatch refuses | `paper/continuous.py:669-688`, `:718-772` |
| Fingerprint | `source`/`synthetic` are part of the hash | `paper/continuous.py:775-804` |
| Trigger | `synthetic=True` refused by default; explicit `allow_synthetic` opt-in; handler repeats the refusal | `orchestrator/paper_triggers.py:391-399`, `:648-677` |
| Acceptance | the report says `synthetic=True` and denies validated/profitable claims | `acceptance/run.py:930-950` |

Where it is **not** enforced: dataset manifest/artifact, evidence, holdout,
promotion, the autonomous paper runner, the orchestrator dataset/analysis
handlers, and every ML/NN run. Unknown (`None`) is not refused anywhere in the
trigger/handler path — only `True` is. So the honest statement is:

> The rule "a known-synthetic batch must not enter research" is implemented for
> the continuous paper path. The rule "unprovenanced or synthetic data must not
> produce promotion evidence" is **not implemented**; the dataset/evidence layer
> has no field to enforce it with.

## 6. Regression tests added (claimed hops only)

Only invariants already promised by docs/tests were pinned. No gap was silently
"fixed" by a test.

| Test | Pins this claim |
|---|---|
| `backend/tests/test_provenance.py::test_store_write_does_not_replace_the_sidecar` | `docs/data-pipeline.md:337`: "the old sidecar is replaced only by an explicit write." |
| `backend/tests/test_api_data_quality.py::test_configured_store_without_a_sidecar_warns` | `docs/data-pipeline.md:296-299`: no sidecar ⇒ `warn`, `status_rule = provenance_missing`. |
| `backend/tests/test_paper_continuous.py::test_session_created_before_provenance_still_resumes` | `docs/paper-trading.md:397-398`: a session recorded before provenance existed (unknown) still resumes; unknown never becomes `False`. |
| `backend/tests/test_paper_ledger.py::test_every_documented_provenance_spelling_is_read` | `docs/orchestrator.md:955-957`: `source`/`provenance.source`/`data.source`/`extra.source` (and the `synthetic` counterparts) still read. |

These pin existing behaviour. They do not claim any gap in section 4 is fixed.

## 7. Exact fix order (proposal, not implemented)

1. G4 (`data derive` flag) — smallest, stops active mislabelling.
2. G1 → G2 (`DatasetManifest` → `CandidateEvidence`) — the invariant the master
   prompt requires; needs a schema-version bump and a decode path that treats
   old bundles as unknown.
3. G3 + G11 (autonomous runner records provenance; unknown policy explicit) —
   closes the route by which unknown evidence currently enters the loop.
4. G5 + G6 (one dataset identity; champion evidence in the session) — makes the
   chain joinable.
5. G7, G10, G12, G13 — reporting and write-boundary integrity.
6. G8, G9 — sidecar strictness and staleness.

## 8. Not proven / residual risk

- No test in this branch runs a synthetic store end to end into evidence and
  promotion to demonstrate the gap. The gap is established by reading the
  dataclasses: neither `DatasetManifest` nor `CandidateEvidence` has the field,
  so there is nothing to assert.
- The audit did not execute the Dukascopy/tick scripts (network/cache
  dependent); their sidecar writes were read, not run.
- A "real" sidecar is self-declared; nothing here authenticates the producer
  (G18).

## 9. Reproduce

```bash
cd backend
PYTHONPATH=src python -m pytest tests/test_provenance.py \
  tests/test_api_data_quality.py tests/test_paper_continuous.py \
  tests/test_paper_ledger.py -q
```

Static checks used for the matrix:

```bash
rg -n "synthetic|source|dataset_id" backend/src/scalping \
  backend/src/scalping/discovery backend/src/scalping/paper
rg -n "save_provenance|build_provenance" backend/src backend/scripts
rg -n "load_provenance" backend/src
```
