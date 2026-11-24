# Knowledge ingestion and hypothesis research

Phase 17, slice 1. This slice adds local, offline ingestion of research
documents with provenance, deterministic text extraction, a searchable SQLite
index and structured hypothesis records. It does not add a language model, a
cloud service, OCR, claim classification or a document-to-order path.

The code lives in `backend/src/scalping/knowledge/`:

| Module | Responsibility |
|---|---|
| `ingest.py` | copy the original, checksum, detect format, extract sections |
| `index.py` | SQLite documents/sections/chunks, FTS5 search, hypothesis tables |
| `hypothesis.py` | hypothesis record, source citations, status, experiment links |
| `routing.py` | experiment-proposal records, approval gate, candidate conversion |

## Supported formats

| Extension | Recorded format | Parser | Version | Notes |
|---|---|---|---|---|
| `.txt` | `text` | `stdlib-text` | `1` | UTF-8 (BOM tolerated); form feeds split pages |
| `.md`, `.markdown` | `markdown` | `stdlib-markdown` | `1` | ATX headings define sections; fenced code is not scanned |
| `.html`, `.htm` | `html` | `html.parser` | `1` | scripts, styles, templates and titles are dropped; `h1`–`h6` define sections |
| `.docx` | `docx` | `zipfile+ElementTree` | `1` | paragraph text from `word/document.xml`; heading styles define sections |
| `.pdf` | `pdf` | none | n/a | `status="unsupported"`: an optional parser must be installed explicitly |

No dependency was added. PDF text extraction needs an optional parser and an
implementation of the `TextExtractor` protocol; until then the ingestor returns
`status="unsupported"` with an error that says so. It never silently reads a
PDF as plain bytes or invents text. Scanned PDFs would additionally need OCR,
which is not implemented.

A DOCX whose `word/document.xml` exceeds 32 MiB is refused before parsing, so a
zip bomb is a recorded extraction failure rather than an out-of-memory crash.

`ingest(path)` returns a record for any existing regular file, including
failures. `extract(path)` is the strict variant: it does not copy and raises
`UnsupportedFormat` or `ExtractionError` instead of returning a failed record.

## Provenance

`DocumentIngestor(root).ingest(path)` returns a `DocumentRecord` and writes the
original to:

```
<root>/sources/<sha256>/<original filename>
```

Rules:

- The SHA-256 of the file bytes is the document identity (`doc_id` and
  `checksum`). The original is copied before extraction, so a parser failure
  never loses it.
- The copy is byte-identical; the copy is written to a temp file in the same
  directory and moved into place, so an interrupted import cannot leave a
  partial document at the final path.
- The same checksum imported again reuses the single copy and reports
  `duplicate=True`, even when the second import uses another filename. A
  different file with the same name gets a different checksum directory; names
  never overwrite each other.
- Recorded metadata: source path (resolved), stored path, filename, bytes,
  mtime, format, parser name and version, import time and duplicate flag.
- A source path containing a `..` component or a NUL byte is rejected. The
  destination is re-checked to stay below `<root>/sources` even though the
  checksum and a single-segment filename already make escape impossible.
- A missing or non-regular source raises; an existing file always produces a
  record. Extraction failures are stored with `status="extraction_failed"` and
  the exception message, never dropped.

Statuses:

| Status | Meaning |
|---|---|
| `indexed` | text was extracted and sections are available |
| `extraction_failed` | a supported parser failed; original is preserved, reason recorded |
| `unsupported` | no approved parser for the format (currently PDF) |

## Index schema

`KnowledgeIndex(path)` uses the same conventions as `scalping.registry`:
SQLite, WAL, `busy_timeout`, `foreign_keys = ON`, `synchronous = NORMAL`,
`BEGIN IMMEDIATE` for writes and a `schema_version` table applied in order.
Opening a database with a newer schema raises instead of writing an older
layout. Opening is idempotent.

| Table | Columns |
|---|---|
| `documents` | `doc_id` PK, unique `checksum`, filename, source/stored path, size, mtime, format, parser, parser version, imported_at, duplicate, status, error |
| `sections` | id, `doc_id` FK, ordinal, heading path, label, text, `char_start`, `char_end`; unique `(doc_id, ordinal)` |
| `chunks` | id, `section_id` FK, ordinal, text, token count, offsets; unique `(section_id, ordinal)` |
| `hypotheses` | id, statement, observed, proposed condition, expected effect, market/timeframe scope, assumptions JSON, falsification test, status, timestamps |
| `hypothesis_sources` | hypothesis FK, position, document FK, section ordinal, quote |
| `hypothesis_experiments` | hypothesis FK, experiment reference, position, linked_at |

Sections and chunks are also deleted with their document (`ON DELETE CASCADE`).

### Offsets

Sections carry character offsets into the canonical document text:

```python
document_text = "\n\n".join(section.text for section in index.sections(doc_id))
document_text[section.char_start:section.char_end] == section.text
```

Chunk offsets are relative to their section, and `split_chunks` returns exact
slices of at most 800 characters broken between words. A single word longer
than the limit stays whole rather than producing a fake offset. The invariant
lets a citation be sliced back out without guessing.

### Search

`index.search(query, limit=10)` searches chunk text and section headings.

- If the Python SQLite build has FTS5, a `chunks_fts` virtual table is created
  with `unicode61` tokenisation and kept in sync by triggers. Ranking uses
  `bm25` (lower is better and exposed as `score`).
- Otherwise the index reports `search_backend == "like"` and uses a
  case-insensitive `LIKE` substring match with `score = 0.0`. The fallback is
  documented, tested and not presented as full-text ranking.
- `search_backend` reports which backend is active. No external search service
  is contacted.
- Query terms are quoted before reaching FTS5, so user text cannot inject
  `MATCH` operators. The LIKE fallback escapes `%`, `_` and the escape
  character.

The FTS object is created outside the versioned migrations: a Python build
without FTS5 must still open the same database, and the index never claims FTS5
unless the virtual table was actually created.

## Hypothesis records

A hypothesis is a record, not a strategy and not an order. It carries:

- `statement`, `observed`, `proposed_condition`, `expected_effect`;
- `market_scope`, `timeframe_scope`;
- `assumptions` (a tuple of strings);
- `falsification_test`;
- `source_refs` (`doc_id`, section ordinal, quote), checked against indexed
  sections before insertion;
- `status` and `experiment_refs`.

```python
from scalping.knowledge import DocumentIngestor, KnowledgeIndex, SourceRef
from scalping.knowledge import create_hypothesis, link_experiment, set_status

index = KnowledgeIndex("runs/knowledge/index.sqlite3")
record = DocumentIngestor("runs/knowledge").ingest("research/order-flow.md")
index.add_document(record)   # status must be "indexed" for citations

ref = SourceRef(doc_id=record.doc_id, section_ordinal=0, quote="sweep of the prior low")
hypothesis = create_hypothesis(
    index,
    "Sweeps of the prior low predict reversals after compression.",
    "The text describes sweeps preceding reversals.",
    "volatility compression plus a sweep of the prior 20-bar low",
    "higher reversal probability within 15 bars",
    "XAUUSD",
    "M15",
    "fails when the sweep continues into a trend day",
    assumptions=["spread below 40 points"],
    source_refs=[ref],
)
link_experiment(index, hypothesis.hypothesis_id, "backtest:atr-expansion:001")
set_status(index, hypothesis.hypothesis_id, "testing")
```

New records default to `proposed`; an importer that is copying an existing
record may pass its current status. Subsequent moves are monotonic for
auditability:

```
proposed -> testing -> supported
proposed -> rejected
testing  -> rejected
supported, rejected -> terminal
```

Repeating the current status is an idempotent no-op. Reopening a terminal state
requires a new hypothesis that cites the old one; the history is not rewritten.
The same experiment reference may be linked twice without duplication.

A source reference must point at a section of a document whose status is
`indexed`. A citation to a failed or unsupported document, an unknown document
or a missing section is rejected before the transaction.

## Sentence classification

`classify_sentence(text)` labels a research sentence with one of five kinds
using deterministic patterns — no LLM, no network:

| Kind | Marker | Example |
|---|---|---|
| `example` | "for example", "e.g.", "such as" | "For example, gold gaps at the London open." |
| `opinion` | first person or hedge | "I think momentum is probably dead." |
| `observation` | measurement/backtest terms or a number with a unit | "A backtest of 215 trades produced +0.159R." |
| `concept` | definition ("refers to", "is a", "means") | "Liquidity refers to resting orders." |
| `claim` | conditional/causal pattern or market statement with a modal | "When volatility expands, momentum signals weaken." |

Precedence is example → opinion → observation → concept → claim. A sentence
without a market term returns `None`, so generic prose is not forced into a
bucket. `classify_document(index, doc_id)` returns `ClassifiedSentence` rows in
section/sentence order.

The hypothesis extractor uses this filter: only `claim` and `observation`
sentences can become drafts; an opinion, definition or example never does. The
classifier is triage, not truth — an `observation` is still prose until a
quantitative experiment reproduces it.

## Hypothesis drafts from prose

`scalping.knowledge.extract_hypotheses(index, doc_id)` proposes hypotheses from
one indexed document with deterministic sentence patterns — no LLM, no network,
no `eval`/`exec`:

| Pattern | Example |
|---|---|
| conditional | `When volatility expands, momentum signals become less reliable.` |
| causal | `A signed order-flow imbalance predicts short-term continuation.` |
| graded | `The more the spread widens, the lower the edge.` |

Every draft:

- starts in `proposed`; the extractor has no way to move it to `testing`;
- stores the exact sentence as the `quote` of a `SourceRef` (document + section
  ordinal), so the citation resolves against the index;
- keeps the condition and expected effect separate, plus a fixed falsification
  template;
- uses `market_scope`/`timeframe_scope` `unspecified` unless the caller says
  otherwise — the extractor does not infer them.

Sentences without a market term are ignored, so generic prose ("If you are new,
read the manual") does not become research. Re-running the extractor skips
`(statement, section)` pairs already stored; it never duplicates the queue.

Boundary: a draft is prose, not evidence. It cannot trade, cannot enter the DSL
compiler, and cannot become `testing` without a human. Only a quantitative
experiment can support or reject it.

```python
from scalping.knowledge import KnowledgeIndex, extract_hypotheses

with KnowledgeIndex("runs/knowledge.sqlite3") as index:
    result = extract_hypotheses(index, doc_id, max_hypotheses=20)
    result.created   # Hypothesis records, status="proposed"
    result.skipped   # duplicates already stored
```

## Experiment proposals and the approval gate

P4 slice 3. `scalping.knowledge.routing` turns one testable hypothesis into an
`ExperimentProposal`, but the proposal is still a record. It contains the
hypothesis id, the original citations, the symbol/timeframe scope, the
hypothesis kind (`claim` or `observation`), the required raw bar columns and
feature groups, a suggested strategy skeleton **as three text fields**
(observation, condition, expected effect), the falsification test and its
governance state.

The proposal id is a lowercase sha256 over the immutable payload (hypothesis,
citations, scope, kind, data requirement, skeleton text), using the
`scalping.knowledge.experiment_proposal.v2` payload schema. Source citations are
sorted before hashing, so the same citations in a different order produce the
same id; the record still shows them in the order the hypothesis supplied.
Status, approval and conversion are excluded, so the same scope always has one
id while governance changes stay auditable. Re-creating the same record returns
the stored row. A rejected id is terminal: creating it again raises
`already_rejected`, so a re-test needs a new hypothesis.

`kind_source` (`classified`, `supplied` or `legacy`) is part of the payload and
the stored row. It says whether `hypothesis_kind` came from the deterministic
sentence classifier, was passed in explicitly for an imported record, or was
migrated from schema v1, which had no such field. It is hashed for new records,
so classified and supplied provenances cannot share an id. `legacy` exists only
on migrated rows and `create()` refuses it.

### Routing rules

`propose_from_hypothesis(hypothesis, *, symbol, timeframe, index)` is the only
constructor:

| Condition | Result |
|---|---|
| status `proposed` or `testing` | accepted |
| status `supported` or `rejected` | `bad_status`; a re-test is a new hypothesis |
| no source refs, or a citation that does not resolve to an `indexed` section | `missing_source` |
| a quote that is not a normalised substring of the cited section text | `unverified_quote` |
| symbol/timeframe outside the project registry | `unknown_scope` |
| statement not classifiable as `claim`/`observation` | `unsupported_kind` |
| a valid scope | `ExperimentProposal(status="proposed")` |

The caller supplies the symbol and timeframe because extracted hypotheses use
`unspecified` scope. Routing does not infer a market from prose and does not
compare the chosen scope with `market_scope`/`timeframe_scope`; that mismatch
remains visible in the two records.

`RoutingError.code` is one of: `missing_source`, `bad_status`, `unknown_scope`,
`already_converted`, `already_rejected`, `not_approved`, `unknown_proposal`,
`unknown_hypothesis`, `unsupported_kind`, `unverified_quote`,
`invalid_proposal`. Payload/provenance codes are `missing_source`,
`unverified_quote`, `unknown_hypothesis`, `unsupported_kind` and
`invalid_proposal`; `unknown_scope` covers registry scope; the rest are
lifecycle and governance refusals.

Required data is recorded rather than fetched. `required_columns` is the bar
schema an experiment needs (OHLC, spread for costs, volume); `required_groups`
starts as `price` + `volatility` and adds groups whose keywords appear as whole
words in the statement. Matching is anchored on word boundaries (with an
optional plural), so `ny` matches `NY` but not `any`, and `sweep` still matches
`sweeps`. This is a planning hint, not a dataset request and not a feature
computation.

### Store and approval

`ProposalStore(path)` follows the same SQLite conventions as the knowledge
index: WAL, `busy_timeout`, foreign keys, `synchronous = NORMAL`,
`BEGIN IMMEDIATE` writes and a versioned `schema_version` table. Schema version
2 adds `kind_source` and the decision log by additive migration. The store keeps
proposals, their source refs and the append-only decision log.

Schema 2 does **not** rewrite schema-v1 primary keys. v1 hashed proposals
without `kind_source` and without sorting citations, so a rehash would change
every id; instead migration keeps the old id, sets `kind_source='legacy'` and
backfills `proposal_decisions` from the `approval_*` columns (an approved or
rejected row gets one row, a converted row gets approved then converted). An
approve-then-reject sequence from v1 cannot be reconstructed because v1
overwrote the approval; migration records the surviving decision only.
`create()` looks up both the schema-v2 id and the computed v1 id for the same
payload, so the post-migration row is found, a rejected idea stays rejected and
the same scope cannot be stored twice. Because a migrated row keeps its v1 id,
use the record returned by `create()` for later governance calls rather than the
id of the draft you passed in.

`create(proposal, *, index)` is the persistence boundary. It re-checks
provenance against the knowledge index even when the record came from
`propose_from_hypothesis`: the hypothesis must be `proposed` or `testing`, its
citation multiset must equal the proposal's citations, every cited document must be
`indexed`, every section must exist, and every quote must be a normalised
(case-folded, whitespace-collapsed) substring of that section's text. Comparing
citation multisets is structural; the module does not prove that the free-text DSL
skeleton paraphrases the hypothesis. `required_columns` and `required_groups`
must be non-empty, duplicate-free tuples from the documented vocabularies, and
the row is read back inside the insert transaction so a malformed payload rolls
back instead of leaving a corrupt row. It accepts only `status="proposed"`
records with no approval and no conversion. Approval is a separate, audited
call; a hand-built approved record cannot bypass the gate. A rejected proposal
id stays rejected: re-creating it raises `already_rejected`.

`proposal_decisions(proposal_id, from_status, to_status, actor, reason, at)` is
append-only and is written in the same `BEGIN IMMEDIATE` transaction as the
status update. Rejecting an approved proposal adds a second row
(`approved -> rejected`) instead of erasing the approval. `decisions(proposal_id)`
returns the log oldest first, and the current `status` always equals the last
row's `to_status`.

The append-only guarantee is an **API-level boundary**, not a database-enforced
one. `ProposalStore.connection` exposes the raw SQLite handle and
`proposal_decisions` cascades from `proposals`, so direct SQL can edit or delete
history. Governance goes through the methods; raw SQL is for read-only
diagnostics.

| Status | Entered by | Approval fields | In queue |
|---|---|---|---|
| `proposed` | `create(proposal, index=...)` | empty | no |
| `approved` | `approve(reason, actor)` | actor, reason, timestamp | yes |
| `rejected` | `reject(reason, actor)` | actor, reason, timestamp | no |
| `converted` | `link_candidate(hash, dataset_id, actor=...)` | preserved from approval | no |

- `approve` and `reject` require a non-empty reason and actor; the latest
  decision is mirrored into `approval` and appended to `proposal_decisions`.
  Re-approving an approved proposal is an idempotent no-op (after argument
  validation). An approved but not-yet-converted proposal may still be
  rejected, and both decisions remain in the log. Rejected and converted
  proposals cannot be approved, because that would rewrite history.
- `link_candidate(hash, dataset_id, *, actor)` is allowed once, only from
  `approved`. It requires a sha256 hex candidate hash and a non-empty actor, and
  appends `approved -> converted` with the hash, dataset id and actor. A bad
  hash is `invalid_proposal`; a second call is `already_converted`; a call
  before approval is `not_approved`.
- Nothing auto-approves. `proposals_for_queue(store, limit)` returns approved
  proposals oldest first and never changes a status.

"Converted" means one discovery candidate hash was linked to one dataset id
for this proposal. It is identity, not evidence: conversion does not mean the
candidate was validated, promoted, paper-traded or profitable.

```python
from scalping.knowledge import (
    KnowledgeIndex,
    ProposalStore,
    propose_from_hypothesis,
    proposals_for_queue,
)

with KnowledgeIndex("runs/knowledge.sqlite3") as index, ProposalStore(
    "runs/proposals.sqlite3"
) as proposals:
    draft = propose_from_hypothesis(
        hypothesis, symbol="XAUUSD", timeframe="M15", index=index
    )
    proposals.create(draft, index=index)          # status="proposed"
    proposals.approve(draft.proposal_id, "scope and data are clear", "owner")
    ready = proposals_for_queue(proposals, 10)    # approved only
```

### No prose-to-order path

The routing module never compiles or executes the skeleton, never builds DSL
nodes, never reads market data and never imports the strategy, discovery, API,
paper-trading, backtest or orchestrator packages. It also has no network
import. A proposal carries no direction, size, stop, target or order type, so
there is nothing to execute.

Approval authorizes a quantitative experiment, not a trade. The Strategy
Discovery Engine still owns DSL/AST validation, the evidence gate owns
promotion, and the paper runner is the only execution target. A supported
hypothesis does not become a champion; a converted proposal does not become an
order.

## Orchestrator bridge (approved proposals only)

The queue bridge lives outside this package, in
`backend/src/scalping/orchestrator/proposal_handlers.py`. It is the only path
from an approved proposal to a `discovery.run` job:

- `ProposalQueueTriggerSource` calls `proposals_for_queue(store, limit)` and
  reads **approved** records only. Proposed, rejected and converted records are
  never bridged, and the source never calls `approve`, `reject` or
  `link_candidate`.
- A record is eligible when its symbol/timeframe is in the project registry, its
  required bar columns and feature groups are non-empty, and an optional
  `dataset_id` hint does not name a locked test/holdout split. A missing or
  unknown scope, a missing data requirement and a holdout-looking hint each
  produce one `proposal.not_queued` audit event instead of a job.
- The emitted payload is the same explicit `DiscoveryRequest` the production
  handler validates (`symbol`, `timeframe`, `max_trials`, `seed`, `stress`,
  `gate`, `population`, `generations`, `patience`, `horizon`). The search policy
  comes from a caller-supplied `DiscoveryTemplate`; nothing is inherited.
- Dedupe is `discovery.run:proposal:<proposal id>:<payload sha256>`, backed by a
  cursor in the job store. A repeat scan or a process restart does not create a
  second job, and changing the template is a new experiment rather than a
  deduplicated old one.
- The optional `knowledge.proposals` handler records a proposal snapshot and
  returns `queue_discovery` with `queued: false` / `executed: false`. It does not
  read the proposal store, run discovery or enqueue work.

The bridge does not change this package's status rules. `approved` still enters
the queue, `rejected` never does, and `link_candidate` remains the only way to
reach `converted`. A proposal converted after its job was queued does not retract
the job; the owner still controls that queue decision. The full trigger contract,
refusal vocabulary and honest limits are in
[`docs/orchestrator.md`](orchestrator.md#approved-proposals-as-discovery-triggers-knowledge-bridge).

## Boundaries

- **Document text is never a trusted strategy or an order.** Nothing in this
  package imports the strategy, backtest, paper-trading or API modules. A
  hypothesis can only be tested by a separate quantitative experiment; its
  prose is never executed.
- **No LLM.** Parsing is deterministic stdlib code. The system does not claim to
  understand a book like a human, to have trained on a PDF, or to have language
  model embeddings. Ingestion is not training.
- **No cloud.** No network, credential or model-provider import exists in the
  package. The index is a local file.
- **No invented content.** If extraction fails, the record says so. Scanned
  PDF/OCR and unsupported formats are reported, not filled in.
- **Claims are not facts.** Classification and hypothesis extraction are
  heuristic triage. A routed proposal names an experiment; it does not prove
  the claim or promote anything.

## Tests

`backend/tests/test_knowledge_ingest.py` covers txt/md/html/docx extraction,
checksum and duplicate handling, byte-identical preservation, PDF reporting,
recorded failures, path traversal, and the absence of network imports,
`eval`/`exec` and trading imports.

`backend/tests/test_knowledge_index.py` covers schema idempotency, exact
section/chunk offsets, FTS5 and LIKE search with limits, duplicate documents,
failure records, hypothesis round trips with citations, status transitions and
experiment links.

`backend/tests/test_hypothesis_routing.py` covers proposal creation from an
extracted hypothesis, missing/unresolvable sources, persistence-boundary
provenance checks (unknown hypothesis, terminal hypothesis status, missing
section, citation ownership, fabricated quote, normalised quote match), refusal
of supported and rejected hypotheses, unknown symbol/timeframe/kind,
approve/reject audit including approve→reject history, one-time candidate
linking with a required hash and actor, queue filtering, deterministic
order-independent ids, `kind_source` recording, required-data validation,
read-back rollback, schema-v1→v2 migration with backfilled decisions,
concurrent rejected inserts, reopen persistence and the absence of
strategy/order/network imports.

The bridge is covered separately in
`backend/tests/test_proposal_bridge.py`: approved-only emission, payload
validation against the real discovery handler, crash/restart dedupe, refusal
reasons, holdout-hint refusal, provenance in the audit log and the
proposal-only `knowledge.proposals` handler.

Run:

```sh
cd backend
uv run pytest tests/test_knowledge_ingest.py tests/test_knowledge_index.py \
  tests/test_knowledge_extract.py tests/test_hypothesis_routing.py \
  tests/test_proposal_bridge.py -q
uv run ruff check src tests scripts
uv run mypy src
```

## Limitations

- PDF is unsupported until an optional parser is explicitly installed and
  registered. Scanned documents need OCR, which is absent.
- DOCX extraction reads paragraph text; tables become paragraph text but lose
  their cell structure, and images are ignored.
- Markdown relies on ATX headings; Setext headings are treated as body text.
- HTML text is normalised (horizontal whitespace collapsed, blank lines
  dropped); exact layout is not preserved.
- The index stores the extracted text, not the source bytes; the preserved
  original remains the authority.
- Classification and extraction are pattern-based, not semantic; there is no
  LLM, no strategy routing and no feedback ingestion of backtest/paper logs in
  this slice.
- Proposal scope is caller-supplied and is not reconciled with the hypothesis'
  `market_scope`/`timeframe_scope`. Dataset availability is not checked at
  routing time; a proposal can name a valid registry scope for which no bars
  are stored yet.
- The queue function only reads approved rows. Wiring it into the orchestrator,
  building a dataset or compiling a DSL candidate from the skeleton is outside
  this slice.
- Proposal ids are deterministic, so a rejected payload cannot be re-proposed
  under the same hypothesis and scope: `create()` raises `already_rejected`. A
  re-test requires a new hypothesis and therefore a new id.
- Migrated v1 rows keep their v1 ids. The dual lookup matches a new proposal to
  a v1 row only when the rest of the payload is identical; a v1 row created
  with a caller-supplied kind that the classifier now derives differently is a
  different payload and will not match.
- The v1 decision backfill cannot invent history. A row that was approved and
  then rejected before schema 2 shows only the surviving rejection; converted
  rows show the reconstructed approval and conversion.
- The append-only decision log is an API-level boundary. `connection` exposes
  raw SQL and the table cascades from `proposals`, so direct SQL can edit or
  delete it.
- The decision log is append-only and unbounded; there is no pruning or
  archival policy yet. It records governance moves, not experiment evidence.
- Quote verification is a normalised substring test. It catches fabricated or
  paraphrased citations but cannot detect a real quote used out of context.
- Citation-ownership checking compares multisets; it does not prove that the DSL
  skeleton text paraphrases the hypothesis.
- Keyword-based `required_groups` is a planning hint and can over- or
  under-select feature groups. It never changes an experiment's actual data.
