# Knowledge ingestion and hypothesis research

Phase 17, slice 1 added local, offline ingestion of research documents with
provenance, deterministic text extraction, a searchable SQLite index and
structured hypothesis records. Slice 2 adds structural table preservation for
DOCX, HTML and Markdown (schema version 2) without OCR, a model or a new
dependency.

The code lives in `backend/src/scalping/knowledge/`:

| Module | Responsibility |
|---|---|
| `ingest.py` | copy the original, checksum, detect format, extract sections and tables |
| `tables.py` | structural DOCX/HTML/Markdown table extraction |
| `index.py` | SQLite documents/sections/chunks/tables, FTS5 search, hypothesis tables |
| `hypothesis.py` | hypothesis record, source citations, status, experiment links |

## Supported formats

| Extension | Recorded format | Parser | Version | Notes |
|---|---|---|---|---|
| `.txt` | `text` | `stdlib-text` | `2` | UTF-8 (BOM tolerated); form feeds split pages; no tables |
| `.md`, `.markdown` | `markdown` | `stdlib-markdown` | `2` | ATX headings define sections; fenced code is not scanned; pipe tables are extracted |
| `.html`, `.htm` | `html` | `html.parser` | `2` | scripts, styles, templates and titles are dropped; `h1`–`h6` define sections; `<table>` is extracted |
| `.docx` | `docx` | `zipfile+ElementTree` | `2` | paragraph text and `w:tbl` tables from `word/document.xml`; heading styles define sections |
| `.pdf` | `pdf` | none | n/a | `status="unsupported"`: an optional parser must be installed explicitly |

Version 2 adds table records; the paragraph text produced by every parser is
unchanged from version 1.

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
| `tables` | id, `doc_id` FK, ordinal, section ordinal, kind, caption, heading, `header_json`, `rows_json`, search text, row/col counts, `char_start`, `char_end`; unique `(doc_id, ordinal)` |
| `hypotheses` | id, statement, observed, proposed condition, expected effect, market/timeframe scope, assumptions JSON, falsification test, status, timestamps |
| `hypothesis_sources` | hypothesis FK, position, document FK, section ordinal, quote |
| `hypothesis_experiments` | hypothesis FK, experiment reference, position, linked_at |

Sections, chunks and tables are deleted with their document (`ON DELETE
CASCADE`).

### Schema history

| Version | Change |
|---|---|
| 1 | documents, sections, chunks, hypotheses, hypothesis sources/experiments |
| 2 | `tables` (document tables with caption, header/rows JSON, counts, offsets) |

Opening a version-1 database applies migration 2 in place; existing documents,
sections, chunks and hypotheses are untouched and `tables(doc_id)` returns an
empty list for them. Re-ingesting the same bytes (a duplicate record) inserts
the missing tables, and the `knowledge.ingest` handler calls
`KnowledgeIndex.backfill_tables(record)` on its duplicate branch, so the
production path backfills too. A partially populated document is completed per
ordinal; stored tables are never deleted or rewritten. A database with a newer
version is still refused.

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

### Tables

`scalping.knowledge.tables.extract_tables(path, sections=None)` returns a
`TableExtraction` with two parts:

- `tables`: `TableRecord` values in document order (deterministic, never
  reordered by size or score).
- `problems`: one `TableProblem(kind, reason, char_start)` per malformed table.
  Extraction reports and continues; it never raises for malformed content.

A `TableRecord` carries:

| Field | Meaning |
|---|---|
| `kind` | `docx`, `html` or `markdown` |
| `caption` | best effort; `""` when the format has none |
| `heading` | heading path at the table, same `" > "` convention as sections |
| `header` | header row, or `None` |
| `rows` | body rows only; a tuple of tuples of cell strings |
| `section_ordinal` | section the table belongs to, best effort; `-1` when unknown |
| `row_count` | header (when present) plus body rows |
| `col_count` | widest row after merged-cell expansion |
| `char_start`, `char_end` | citation span, see below |

Format rules:

- **DOCX** — `w:tbl`/`w:tr`/`w:tc`. The first row is the header only when it
  carries `w:tblHeader`; no style inspection guesses a header. A cell with
  `w:gridSpan` keeps its text in the first slot; the extra slots are
  placeholders. A `w:vMerge` continuation cell becomes one placeholder,
  whatever its (normally empty) text. Rows shorter than the widest row are
  padded with empty strings. Nested tables are not descended into. A table
  whose `<w:tbl>` span cannot be located in the payload is reported, not stored
  with an empty range.
- **HTML** — `html.parser` with `<table>`, `<caption>`, `<thead>`, `<tbody>`,
  `<tr>`, `<th>`, `<td>`. All rows are expanded in one grid pass, then the
  `<thead>` rows are separated by index. The first `<thead>` row, or the first
  body row when every cell is `<th>`, is the header; a multi-row `<thead>`
  keeps its first row as the header and reports the extra header rows (they are
  not stored in `rows`). `colspan` expands to placeholders; `rowspan` reserves
  covered slots in later rows, including rows after a `<thead>`, and is clipped
  to the real table height so no phantom rows appear. `script`, `style`,
  `noscript` and `template` content is ignored.
- **Markdown** — a header line with `|` followed by an alignment delimiter row
  (`---`, `:--`, `--:`, `:-:`); alignment is validated but not stored. Fenced
  code blocks are skipped entirely. Cells are trimmed and `\|` escapes a pipe.
  Ragged body rows are truncated to the header width and reported; every row is
  padded to the table width in one place. Inline code spans containing `|` are
  not protected; a pipe inside backticks still splits the cell.

Merged-cell placeholder policy: the covering cell keeps the text and every
covered slot holds the literal `[[merged]]` (`MERGED_CELL`). The extractor
never copies a value into slots it did not occupy, so "merged" is
distinguishable from "repeated".

Limits, enforced per table and reported as a `TableProblem`:

- more than `MAX_TABLE_COLUMNS` (512) columns after expansion;
- more than `MAX_TABLE_ROWS` (10,000) rows including the header;
- a DOCX payload larger than `MAX_DOCX_XML_BYTES` (32 MiB) or one containing a
  DTD/entity declaration is refused before `ElementTree` sees it.

Character offsets slice the raw text the extractor read:

- Markdown and HTML: the decoded file text with any UTF-8 BOM removed
  (`utf-8-sig`), starting at the header line or `<table>`.
- DOCX: the decoded `word/document.xml` payload inside the preserved archive,
  starting at `<w:tbl>`. The ingestor reads that payload once and passes it to
  table extraction (`read_docx_payload`), so a DOCX import opens the archive a
  single time.

The DOCX body has no stable plain-text rendering, so pointing at the payload is
the honest option. The invariant is `0 <= char_start < char_end`: a table with
no locatable span becomes a problem instead of an empty range. Offsets are
citations, not a promise that re-parsing the slice alone reproduces the record.

Table text is not added to section text by this module. The existing paragraph
extractors already include cell text (DOCX cell paragraphs, HTML visible text,
Markdown source lines), so a value appears in a section exactly once and the
table record is the structured view of the same source.

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

`index.search_tables(query, limit=10)` is a separate path over table text: the
flattened rows (`flatten_table`, tabs between cells, newlines between rows),
captions and heading paths. It returns `TableHit` values (`doc_id`, filename,
`table_ordinal`, `section_ordinal`, kind, caption, heading, text, score) and
uses `tables_fts` with the same `bm25`/`LIKE` policy, reported by
`tables_search_backend`. Tables are not appended to `chunks`: a chunk belongs
to a section and keeps exact section offsets, which a table chunk would break.
A cell value is therefore searchable through `search_tables`, and often through
`search` too because the source paragraphs/lines remain in section text.

Both FTS objects are created outside the versioned migrations: a Python build
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

## Boundaries

- **Document text is never a trusted strategy or an order.** Nothing in this
  package imports the strategy, backtest, paper-trading or API modules. A
  hypothesis can only be tested by a separate quantitative experiment; its
  prose is never executed.
- **Tables are structure, not interpretation.** A table record reproduces
  cells and their positions; it does not classify, score or trust them, and a
  merged-cell placeholder is metadata rather than data.
- **No LLM.** Parsing is deterministic stdlib code. The system does not claim to
  understand a book like a human, to have trained on a PDF, or to have language
  model embeddings. Ingestion is not training.
- **No cloud.** No network, credential or model-provider import exists in the
  package. The index is a local file.
- **No invented content.** If extraction fails, the record says so. Scanned
  PDF/OCR and unsupported formats are reported, not filled in.
- **Claims are not facts.** This slice stores text and hypotheses but does not
  yet classify claims, definitions, examples or opinions. That classification
  and the route into the Strategy Discovery Engine are later slices.

## Tests

`backend/tests/test_knowledge_ingest.py` covers txt/md/html/docx extraction,
checksum and duplicate handling, byte-identical preservation, PDF reporting,
recorded failures, path traversal, and the absence of network imports,
`eval`/`exec` and trading imports.

`backend/tests/test_knowledge_index.py` covers schema idempotency, exact
section/chunk offsets, FTS5 and LIKE search with limits, duplicate documents,
failure records, hypothesis round trips with citations, status transitions and
experiment links.

`backend/tests/test_knowledge_tables.py` covers DOCX header/`w:tblHeader` and
merged cells, HTML caption/`thead`/`colspan`/`rowspan`, Markdown alignment and
fenced-code exclusion, odd line separators (`\f`, `\v`, `\u2028`, lone `\r`) and
BOM offsets, malformed-table reporting, DOCX empty-table span alignment,
broken zip/DTD rejection, deterministic order, index round trips, table search
(FTS5 and a forced LIKE fallback), row limits, FTS trigger synchronisation,
partial backfill, v1→v2 migration with existing sections/chunks, cascade
deletion, and the absence of network/`eval`/trading imports.

`backend/tests/test_orchestrator_handlers.py` covers the production duplicate
path: a version-1 index plus the same file ingested through the handler
backfills the table rows.

Run:

```sh
cd backend
uv run pytest tests/test_knowledge_ingest.py tests/test_knowledge_index.py \
  tests/test_knowledge_tables.py -q
uv run ruff check src tests scripts
uv run mypy src
```

## Limitations

- PDF is unsupported until an optional parser is explicitly installed and
  registered. Scanned documents need OCR, which is absent. There is no OCR for
  table images in any format.
- Table extraction is structural, not semantic: complex layouts (nested DOCX
  tables, floating/positioned tables, multi-page tables, vertical merges that
  also carry content) may flatten or lose a cell. DOCX nested tables are not
  descended into.
- A multi-row HTML `<thead>` stores only its first row as `header`; the extra
  header rows are reported and are not part of `rows`. A `rowspan` past the last
  row is clipped rather than materialised as a phantom row.
- Tables above `MAX_TABLE_ROWS` (10,000 rows) or `MAX_TABLE_COLUMNS` (512
  columns after expansion) are reported instead of stored; a DOCX payload with
  a DTD/entity declaration is refused unparsed.
- DOCX tables are detected by `w:tbl`/`w:tr`/`w:tc` only. Rows wrapped in
  content controls (`w:sdt`) can be missed; column widths and cell borders are
  not stored.
- The DOCX offset points into `word/document.xml`, not at a rendered page.
- Markdown pipe tables are recognised by the delimiter row; Setext headings,
  tables inside block quotes and pipes inside inline code are not handled
  specially.
- Markdown relies on ATX headings; Setext headings are treated as body text.
- HTML text is normalised (horizontal whitespace collapsed, blank lines
  dropped); exact layout is not preserved. A table without a closing
  `</table>` is extracted up to its last parsed row and reported.
- The index stores the extracted text and table records, not the source bytes;
  the preserved original remains the authority.
- Classification and extraction are pattern-based, not semantic; there is no
  LLM, no strategy routing and no feedback ingestion of backtest/paper logs in
  this slice.
