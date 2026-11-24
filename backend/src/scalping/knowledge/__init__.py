"""Local knowledge ingestion, search and hypothesis records (Phase 17, slice 1).

The package turns local research documents into provenance records, extracted
sections, a searchable SQLite index and structured hypotheses with citations.
It is deliberately deterministic and offline:

* only the Python standard library parses documents; PDF needs an optional
  parser and is reported as unsupported rather than quietly skipped;
* the index is SQLite FTS5 when available, otherwise a documented ``LIKE``
  fallback, never an external search service;
* no language model is used or required, and extraction is not training;
* document text is never a trusted strategy or an order. A hypothesis is a
  record that a separate quantitative experiment must test.

See ``docs/knowledge-ingestion.md`` for the formats, schema and boundaries.
"""

from scalping.knowledge.classify import (
    CLAIM_KINDS,
    ClaimKind,
    ClassifiedSentence,
    classify_document,
    classify_sentence,
)
from scalping.knowledge.extract import (
    DEFAULT_FALSIFICATION,
    ExtractionResult,
    extract_hypotheses,
)
from scalping.knowledge.hypothesis import (
    HYPOTHESIS_STATUSES,
    STATUS_TRANSITIONS,
    Hypothesis,
    HypothesisError,
    SourceRef,
    create_hypothesis,
    get_hypothesis,
    link_experiment,
    list_hypotheses,
    set_status,
)
from scalping.knowledge.index import (
    CHUNK_CHARS,
    SCHEMA_VERSION,
    SEARCH_BACKENDS,
    ChunkRecord,
    DocumentSummary,
    KnowledgeIndex,
    KnowledgeIndexError,
    SearchHit,
    SectionRecord,
    split_chunks,
)
from scalping.knowledge.ingest import (
    DocumentIngestionError,
    DocumentIngestor,
    DocumentRecord,
    ExtractionError,
    Section,
    TextExtractor,
    UnsupportedFormat,
)

__all__ = [
    "CHUNK_CHARS",
    "CLAIM_KINDS",
    "DEFAULT_FALSIFICATION",
    "HYPOTHESIS_STATUSES",
    "SCHEMA_VERSION",
    "SEARCH_BACKENDS",
    "STATUS_TRANSITIONS",
    "ChunkRecord",
    "ClaimKind",
    "ClassifiedSentence",
    "DocumentIngestionError",
    "DocumentIngestor",
    "DocumentRecord",
    "DocumentSummary",
    "ExtractionError",
    "ExtractionResult",
    "Hypothesis",
    "HypothesisError",
    "KnowledgeIndex",
    "KnowledgeIndexError",
    "SearchHit",
    "Section",
    "SectionRecord",
    "SourceRef",
    "TextExtractor",
    "UnsupportedFormat",
    "classify_document",
    "classify_sentence",
    "create_hypothesis",
    "extract_hypotheses",
    "get_hypothesis",
    "link_experiment",
    "list_hypotheses",
    "set_status",
    "split_chunks",
]
