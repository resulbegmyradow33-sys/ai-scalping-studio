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
from scalping.knowledge.tables import (
    MERGED_CELL,
    TABLE_KINDS,
    TableExtraction,
    TableProblem,
    TableRecord,
    extract_tables,
    flatten_table,
)
from scalping.knowledge.routing import (
    APPROVAL_STATUSES,
    PROPOSAL_BAR_COLUMNS,
    PROPOSAL_GROUPS,
    PROPOSAL_KIND_SOURCES,
    PROPOSAL_SCHEMA_VERSION,
    PROPOSAL_STATUSES,
    ROUTING_ERROR_CODES,
    Approval,
    ExperimentProposal,
    ProposalDecision,
    ProposalStore,
    RoutingError,
    canonical_proposal_id,
    legacy_proposal_id,
    proposals_for_queue,
    propose_from_hypothesis,
)

__all__ = [
    "flatten_table",
    "extract_tables",
    "TableRecord",
    "TableProblem",
    "TableExtraction",
    "TABLE_KINDS",
    "MERGED_CELL",
    "APPROVAL_STATUSES",
    "CHUNK_CHARS",
    "CLAIM_KINDS",
    "DEFAULT_FALSIFICATION",
    "HYPOTHESIS_STATUSES",
    "PROPOSAL_BAR_COLUMNS",
    "PROPOSAL_GROUPS",
    "PROPOSAL_KIND_SOURCES",
    "PROPOSAL_SCHEMA_VERSION",
    "PROPOSAL_STATUSES",
    "ROUTING_ERROR_CODES",
    "SCHEMA_VERSION",
    "SEARCH_BACKENDS",
    "STATUS_TRANSITIONS",
    "Approval",
    "ChunkRecord",
    "ClaimKind",
    "ClassifiedSentence",
    "DocumentIngestionError",
    "DocumentIngestor",
    "DocumentRecord",
    "DocumentSummary",
    "ExperimentProposal",
    "ExtractionError",
    "ExtractionResult",
    "Hypothesis",
    "HypothesisError",
    "KnowledgeIndex",
    "KnowledgeIndexError",
    "ProposalDecision",
    "ProposalStore",
    "RoutingError",
    "SearchHit",
    "Section",
    "SectionRecord",
    "SourceRef",
    "TextExtractor",
    "UnsupportedFormat",
    "canonical_proposal_id",
    "classify_document",
    "classify_sentence",
    "create_hypothesis",
    "extract_hypotheses",
    "get_hypothesis",
    "legacy_proposal_id",
    "link_experiment",
    "list_hypotheses",
    "proposals_for_queue",
    "propose_from_hypothesis",
    "set_status",
    "split_chunks",
]
