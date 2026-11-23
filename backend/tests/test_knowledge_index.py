"""SQLite knowledge index: schema, offsets, search and hypothesis records."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scalping.knowledge import (
    HYPOTHESIS_STATUSES,
    SCHEMA_VERSION,
    DocumentIngestor,
    DocumentRecord,
    Hypothesis,
    HypothesisError,
    KnowledgeIndex,
    KnowledgeIndexError,
    SourceRef,
    create_hypothesis,
    get_hypothesis,
    link_experiment,
    list_hypotheses,
    set_status,
    split_chunks,
)


def _ingest(tmp_path: Path, name: str, text: str, *, root: Path | None = None) -> DocumentRecord:
    source = tmp_path / name
    source.write_text(text, encoding="utf-8")
    return DocumentIngestor(root or tmp_path / "knowledge").ingest(source)


def _index(tmp_path: Path) -> KnowledgeIndex:
    return KnowledgeIndex(tmp_path / "knowledge.sqlite3")


def _document(tmp_path: Path, index: KnowledgeIndex, name: str, text: str) -> DocumentRecord:
    record = _ingest(tmp_path, name, text)
    index.add_document(record)
    return record


def test_schema_is_idempotent_and_reports_search_backend(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.sqlite3"
    first = KnowledgeIndex(path)
    assert first.schema_version == SCHEMA_VERSION
    assert first.search_backend in {"fts5", "like"}
    first.close()

    second = KnowledgeIndex(path)
    assert second.schema_version == SCHEMA_VERSION
    rows = second.connection.execute("SELECT version FROM schema_version").fetchall()
    assert [int(row[0]) for row in rows] == [SCHEMA_VERSION]
    assert second.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    second.close()


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite3"
    index = KnowledgeIndex(path)
    index.close()
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (?, '2026-01-01')", (99,))
    conn.commit()
    conn.close()

    with pytest.raises(KnowledgeIndexError, match="newer"):
        KnowledgeIndex(path)


def test_sections_and_chunks_have_exact_offsets(tmp_path: Path) -> None:
    index = _index(tmp_path)
    text = "# Structure\n\n" + "Swing highs define structure. " * 80 + "\n\n## Liquidity\n\nSweep."
    record = _document(tmp_path, index, "book.md", text)
    sections = index.sections(record.doc_id)

    assert [section.ordinal for section in sections] == [0, 1]
    assert sections[0].heading == "Structure"
    assert sections[1].heading == "Structure > Liquidity"
    document_text = "\n\n".join(section.text for section in sections)
    for section in sections:
        assert document_text[section.char_start : section.char_end] == section.text
    assert index.document_text(record.doc_id) == document_text

    chunks = index.chunks(sections[0].id)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert len(chunks) > 1
    for chunk in chunks:
        assert sections[0].text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.token_count == len(chunk.text.split())
        assert len(chunk.text) <= 800
    index.close()


def test_split_chunks_keeps_long_words_whole() -> None:
    text = "short " + "x" * 900
    chunks = split_chunks(text, max_chars=50)
    assert len(chunks) == 2
    assert chunks[0][0] == "short"
    assert chunks[1][0] == "x" * 900
    for chunk, start, end in chunks:
        assert text[start:end] == chunk


def test_search_returns_citable_hits_and_respects_limit(tmp_path: Path) -> None:
    index = _index(tmp_path)
    liquidity = _document(
        tmp_path,
        index,
        "liquidity.md",
        "# Liquidity\n\nA liquidity sweep takes the prior low before reversing.",
    )
    _document(tmp_path, index, "volume.md", "# Volume\n\nVolume confirms a breakout.")
    _document(tmp_path, index, "footer.md", "# Notes\n\nNothing about the matching term.")

    hits = index.search("liquidity sweep")
    assert hits
    assert {hit.doc_id for hit in hits} == {liquidity.doc_id}
    assert hits[0].filename == "liquidity.md"
    assert hits[0].heading == "Liquidity"
    assert "prior low" in hits[0].text

    limited = index.search("liquidity", limit=1)
    assert len(limited) == 1
    with pytest.raises(ValueError):
        index.search("liquidity", limit=0)
    with pytest.raises(ValueError):
        index.search("   ")
    index.close()


def test_search_tolerates_fts_operators_in_the_query(tmp_path: Path) -> None:
    index = _index(tmp_path)
    _document(tmp_path, index, "quotes.md", "# Quotes\n\nThe order flow quote is inert.")
    hits = index.search('order-flow "quote"')
    assert hits
    index.close()


def test_like_fallback_finds_substrings(tmp_path: Path) -> None:
    index = _index(tmp_path)
    _document(tmp_path, index, "fallback.txt", "A liquidity sweep precedes a reversal.")
    if index.search_backend != "like":
        index._search_backend = "like"  # exercise the documented fallback

    hits = index.search("liquidity", limit=5)
    assert hits
    assert hits[0].score == 0.0
    assert "liquidity" in hits[0].text.lower()
    assert index.search("100% missing") == []
    assert index.search("no_such_term") == []
    index.close()


def test_duplicate_document_is_not_indexed_twice(tmp_path: Path) -> None:
    index = _index(tmp_path)
    source = tmp_path / "same.txt"
    source.write_text("Only one row for these bytes.", encoding="utf-8")
    ingestor = DocumentIngestor(tmp_path / "knowledge")
    first = ingestor.ingest(source)
    second = ingestor.ingest(source)
    assert second.duplicate is True

    stored_first = index.add_document(first)
    stored_second = index.add_document(second)
    assert stored_first == stored_second
    assert stored_second.imported_at == stored_first.imported_at
    assert stored_second.duplicate is False
    count = index.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count == 1
    sections = index.sections(first.doc_id)
    assert len(sections) == 1
    index.close()


def test_failed_extraction_is_stored_without_sections(tmp_path: Path) -> None:
    index = _index(tmp_path)
    source = tmp_path / "broken.docx"
    source.write_bytes(b"not a zip")
    record = DocumentIngestor(tmp_path / "knowledge").ingest(source)
    stored = index.add_document(record)

    assert stored.status == "extraction_failed"
    assert stored.error is not None
    assert index.sections(record.doc_id) == []
    assert [item.doc_id for item in index.list_documents(status="extraction_failed")] == [
        record.doc_id
    ]
    index.close()


def test_unknown_and_unsupported_documents_are_reported(tmp_path: Path) -> None:
    index = _index(tmp_path)
    assert index.document("0" * 64) is None
    assert index.document_by_checksum("0" * 64) is None
    with pytest.raises(TypeError):
        index.add_document(object())  # type: ignore[arg-type]
    index.close()


def test_hypothesis_round_trip_with_sources(tmp_path: Path) -> None:
    index = _index(tmp_path)
    record = _document(
        tmp_path,
        index,
        "sweeps.md",
        "# Sweeps\n\nA sweep of the prior low often precedes a reversal.",
    )
    section = index.sections(record.doc_id)[0]
    ref = SourceRef(
        doc_id=record.doc_id,
        section_ordinal=section.ordinal,
        quote="A sweep of the prior low",
    )

    created = create_hypothesis(
        index,
        "Prior-low sweeps on M15 predict short-term reversals after compression.",
        "The source describes sweeps preceding reversals.",
        "volatility compression and a sweep of the prior 20-bar low",
        "higher probability of a reversal within 15 bars",
        "XAUUSD",
        "M15",
        "failed when the sweep continues into a trend day",
        assumptions=["spread stays below 40 points", "no news window"],
        source_refs=[ref],
    )

    assert isinstance(created, Hypothesis)
    assert created.status == "proposed"
    assert created.assumptions == ("spread stays below 40 points", "no news window")
    assert created.source_refs == (ref,)
    assert created.experiment_refs == ()
    assert create_hypothesis.__module__.startswith("scalping.knowledge")
    assert get_hypothesis(index, created.hypothesis_id) == created
    assert list_hypotheses(index) == [created]
    assert list_hypotheses(index, status="proposed") == [created]
    assert list_hypotheses(index, status="rejected") == []
    index.close()


def test_hypothesis_status_transitions_and_experiment_link(tmp_path: Path) -> None:
    index = _index(tmp_path)
    record = _document(tmp_path, index, "regime.md", "# Regime\n\nCompression precedes expansion.")
    ref: dict[str, object] = {
        "doc_id": record.doc_id,
        "section_ordinal": 0,
        "quote": "Compression precedes expansion.",
    }
    created = create_hypothesis(
        index,
        "Compression predicts expansion.",
        "Compression precedes expansion.",
        "ATR contracts for 10 bars",
        "ATR expands within 5 bars",
        "XAUUSD",
        "M5",
        "no expansion after 5 bars",
        source_refs=[ref],
    )

    testing = set_status(index, created.hypothesis_id, "testing")
    assert testing.status == "testing"
    assert testing.updated_at >= created.updated_at

    linked = link_experiment(index, created.hypothesis_id, "backtest:atr-expansion:001")
    assert linked.experiment_refs == ("backtest:atr-expansion:001",)
    repeat = link_experiment(index, created.hypothesis_id, "backtest:atr-expansion:001")
    assert repeat.experiment_refs == ("backtest:atr-expansion:001",)

    supported = set_status(index, created.hypothesis_id, "supported")
    assert supported.status == "supported"
    with pytest.raises(HypothesisError, match="cannot move"):
        set_status(index, created.hypothesis_id, "testing")
    with pytest.raises(ValueError):
        set_status(index, created.hypothesis_id, "unknown")
    index.close()


def test_hypothesis_rejects_untraceable_or_invalid_input(tmp_path: Path) -> None:
    index = _index(tmp_path)
    _document(tmp_path, index, "one.md", "# One\n\nSome text.")
    base = {
        "statement": "A statement.",
        "observed": "Observed.",
        "proposed_condition": "Condition.",
        "expected_effect": "Effect.",
        "market_scope": "XAUUSD",
        "timeframe_scope": "M5",
        "falsification_test": "It fails.",
    }

    with pytest.raises(HypothesisError, match="unknown source document"):
        create_hypothesis(
            index,
            source_refs=[SourceRef(doc_id="0" * 64, section_ordinal=0, quote="x")],
            **base,
        )
    with pytest.raises(HypothesisError, match="no section"):
        record = index.list_documents()[0]
        create_hypothesis(
            index,
            source_refs=[SourceRef(doc_id=record.doc_id, section_ordinal=7, quote="x")],
            **base,
        )
    with pytest.raises(ValueError, match="must not be empty"):
        create_hypothesis(
            index, statement="   ", **{k: v for k, v in base.items() if k != "statement"}
        )
    assert list_hypotheses(index) == []
    assert HYPOTHESIS_STATUSES == {"proposed", "testing", "supported", "rejected"}
    index.close()


def test_hypothesis_sources_cannot_cite_a_failed_document(tmp_path: Path) -> None:
    index = _index(tmp_path)
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.4")
    record = DocumentIngestor(tmp_path / "knowledge").ingest(source)
    index.add_document(record)

    with pytest.raises(HypothesisError, match="not 'indexed'"):
        create_hypothesis(
            index,
            "Statement.",
            "Observed.",
            "Condition.",
            "Effect.",
            "XAUUSD",
            "M5",
            "Fails.",
            source_refs=[SourceRef(doc_id=record.doc_id, section_ordinal=0, quote="x")],
        )
    index.close()


def test_unknown_hypothesis_operations_raise(tmp_path: Path) -> None:
    index = _index(tmp_path)
    with pytest.raises(HypothesisError, match="unknown hypothesis"):
        set_status(index, "missing", "testing")
    with pytest.raises(HypothesisError, match="unknown hypothesis"):
        link_experiment(index, "missing", "experiment")
    assert get_hypothesis(index, "missing") is None
    index.close()


def test_closed_index_rejects_operations(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.close()
    with pytest.raises(KnowledgeIndexError, match="closed"):
        index.list_documents()
