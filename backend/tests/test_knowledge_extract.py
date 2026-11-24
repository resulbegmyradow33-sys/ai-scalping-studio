"""Deterministic prose → hypothesis drafts: patterns, sources and dedupe."""

from __future__ import annotations

from pathlib import Path

import pytest

from scalping.knowledge import (
    DocumentIngestor,
    DocumentRecord,
    KnowledgeIndex,
    KnowledgeIndexError,
    extract_hypotheses,
    list_hypotheses,
)

TEXT = """# Claims

When volatility expands, momentum signals become less reliable. A signed order-flow imbalance predicts short-term continuation. The more the spread widens, the lower the edge. Gold is a metal. If you are new, read the manual.

## Liquidity

Sweeps of session lows lead to reversals.
"""


def _document(tmp_path: Path, index: KnowledgeIndex, text: str = TEXT) -> DocumentRecord:
    source = tmp_path / "notes.md"
    source.write_text(text, encoding="utf-8")
    record = DocumentIngestor(tmp_path / "knowledge").ingest(source)
    index.add_document(record)
    return record


def _fingerprint(result) -> list[tuple]:
    return [
        (
            item.statement,
            item.proposed_condition,
            item.expected_effect,
            item.observed,
            tuple((ref.doc_id, ref.section_ordinal, ref.quote) for ref in item.source_refs),
            item.status,
        )
        for item in result.created
    ]


def test_extracts_patterns_in_section_order(tmp_path: Path) -> None:
    index = KnowledgeIndex(tmp_path / "knowledge.sqlite3")
    record = _document(tmp_path, index)

    result = extract_hypotheses(index, record.doc_id)

    assert len(result.created) == 4
    assert [item.proposed_condition for item in result.created] == [
        "volatility expands",
        "A signed order-flow imbalance",
        "more spread widens",
        "Sweeps of session lows",
    ]
    assert [item.expected_effect for item in result.created] == [
        "momentum signals become less reliable",
        "short-term continuation",
        "lower the edge",
        "reversals",
    ]
    assert all(item.status == "proposed" for item in result.created)
    assert result.created[0].source_refs[0].quote == (
        "When volatility expands, momentum signals become less reliable"
    )
    assert result.created[3].source_refs[0].section_ordinal == 1
    assert result.created[0].observed.startswith("Claims")
    assert "Liquidity" in result.created[3].observed
    assert "out-of-sample" in result.created[0].falsification_test


def test_second_run_skips_duplicates(tmp_path: Path) -> None:
    index = KnowledgeIndex(tmp_path / "knowledge.sqlite3")
    record = _document(tmp_path, index)

    first = extract_hypotheses(index, record.doc_id)
    second = extract_hypotheses(index, record.doc_id)

    assert len(first.created) == 4
    assert second.created == ()
    assert second.skipped == 4
    assert len(list_hypotheses(index)) == 4


def test_generic_prose_is_not_turned_into_research(tmp_path: Path) -> None:
    index = KnowledgeIndex(tmp_path / "knowledge.sqlite3")
    record = _document(
        tmp_path,
        index,
        "# Intro\n\nIf you are new, read the manual. Gold is a metal. The market is open.\n",
    )

    result = extract_hypotheses(index, record.doc_id)

    assert result.created == ()
    assert result.scanned_sentences >= 2
    assert list_hypotheses(index) == []


def test_extraction_is_deterministic_and_capped(tmp_path: Path) -> None:
    first_index = KnowledgeIndex(tmp_path / "a.sqlite3")
    first_doc = _document(tmp_path, first_index)
    first = extract_hypotheses(first_index, first_doc.doc_id)

    other = tmp_path / "other"
    other.mkdir()
    second_index = KnowledgeIndex(tmp_path / "b.sqlite3")
    second_doc = _document(other, second_index)
    second = extract_hypotheses(second_index, second_doc.doc_id)

    assert _fingerprint(first) == _fingerprint(second)

    capped = extract_hypotheses(first_index, first_doc.doc_id, max_hypotheses=1)
    assert capped.created == ()  # already stored above
    with pytest.raises(ValueError, match="max_hypotheses"):
        extract_hypotheses(first_index, first_doc.doc_id, max_hypotheses=0)


def test_unknown_document_is_rejected(tmp_path: Path) -> None:
    index = KnowledgeIndex(tmp_path / "knowledge.sqlite3")

    with pytest.raises(KnowledgeIndexError, match="unknown document"):
        extract_hypotheses(index, "does-not-exist")
