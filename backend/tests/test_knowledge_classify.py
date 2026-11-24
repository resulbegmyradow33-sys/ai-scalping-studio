"""Deterministic sentence classification and its effect on extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from scalping.knowledge import (
    CLAIM_KINDS,
    DocumentIngestor,
    DocumentRecord,
    KnowledgeIndex,
    KnowledgeIndexError,
    classify_document,
    classify_sentence,
    extract_hypotheses,
)


def _document(tmp_path: Path, index: KnowledgeIndex, text: str) -> DocumentRecord:
    source = tmp_path / "notes.md"
    source.write_text(text, encoding="utf-8")
    record = DocumentIngestor(tmp_path / "knowledge").ingest(source)
    index.add_document(record)
    return record


@pytest.mark.parametrize(
    ("sentence", "kind"),
    [
        ("When volatility expands, momentum signals become less reliable.", "claim"),
        ("Wide spreads can reduce the edge of a scalp.", "claim"),
        ("Liquidity refers to resting orders in the book.", "concept"),
        ("A backtest of 215 trades produced +0.159R expectancy.", "observation"),
        ("I think momentum is probably dead in this market.", "opinion"),
        ("For example, gold often gaps at the London open.", "example"),
        ("If you are new, read the manual.", None),
    ],
)
def test_classification_examples(sentence: str, kind: str | None) -> None:
    assert classify_sentence(sentence) == kind


def test_example_wins_over_an_embedded_claim() -> None:
    sentence = "For example, when spreads widen the edge can fall."

    assert classify_sentence(sentence) == "example"


def test_document_classification_is_ordered_and_traceable(tmp_path: Path) -> None:
    index = KnowledgeIndex(tmp_path / "knowledge.sqlite3")
    record = _document(
        tmp_path,
        index,
        "# Claims\n\n"
        "When volatility expands, momentum signals become less reliable. "
        "I think this edge is probably gone. "
        "Liquidity refers to resting orders in the book.\n\n"
        "## Evidence\n\nA backtest of 215 trades produced +0.159R expectancy.\n",
    )

    rows = classify_document(index, record.doc_id)

    assert [row.kind for row in rows] == ["claim", "opinion", "concept", "observation"]
    assert [row.section_ordinal for row in rows] == [0, 0, 0, 1]
    assert rows[0].sentence.startswith("When volatility expands")
    assert {row.kind for row in rows}.issubset(set(CLAIM_KINDS))


def test_unknown_document_is_rejected(tmp_path: Path) -> None:
    index = KnowledgeIndex(tmp_path / "knowledge.sqlite3")

    with pytest.raises(KnowledgeIndexError, match="unknown document"):
        classify_document(index, "missing")


def test_extraction_only_uses_claims_and_observations(tmp_path: Path) -> None:
    index = KnowledgeIndex(tmp_path / "knowledge.sqlite3")
    record = _document(
        tmp_path,
        index,
        "# Mixed\n\n"
        "When volatility expands, momentum signals become less reliable. "
        "I think momentum is probably dead in this market. "
        "Liquidity refers to resting orders in the book. "
        "For example, when spreads widen the edge can fall.\n",
    )

    result = extract_hypotheses(index, record.doc_id)

    assert [item.statement for item in result.created] == [
        "When volatility expands, momentum signals become less reliable"
    ]
    assert all(item.status == "proposed" for item in result.created)
