"""Deterministic hypothesis drafts from prose (Phase 17, slice 2).

No language model, no ``eval``/``exec`` and no network: a small set of
sentence patterns proposes ``proposed`` hypotheses that cite the exact section
and sentence they came from. Everything is pattern-matched, so the same input
always yields the same drafts in the same order.

The extractor cannot test, promote or trade. A draft starts in ``proposed``;
only a human moves it to ``testing``, and a quantitative experiment decides the
outcome. Sentences without a market term are ignored, which keeps ordinary
prose ("if you are new, read the manual") out of the research queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from scalping.knowledge.classify import (
    classify_sentence,
    is_market_claim,
    match_claim_pattern,
    split_sentences,
)
from scalping.knowledge.hypothesis import Hypothesis, SourceRef, create_hypothesis, list_hypotheses
from scalping.knowledge.index import KnowledgeIndex, KnowledgeIndexError, SectionRecord

__all__ = ["DEFAULT_FALSIFICATION", "ExtractionResult", "extract_hypotheses"]

DEFAULT_FALSIFICATION: Final[str] = (
    "Test the claimed relation out-of-sample with simulator costs; reject it if "
    "the effect is absent or the sign is reversed."
)

@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Drafts created in order, plus how many candidates were skipped."""

    created: tuple[Hypothesis, ...]
    skipped: int
    scanned_sentences: int

    def to_dict(self) -> dict[str, object]:
        return {
            "created": [item.to_dict() for item in self.created],
            "skipped": self.skipped,
            "scanned_sentences": self.scanned_sentences,
        }


def _existing_keys(index: KnowledgeIndex) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for hypothesis in list_hypotheses(index):
        statement = hypothesis.statement.casefold()
        for ref in hypothesis.source_refs:
            keys.add((statement, ref.doc_id, ref.section_ordinal))
    return keys


def _observed(section: SectionRecord) -> str:
    heading = section.heading or section.label or f"section {section.ordinal}"
    return f"{heading} (section {section.ordinal})"


def extract_hypotheses(
    index: KnowledgeIndex,
    doc_id: str,
    *,
    max_hypotheses: int = 50,
    market_scope: str = "unspecified",
    timeframe_scope: str = "unspecified",
) -> ExtractionResult:
    """Propose hypotheses from one indexed document, deterministically.

    Duplicate ``(statement, section)`` pairs already stored are skipped, so
    running the extractor twice does not multiply the research queue.
    """
    if not isinstance(index, KnowledgeIndex):
        raise TypeError("index must be a KnowledgeIndex")
    if max_hypotheses < 1:
        raise ValueError("max_hypotheses must be >= 1")
    document = index.document(doc_id)
    if document is None:
        raise KnowledgeIndexError(f"unknown document {doc_id!r}")
    if document.status != "indexed":
        raise KnowledgeIndexError(
            f"document {doc_id!r} has status {document.status!r}, not 'indexed'"
        )

    seen = _existing_keys(index)
    created: list[Hypothesis] = []
    skipped = 0
    scanned = 0
    for section in index.sections(doc_id):
        for sentence in split_sentences(section.text):
            scanned += 1
            if len(created) >= max_hypotheses:
                break
            if not is_market_claim(sentence):
                continue
            match = match_claim_pattern(sentence)
            if match is None:
                continue
            condition, effect = match
            if not condition or not effect or not is_market_claim(f"{condition} {effect}"):
                continue
            # Opinions, definitions and examples are research context, not
            # testable claims; only claims/observations may become drafts.
            if classify_sentence(sentence) not in {"claim", "observation"}:
                continue
            key = (sentence.casefold(), doc_id, section.ordinal)
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            created.append(
                create_hypothesis(
                    index,
                    statement=sentence,
                    observed=_observed(section),
                    proposed_condition=condition,
                    expected_effect=effect,
                    market_scope=market_scope,
                    timeframe_scope=timeframe_scope,
                    falsification_test=DEFAULT_FALSIFICATION,
                    assumptions=("Extracted from prose; untested.",),
                    source_refs=(SourceRef(doc_id, section.ordinal, sentence),),
                    status="proposed",
                )
            )
        if len(created) >= max_hypotheses:
            break
    return ExtractionResult(created=tuple(created), skipped=skipped, scanned_sentences=scanned)
