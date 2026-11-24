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

import re
from dataclasses import dataclass
from typing import Final

from scalping.knowledge.hypothesis import Hypothesis, SourceRef, create_hypothesis, list_hypotheses
from scalping.knowledge.index import KnowledgeIndex, KnowledgeIndexError, SectionRecord

__all__ = ["DEFAULT_FALSIFICATION", "ExtractionResult", "extract_hypotheses"]

#: Term(s) that make a sentence a market claim rather than generic prose.
_MARKET_TERMS: Final[frozenset[str]] = frozenset(
    {
        "bar",
        "breakout",
        "candle",
        "close",
        "drawdown",
        "edge",
        "entry",
        "exit",
        "flow",
        "gap",
        "high",
        "level",
        "liquidity",
        "low",
        "market",
        "momentum",
        "open",
        "price",
        "range",
        "return",
        "risk",
        "session",
        "spread",
        "stop",
        "support",
        "resistance",
        "trend",
        "volatility",
        "volume",
        "vwap",
    }
)

DEFAULT_FALSIFICATION: Final[str] = (
    "Test the claimed relation out-of-sample with simulator costs; reject it if "
    "the effect is absent or the sign is reversed."
)

_CONDITIONAL: Final[re.Pattern[str]] = re.compile(
    r"^(?:when|whenever|if|after|once)\s+(?P<condition>.{3,200}?)\s*,\s*"
    r"(?:then\s+)?(?P<effect>.{3,280}?)\.?$",
    re.IGNORECASE,
)
_CAUSAL: Final[re.Pattern[str]] = re.compile(
    r"^(?P<condition>.{3,200}?)\s+"
    r"(?:predicts|leads?\s+to|implies|causes|results?\s+in|is\s+associated\s+with|"
    r"correlates\s+with|signals)\s+"
    r"(?P<effect>.{3,280}?)\.?$",
    re.IGNORECASE,
)
_GRADED: Final[re.Pattern[str]] = re.compile(
    r"^the\s+more\s+(?P<condition>.{3,200}?)\s*,\s*"
    r"the\s+(?P<direction>more|less|higher|lower|stronger|weaker)\s+"
    r"(?P<effect>.{3,280}?)\.?$",
    re.IGNORECASE,
)

_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


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


def _clean(text: str) -> str:
    value = text.replace("*", "").replace("`", "").replace("_", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value.rstrip(" .,;:")


def _market_claim(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower()))
    stems = {word[:-1] if word.endswith("s") and len(word) > 3 else word for word in words}
    return bool((words | stems) & _MARKET_TERMS)


def _sentences(text: str) -> list[str]:
    # Section text may carry its Markdown heading; a heading is not a claim and
    # would otherwise glue two sentences across a section boundary.
    body = re.sub(r"(?m)^\s{0,3}#{1,6}\s.*$", "", text)
    out: list[str] = []
    for raw in _SENTENCE_SPLIT.split(body.replace("\n", " ")):
        sentence = _clean(raw)
        if 15 <= len(sentence) <= 400:
            out.append(sentence)
    return out


def _match(sentence: str) -> tuple[str, str] | None:
    """Return ``(condition, expected_effect)`` or ``None``."""
    conditional = _CONDITIONAL.match(sentence)
    if conditional is not None:
        return _clean(conditional.group("condition")), _clean(conditional.group("effect"))
    causal = _CAUSAL.match(sentence)
    if causal is not None:
        return _clean(causal.group("condition")), _clean(causal.group("effect"))
    graded = _GRADED.match(sentence)
    if graded is not None:
        condition = _clean(graded.group("condition"))
        condition = condition.removeprefix("the ")
        return (
            f"more {condition}",
            f"{graded.group('direction').lower()} {_clean(graded.group('effect'))}",
        )
    return None


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
        for sentence in _sentences(section.text):
            scanned += 1
            if len(created) >= max_hypotheses:
                break
            if not _market_claim(sentence):
                continue
            match = _match(sentence)
            if match is None:
                continue
            condition, effect = match
            if not condition or not effect or not _market_claim(f"{condition} {effect}"):
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
