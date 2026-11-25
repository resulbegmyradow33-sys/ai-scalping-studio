"""Deterministic sentence classification for research documents.

No language model: patterns decide whether a sentence is a claim, a definition,
an example, an opinion or an empirical observation. The classes are useful for
triage, not for truth:

* a **claim** may become a hypothesis draft, but only after a human moves it;
* an **observation** cites a measurement (backtest, sample, percentage) and can
  seed a hypothesis, but prose is still not evidence;
* **concept**, **example** and **opinion** sentences never create hypotheses.

The classifier is intentionally conservative: an unclassified sentence returns
``None`` instead of being forced into a bucket.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from scalping.knowledge.index import KnowledgeIndex, KnowledgeIndexError

__all__ = [
    "CLAIM_KINDS",
    "ClaimKind",
    "ClassifiedSentence",
    "classify_document",
    "classify_sentence",
    "clean_sentence",
    "is_market_claim",
    "match_claim_pattern",
    "split_sentences",
]

ClaimKind = Literal["claim", "concept", "example", "opinion", "observation"]
CLAIM_KINDS: Final[tuple[ClaimKind, ...]] = (
    "claim",
    "concept",
    "example",
    "opinion",
    "observation",
)

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

_EXAMPLE: Final[re.Pattern[str]] = re.compile(
    r"\b(for\s+example|for\s+instance|e\.g\.|such\s+as|example:|examples?:)\b",
    re.IGNORECASE,
)
_OPINION: Final[re.Pattern[str]] = re.compile(
    r"\b((i|we)\s+(think|believe|expect|suspect)|in\s+my\s+(view|opinion)|"
    r"arguably|probably|presumably|seems?|should)\b",
    re.IGNORECASE,
)
_OBSERVATION: Final[re.Pattern[str]] = re.compile(
    r"\b(backtest|sample|measured|observed|win\s+rate|expectancy|profit\s+factor|"
    r"drawdown|p-value|statistically|trades?|bars?)\b"
    r"|\b\d+(?:\.\d+)?\s*(%|r\b|pips?|points?|days?|weeks?|months?|hours?|minutes?)",
    re.IGNORECASE,
)
_CONCEPT: Final[re.Pattern[str]] = re.compile(
    r"\b(refers?\s+to|is\s+defined\s+as|is\s+called|means\s+that)\b"
    r"|\b(is|are)\s+(a|an|the)\s+[a-z]",
    re.IGNORECASE,
)
_CLAIM_MODAL: Final[re.Pattern[str]] = re.compile(
    r"\b(can|could|will|would|may|might|tends?\s+to|increases?|decreases?|"
    r"reduces?|improves?|worsens?|drives?)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


@dataclass(frozen=True, slots=True)
class ClassifiedSentence:
    """One classified sentence and the section it came from."""

    section_ordinal: int
    sentence: str
    kind: ClaimKind

    def to_dict(self) -> dict[str, object]:
        return {
            "section_ordinal": self.section_ordinal,
            "sentence": self.sentence,
            "kind": self.kind,
        }


def clean_sentence(text: str) -> str:
    """Drop Markdown emphasis and collapse whitespace; no other rewriting."""
    value = text.replace("*", "").replace("`", "").replace("_", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value.rstrip(" .,;:")


def split_sentences(text: str) -> list[str]:
    """Split a section into sentences, ignoring Markdown headings.

    Headings are not claims and would otherwise glue sentences together across
    a section boundary.
    """
    body = re.sub(r"(?m)^\s{0,3}#{1,6}\s.*$", "", text)
    out: list[str] = []
    for raw in _SENTENCE_SPLIT.split(body.replace("\n", " ")):
        sentence = clean_sentence(raw)
        if 15 <= len(sentence) <= 400:
            out.append(sentence)
    return out


def is_market_claim(text: str) -> bool:
    """True when a sentence speaks about a market/price/trading concept."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    stems = {word[:-1] if word.endswith("s") and len(word) > 3 else word for word in words}
    return bool((words | stems) & _MARKET_TERMS)


def match_claim_pattern(sentence: str) -> tuple[str, str] | None:
    """Return ``(condition, expected_effect)`` for a structured claim."""
    conditional = _CONDITIONAL.match(sentence)
    if conditional is not None:
        return clean_sentence(conditional.group("condition")), clean_sentence(
            conditional.group("effect")
        )
    causal = _CAUSAL.match(sentence)
    if causal is not None:
        return clean_sentence(causal.group("condition")), clean_sentence(
            causal.group("effect")
        )
    graded = _GRADED.match(sentence)
    if graded is not None:
        condition = clean_sentence(graded.group("condition"))
        condition = condition.removeprefix("the ")
        return (
            f"more {condition}",
            f"{graded.group('direction').lower()} {clean_sentence(graded.group('effect'))}",
        )
    return None


def classify_sentence(text: str) -> ClaimKind | None:
    """Classify one sentence, or ``None`` when it is not research prose."""
    sentence = clean_sentence(text)
    if len(sentence) < 15:
        return None
    if _EXAMPLE.search(sentence):
        return "example"
    if _OPINION.search(sentence):
        return "opinion"
    if _OBSERVATION.search(sentence):
        return "observation"
    if _CONCEPT.search(sentence):
        return "concept"
    if not is_market_claim(sentence):
        return None
    if match_claim_pattern(sentence) is not None:
        return "claim"
    if _CLAIM_MODAL.search(sentence):
        return "claim"
    return None


def classify_document(
    index: KnowledgeIndex,
    doc_id: str,
) -> list[ClassifiedSentence]:
    """Classify every research sentence in one indexed document, in order."""
    if not isinstance(index, KnowledgeIndex):
        raise TypeError("index must be a KnowledgeIndex")
    document = index.document(doc_id)
    if document is None:
        raise KnowledgeIndexError(f"unknown document {doc_id!r}")
    if document.status != "indexed":
        raise KnowledgeIndexError(
            f"document {doc_id!r} has status {document.status!r}, not 'indexed'"
        )
    rows: list[ClassifiedSentence] = []
    for section in index.sections(doc_id):
        for sentence in split_sentences(section.text):
            kind = classify_sentence(sentence)
            if kind is not None:
                rows.append(ClassifiedSentence(section.ordinal, sentence, kind))
    return rows
