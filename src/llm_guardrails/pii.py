"""Regex-based PII detection and redaction.

This module intentionally uses pattern matching rather than a statistical
NER model: it is fast, has zero dependencies, and its behavior is fully
predictable/auditable. The tradeoff is recall - see the "Limitations"
section of the README for what this will and will not catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Pattern


class PIIType(str, Enum):
    """Categories of PII this module knows how to find."""

    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"


@dataclass(frozen=True)
class PIIMatch:
    """A single redacted span."""

    pii_type: PIIType
    original: str
    start: int
    end: int


@dataclass
class RedactionResult:
    """Outcome of running :meth:`PIIRedactor.redact` over a string."""

    text: str
    """The (possibly) redacted text."""

    matches: list[PIIMatch] = field(default_factory=list)
    """Every span that was redacted, in original-text order."""

    @property
    def found_pii(self) -> bool:
        return len(self.matches) > 0

    def counts(self) -> dict[PIIType, int]:
        counts: dict[PIIType, int] = {}
        for match in self.matches:
            counts[match.pii_type] = counts.get(match.pii_type, 0) + 1
        return counts


# --- Patterns -----------------------------------------------------------
#
# Order matters only in that CREDIT_CARD and PHONE candidate spans overlap
# heavily (a 13-16 digit run with separators looks like both). We resolve
# overlaps in `PIIRedactor.redact` rather than relying on pattern order.

_EMAIL_RE: Pattern[str] = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

_SSN_RE: Pattern[str] = re.compile(
    r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
)

# Candidate credit-card-shaped runs: 13-19 digits, optionally grouped by
# spaces or dashes in blocks of 4 (with a shorter final block). Validity is
# confirmed separately via the Luhn checksum to cut down on false positives
# from things like phone numbers or order IDs.
_CREDIT_CARD_CANDIDATE_RE: Pattern[str] = re.compile(
    r"\b(?:\d[ -]?){13,19}\b"
)

# US-style phone numbers, with or without country code / separators.
_PHONE_RE: Pattern[str] = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
)


def _luhn_checksum(digits: str) -> bool:
    """Standard Luhn (mod 10) check used by all major card networks."""
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _find_credit_cards(text: str) -> list[re.Match[str]]:
    matches = []
    for m in _CREDIT_CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"[ -]", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_checksum(digits):
            matches.append(m)
    return matches


DEFAULT_TOKEN_FORMAT = "[REDACTED:{type}]"


class PIIRedactor:
    """Finds and redacts PII in free text.

    Example:
        >>> redactor = PIIRedactor()
        >>> result = redactor.redact("Email me at jane@example.com")
        >>> result.text
        'Email me at [REDACTED:EMAIL]'
    """

    def __init__(
        self,
        enabled_types: set[PIIType] | None = None,
        token_format: str = DEFAULT_TOKEN_FORMAT,
    ) -> None:
        self.enabled_types = enabled_types or set(PIIType)
        self.token_format = token_format

    def _token(self, pii_type: PIIType) -> str:
        return self.token_format.format(type=pii_type.value)

    def _all_candidate_matches(
        self, text: str
    ) -> list[tuple[PIIType, re.Match[str]]]:
        candidates: list[tuple[PIIType, re.Match[str]]] = []

        if PIIType.EMAIL in self.enabled_types:
            candidates += [(PIIType.EMAIL, m) for m in _EMAIL_RE.finditer(text)]
        if PIIType.SSN in self.enabled_types:
            candidates += [(PIIType.SSN, m) for m in _SSN_RE.finditer(text)]
        if PIIType.CREDIT_CARD in self.enabled_types:
            candidates += [
                (PIIType.CREDIT_CARD, m) for m in _find_credit_cards(text)
            ]
        if PIIType.PHONE in self.enabled_types:
            candidates += [(PIIType.PHONE, m) for m in _PHONE_RE.finditer(text)]

        return candidates

    def find(self, text: str) -> list[PIIMatch]:
        """Return non-overlapping PII matches without modifying the text."""
        candidates = self._all_candidate_matches(text)

        # Resolve overlaps: earliest start wins, then longest span. This
        # keeps e.g. a valid credit-card span from being partially
        # shadowed by a phone-number candidate at the same offset.
        candidates.sort(key=lambda c: (c[1].start(), -(c[1].end() - c[1].start())))

        selected: list[PIIMatch] = []
        last_end = -1
        for pii_type, m in candidates:
            if m.start() < last_end:
                continue
            selected.append(PIIMatch(pii_type, m.group(), m.start(), m.end()))
            last_end = m.end()

        return selected

    def redact(self, text: str) -> RedactionResult:
        """Redact all detected PII, replacing each span with a token."""
        matches = self.find(text)
        if not matches:
            return RedactionResult(text=text, matches=[])

        pieces: list[str] = []
        cursor = 0
        for match in matches:
            pieces.append(text[cursor : match.start])
            pieces.append(self._token(match.pii_type))
            cursor = match.end
        pieces.append(text[cursor:])

        return RedactionResult(text="".join(pieces), matches=matches)


def redact(text: str, enabled_types: set[PIIType] | None = None) -> RedactionResult:
    """Convenience wrapper around ``PIIRedactor(enabled_types).redact(text)``."""
    return PIIRedactor(enabled_types=enabled_types).redact(text)
