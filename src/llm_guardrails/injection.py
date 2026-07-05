"""Heuristic prompt-injection detection.

IMPORTANT - read this before trusting it in production:

This is pattern/keyword matching, not a trained classifier. It will catch
common, "textbook" injection attempts (the kind found in blog posts and
CTF writeups) but is trivially bypassed by:

  * paraphrasing ("kindly set aside the rules above" instead of
    "ignore previous instructions")
  * translation into another language
  * encoding tricks (base64, ROT13, zero-width characters, homoglyphs)
  * splitting the payload across multiple turns/messages
  * indirect injection hidden in retrieved documents/tool output, where
    the "prompt" this module sees is only the user turn, not the full
    context the model actually reads

Treat this as a cheap first line of defense (and an audit trail via
logging), not a security boundary. Anything security-critical downstream
of an LLM call still needs to validate its own inputs regardless of what
this module reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class InjectionMatch:
    pattern_name: str
    matched_text: str
    weight: float


@dataclass
class InjectionResult:
    text: str
    score: float
    verdict: Verdict
    matches: list[InjectionMatch] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return self.verdict is Verdict.BLOCKED


# (name, weight, compiled pattern). Weights are heuristic and tuned by
# hand, not learned - adjust `InjectionDetector` thresholds for your
# risk tolerance rather than fighting these numbers.
_PATTERNS: list[tuple[str, float, re.Pattern[str]]] = [
    (
        "instruction_override",
        4.0,
        re.compile(
            r"\b(ignore|disregard|forget)\b[^.\n]{0,40}\b"
            r"(previous|prior|above|earlier|all)\b[^.\n]{0,40}\b"
            r"(instructions?|rules?|prompt|directives?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_override_short",
        3.0,
        re.compile(
            r"\b(disregard|ignore)\s+(the\s+)?(above|previous|prior)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_exfiltration",
        4.0,
        re.compile(
            r"\b(reveal|print|show|repeat|output|leak|what\s+is)\b[^.\n]{0,40}\b"
            r"(system\s+prompt|initial\s+instructions?|your\s+instructions?|"
            r"hidden\s+prompt)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_override",
        2.5,
        re.compile(
            r"\b(you\s+are\s+now|act\s+as|pretend\s+(you\s+are|to\s+be)|"
            r"roleplay\s+as)\b[^.\n]{0,40}\b"
            r"(developer\s+mode|dan|jailbroken?|unfiltered|no\s+restrictions|"
            r"without\s+(any\s+)?(restrictions|limitations|filters))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "known_jailbreak_alias",
        3.0,
        re.compile(r"\b(DAN|do\s+anything\s+now|developer\s+mode\s+enabled)\b"),
    ),
    (
        "fake_delimiter_injection",
        2.0,
        re.compile(
            r"(<\|.*?\|>|\[/?system\]|\[/?INST\]|###\s*(system|instruction)|"
            r"end\s+of\s+(prompt|instructions|system\s+message))",
            re.IGNORECASE,
        ),
    ),
    (
        "exfiltration_request",
        2.5,
        re.compile(
            r"\b(send|email|post|forward)\b[^.\n]{0,40}\b"
            r"(this|the\s+above|conversation|data|following)\b[^.\n]{0,40}\bto\b",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_or_secret_probe",
        2.0,
        re.compile(
            r"\b(api\s*key|password|secret\s+key|access\s+token|credentials)\b"
            r"[^.\n]{0,20}\b(is|are|:)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "suspicious_base64_blob",
        1.5,
        re.compile(r"\b(?:[A-Za-z0-9+/]{4}){10,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\b"),
    ),
]


class InjectionDetector:
    """Scores free text for prompt-injection heuristics.

    See the module docstring for what this can and cannot catch.
    """

    def __init__(
        self,
        suspicious_threshold: float = 2.0,
        block_threshold: float = 4.0,
        patterns: list[tuple[str, float, re.Pattern[str]]] | None = None,
    ) -> None:
        if suspicious_threshold > block_threshold:
            raise ValueError("suspicious_threshold must be <= block_threshold")
        self.suspicious_threshold = suspicious_threshold
        self.block_threshold = block_threshold
        self.patterns = patterns if patterns is not None else _PATTERNS

    def scan(self, text: str) -> InjectionResult:
        matches: list[InjectionMatch] = []
        score = 0.0

        for name, weight, pattern in self.patterns:
            for m in pattern.finditer(text):
                matches.append(InjectionMatch(name, m.group(), weight))
                score += weight

        if score >= self.block_threshold:
            verdict = Verdict.BLOCKED
        elif score >= self.suspicious_threshold:
            verdict = Verdict.SUSPICIOUS
        else:
            verdict = Verdict.SAFE

        return InjectionResult(text=text, score=score, verdict=verdict, matches=matches)


def scan(text: str) -> InjectionResult:
    """Convenience wrapper around ``InjectionDetector().scan(text)``."""
    return InjectionDetector().scan(text)
