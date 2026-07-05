"""Shared exception hierarchy for llm-guardrails."""

from __future__ import annotations


class GuardrailsError(Exception):
    """Base class for all errors raised by llm-guardrails."""


class InjectionBlockedError(GuardrailsError):
    """Raised when a prompt is blocked by :class:`~llm_guardrails.injection.InjectionDetector`."""

    def __init__(self, message: str, score: float) -> None:
        super().__init__(message)
        self.score = score
