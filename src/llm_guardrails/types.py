"""Shared result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel

from llm_guardrails.injection import InjectionResult
from llm_guardrails.pii import RedactionResult

T = TypeVar("T", bound=BaseModel)


@dataclass
class GuardedResponse(Generic[T]):
    """Everything a guarded call produced, in one place."""

    raw: Any
    """The underlying provider response object, untouched."""

    text: str
    """The model's text output, after PII redaction (if enabled)."""

    parsed: Optional[T]
    """Parsed & validated Pydantic instance, if a ``response_model`` was given."""

    input_redaction: RedactionResult
    output_redaction: RedactionResult
    injection: InjectionResult
    latency_ms: float
