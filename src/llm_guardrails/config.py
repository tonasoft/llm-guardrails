"""Configuration for :class:`~llm_guardrails.client.GuardedClient` and
:func:`~llm_guardrails.decorator.guarded_llm_call`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Set, Type

from pydantic import BaseModel

from llm_guardrails.pii import PIIType
from llm_guardrails.usage_logger import UsageRecord


class OnInjection(str, Enum):
    """What to do when the injection detector flags a prompt."""

    ALLOW = "allow"
    """Never block. Verdict/score are still computed and logged."""

    WARN = "warn"
    """Same as ALLOW - the call proceeds. Distinct value for callers who
    want to branch on configured intent rather than the verdict itself."""

    BLOCK = "block"
    """Raise :class:`~llm_guardrails.exceptions.InjectionBlockedError`
    instead of calling the underlying model when the verdict is BLOCKED."""


@dataclass
class GuardrailsConfig:
    redact_input: bool = True
    redact_output: bool = True
    pii_types: Optional[Set[PIIType]] = None

    on_injection: OnInjection = OnInjection.BLOCK
    injection_suspicious_threshold: float = 2.0
    injection_block_threshold: float = 4.0

    response_model: Optional[Type[BaseModel]] = None
    raise_on_schema_error: bool = True

    log_file: Optional[str | Path] = None
    log_callback: Optional[Callable[[UsageRecord], None]] = None
