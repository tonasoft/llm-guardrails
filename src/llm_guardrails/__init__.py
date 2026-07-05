"""llm-guardrails: a drop-in wrapper adding PII redaction, prompt-injection
detection, output schema validation, and usage logging to LLM API calls.
"""

from llm_guardrails.client import GuardedClient
from llm_guardrails.config import GuardrailsConfig, OnInjection
from llm_guardrails.decorator import guarded_llm_call
from llm_guardrails.exceptions import GuardrailsError, InjectionBlockedError
from llm_guardrails.injection import InjectionDetector, InjectionResult, Verdict
from llm_guardrails.pii import PIIRedactor, PIIType, RedactionResult
from llm_guardrails.schema import SchemaValidationError, validate_output
from llm_guardrails.types import GuardedResponse
from llm_guardrails.usage_logger import UsageLogger, UsageRecord

__version__ = "0.1.0"

__all__ = [
    "GuardedClient",
    "GuardedResponse",
    "GuardrailsConfig",
    "GuardrailsError",
    "InjectionBlockedError",
    "InjectionDetector",
    "InjectionResult",
    "OnInjection",
    "PIIRedactor",
    "PIIType",
    "RedactionResult",
    "SchemaValidationError",
    "UsageLogger",
    "UsageRecord",
    "Verdict",
    "guarded_llm_call",
    "validate_output",
    "__version__",
]
