"""The guarded-call pipeline shared by :class:`~llm_guardrails.client.GuardedClient`
and :func:`~llm_guardrails.decorator.guarded_llm_call`.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from llm_guardrails.config import GuardrailsConfig, OnInjection
from llm_guardrails.exceptions import InjectionBlockedError
from llm_guardrails.injection import InjectionDetector, Verdict
from llm_guardrails.pii import PIIRedactor, RedactionResult
from llm_guardrails.schema import SchemaValidationError, validate_output
from llm_guardrails.types import GuardedResponse
from llm_guardrails.usage_logger import UsageLogger, UsageRecord

T = TypeVar("T", bound=BaseModel)


def run_guarded_pipeline(
    *,
    prompt_text: str,
    call: Callable[[str], Any],
    extract_response_text: Callable[[Any], str],
    provider: str,
    config: GuardrailsConfig,
    pii_redactor: PIIRedactor,
    injection_detector: InjectionDetector,
    logger: UsageLogger,
    response_model: Optional[Type[T]] = None,
) -> GuardedResponse[T]:
    """Run the full guardrails pipeline around a single LLM call.

    ``call`` receives the (possibly redacted) prompt text and must return
    the raw provider response. ``extract_response_text`` turns that raw
    response into plain text for redaction/schema validation.
    """
    injection_result = injection_detector.scan(prompt_text)
    if injection_result.verdict is Verdict.BLOCKED and config.on_injection is OnInjection.BLOCK:
        logger.log(
            UsageRecord(
                provider=provider,
                input_length=len(prompt_text),
                injection_verdict=injection_result.verdict.value,
                injection_score=injection_result.score,
                error="InjectionBlockedError",
            )
        )
        raise InjectionBlockedError(
            f"Prompt blocked by injection detector (score={injection_result.score:.1f})",
            score=injection_result.score,
        )

    input_redaction = RedactionResult(text=prompt_text, matches=[])
    call_prompt = prompt_text
    if config.redact_input:
        input_redaction = pii_redactor.redact(prompt_text)
        call_prompt = input_redaction.text

    start = time.perf_counter()
    try:
        raw_response = call(call_prompt)
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.log(
            UsageRecord(
                provider=provider,
                latency_ms=latency_ms,
                input_length=len(prompt_text),
                pii_redacted_input_count=len(input_redaction.matches),
                injection_verdict=injection_result.verdict.value,
                injection_score=injection_result.score,
                error=repr(exc),
            )
        )
        raise
    latency_ms = (time.perf_counter() - start) * 1000

    output_text = extract_response_text(raw_response)
    output_redaction = RedactionResult(text=output_text, matches=[])
    if config.redact_output:
        output_redaction = pii_redactor.redact(output_text)
    final_text = output_redaction.text

    parsed: Optional[T] = None
    schema_valid: Optional[bool] = None
    schema_error: Optional[SchemaValidationError] = None
    if response_model is not None:
        try:
            parsed = validate_output(final_text, response_model)
            schema_valid = True
        except SchemaValidationError as exc:
            schema_valid = False
            schema_error = exc

    logger.log(
        UsageRecord(
            provider=provider,
            latency_ms=latency_ms,
            input_length=len(prompt_text),
            output_length=len(output_text),
            pii_redacted_input_count=len(input_redaction.matches),
            pii_redacted_output_count=len(output_redaction.matches),
            injection_verdict=injection_result.verdict.value,
            injection_score=injection_result.score,
            schema_valid=schema_valid,
            error=str(schema_error) if schema_error else None,
        )
    )

    if schema_error is not None and config.raise_on_schema_error:
        raise schema_error

    return GuardedResponse(
        raw=raw_response,
        text=final_text,
        parsed=parsed,
        input_redaction=input_redaction,
        output_redaction=output_redaction,
        injection=injection_result,
        latency_ms=latency_ms,
    )
