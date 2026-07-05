"""``@guarded_llm_call``: decorator version of the guardrails pipeline for
plain ``fn(prompt: str, ...) -> str`` callables.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Type, TypeVar, overload

from pydantic import BaseModel

from llm_guardrails.config import GuardrailsConfig
from llm_guardrails.core import run_guarded_pipeline
from llm_guardrails.injection import InjectionDetector
from llm_guardrails.pii import PIIRedactor
from llm_guardrails.types import GuardedResponse
from llm_guardrails.usage_logger import UsageLogger

T = TypeVar("T", bound=BaseModel)
F = Callable[..., str]


@overload
def guarded_llm_call(func: F) -> Callable[..., GuardedResponse[Any]]: ...
@overload
def guarded_llm_call(
    func: None = None,
    *,
    config: Optional[GuardrailsConfig] = None,
    response_model: Optional[Type[BaseModel]] = None,
    provider: str = "custom",
) -> Callable[[F], Callable[..., GuardedResponse[Any]]]: ...


def guarded_llm_call(
    func: Optional[F] = None,
    *,
    config: Optional[GuardrailsConfig] = None,
    response_model: Optional[Type[BaseModel]] = None,
    provider: str = "custom",
) -> Any:
    """Wrap any ``fn(prompt: str, *args, **kwargs) -> str`` in PII
    redaction, injection detection, output schema validation, and usage
    logging.

    Usable bare (``@guarded_llm_call``) or with options
    (``@guarded_llm_call(config=..., response_model=...)``). Works with
    any provider - put the SDK-specific call inside ``func`` and return
    its text output; this decorator only ever sees strings in and out.

    Example::

        @guarded_llm_call
        def call_llm(prompt: str) -> str:
            return openai_client.responses.create(input=prompt).output_text

        result = call_llm("Hello!")
        print(result.text)
    """

    def decorator(fn: F) -> Callable[..., GuardedResponse[Any]]:
        cfg = config or GuardrailsConfig()
        model = response_model or cfg.response_model
        pii = PIIRedactor(enabled_types=cfg.pii_types)
        injector = InjectionDetector(
            suspicious_threshold=cfg.injection_suspicious_threshold,
            block_threshold=cfg.injection_block_threshold,
        )
        logger = UsageLogger(file_path=cfg.log_file, callback=cfg.log_callback)

        @functools.wraps(fn)
        def wrapper(prompt: str, *args: Any, **kwargs: Any) -> GuardedResponse[Any]:
            return run_guarded_pipeline(
                prompt_text=prompt,
                call=lambda redacted_prompt: fn(redacted_prompt, *args, **kwargs),
                extract_response_text=str,
                provider=provider,
                config=cfg,
                pii_redactor=pii,
                injection_detector=injector,
                logger=logger,
                response_model=model,
            )

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
