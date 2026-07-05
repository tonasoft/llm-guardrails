"""``GuardedClient``: drop-in wrapper around an existing OpenAI/Anthropic/
Gemini client instance.
"""

from __future__ import annotations

from typing import Any, Optional, Type, TypeVar, cast

from pydantic import BaseModel

from llm_guardrails.adapters import ProviderAdapter
from llm_guardrails.config import GuardrailsConfig
from llm_guardrails.core import run_guarded_pipeline
from llm_guardrails.injection import InjectionDetector
from llm_guardrails.pii import PIIRedactor
from llm_guardrails.types import GuardedResponse
from llm_guardrails.usage_logger import UsageLogger

T = TypeVar("T", bound=BaseModel)


class GuardedClient:
    """Wraps an existing LLM client with PII redaction, injection
    detection, output schema validation, and usage logging.

    Example::

        from openai import OpenAI
        from llm_guardrails import GuardedClient
        from llm_guardrails.adapters import OpenAIChatAdapter

        client = GuardedClient(OpenAI(), adapter=OpenAIChatAdapter())
        response = client.call(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.text)

    The wrapped client and its call signature are untouched - ``call()``
    simply forwards any keyword arguments to the adapter, which knows how
    to find the prompt text, invoke the client, and extract response text
    for that particular SDK.
    """

    def __init__(
        self,
        client: Any,
        adapter: ProviderAdapter,
        config: Optional[GuardrailsConfig] = None,
    ) -> None:
        self.client = client
        self.adapter = adapter
        self.config = config or GuardrailsConfig()
        self._pii = PIIRedactor(enabled_types=self.config.pii_types)
        self._injection = InjectionDetector(
            suspicious_threshold=self.config.injection_suspicious_threshold,
            block_threshold=self.config.injection_block_threshold,
        )
        self._logger = UsageLogger(
            file_path=self.config.log_file, callback=self.config.log_callback
        )

    def call(
        self, response_model: Optional[Type[T]] = None, **kwargs: Any
    ) -> GuardedResponse[T]:
        """Make a guarded call. ``kwargs`` are forwarded to the adapter/client
        exactly as you'd pass them to the underlying SDK (e.g. ``model=``,
        ``messages=`` for OpenAI/Anthropic, or ``contents=`` for Gemini).
        """
        model = response_model or self.config.response_model
        prompt_text = self.adapter.get_prompt_text(kwargs)

        def do_call(redacted_prompt: str) -> Any:
            call_kwargs = kwargs
            if redacted_prompt != prompt_text:
                call_kwargs = self.adapter.set_prompt_text(kwargs, redacted_prompt)
            return self.adapter.call(self.client, **call_kwargs)

        return run_guarded_pipeline(
            prompt_text=prompt_text,
            call=do_call,
            extract_response_text=self.adapter.get_response_text,
            provider=self.adapter.name,
            config=self.config,
            pii_redactor=self._pii,
            injection_detector=self._injection,
            logger=self._logger,
            response_model=cast(Optional[Type[T]], model),
        )
