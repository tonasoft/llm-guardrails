"""Provider adapters: translate between llm-guardrails' generic pipeline
and each SDK's specific request/response shape.

These are duck-typed on purpose - llm-guardrails does not depend on the
openai/anthropic/google-generativeai packages. An adapter only needs to
know the shape of the kwargs you pass and the response object you get
back, so it works against any client that follows that SDK's conventions
(including test doubles).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProviderAdapter(Protocol):
    """Interface a :class:`~llm_guardrails.client.GuardedClient` needs."""

    name: str

    def get_prompt_text(self, kwargs: dict[str, Any]) -> str:
        """Extract the user-facing prompt text from call kwargs."""
        ...

    def set_prompt_text(self, kwargs: dict[str, Any], new_text: str) -> dict[str, Any]:
        """Return a copy of kwargs with the prompt text replaced (e.g. after redaction)."""
        ...

    def call(self, client: Any, **kwargs: Any) -> Any:
        """Perform the actual provider call."""
        ...

    def get_response_text(self, response: Any) -> str:
        """Extract plain text from the provider's response object."""
        ...


def _last_user_message_index(messages: list[dict[str, Any]]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


class OpenAIChatAdapter:
    """For ``client.chat.completions.create(messages=[...], ...)``."""

    name = "openai"

    def get_prompt_text(self, kwargs: dict[str, Any]) -> str:
        messages = kwargs.get("messages", [])
        idx = _last_user_message_index(messages)
        return str(messages[idx].get("content", "")) if idx != -1 else ""

    def set_prompt_text(self, kwargs: dict[str, Any], new_text: str) -> dict[str, Any]:
        new_kwargs = dict(kwargs)
        messages = [dict(m) for m in new_kwargs.get("messages", [])]
        idx = _last_user_message_index(messages)
        if idx != -1:
            messages[idx]["content"] = new_text
        new_kwargs["messages"] = messages
        return new_kwargs

    def call(self, client: Any, **kwargs: Any) -> Any:
        return client.chat.completions.create(**kwargs)

    def get_response_text(self, response: Any) -> str:
        return str(response.choices[0].message.content or "")


class AnthropicMessagesAdapter:
    """For ``client.messages.create(messages=[...], ...)``."""

    name = "anthropic"

    def get_prompt_text(self, kwargs: dict[str, Any]) -> str:
        messages = kwargs.get("messages", [])
        idx = _last_user_message_index(messages)
        if idx == -1:
            return ""
        content = messages[idx].get("content", "")
        return content if isinstance(content, str) else str(content)

    def set_prompt_text(self, kwargs: dict[str, Any], new_text: str) -> dict[str, Any]:
        new_kwargs = dict(kwargs)
        messages = [dict(m) for m in new_kwargs.get("messages", [])]
        idx = _last_user_message_index(messages)
        if idx != -1:
            messages[idx]["content"] = new_text
        new_kwargs["messages"] = messages
        return new_kwargs

    def call(self, client: Any, **kwargs: Any) -> Any:
        return client.messages.create(**kwargs)

    def get_response_text(self, response: Any) -> str:
        blocks = getattr(response, "content", [])
        return "".join(str(getattr(b, "text", "")) for b in blocks)


class GeminiAdapter:
    """For ``model.generate_content(contents=..., ...)`` (google-generativeai)."""

    name = "gemini"

    def get_prompt_text(self, kwargs: dict[str, Any]) -> str:
        contents = kwargs.get("contents", "")
        return contents if isinstance(contents, str) else str(contents)

    def set_prompt_text(self, kwargs: dict[str, Any], new_text: str) -> dict[str, Any]:
        new_kwargs = dict(kwargs)
        new_kwargs["contents"] = new_text
        return new_kwargs

    def call(self, client: Any, **kwargs: Any) -> Any:
        return client.generate_content(**kwargs)

    def get_response_text(self, response: Any) -> str:
        return str(getattr(response, "text", ""))
