"""Output schema validation via Pydantic.

LLMs are asked for JSON but frequently wrap it in prose or markdown code
fences ("Sure, here's the JSON:\\n```json\\n{...}\\n```"). This module
tries the raw text first and falls back to extracting a fenced or
brace-delimited JSON blob before giving up.
"""

from __future__ import annotations

import re
from typing import Any, Type, TypeVar, cast

from pydantic import BaseModel, ValidationError

from llm_guardrails.exceptions import GuardrailsError

T = TypeVar("T", bound=BaseModel)

_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class SchemaValidationError(GuardrailsError):
    """Raised when model output cannot be validated against the schema."""

    def __init__(self, message: str, raw_output: str, errors: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.errors = errors


def _extract_fenced_json(text: str) -> str | None:
    match = _FENCED_BLOCK_RE.search(text)
    return match.group(1).strip() if match else None


def _extract_braces_json(text: str) -> str | None:
    """Fall back to the first top-level {...} or [...] span in the text."""
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
    return None


def validate_output(raw: str, model: Type[T]) -> T:
    """Parse and validate ``raw`` LLM output against a Pydantic model.

    Tries, in order: the raw text as-is, a fenced ```json``` code block,
    and the first brace-delimited span in the text. Raises
    :class:`SchemaValidationError` (including the original raw text and
    all validation errors encountered) if none of them validate.
    """
    text = raw.strip()

    candidates = [text]
    fenced = _extract_fenced_json(text)
    if fenced and fenced not in candidates:
        candidates.append(fenced)
    braces = _extract_braces_json(text)
    if braces and braces not in candidates:
        candidates.append(braces)

    last_error: ValidationError | None = None
    for candidate in candidates:
        try:
            return model.model_validate_json(candidate)
        except ValidationError as exc:
            last_error = exc

    assert last_error is not None  # at least one candidate always exists
    raise SchemaValidationError(
        f"Output failed to validate against {model.__name__}: {last_error}",
        raw_output=raw,
        errors=cast(list[dict[str, Any]], last_error.errors()),
    )
