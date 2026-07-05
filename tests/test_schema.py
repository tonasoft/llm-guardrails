import pytest
from pydantic import BaseModel

from llm_guardrails.schema import SchemaValidationError, validate_output


class Person(BaseModel):
    name: str
    age: int


def test_validates_plain_json() -> None:
    person = validate_output('{"name": "Ada", "age": 30}', Person)
    assert person.name == "Ada"
    assert person.age == 30


def test_validates_json_inside_markdown_fence() -> None:
    raw = 'Sure, here you go:\n```json\n{"name": "Grace", "age": 45}\n```\nHope that helps!'
    person = validate_output(raw, Person)
    assert person.name == "Grace"
    assert person.age == 45


def test_validates_json_inside_unlabeled_fence() -> None:
    raw = '```\n{"name": "Alan", "age": 41}\n```'
    person = validate_output(raw, Person)
    assert person.name == "Alan"


def test_validates_json_with_surrounding_prose_no_fence() -> None:
    raw = 'The answer is: {"name": "Barbara", "age": 52} - let me know if you need more.'
    person = validate_output(raw, Person)
    assert person.name == "Barbara"


def test_raises_schema_validation_error_on_missing_field() -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_output('{"name": "Ada"}', Person)
    err = exc_info.value
    assert "Ada" not in err.raw_output or err.raw_output == '{"name": "Ada"}'
    assert any(e["loc"] == ("age",) for e in err.errors)


def test_raises_schema_validation_error_on_garbage_input() -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_output("This is not JSON at all.", Person)
    assert exc_info.value.raw_output == "This is not JSON at all."


def test_schema_validation_error_preserves_raw_output() -> None:
    raw = "definitely not json {{{"
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_output(raw, Person)
    assert exc_info.value.raw_output == raw
