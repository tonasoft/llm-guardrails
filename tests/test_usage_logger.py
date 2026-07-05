import json
from pathlib import Path

from llm_guardrails.usage_logger import UsageLogger, UsageRecord


def test_logs_to_file_as_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "usage.jsonl"
    logger = UsageLogger(file_path=log_path)

    logger.log(UsageRecord(provider="openai", model="gpt-4o-mini", latency_ms=123.4))
    logger.log(UsageRecord(provider="anthropic", model="claude-sonnet-5", latency_ms=88.0))

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["provider"] == "openai"
    assert first["model"] == "gpt-4o-mini"
    assert first["latency_ms"] == 123.4


def test_creates_parent_directories(tmp_path: Path) -> None:
    nested_path = tmp_path / "nested" / "dir" / "usage.jsonl"
    logger = UsageLogger(file_path=nested_path)
    logger.log(UsageRecord(provider="gemini"))
    assert nested_path.exists()


def test_invokes_callback() -> None:
    received: list[UsageRecord] = []
    logger = UsageLogger(callback=received.append)

    record = UsageRecord(provider="openai", injection_verdict="SAFE")
    logger.log(record)

    assert len(received) == 1
    assert received[0] is record


def test_file_and_callback_both_fire(tmp_path: Path) -> None:
    log_path = tmp_path / "usage.jsonl"
    received: list[UsageRecord] = []
    logger = UsageLogger(file_path=log_path, callback=received.append)

    logger.log(UsageRecord(provider="openai"))

    assert len(received) == 1
    assert len(log_path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_no_sinks_configured_is_a_silent_noop() -> None:
    logger = UsageLogger()
    logger.log(UsageRecord(provider="openai"))  # should not raise


def test_record_to_dict_and_to_json_roundtrip() -> None:
    record = UsageRecord(
        provider="openai",
        model="gpt-4o-mini",
        latency_ms=42.0,
        pii_redacted_input_count=2,
        injection_verdict="BLOCKED",
        injection_score=5.5,
        schema_valid=False,
        error="SchemaValidationError",
        metadata={"request_id": "abc123"},
    )
    parsed = json.loads(record.to_json())
    assert parsed["pii_redacted_input_count"] == 2
    assert parsed["metadata"]["request_id"] == "abc123"
