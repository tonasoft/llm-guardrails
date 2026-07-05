from llm_guardrails.pii import PIIRedactor, PIIType, redact


def test_redacts_email() -> None:
    result = redact("Contact me at jane.doe@example.com please")
    assert "[REDACTED:EMAIL]" in result.text
    assert "jane.doe@example.com" not in result.text
    assert result.counts()[PIIType.EMAIL] == 1


def test_redacts_ssn() -> None:
    result = redact("My SSN is 219-09-9999")
    assert result.text == "My SSN is [REDACTED:SSN]"
    assert result.counts() == {PIIType.SSN: 1}


def test_redacts_valid_credit_card_luhn_passes() -> None:
    # 4111 1111 1111 1111 is the standard Visa test number (Luhn-valid).
    result = redact("Card: 4111 1111 1111 1111")
    assert "[REDACTED:CREDIT_CARD]" in result.text
    assert "4111" not in result.text


def test_does_not_redact_luhn_invalid_digit_run() -> None:
    # 16 digits that fail the Luhn checksum should not be flagged as a card.
    result = redact("Order id: 1234567890123456")
    assert result.found_pii is False


def test_redacts_us_phone_number() -> None:
    result = redact("Call me at (415) 555-2671")
    assert "[REDACTED:PHONE]" in result.text
    assert "555-2671" not in result.text


def test_redacts_multiple_types_in_one_string() -> None:
    text = "Email jane@example.com or call 415-555-2671, SSN 219-09-9999"
    result = redact(text)
    counts = result.counts()
    assert counts[PIIType.EMAIL] == 1
    assert counts[PIIType.PHONE] == 1
    assert counts[PIIType.SSN] == 1


def test_no_pii_returns_original_text_unchanged() -> None:
    text = "There is nothing sensitive in this sentence."
    result = redact(text)
    assert result.text == text
    assert result.found_pii is False
    assert result.matches == []


def test_enabled_types_filters_detection() -> None:
    redactor = PIIRedactor(enabled_types={PIIType.EMAIL})
    result = redactor.redact("Email jane@example.com, SSN 219-09-9999")
    assert "[REDACTED:EMAIL]" in result.text
    assert "219-09-9999" in result.text  # SSN untouched, type disabled


def test_custom_token_format() -> None:
    redactor = PIIRedactor(token_format="<<{type}>>")
    result = redactor.redact("jane@example.com")
    assert result.text == "<<EMAIL>>"


def test_find_does_not_mutate_text() -> None:
    text = "jane@example.com"
    redactor = PIIRedactor()
    matches = redactor.find(text)
    assert len(matches) == 1
    assert matches[0].original == "jane@example.com"
