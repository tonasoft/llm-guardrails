"""Usage/latency logging: append JSON Lines to a local file and/or invoke
a user-supplied callback for every guarded call.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class UsageRecord:
    """A single guarded-call record, ready to be logged or shipped elsewhere."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provider: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    input_length: Optional[int] = None
    output_length: Optional[int] = None
    pii_redacted_input_count: int = 0
    pii_redacted_output_count: int = 0
    injection_verdict: Optional[str] = None
    injection_score: Optional[float] = None
    schema_valid: Optional[bool] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class UsageLogger:
    """Writes :class:`UsageRecord`\\ s to a JSONL file and/or a callback.

    Both sinks are optional and independent - pass whichever fit your
    setup (e.g. a file for local debugging, a callback that forwards to
    your metrics/observability stack, or both).
    """

    def __init__(
        self,
        file_path: str | Path | None = None,
        callback: Callable[[UsageRecord], None] | None = None,
    ) -> None:
        self.file_path = Path(file_path) if file_path else None
        self.callback = callback
        self._lock = threading.Lock()
        if self.file_path is not None:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: UsageRecord) -> None:
        if self.file_path is not None:
            with self._lock:
                with open(self.file_path, "a", encoding="utf-8") as f:
                    f.write(record.to_json() + "\n")
        if self.callback is not None:
            self.callback(record)
