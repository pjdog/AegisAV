"""Telemetry Logger.

Structured logging of drone telemetry for future analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TelemetryLogEntry:
    """Single telemetry log entry."""

    timestamp: str
    vehicle_id: str
    source: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class TelemetryLogger:
    """Append-only telemetry logger."""

    def __init__(self, log_dir: Path) -> None:
        """Initialize the TelemetryLogger.

        Args:
            log_dir: Directory path where telemetry logs will be stored.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._current_run: str | None = None
        self._run_file: Path | None = None

    def start_run(self, run_id: str | None = None) -> str:
        """Start a new logging run."""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._current_run = run_id
        self._run_file = self.log_dir / f"telemetry_{run_id}.jsonl"
        logger.info("Started telemetry logging run: %s", run_id)
        return run_id

    def end_run(self) -> None:
        """End the current logging run."""
        if self._current_run:
            logger.info("Ended telemetry logging run: %s", self._current_run)
            self._current_run = None
            self._run_file = None

    def log_telemetry(self, vehicle_id: str, payload: dict[str, Any], source: str = "http") -> None:
        """Log telemetry payload.

        Args:
            vehicle_id: Vehicle identifier.
            payload: Telemetry payload.
            source: Source channel (http/async/etc).
        """
        if not self._run_file:
            self.start_run()

        entry = TelemetryLogEntry(
            timestamp=datetime.now().isoformat(),
            vehicle_id=vehicle_id,
            source=source,
            payload=payload,
        )

        with open(self._run_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
