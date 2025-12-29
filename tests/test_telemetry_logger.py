"""Tests for telemetry logging."""

import json

from metrics.telemetry_logger import TelemetryLogger


def test_log_telemetry_creates_run(tmp_path) -> None:
    """Logging telemetry should create a run file and write JSON lines."""
    logger = TelemetryLogger(tmp_path)
    logger.log_telemetry("vehicle_1", {"speed": 5}, source="async")

    files = list(tmp_path.glob("telemetry_*.jsonl"))
    assert len(files) == 1

    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["vehicle_id"] == "vehicle_1"
    assert payload["source"] == "async"
    assert payload["payload"] == {"speed": 5}
    assert "timestamp" in payload


def test_start_and_end_run(tmp_path) -> None:
    """Explicit runs write to the expected file and reset state."""
    logger = TelemetryLogger(tmp_path)
    run_id = logger.start_run("run_123")
    logger.log_telemetry("vehicle_2", {"battery": 80})
    logger.end_run()

    assert run_id == "run_123"
    run_file = tmp_path / "telemetry_run_123.jsonl"
    assert run_file.exists()
