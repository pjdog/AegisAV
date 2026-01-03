"""Demo Recorder for AegisAV Lightweight Simulator.

Records simulation events to JSONL files for playback without
requiring the agent server.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DemoEvent:
    """A single event in a demo recording."""

    timestamp_ms: float
    event_type: str
    data: dict


@dataclass
class DemoMetadata:
    """Metadata for a demo recording."""

    scenario_id: str
    scenario_name: str
    recorded_at: str
    duration_ms: float
    event_count: int
    edge_profile: str = "SBC_CPU"
    time_scale: float = 1.0


class DemoRecorder:
    """Records simulation and agent events to a JSONL file.

    Usage:
        recorder = DemoRecorder()
        recorder.start("normal_ops_001", "Normal Operations")
        recorder.record_event("state_update", {...})
        recorder.record_event("agent_event", {...})
        path = recorder.stop()
    """

    def __init__(self, output_dir: Path | None = None):
        """Initialize the demo recorder.

        Args:
            output_dir: Directory to save demos. Defaults to data/demos/
        """
        self.output_dir = output_dir or Path(__file__).parent.parent.parent / "data" / "demos"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._recording = False
        self._events: list[DemoEvent] = []
        self._start_time_ms: float = 0
        self._scenario_id: str = ""
        self._scenario_name: str = ""
        self._edge_profile: str = "SBC_CPU"
        self._time_scale: float = 1.0

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def start(
        self,
        scenario_id: str,
        scenario_name: str,
        edge_profile: str = "SBC_CPU",
        time_scale: float = 1.0,
    ) -> None:
        """Start recording a demo.

        Args:
            scenario_id: Scenario identifier
            scenario_name: Human-readable scenario name
            edge_profile: Edge compute profile
            time_scale: Simulation time scale
        """
        if self._recording:
            logger.warning("Already recording - stopping current recording")
            self.stop()

        self._recording = True
        self._events = []
        self._start_time_ms = datetime.now().timestamp() * 1000
        self._scenario_id = scenario_id
        self._scenario_name = scenario_name
        self._edge_profile = edge_profile
        self._time_scale = time_scale

        logger.info(f"Started recording demo for scenario: {scenario_id}")

    def record_event(self, event_type: str, data: dict) -> None:
        """Record an event.

        Args:
            event_type: Type of event (state_update, agent_event, etc.)
            data: Event data
        """
        if not self._recording:
            return

        timestamp_ms = datetime.now().timestamp() * 1000 - self._start_time_ms

        event = DemoEvent(
            timestamp_ms=timestamp_ms,
            event_type=event_type,
            data=data,
        )
        self._events.append(event)

    def stop(self) -> Path | None:
        """Stop recording and save the demo.

        Returns:
            Path to the saved demo file, or None if not recording
        """
        if not self._recording:
            return None

        self._recording = False

        if not self._events:
            logger.warning("No events recorded")
            return None

        # Calculate duration
        duration_ms = self._events[-1].timestamp_ms if self._events else 0

        # Create metadata
        metadata = DemoMetadata(
            scenario_id=self._scenario_id,
            scenario_name=self._scenario_name,
            recorded_at=datetime.now().isoformat(),
            duration_ms=duration_ms,
            event_count=len(self._events),
            edge_profile=self._edge_profile,
            time_scale=self._time_scale,
        )

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._scenario_id}_{timestamp}.jsonl"
        filepath = self.output_dir / filename

        # Write to file
        with open(filepath, "w") as f:
            # First line is metadata
            f.write(json.dumps({"metadata": metadata.__dict__}) + "\n")

            # Subsequent lines are events
            for event in self._events:
                f.write(
                    json.dumps({
                        "t": event.timestamp_ms,
                        "type": event.event_type,
                        "data": event.data,
                    })
                    + "\n"
                )

        logger.info(
            f"Saved demo to {filepath} ({len(self._events)} events, {duration_ms / 1000:.1f}s)"
        )

        return filepath

    def get_stats(self) -> dict:
        """Get recording statistics."""
        if not self._recording:
            return {"recording": False}

        elapsed_ms = datetime.now().timestamp() * 1000 - self._start_time_ms
        return {
            "recording": True,
            "scenario_id": self._scenario_id,
            "elapsed_ms": elapsed_ms,
            "event_count": len(self._events),
        }


class DemoPlayer:
    """Plays back recorded demo files.

    Usage:
        player = DemoPlayer()
        player.load("path/to/demo.jsonl")
        async for event in player.play(speed=1.0):
            # Process event
            pass
    """

    def __init__(self):
        """Initialize the demo player."""
        self._events: list[dict] = []
        self._metadata: dict = {}
        self._loaded = False
        self._playing = False
        self._paused = False
        self._current_index = 0
        self._playback_start_time: float = 0

    @property
    def is_loaded(self) -> bool:
        """Check if a demo is loaded."""
        return self._loaded

    @property
    def is_playing(self) -> bool:
        """Check if playback is in progress."""
        return self._playing

    @property
    def is_paused(self) -> bool:
        """Check if playback is paused."""
        return self._paused

    def load(self, path: Path | str) -> dict:
        """Load a demo file.

        Args:
            path: Path to the demo JSONL file

        Returns:
            Demo metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Demo file not found: {path}")

        self._events = []
        self._metadata = {}

        with open(path) as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())

                if i == 0 and "metadata" in data:
                    self._metadata = data["metadata"]
                else:
                    self._events.append(data)

        if not self._metadata:
            raise ValueError("Demo file missing metadata")

        self._loaded = True
        self._current_index = 0

        logger.info(f"Loaded demo: {path.name} ({len(self._events)} events)")

        return self._metadata

    async def play(self, speed: float = 1.0):
        """Play back the demo, yielding events with timing.

        Args:
            speed: Playback speed multiplier (1.0 = real-time)

        Yields:
            Event dictionaries in order with proper timing
        """
        if not self._loaded:
            raise RuntimeError("No demo loaded")

        self._playing = True
        self._paused = False
        self._current_index = 0
        self._playback_start_time = datetime.now().timestamp() * 1000

        prev_timestamp = 0

        try:
            while self._current_index < len(self._events) and self._playing:
                if self._paused:
                    await asyncio.sleep(0.1)
                    continue

                event = self._events[self._current_index]
                timestamp = event.get("t", 0)

                # Calculate delay
                delay_ms = (timestamp - prev_timestamp) / speed
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000)

                prev_timestamp = timestamp
                self._current_index += 1

                yield event

        finally:
            self._playing = False

    def pause(self) -> None:
        """Pause playback."""
        self._paused = True

    def resume(self) -> None:
        """Resume playback."""
        self._paused = False

    def stop(self) -> None:
        """Stop playback."""
        self._playing = False

    def seek(self, time_ms: float) -> None:
        """Seek to a specific time in the demo.

        Args:
            time_ms: Target time in milliseconds
        """
        for i, event in enumerate(self._events):
            if event.get("t", 0) >= time_ms:
                self._current_index = i
                return
        self._current_index = len(self._events)

    def get_progress(self) -> dict:
        """Get playback progress."""
        if not self._loaded:
            return {"loaded": False}

        duration_ms = self._metadata.get("duration_ms", 0)
        current_time_ms = 0
        if self._current_index < len(self._events):
            current_time_ms = self._events[self._current_index].get("t", 0)

        return {
            "loaded": True,
            "playing": self._playing,
            "paused": self._paused,
            "current_index": self._current_index,
            "total_events": len(self._events),
            "current_time_ms": current_time_ms,
            "duration_ms": duration_ms,
            "progress": current_time_ms / duration_ms if duration_ms > 0 else 0,
        }


def list_demos(demo_dir: Path | None = None) -> list[dict]:
    """List available demo files.

    Args:
        demo_dir: Directory to search. Defaults to data/demos/

    Returns:
        List of demo metadata dictionaries
    """
    demo_dir = demo_dir or Path(__file__).parent.parent.parent / "data" / "demos"

    if not demo_dir.exists():
        return []

    demos = []
    for path in sorted(demo_dir.glob("*.jsonl"), reverse=True):
        try:
            with open(path) as f:
                first_line = f.readline()
                data = json.loads(first_line)
                if "metadata" in data:
                    demos.append({
                        "id": path.stem,
                        "filename": path.name,
                        "path": str(path),
                        **data["metadata"],
                    })
        except Exception as e:
            logger.warning(f"Failed to read demo {path}: {e}")

    return demos
