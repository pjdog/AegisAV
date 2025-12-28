"""
Decision Logger

Structured logging of agent decisions for analysis and explainability.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DecisionLogEntry:
    """A single decision log entry with full context."""

    # pylint: disable=too-many-instance-attributes

    # Decision info
    timestamp: str
    decision_id: str
    action: str
    parameters: dict
    confidence: float
    reasoning: str

    # Risk context
    risk_level: str
    risk_score: float
    risk_factors: dict

    # World state context
    vehicle_position: dict
    battery_percent: float
    mode: str
    armed: bool

    # Goal context
    goal_type: str
    target_asset: str | None

    # Phase 2: Critic context
    critic_approved: bool = True
    critic_concerns: list = None
    escalation_level: str | None = None
    escalation_reason: str | None = None

    # Visualization
    assets: list[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DecisionLogContext:
    """Inputs required to log a decision."""

    decision: Any
    risk: Any
    world: Any
    goal: Any | None
    escalation: Any | None = None


class DecisionLogger:
    """
    Logs agent decisions with full context for analysis.

    Each decision is logged as a structured JSON entry containing:
    - The decision itself (action, parameters, reasoning)
    - Risk assessment at time of decision
    - World state snapshot
    - Goal that led to the decision

    Logs are written to the logs/ directory with one file per run.

    Example:
        logger = DecisionLogger(log_dir)
        logger.start_run("mission-001")

        # Log each decision
        logger.log_decision(DecisionLogContext(decision, risk, world_snapshot, goal))

        logger.end_run()
    """

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._current_run: str | None = None
        self._run_file: Path | None = None
        self._entries: list[DecisionLogEntry] = []

    def start_run(self, run_id: str | None = None) -> str:
        """
        Start a new logging run.

        Args:
            run_id: Optional run identifier (default: timestamp)

        Returns:
            The run ID
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._current_run = run_id
        self._run_file = self.log_dir / f"decisions_{run_id}.jsonl"
        self._entries = []

        logger.info(f"Started logging run: {run_id}")
        return run_id

    def end_run(self) -> None:
        """End the current logging run."""
        if self._current_run:
            logger.info(f"Ended logging run: {self._current_run}, {len(self._entries)} decisions")
            self._current_run = None
            self._run_file = None

    def log_decision(self, context: DecisionLogContext) -> None:
        """
        Log a decision with full context.

        Args:
            context: Aggregated context for the decision
        """
        if not self._run_file:
            # Auto-start if not started
            self.start_run()

        decision = context.decision
        risk = context.risk
        world = context.world
        goal = context.goal
        escalation = context.escalation

        # Extract critic concerns if escalation exists
        critic_concerns = []
        if escalation:
            for response in escalation.critic_responses:
                critic_concerns.extend(response.concerns)

        entry = DecisionLogEntry(
            timestamp=datetime.now().isoformat(),
            decision_id=decision.decision_id,
            action=decision.action.value,
            parameters=decision.parameters,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            risk_level=risk.overall_level.value,
            risk_score=risk.overall_score,
            risk_factors={
                name: {"value": f.value, "level": f.level.value} for name, f in risk.factors.items()
            },
            vehicle_position={
                "lat": world.vehicle.position.latitude,
                "lon": world.vehicle.position.longitude,
                "alt": world.vehicle.position.altitude_msl,
            },
            battery_percent=world.vehicle.battery.remaining_percent,
            mode=world.vehicle.mode.value,
            armed=world.vehicle.armed,
            goal_type=goal.goal_type.value if goal else "unknown",
            target_asset=goal.target_asset.asset_id if goal and goal.target_asset else None,
            # Phase 2: Critic info
            critic_approved=escalation.approved if escalation else True,
            critic_concerns=critic_concerns if critic_concerns else None,
            escalation_level=escalation.escalation_level.value if escalation else None,
            escalation_reason=escalation.reason if escalation else None,
            # Phase 4: Spatial context for visualization
            assets=[
                {
                    "id": a.asset_id,
                    "type": a.asset_type.value,
                    "lat": a.position.latitude,
                    "lon": a.position.longitude,
                    "alt": a.position.altitude_msl,
                }
                for a in world.assets
            ],
        )

        self._entries.append(entry)

        # Write to file (append mode, one JSON per line)
        with open(self._run_file, "a", encoding="utf-8") as f:
            f.write(entry.to_json().replace("\n", " ") + "\n")

    def get_entries(self) -> list[DecisionLogEntry]:
        """Get all entries from current run."""
        return list(self._entries)

    def load_run(self, run_id: str) -> list[dict]:
        """
        Load entries from a previous run.

        Args:
            run_id: The run ID to load

        Returns:
            List of decision entries as dicts
        """
        run_file = self.log_dir / f"decisions_{run_id}.jsonl"
        if not run_file.exists():
            return []

        entries = []
        with open(run_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        return entries

    def list_runs(self) -> list[str]:
        """List all available run IDs."""
        runs = []
        for path in self.log_dir.glob("decisions_*.jsonl"):
            run_id = path.stem.replace("decisions_", "")
            runs.append(run_id)
        return sorted(runs)


def create_summary_report(entries: list[dict]) -> str:
    """
    Create a summary report from decision log entries.

    Args:
        entries: List of decision entries

    Returns:
        Markdown-formatted summary report
    """
    if not entries:
        return "No decisions logged."

    # Count actions
    action_counts: dict[str, int] = {}
    for entry in entries:
        action = entry.get("action", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1

    # Risk stats
    risk_scores = [e.get("risk_score", 0) for e in entries]
    avg_risk = sum(risk_scores) / len(risk_scores)
    max_risk = max(risk_scores)

    # Battery range
    battery_levels = [e.get("battery_percent", 0) for e in entries]
    battery_range = f"{min(battery_levels):.1f}% - {max(battery_levels):.1f}%"

    report = f"""# Decision Summary Report

## Overview
- **Total Decisions**: {len(entries)}
- **Duration**: {entries[0].get("timestamp", "?")} to {entries[-1].get("timestamp", "?")}

## Action Distribution
| Action | Count | Percentage |
|--------|-------|------------|
"""

    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / len(entries) * 100
        report += f"| {action} | {count} | {pct:.1f}% |\n"

    report += f"""
## Risk Statistics
- **Average Risk Score**: {avg_risk:.3f}
- **Maximum Risk Score**: {max_risk:.3f}

## Battery
- **Range**: {battery_range}
"""

    return report
