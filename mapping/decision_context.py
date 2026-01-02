"""Decision context with map usage metadata.

Phase 5 Worker B: Add decision metadata entries for map usage.

This module provides:
- Map context for decision making
- Metadata entries: map_version, obstacle_count, map_age_s
- Event/log emission for map-based decisions
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MapContext:
    """Context about the current navigation map for decision making.

    This is attached to decisions to track what map state was used
    when making navigation decisions.
    """

    # Map identification
    map_version: int = 0
    map_id: str | None = None
    scenario_id: str | None = None

    # Map state
    map_available: bool = False
    map_age_s: float = float("inf")
    obstacle_count: int = 0

    # Quality metrics
    map_quality_score: float = 0.0
    slam_confidence: float = 0.0
    splat_quality: float = 0.0

    # Source tracking
    map_source: str = "none"  # scenario, slam, splat, fused
    last_update: str | None = None

    # Safety flags
    map_stale: bool = True
    map_valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for decision metadata."""
        return {
            "map_version": self.map_version,
            "map_id": self.map_id,
            "scenario_id": self.scenario_id,
            "map_available": self.map_available,
            "map_age_s": self.map_age_s,
            "obstacle_count": self.obstacle_count,
            "map_quality_score": self.map_quality_score,
            "slam_confidence": self.slam_confidence,
            "splat_quality": self.splat_quality,
            "map_source": self.map_source,
            "last_update": self.last_update,
            "map_stale": self.map_stale,
            "map_valid": self.map_valid,
        }

    @classmethod
    def from_navigation_map(
        cls,
        nav_map: dict[str, Any] | None,
        stale_threshold_s: float = 60.0,
        min_quality_score: float = 0.3,
    ) -> MapContext:
        """Create MapContext from a navigation map.

        Args:
            nav_map: The navigation map from server_state.
            stale_threshold_s: Age in seconds after which map is considered stale.
            min_quality_score: Minimum quality score for map to be valid.

        Returns:
            MapContext instance.
        """
        if not nav_map:
            return cls(
                map_available=False,
                map_stale=True,
                map_valid=False,
            )

        # Calculate map age
        generated_at = nav_map.get("generated_at") or nav_map.get("last_updated")
        map_age = float("inf")
        if generated_at:
            try:
                gen_time = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                map_age = (datetime.now(gen_time.tzinfo) - gen_time).total_seconds()
                map_age = max(0.0, map_age)
            except Exception:
                pass

        # Extract metadata
        metadata = nav_map.get("metadata", {})
        obstacles = nav_map.get("obstacles", [])

        quality_score = metadata.get("map_quality_score", 1.0)
        slam_confidence = metadata.get("slam_confidence", 0.0)
        splat_quality = metadata.get("splat_quality", 0.0)

        # Determine validity
        map_stale = map_age > stale_threshold_s
        map_valid = not map_stale and quality_score >= min_quality_score

        return cls(
            map_version=metadata.get("version", 1),
            map_id=metadata.get("map_id", nav_map.get("scenario_id")),
            scenario_id=nav_map.get("scenario_id"),
            map_available=True,
            map_age_s=map_age,
            obstacle_count=len(obstacles),
            map_quality_score=quality_score,
            slam_confidence=slam_confidence,
            splat_quality=splat_quality,
            map_source=nav_map.get("source", "unknown"),
            last_update=nav_map.get("last_updated", generated_at),
            map_stale=map_stale,
            map_valid=map_valid,
        )


@dataclass
class MapDecisionEvent:
    """Event logged when a map-based decision is made."""

    event_type: str  # map_update, map_query, obstacle_avoidance, path_replan, planner_gate, map_decision
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    decision_id: str | None = None

    # Map state at decision time
    map_context: MapContext | None = None

    # Decision details
    action: str | None = None
    obstacle_ids_considered: list[str] = field(default_factory=list)
    path_modified: bool = False
    avoidance_triggered: bool = False

    # Performance
    query_latency_ms: float = 0.0
    gate_result: str | None = None
    gate_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "decision_id": self.decision_id,
            "map_context": self.map_context.to_dict() if self.map_context else None,
            "action": self.action,
            "obstacle_ids_considered": self.obstacle_ids_considered,
            "path_modified": self.path_modified,
            "avoidance_triggered": self.avoidance_triggered,
            "query_latency_ms": self.query_latency_ms,
            "gate_result": self.gate_result,
            "gate_reason": self.gate_reason,
        }


class MapDecisionLogger:
    """Logs map-related decision events.

    Provides structured logging of map usage in decision making
    for debugging, monitoring, and post-hoc analysis.

    Usage:
        logger = MapDecisionLogger()

        # Log a map update event
        logger.log_map_update(map_context, "slam")

        # Log an obstacle query
        with logger.timed_query() as query:
            obstacles = get_nearby_obstacles(position)
        logger.log_obstacle_query(
            map_context,
            decision_id="dec_001",
            obstacles_found=len(obstacles),
            query_latency_ms=query.elapsed_ms,
        )
    """

    def __init__(self) -> None:
        """Initialize the decision logger."""
        self._events: list[MapDecisionEvent] = []
        self._max_events = 1000

    def log_map_update(
        self,
        map_context: MapContext,
        source: str,
        obstacle_count_delta: int = 0,
    ) -> MapDecisionEvent:
        """Log a map update event.

        Args:
            map_context: Current map context.
            source: Source of the update (slam, splat, scenario).
            obstacle_count_delta: Change in obstacle count.

        Returns:
            The logged event.
        """
        event = MapDecisionEvent(
            event_type="map_update",
            map_context=map_context,
        )

        self._add_event(event)

        logger.info(
            "map_decision_event",
            event_type="map_update",
            source=source,
            map_version=map_context.map_version,
            obstacle_count=map_context.obstacle_count,
            obstacle_count_delta=obstacle_count_delta,
            map_quality=map_context.map_quality_score,
        )

        return event

    def log_obstacle_query(
        self,
        map_context: MapContext,
        decision_id: str | None = None,
        obstacles_found: int = 0,
        obstacle_ids: list[str] | None = None,
        query_latency_ms: float = 0.0,
    ) -> MapDecisionEvent:
        """Log an obstacle query event.

        Args:
            map_context: Current map context.
            decision_id: Associated decision ID.
            obstacles_found: Number of obstacles found.
            obstacle_ids: IDs of obstacles found.
            query_latency_ms: Query latency in milliseconds.

        Returns:
            The logged event.
        """
        event = MapDecisionEvent(
            event_type="map_query",
            decision_id=decision_id,
            map_context=map_context,
            obstacle_ids_considered=obstacle_ids or [],
            query_latency_ms=query_latency_ms,
        )

        self._add_event(event)

        logger.debug(
            "map_decision_event",
            event_type="map_query",
            decision_id=decision_id,
            obstacles_found=obstacles_found,
            query_latency_ms=query_latency_ms,
            map_valid=map_context.map_valid,
        )

        return event

    def log_avoidance(
        self,
        map_context: MapContext,
        decision_id: str,
        action: str,
        obstacle_ids: list[str],
        path_modified: bool = True,
    ) -> MapDecisionEvent:
        """Log an obstacle avoidance event.

        Args:
            map_context: Current map context.
            decision_id: Associated decision ID.
            action: The action taken.
            obstacle_ids: IDs of obstacles being avoided.
            path_modified: Whether the path was modified.

        Returns:
            The logged event.
        """
        event = MapDecisionEvent(
            event_type="obstacle_avoidance",
            decision_id=decision_id,
            map_context=map_context,
            action=action,
            obstacle_ids_considered=obstacle_ids,
            path_modified=path_modified,
            avoidance_triggered=True,
        )

        self._add_event(event)

        logger.info(
            "map_decision_event",
            event_type="obstacle_avoidance",
            decision_id=decision_id,
            action=action,
            obstacles_avoided=len(obstacle_ids),
            path_modified=path_modified,
        )

        return event

    def log_path_replan(
        self,
        map_context: MapContext,
        decision_id: str,
        reason: str,
        obstacles_in_path: list[str],
    ) -> MapDecisionEvent:
        """Log a path replanning event.

        Args:
            map_context: Current map context.
            decision_id: Associated decision ID.
            reason: Reason for replanning.
            obstacles_in_path: Obstacles that triggered replanning.

        Returns:
            The logged event.
        """
        event = MapDecisionEvent(
            event_type="path_replan",
            decision_id=decision_id,
            map_context=map_context,
            action=reason,
            obstacle_ids_considered=obstacles_in_path,
            path_modified=True,
        )

        self._add_event(event)

        logger.info(
            "map_decision_event",
            event_type="path_replan",
            decision_id=decision_id,
            reason=reason,
            obstacles_triggering=len(obstacles_in_path),
            map_age_s=map_context.map_age_s,
        )

        return event

    def log_planner_gate(
        self,
        map_context: MapContext,
        decision_id: str | None,
        action: str,
        gate_result: Any,
    ) -> MapDecisionEvent:
        """Log a planner safety gate decision event."""
        event = MapDecisionEvent(
            event_type="planner_gate",
            decision_id=decision_id,
            map_context=map_context,
            action=action,
            gate_result=getattr(gate_result, "result", gate_result).value
            if hasattr(gate_result, "result")
            else str(gate_result),
            gate_reason=getattr(gate_result, "reason", None),
        )

        self._add_event(event)

        logger.info(
            "map_decision_event",
            event_type="planner_gate",
            decision_id=decision_id,
            action=action,
            gate_result=event.gate_result,
            gate_reason=event.gate_reason,
            map_valid=map_context.map_valid,
            map_stale=map_context.map_stale,
        )

        return event

    def log_map_decision(
        self,
        map_context: MapContext,
        decision_id: str | None,
        action: str,
    ) -> MapDecisionEvent:
        """Log a decision that used map context."""
        event = MapDecisionEvent(
            event_type="map_decision",
            decision_id=decision_id,
            map_context=map_context,
            action=action,
        )

        self._add_event(event)

        logger.info(
            "map_decision_event",
            event_type="map_decision",
            decision_id=decision_id,
            action=action,
            map_version=map_context.map_version,
            obstacle_count=map_context.obstacle_count,
            map_quality=map_context.map_quality_score,
        )

        return event

    def _add_event(self, event: MapDecisionEvent) -> None:
        """Add event to history, maintaining max size."""
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    def get_recent_events(self, count: int = 100) -> list[MapDecisionEvent]:
        """Get recent events."""
        return self._events[-count:]

    def get_stats(self) -> dict[str, Any]:
        """Get logging statistics."""
        if not self._events:
            return {
                "total_events": 0,
                "event_types": {},
                "avg_query_latency_ms": 0.0,
            }

        event_types: dict[str, int] = {}
        total_latency = 0.0
        query_count = 0

        for event in self._events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            if event.event_type == "map_query":
                total_latency += event.query_latency_ms
                query_count += 1

        return {
            "total_events": len(self._events),
            "event_types": event_types,
            "avg_query_latency_ms": total_latency / max(query_count, 1),
            "avoidance_events": event_types.get("obstacle_avoidance", 0),
            "replan_events": event_types.get("path_replan", 0),
        }


class TimedQuery:
    """Context manager for timing map queries."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> TimedQuery:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def add_map_context_to_decision(
    decision_dict: dict[str, Any],
    map_context: MapContext,
) -> dict[str, Any]:
    """Add map context to a decision dictionary.

    Args:
        decision_dict: The decision as a dictionary.
        map_context: Map context to add.

    Returns:
        Updated decision dictionary with map context.
    """
    decision_dict["map_context"] = map_context.to_dict()

    # Also add key fields to risk_factors for compatibility
    risk_factors = decision_dict.get("risk_factors", {})
    if map_context.map_stale:
        risk_factors["stale_map"] = map_context.map_age_s / 60.0  # Risk factor based on age
    if not map_context.map_valid:
        risk_factors["invalid_map"] = 1.0 - map_context.map_quality_score

    decision_dict["risk_factors"] = risk_factors

    return decision_dict


# Global logger instance
map_decision_logger = MapDecisionLogger()
