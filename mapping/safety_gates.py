"""Safety gating for map updates and planning.

Phase 7 Worker B: Implement safety gating.

This module provides:
- Map update validation (reject if confidence < threshold)
- Planner freeze logic (if map inconsistent or stale)
- Automated tests for map generation pipeline outputs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from mapping.decision_context import MapContext

logger = structlog.get_logger(__name__)


class SafetyGateResult(Enum):
    """Result of a safety gate check."""

    PASS = "pass"
    REJECT = "reject"
    FREEZE = "freeze"
    WARN = "warn"


@dataclass
class SafetyGateConfig:
    """Configuration for safety gates.

    Attributes:
        min_map_confidence: Minimum overall map confidence to accept updates.
        min_slam_confidence: Minimum SLAM tracking confidence.
        min_splat_quality: Minimum splat reconstruction quality.
        max_map_age_s: Maximum age in seconds before map is considered stale.
        max_obstacle_change_rate: Max obstacle count change per update (fraction).
        min_obstacle_overlap: Minimum overlap with previous obstacles.
        freeze_on_stale: Whether to freeze planner on stale map.
        freeze_on_low_quality: Whether to freeze on low quality map.
        require_slam_for_planning: Whether SLAM is required for planning.
    """

    min_map_confidence: float = 0.5
    min_slam_confidence: float = 0.3
    min_splat_quality: float = 0.3
    max_map_age_s: float = 60.0
    max_obstacle_change_rate: float = 0.5  # 50% change triggers review
    min_obstacle_overlap: float = 0.3  # 30% of obstacles should persist
    freeze_on_stale: bool = True
    freeze_on_low_quality: bool = True
    require_slam_for_planning: bool = False


@dataclass
class GateCheckResult:
    """Result of a safety gate check."""

    result: SafetyGateResult
    gate_name: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result": self.result.value,
            "gate_name": self.gate_name,
            "reason": self.reason,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class MapUpdateGate:
    """Safety gate for map updates.

    Validates incoming map updates before they are accepted into
    the navigation system. Rejects updates that don't meet
    quality or consistency thresholds.

    Usage:
        gate = MapUpdateGate(config)

        # Check if update should be accepted
        result = gate.check_update(new_map, previous_map)

        if result.result == SafetyGateResult.PASS:
            apply_map_update(new_map)
        elif result.result == SafetyGateResult.REJECT:
            log_rejected_update(result)
        elif result.result == SafetyGateResult.WARN:
            apply_map_update(new_map)
            alert_operator(result)
    """

    def __init__(self, config: SafetyGateConfig | None = None) -> None:
        """Initialize the gate.

        Args:
            config: Gate configuration.
        """
        self.config = config or SafetyGateConfig()
        self._check_history: list[GateCheckResult] = []

    def check_update(
        self,
        new_map: dict[str, Any],
        previous_map: dict[str, Any] | None = None,
    ) -> GateCheckResult:
        """Check if a map update should be accepted.

        Args:
            new_map: The new map data.
            previous_map: The previous map data (for consistency checks).

        Returns:
            GateCheckResult indicating pass/reject/warn.
        """
        # Extract metadata
        metadata = new_map.get("metadata", {})
        quality_score = metadata.get("map_quality_score", 1.0)
        slam_confidence = metadata.get("slam_confidence", 1.0)
        splat_quality = metadata.get("splat_quality", 1.0)
        obstacles = new_map.get("obstacles", [])

        # Check minimum confidence
        if quality_score < self.config.min_map_confidence:
            result = GateCheckResult(
                result=SafetyGateResult.REJECT,
                gate_name="map_confidence",
                reason=f"Map quality {quality_score:.2f} below threshold {self.config.min_map_confidence}",
                details={
                    "quality_score": quality_score,
                    "threshold": self.config.min_map_confidence,
                },
            )
            self._log_check(result)
            return result

        # Check SLAM confidence
        if slam_confidence < self.config.min_slam_confidence:
            result = GateCheckResult(
                result=SafetyGateResult.REJECT,
                gate_name="slam_confidence",
                reason=f"SLAM confidence {slam_confidence:.2f} below threshold",
                details={
                    "slam_confidence": slam_confidence,
                    "threshold": self.config.min_slam_confidence,
                },
            )
            self._log_check(result)
            return result

        # Check splat quality if present
        if splat_quality > 0 and splat_quality < self.config.min_splat_quality:
            result = GateCheckResult(
                result=SafetyGateResult.WARN,
                gate_name="splat_quality",
                reason=f"Splat quality {splat_quality:.2f} below threshold (accepted with warning)",
                details={
                    "splat_quality": splat_quality,
                    "threshold": self.config.min_splat_quality,
                },
            )
            self._log_check(result)
            return result

        # Consistency check with previous map
        if previous_map:
            consistency_result = self._check_consistency(obstacles, previous_map)
            if consistency_result.result != SafetyGateResult.PASS:
                self._log_check(consistency_result)
                return consistency_result

        # All checks passed
        result = GateCheckResult(
            result=SafetyGateResult.PASS,
            gate_name="all_checks",
            reason="Map update accepted",
            details={
                "quality_score": quality_score,
                "obstacle_count": len(obstacles),
            },
        )
        self._log_check(result)
        return result

    def _check_consistency(
        self,
        new_obstacles: list[dict],
        previous_map: dict[str, Any],
    ) -> GateCheckResult:
        """Check consistency between new and previous obstacles."""
        prev_obstacles = previous_map.get("obstacles", [])

        if not prev_obstacles:
            # No previous obstacles to compare
            return GateCheckResult(
                result=SafetyGateResult.PASS,
                gate_name="consistency",
                reason="No previous map for consistency check",
            )

        # Check obstacle count change rate
        prev_count = len(prev_obstacles)
        new_count = len(new_obstacles)
        if prev_count > 0:
            change_rate = abs(new_count - prev_count) / prev_count
            if change_rate > self.config.max_obstacle_change_rate:
                return GateCheckResult(
                    result=SafetyGateResult.WARN,
                    gate_name="obstacle_change_rate",
                    reason=f"Obstacle count changed by {change_rate:.0%} (threshold: {self.config.max_obstacle_change_rate:.0%})",
                    details={
                        "previous_count": prev_count,
                        "new_count": new_count,
                        "change_rate": change_rate,
                    },
                )

        # Check obstacle overlap (simplified - by ID)
        prev_ids = {o.get("obstacle_id") or o.get("asset_id") for o in prev_obstacles}
        new_ids = {o.get("obstacle_id") or o.get("asset_id") for o in new_obstacles}

        if prev_ids and new_ids:
            overlap = len(prev_ids & new_ids) / len(prev_ids)
            if overlap < self.config.min_obstacle_overlap:
                return GateCheckResult(
                    result=SafetyGateResult.WARN,
                    gate_name="obstacle_overlap",
                    reason=f"Only {overlap:.0%} obstacle overlap with previous map",
                    details={
                        "overlap": overlap,
                        "threshold": self.config.min_obstacle_overlap,
                        "previous_ids": len(prev_ids),
                        "new_ids": len(new_ids),
                    },
                )

        return GateCheckResult(
            result=SafetyGateResult.PASS,
            gate_name="consistency",
            reason="Consistency checks passed",
        )

    def _log_check(self, result: GateCheckResult) -> None:
        """Log a gate check result."""
        self._check_history.append(result)
        if len(self._check_history) > 100:
            self._check_history = self._check_history[-100:]

        log_level = "info" if result.result == SafetyGateResult.PASS else "warning"
        getattr(logger, log_level)(
            "map_update_gate_check",
            result=result.result.value,
            gate=result.gate_name,
            reason=result.reason,
        )

    def get_check_history(self) -> list[GateCheckResult]:
        """Get recent check history."""
        return list(self._check_history)


class PlannerSafetyGate:
    """Safety gate for the planning system.

    Monitors map state and can freeze the planner when
    the map is stale, inconsistent, or low quality.

    Usage:
        gate = PlannerSafetyGate(config)

        # Check before planning
        result = gate.check_planning_allowed(map_context)

        if result.result == SafetyGateResult.FREEZE:
            use_fallback_behavior()
        else:
            proceed_with_planning()
    """

    def __init__(self, config: SafetyGateConfig | None = None) -> None:
        """Initialize the gate.

        Args:
            config: Gate configuration.
        """
        self.config = config or SafetyGateConfig()
        self._frozen = False
        self._freeze_reason: str | None = None
        self._last_check: GateCheckResult | None = None

    @property
    def is_frozen(self) -> bool:
        """Whether the planner is currently frozen."""
        return self._frozen

    @property
    def freeze_reason(self) -> str | None:
        """Reason for current freeze, if any."""
        return self._freeze_reason

    def check_planning_allowed(
        self,
        map_context: MapContext,
    ) -> GateCheckResult:
        """Check if planning is allowed given current map state.

        Args:
            map_context: Current map context.

        Returns:
            GateCheckResult indicating if planning should proceed.
        """
        # Check if map is available
        if not map_context.map_available:
            if self.config.require_slam_for_planning:
                result = GateCheckResult(
                    result=SafetyGateResult.FREEZE,
                    gate_name="map_availability",
                    reason="No map available for planning",
                )
                self._freeze(result.reason)
                self._last_check = result
                return result
            else:
                # Allow planning without map (with warning)
                result = GateCheckResult(
                    result=SafetyGateResult.WARN,
                    gate_name="map_availability",
                    reason="Planning without map data",
                )
                self._unfreeze()
                self._last_check = result
                return result

        # Check map staleness
        if map_context.map_stale:
            if self.config.freeze_on_stale:
                result = GateCheckResult(
                    result=SafetyGateResult.FREEZE,
                    gate_name="map_staleness",
                    reason=f"Map is stale (age: {map_context.map_age_s:.1f}s, threshold: {self.config.max_map_age_s}s)",
                    details={
                        "map_age_s": map_context.map_age_s,
                        "threshold_s": self.config.max_map_age_s,
                    },
                )
                self._freeze(result.reason)
                self._last_check = result
                return result
            else:
                result = GateCheckResult(
                    result=SafetyGateResult.WARN,
                    gate_name="map_staleness",
                    reason="Map is stale but planning continues",
                    details={"map_age_s": map_context.map_age_s},
                )
                self._last_check = result
                return result

        # Check map quality
        if not map_context.map_valid:
            if self.config.freeze_on_low_quality:
                result = GateCheckResult(
                    result=SafetyGateResult.FREEZE,
                    gate_name="map_quality",
                    reason=f"Map quality too low ({map_context.map_quality_score:.2f})",
                    details={
                        "quality_score": map_context.map_quality_score,
                        "threshold": self.config.min_map_confidence,
                    },
                )
                self._freeze(result.reason)
                self._last_check = result
                return result

        # Check SLAM confidence
        if (self.config.require_slam_for_planning and
                map_context.slam_confidence < self.config.min_slam_confidence):
            result = GateCheckResult(
                result=SafetyGateResult.FREEZE,
                gate_name="slam_confidence",
                reason=f"SLAM confidence too low ({map_context.slam_confidence:.2f})",
                details={
                    "slam_confidence": map_context.slam_confidence,
                    "threshold": self.config.min_slam_confidence,
                },
            )
            self._freeze(result.reason)
            self._last_check = result
            return result

        # All checks passed
        self._unfreeze()
        result = GateCheckResult(
            result=SafetyGateResult.PASS,
            gate_name="all_checks",
            reason="Planning allowed",
            details={
                "map_age_s": map_context.map_age_s,
                "quality_score": map_context.map_quality_score,
                "obstacle_count": map_context.obstacle_count,
            },
        )
        self._last_check = result
        return result

    def _freeze(self, reason: str) -> None:
        """Freeze the planner."""
        if not self._frozen:
            logger.warning("planner_frozen", reason=reason)
        self._frozen = True
        self._freeze_reason = reason

    def _unfreeze(self) -> None:
        """Unfreeze the planner."""
        if self._frozen:
            logger.info("planner_unfrozen", previous_reason=self._freeze_reason)
        self._frozen = False
        self._freeze_reason = None

    def force_unfreeze(self) -> None:
        """Manually unfreeze the planner (operator override)."""
        logger.warning("planner_force_unfrozen", previous_reason=self._freeze_reason)
        self._frozen = False
        self._freeze_reason = None

    def get_status(self) -> dict[str, Any]:
        """Get current gate status."""
        return {
            "frozen": self._frozen,
            "freeze_reason": self._freeze_reason,
            "last_check": self._last_check.to_dict() if self._last_check else None,
            "config": {
                "min_map_confidence": self.config.min_map_confidence,
                "max_map_age_s": self.config.max_map_age_s,
                "freeze_on_stale": self.config.freeze_on_stale,
                "freeze_on_low_quality": self.config.freeze_on_low_quality,
            },
        }


# -----------------------------------------------------------------------------
# Test Helpers for Automated Testing
# -----------------------------------------------------------------------------


def validate_map_output(
    nav_map: dict[str, Any],
    require_obstacles: bool = False,
    min_obstacles: int = 0,
) -> tuple[bool, list[str]]:
    """Validate navigation map output format.

    Args:
        nav_map: Navigation map to validate.
        require_obstacles: Whether obstacles are required.
        min_obstacles: Minimum number of obstacles required.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors: list[str] = []

    if not nav_map:
        errors.append("Map is None or empty")
        return False, errors

    # Check required fields
    if "obstacles" not in nav_map:
        errors.append("Missing 'obstacles' field")

    if "generated_at" not in nav_map and "last_updated" not in nav_map:
        errors.append("Missing timestamp field (generated_at or last_updated)")

    # Check obstacles format
    obstacles = nav_map.get("obstacles", [])
    if require_obstacles and not obstacles:
        errors.append("No obstacles in map (required)")

    if len(obstacles) < min_obstacles:
        errors.append(f"Insufficient obstacles: {len(obstacles)} < {min_obstacles}")

    for i, obs in enumerate(obstacles):
        obs_errors = _validate_obstacle(obs, i)
        errors.extend(obs_errors)

    # Check metadata if present
    metadata = nav_map.get("metadata", {})
    if metadata:
        if "map_quality_score" in metadata:
            score = metadata["map_quality_score"]
            if not (0.0 <= score <= 1.0):
                errors.append(f"Invalid map_quality_score: {score} (must be 0-1)")

    return len(errors) == 0, errors


def _validate_obstacle(obs: dict[str, Any], index: int) -> list[str]:
    """Validate a single obstacle entry."""
    errors: list[str] = []
    prefix = f"Obstacle[{index}]"

    # Check required geometry fields
    if "radius_m" not in obs:
        errors.append(f"{prefix}: Missing radius_m")
    elif obs["radius_m"] <= 0:
        errors.append(f"{prefix}: Invalid radius_m: {obs['radius_m']}")

    if "height_m" not in obs:
        errors.append(f"{prefix}: Missing height_m")
    elif obs["height_m"] <= 0:
        errors.append(f"{prefix}: Invalid height_m: {obs['height_m']}")

    # Check position (either geo or NED)
    has_geo = "latitude" in obs and "longitude" in obs
    has_ned = "x_ned" in obs and "y_ned" in obs

    if not has_geo and not has_ned:
        errors.append(f"{prefix}: Missing position (need lat/lon or x_ned/y_ned)")

    # Validate lat/lon ranges if present
    if has_geo:
        lat = obs.get("latitude")
        lon = obs.get("longitude")
        if lat is not None and not (-90 <= lat <= 90):
            errors.append(f"{prefix}: Invalid latitude: {lat}")
        if lon is not None and not (-180 <= lon <= 180):
            errors.append(f"{prefix}: Invalid longitude: {lon}")

    return errors


def validate_slam_output(
    slam_status: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate SLAM status output format.

    Args:
        slam_status: SLAM status to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors: list[str] = []

    if not slam_status:
        errors.append("SLAM status is None or empty")
        return False, errors

    # Check required fields
    required_fields = ["enabled", "running", "tracking_state"]
    for field in required_fields:
        if field not in slam_status:
            errors.append(f"Missing required field: {field}")

    # Validate confidence values
    if "pose_confidence" in slam_status:
        conf = slam_status["pose_confidence"]
        if not (0.0 <= conf <= 1.0):
            errors.append(f"Invalid pose_confidence: {conf}")

    # Validate counts
    for count_field in ["keyframe_count", "map_point_count", "loop_closure_count"]:
        if count_field in slam_status:
            val = slam_status[count_field]
            if not isinstance(val, int) or val < 0:
                errors.append(f"Invalid {count_field}: {val}")

    return len(errors) == 0, errors
