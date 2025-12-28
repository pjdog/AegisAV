"""
Risk Evaluator

Assesses operational risks and provides go/no-go decisions.
This component gates all decisions through a risk assessment
to ensure safe operation.
"""

import logging
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from agent.server.world_model import WorldSnapshot

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Overall risk level assessment."""

    LOW = "low"  # Normal operations
    MODERATE = "moderate"  # Proceed with caution
    HIGH = "high"  # Consider aborting
    CRITICAL = "critical"  # Must abort


class RiskFactor(BaseModel):
    """Individual risk factor assessment."""

    name: str
    value: float  # 0.0 - 1.0, higher = more risk
    threshold: float  # Value at which this becomes concerning
    critical: float  # Value at which this triggers abort
    description: str = ""

    @property
    def level(self) -> RiskLevel:
        """Get risk level for this factor."""
        if self.value >= self.critical:
            return RiskLevel.CRITICAL
        if self.value >= self.threshold:
            return RiskLevel.HIGH
        if self.value >= (self.threshold * 0.7):
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    @property
    def is_critical(self) -> bool:
        """Check if this factor is critical."""
        return self.value >= self.critical

    @property
    def is_concerning(self) -> bool:
        """Check if this factor is concerning (Moderate or higher)."""
        return self.value >= (self.threshold * 0.7)


class RiskAssessment(BaseModel):
    """
    Complete risk assessment for current state.

    Contains individual risk factors and overall assessment.
    Used to gate decisions and trigger aborts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    overall_level: RiskLevel
    overall_score: float  # 0.0 - 1.0

    factors: dict[str, RiskFactor] = Field(default_factory=dict)

    abort_recommended: bool = False
    abort_reason: str | None = None

    warnings: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "overall_level": self.overall_level.value,
            "overall_score": self.overall_score,
            "abort_recommended": self.abort_recommended,
            "abort_reason": self.abort_reason,
            "warnings": self.warnings,
            "factors": {
                name: {"value": f.value, "level": f.level.value, "description": f.description}
                for name, f in self.factors.items()
            },
        }


class RiskThresholds(BaseModel):
    """Configurable risk thresholds."""

    # Battery
    battery_warning_percent: float = 30.0
    battery_critical_percent: float = 15.0

    # Wind
    wind_warning_ms: float = 8.0
    wind_abort_ms: float = 12.0

    # Distance from dock
    max_distance_m: float = 5000.0

    # GPS
    min_satellites: int = 6
    max_hdop: float = 2.0

    # Connectivity (staleness of data)
    data_stale_warning_s: float = 5.0
    data_stale_critical_s: float = 30.0


class RiskEvaluator:
    """
    Evaluates operational risks based on world state.

    The risk evaluator examines multiple risk factors:
    - Battery remaining vs. distance to dock
    - Weather conditions (wind, visibility)
    - GPS quality
    - Vehicle health
    - Communication quality (data freshness)

    Each factor produces a normalized risk score (0-1), and
    the overall risk is computed as a weighted combination.

    Example:
        evaluator = RiskEvaluator(thresholds)
        assessment = evaluator.evaluate(world_snapshot)

        if assessment.abort_recommended:
            # Handle abort
        else:
            for warning in assessment.warnings:
                logger.warning(warning)
    """

    def __init__(self, thresholds: RiskThresholds | None = None):
        self.thresholds = thresholds or RiskThresholds()

        # Factor weights for overall score
        self.weights = {
            "battery": 0.25,
            "wind": 0.15,
            "gps": 0.20,
            "health": 0.25,
            "distance": 0.15,
        }

    def evaluate(self, world: WorldSnapshot) -> RiskAssessment:
        """
        Perform complete risk assessment.

        Args:
            world: Current world snapshot

        Returns:
            RiskAssessment with all factors and overall score
        """
        factors: dict[str, RiskFactor] = {}
        warnings: list[str] = []
        abort_reasons: list[str] = []

        for name, assessor in (
            ("battery", self._assess_battery),
            ("wind", self._assess_wind),
            ("gps", self._assess_gps),
            ("health", self._assess_health),
            ("distance", self._assess_distance),
        ):
            factor = assessor(world)
            factors[name] = factor
            if factor.is_critical:
                abort_reasons.append(factor.description)
            elif factor.is_concerning:
                warnings.append(factor.description)

        # Calculate overall score
        overall_score = sum(
            factors[name].value * weight for name, weight in self.weights.items() if name in factors
        )

        # Determine overall level
        if any(f.is_critical for f in factors.values()):
            overall_level = RiskLevel.CRITICAL
        elif overall_score > 0.7:
            overall_level = RiskLevel.HIGH
        elif overall_score > 0.4:
            overall_level = RiskLevel.MODERATE
        else:
            overall_level = RiskLevel.LOW

        return RiskAssessment(
            overall_level=overall_level,
            overall_score=overall_score,
            factors=factors,
            abort_recommended=len(abort_reasons) > 0,
            abort_reason="; ".join(abort_reasons) if abort_reasons else None,
            warnings=warnings,
        )

    def should_abort(self, assessment: RiskAssessment) -> bool:
        """
        Determine if mission should be aborted based on assessment.

        Args:
            assessment: Risk assessment to evaluate

        Returns:
            True if abort is recommended
        """
        return assessment.abort_recommended or assessment.overall_level == RiskLevel.CRITICAL

    def _assess_battery(self, world: WorldSnapshot) -> RiskFactor:
        """Assess battery risk considering distance to dock."""

        battery_percent = world.vehicle.battery.remaining_percent
        distance_to_dock = world.distance_to_dock()

        # Estimate battery needed to return (rough heuristic)
        # Assume ~0.5% battery per 100m at cruise speed
        battery_to_return = (distance_to_dock / 100) * 0.5
        effective_battery = battery_percent - battery_to_return

        # Normalize to 0-1 risk (inverted - lower battery = higher risk)
        if effective_battery < self.thresholds.battery_critical_percent:
            risk = 1.0
            desc = f"Battery critical: {battery_percent:.1f}% with {distance_to_dock:.0f}m to dock"
        elif effective_battery < self.thresholds.battery_warning_percent:
            risk = 0.6 + 0.4 * (1 - effective_battery / self.thresholds.battery_warning_percent)
            desc = f"Battery low: {battery_percent:.1f}%"
        else:
            risk = max(0, 1 - battery_percent / 100)
            desc = f"Battery OK: {battery_percent:.1f}%"

        return RiskFactor(
            name="battery",
            value=min(1.0, risk),
            threshold=0.6,
            critical=0.85,
            description=desc,
        )

    def _assess_wind(self, world: WorldSnapshot) -> RiskFactor:
        """Assess wind risk."""

        wind_speed = world.environment.wind_speed_ms

        if wind_speed >= self.thresholds.wind_abort_ms:
            risk = 1.0
            desc = f"Wind critical: {wind_speed:.1f} m/s"
        elif wind_speed >= self.thresholds.wind_warning_ms:
            risk = 0.6 + 0.4 * (wind_speed - self.thresholds.wind_warning_ms) / (
                self.thresholds.wind_abort_ms - self.thresholds.wind_warning_ms
            )
            desc = f"Wind high: {wind_speed:.1f} m/s"
        else:
            risk = wind_speed / self.thresholds.wind_warning_ms * 0.5
            desc = f"Wind OK: {wind_speed:.1f} m/s"

        return RiskFactor(
            name="wind",
            value=min(1.0, risk),
            threshold=0.6,
            critical=0.9,
            description=desc,
        )

    def _assess_gps(self, world: WorldSnapshot) -> RiskFactor:
        """Assess GPS quality risk."""

        gps = world.vehicle.gps

        if gps is None or not gps.has_fix:
            return RiskFactor(
                name="gps",
                value=1.0,
                threshold=0.6,
                critical=0.9,
                description="GPS: No fix",
            )

        # Combine satellite count and HDOP into risk score
        sat_risk = max(0, 1 - gps.satellites_visible / 12)
        hdop_risk = min(1, gps.hdop / 5)
        risk = (sat_risk + hdop_risk) / 2

        if gps.satellites_visible < self.thresholds.min_satellites:
            risk = max(risk, 0.7)
            desc = f"GPS degraded: {gps.satellites_visible} sats, HDOP {gps.hdop:.1f}"
        elif gps.hdop > self.thresholds.max_hdop:
            desc = f"GPS poor HDOP: {gps.hdop:.1f}"
        else:
            desc = f"GPS OK: {gps.satellites_visible} sats"

        return RiskFactor(
            name="gps",
            value=min(1.0, risk),
            threshold=0.5,
            critical=0.9,
            description=desc,
        )

    def _assess_health(self, world: WorldSnapshot) -> RiskFactor:
        """Assess vehicle health risk."""

        health = world.vehicle.health

        if health is None or not health.is_healthy:
            unhealthy = []
            if health:
                if not health.sensors_healthy:
                    unhealthy.append("sensors")
                if not health.gps_healthy:
                    unhealthy.append("gps")
                if not health.battery_healthy:
                    unhealthy.append("battery")
                if not health.motors_healthy:
                    unhealthy.append("motors")
                if not health.ekf_healthy:
                    unhealthy.append("ekf")
            else:
                unhealthy.append("unknown")

            # Critical if motors or EKF unhealthy (only check if health object exists)
            if health and (not health.motors_healthy or not health.ekf_healthy):
                risk = 1.0
            else:
                risk = 0.3 * len(unhealthy)

            return RiskFactor(
                name="health",
                value=min(1.0, risk),
                threshold=0.5,
                critical=0.9,
                description=f"Health issues: {', '.join(unhealthy)}",
            )

        return RiskFactor(
            name="health",
            value=0.0,
            threshold=0.5,
            critical=0.9,
            description="Health OK",
        )

    def _assess_distance(self, world: WorldSnapshot) -> RiskFactor:
        """Assess distance from dock risk."""

        distance = world.distance_to_dock()

        if distance > self.thresholds.max_distance_m:
            risk = 1.0
            desc = f"Distance critical: {distance:.0f}m from dock"
        elif distance > self.thresholds.max_distance_m * 0.8:
            risk = 0.6 + 0.4 * (distance - self.thresholds.max_distance_m * 0.8) / (
                self.thresholds.max_distance_m * 0.2
            )
            desc = f"Distance high: {distance:.0f}m from dock"
        else:
            risk = distance / self.thresholds.max_distance_m * 0.5
            desc = f"Distance OK: {distance:.0f}m from dock"

        return RiskFactor(
            name="distance",
            value=min(1.0, risk),
            threshold=0.6,
            critical=0.9,
            description=desc,
        )
