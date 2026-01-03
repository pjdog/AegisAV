"""AirSim Action Executor - Translates scenario decisions to AirSim flight commands.

This module bridges the gap between high-level autonomous decisions made by the
scenario runner/goal selector and low-level AirSim flight commands.

When the decision engine decides "INSPECT_ASSET at position (lat, lon)", this
executor translates that into:
1. Convert GPS to NED coordinates
2. Takeoff if not already flying
3. Fly to the asset location
4. Perform an inspection orbit
5. Report completion

Example:
    bridge = RealtimeAirSimBridge(config)
    await bridge.connect()

    geo_ref = GeoReference(47.641468, -122.140165, 0.0)
    executor = AirSimActionExecutor(bridge, geo_ref)

    # Execute a decision from the scenario runner
    result = await executor.execute({
        "action": "inspect_asset",
        "parameters": {
            "position": {"latitude": 47.642, "longitude": -122.139},
            "orbit_radius_m": 20.0,
            "dwell_time_s": 30.0
        },
        "reasoning": "Scheduled inspection of Solar Farm Alpha"
    })
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .realtime_bridge import RealtimeAirSimBridge

from .coordinate_utils import GeoReference

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of action execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of executing an action."""

    status: ExecutionStatus
    action: str
    drone_id: str = ""
    duration_s: float = 0.0
    error: str | None = None
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "action": self.action,
            "drone_id": self.drone_id,
            "duration_s": round(self.duration_s, 2),
            "error": self.error,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FlightConfig:
    """Configuration for flight parameters."""

    default_altitude_agl: float = 30.0  # meters
    default_velocity: float = 5.0  # m/s
    max_velocity: float = 15.0  # m/s
    takeoff_velocity: float = 2.0  # m/s
    landing_velocity: float = 1.5  # m/s
    inspection_orbit_radius: float = 20.0  # meters
    inspection_dwell_time: float = 30.0  # seconds
    anomaly_dwell_multiplier: float = 2.0  # longer inspection for anomalies
    orbit_velocity: float = 3.0  # m/s tangential
    inspection_altitude_agl_cap: float | None = None  # Clamp to get closer inspections
    inspection_orbit_radius_cap: float | None = None  # Clamp to tighter orbits


class AirSimActionExecutor:
    """Translates high-level decisions from the scenario runner into
    low-level AirSim flight commands.

    This is the critical missing link that makes the drone actually fly
    when the decision engine says "INSPECT_ASSET" or "RETURN_LOW_BATTERY".
    """

    def __init__(
        self,
        bridge: RealtimeAirSimBridge,
        geo_ref: GeoReference,
        config: FlightConfig | None = None,
        drone_id: str = "Drone1",
        on_execution_start: Callable[[str, dict], None] | None = None,
        on_execution_complete: Callable[[str, ExecutionResult], None] | None = None,
    ):
        """Initialize the action executor.

        Args:
            bridge: Connected RealtimeAirSimBridge instance
            geo_ref: Geographic reference for coordinate conversion
            config: Flight configuration parameters
            drone_id: Drone identifier for logging
            on_execution_start: Callback when action starts
            on_execution_complete: Callback when action completes
        """
        self.bridge = bridge
        self.geo_ref = geo_ref
        self.config = config or FlightConfig()
        self.drone_id = drone_id
        self.on_execution_start = on_execution_start
        self.on_execution_complete = on_execution_complete

        # State tracking
        self._current_action: str | None = None
        self._is_flying = False
        self._is_armed = False
        self._home_ned: tuple[float, float, float] | None = None
        self._current_task: asyncio.Task | None = None

        # Action handlers mapping
        self._handlers = {
            # GoalType values (from goals.py)
            "inspect_asset": self._handle_inspect_asset,
            "inspect_anomaly": self._handle_inspect_anomaly,
            "return_low_battery": self._handle_return,
            "return_complete": self._handle_return,
            "return_weather": self._handle_return,
            "wait": self._handle_wait,
            "abort": self._handle_abort,
            "recharge": self._handle_recharge,
            "none": self._handle_none,
            # ActionType values (from api_models.py)
            "goto": self._handle_goto,
            "takeoff": self._handle_takeoff,
            "land": self._handle_land,
            "rtl": self._handle_rtl,
            "inspect": self._handle_inspect_asset,
            "orbit": self._handle_orbit,
            "dock": self._handle_return,
            "return": self._handle_return,
        }

        logger.info(
            f"AirSimActionExecutor initialized for {drone_id} "
            f"(ref: {geo_ref.latitude:.6f}, {geo_ref.longitude:.6f})"
        )

        self._avoid_zones: list[dict[str, Any]] = []
        self._avoidance_buffer_m = 10.0

    @property
    def is_flying(self) -> bool:
        """Check if drone is currently flying."""
        return self._is_flying

    @property
    def current_action(self) -> str | None:
        """Get currently executing action."""
        return self._current_action

    def set_avoid_zones(
        self,
        zones: list[dict[str, Any]],
        buffer_m: float | None = None,
    ) -> None:
        """Set avoidance zones for simple collision avoidance."""
        self._avoid_zones = list(zones or [])
        if buffer_m is not None:
            self._avoidance_buffer_m = max(0.0, float(buffer_m))
        logger.info("avoid_zones_updated", count=len(self._avoid_zones))

    def set_navigation_map(self, navigation_map: dict[str, Any]) -> None:
        """Update avoidance zones using map-derived buffers."""
        metadata = navigation_map.get("metadata", {}) if navigation_map else {}
        resolution = float(metadata.get("resolution_m", 0.0) or 0.0)
        voxel_size = metadata.get("voxel_size_m")

        buffer_m = self._avoidance_buffer_m
        if resolution > 0.0:
            buffer_m = max(buffer_m, resolution * 2.0)
        if voxel_size:
            try:
                buffer_m = max(buffer_m, float(voxel_size) * 2.0)
            except (TypeError, ValueError):
                pass

        self.set_avoid_zones(navigation_map.get("obstacles", []), buffer_m=buffer_m)

    async def execute(self, decision: dict) -> ExecutionResult:
        """Execute a decision from the scenario runner.

        This is the main entry point. It:
        1. Parses the decision
        2. Calls the appropriate handler
        3. Returns the result

        Args:
            decision: Decision dict with keys:
                - action: str - The action type (e.g., "inspect_asset", "return_low_battery")
                - parameters: dict - Action-specific parameters
                - confidence: float - Decision confidence (0-1)
                - reasoning: str - Human-readable explanation
                - target_asset: dict - Optional asset details
                - drone_id: str - Optional drone identifier

        Returns:
            ExecutionResult with status and details
        """
        action = decision.get("action", "none")

        # Handle enum types
        if hasattr(action, "value"):
            action = action.value

        # Normalize action string
        action = str(action).lower().replace("-", "_")

        start_time = asyncio.get_event_loop().time()
        self._current_action = action

        drone_id = decision.get("drone_id", self.drone_id)

        logger.info(
            f"[{drone_id}] Executing action: {action} "
            f"(confidence: {decision.get('confidence', 1.0):.2f})"
        )

        if self.on_execution_start:
            try:
                self.on_execution_start(action, decision)
            except Exception as e:
                logger.warning(f"Execution start callback error: {e}")

        # Get handler
        handler = self._handlers.get(action, self._handle_unknown)

        try:
            result = await handler(decision)
            result.duration_s = asyncio.get_event_loop().time() - start_time
            result.drone_id = drone_id

            logger.info(
                f"[{drone_id}] Action {action} completed: {result.status.value} "
                f"({result.duration_s:.1f}s)"
            )

            if self.on_execution_complete:
                try:
                    self.on_execution_complete(action, result)
                except Exception as e:
                    logger.warning(f"Execution complete callback error: {e}")

            return result

        except asyncio.CancelledError:
            logger.warning(f"[{drone_id}] Action {action} was cancelled")
            return ExecutionResult(
                status=ExecutionStatus.CANCELLED,
                action=action,
                drone_id=drone_id,
                duration_s=asyncio.get_event_loop().time() - start_time,
            )

        except Exception as e:
            logger.exception(f"[{drone_id}] Action {action} failed: {e}")
            result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                action=action,
                drone_id=drone_id,
                duration_s=asyncio.get_event_loop().time() - start_time,
                error=str(e),
            )

            if self.on_execution_complete:
                try:
                    self.on_execution_complete(action, result)
                except Exception as cb_err:
                    logger.warning(f"Execution complete callback error: {cb_err}")

            return result

        finally:
            self._current_action = None

    async def cancel(self) -> None:
        """Cancel the currently executing action."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass

    # =========================================================================
    # Action Handlers
    # =========================================================================

    async def _handle_takeoff(self, decision: dict) -> ExecutionResult:
        """Take off to specified or default altitude."""
        params = decision.get("parameters", {})
        altitude = params.get("altitude_agl", self.config.default_altitude_agl)

        if self._is_flying:
            return ExecutionResult(
                status=ExecutionStatus.COMPLETED,
                action="takeoff",
                details={"already_flying": True, "altitude_agl": altitude},
            )

        logger.info(f"Taking off to {altitude}m AGL")
        success = await self.bridge.takeoff(altitude)

        if success:
            self._is_flying = True
            self._is_armed = True

            # Store home position using bridge's async method
            try:
                if self.bridge.client:
                    pos = await self.bridge.get_position()
                    if pos:
                        self._home_ned = (pos.x_val, pos.y_val, pos.z_val)
                        logger.info(
                            f"Home position set: NED({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})"
                        )
                    else:
                        self._home_ned = (0, 0, 0)
                else:
                    self._home_ned = (0, 0, 0)
            except Exception as e:
                logger.warning(f"Could not store home position: {e}")
                self._home_ned = (0, 0, 0)  # Default to origin

            return ExecutionResult(
                status=ExecutionStatus.COMPLETED,
                action="takeoff",
                details={"altitude_agl": altitude},
            )
        else:
            return ExecutionResult(
                status=ExecutionStatus.FAILED, action="takeoff", error="Takeoff command failed"
            )

    async def _handle_inspect_asset(self, decision: dict) -> ExecutionResult:
        """Fly to asset and perform inspection orbit."""
        params = decision.get("parameters", {})

        # Extract target position from various possible formats
        target_asset = decision.get("target_asset") or params.get("target_asset")
        position = params.get("position")

        lat, lon, alt, orbit_radius, dwell_time, asset_id = self._extract_target(
            target_asset, position, params
        )

        if lat is None or lon is None:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                action="inspect_asset",
                error="No target position provided in decision",
            )

        if (
            self.config.inspection_altitude_agl_cap is not None
            and alt > self.config.inspection_altitude_agl_cap
        ):
            logger.info(
                "inspection_altitude_clamped",
                asset_id=asset_id,
                requested=alt,
                capped=self.config.inspection_altitude_agl_cap,
            )
            alt = self.config.inspection_altitude_agl_cap

        if (
            self.config.inspection_orbit_radius_cap is not None
            and orbit_radius > self.config.inspection_orbit_radius_cap
        ):
            logger.info(
                "inspection_orbit_radius_clamped",
                asset_id=asset_id,
                requested=orbit_radius,
                capped=self.config.inspection_orbit_radius_cap,
            )
            orbit_radius = self.config.inspection_orbit_radius_cap

        lat, lon, alt, avoidance = self._apply_avoidance(lat, lon, alt, asset_id)

        # Ensure we're airborne
        if not self._is_flying:
            takeoff_result = await self._handle_takeoff({"parameters": {"altitude_agl": alt}})
            if takeoff_result.status != ExecutionStatus.COMPLETED:
                return takeoff_result

        # Convert GPS to NED coordinates
        north, east, down = self.geo_ref.gps_to_ned(lat, lon, self.geo_ref.altitude + alt)

        logger.info(
            f"Flying to asset {asset_id} at GPS ({lat:.6f}, {lon:.6f}), "
            f"NED ({north:.1f}, {east:.1f}, {down:.1f})"
        )

        # Fly to position with obstacle avoidance
        if hasattr(self.bridge, "move_to_position_with_obstacle_avoidance"):
            move_success = await self.bridge.move_to_position_with_obstacle_avoidance(
                north,
                east,
                down,
                velocity=self.config.default_velocity,
                obstacle_distance_m=15.0,
                avoidance_step_m=10.0,
            )
        else:
            # Fallback to standard movement
            move_success = await self.bridge.move_to_position(
                north, east, down, self.config.default_velocity
            )

        if not move_success:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                action="inspect_asset",
                error=f"Failed to fly to asset {asset_id}",
            )

        # Perform inspection orbit
        logger.info(f"Performing inspection orbit: radius={orbit_radius}m, dwell={dwell_time}s")

        orbit_success = await self.bridge.orbit(
            north,
            east,
            down,
            radius=orbit_radius,
            velocity=self.config.orbit_velocity,
            duration=dwell_time,
        )

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED if orbit_success else ExecutionStatus.FAILED,
            action="inspect_asset",
            details={
                "asset_id": asset_id,
                "position_gps": {"latitude": lat, "longitude": lon, "altitude_agl": alt},
                "position_ned": {"north": north, "east": east, "down": down},
                "orbit_radius_m": orbit_radius,
                "dwell_time_s": dwell_time,
                "avoidance": avoidance,
            },
            error=None if orbit_success else "Orbit failed",
        )

    async def _handle_inspect_anomaly(self, decision: dict) -> ExecutionResult:
        """Re-inspect an asset with detected anomaly (more thorough inspection)."""
        # Same as inspect but with longer dwell time
        params = dict(decision.get("parameters", {}))
        current_dwell = params.get("dwell_time_s", self.config.inspection_dwell_time)
        params["dwell_time_s"] = current_dwell * self.config.anomaly_dwell_multiplier

        modified_decision = dict(decision)
        modified_decision["parameters"] = params

        result = await self._handle_inspect_asset(modified_decision)
        result.action = "inspect_anomaly"
        return result

    async def _handle_return(self, decision: dict) -> ExecutionResult:
        """Return to dock/home position."""
        reason = decision.get("reason") or decision.get("reasoning", "return requested")

        logger.info(f"Returning to home: {reason}")

        if not self._is_flying:
            return ExecutionResult(
                status=ExecutionStatus.COMPLETED,
                action="return",
                details={"reason": reason, "already_landed": True},
            )

        # Return to home position (origin in NED)
        safe_altitude = -self.config.default_altitude_agl

        # First go to safe altitude above home with obstacle avoidance
        if hasattr(self.bridge, "move_to_position_with_obstacle_avoidance"):
            await self.bridge.move_to_position_with_obstacle_avoidance(
                0,
                0,
                safe_altitude,
                velocity=self.config.default_velocity,
                obstacle_distance_m=15.0,
                avoidance_step_m=10.0,
            )
        else:
            await self.bridge.move_to_position(0, 0, safe_altitude, self.config.default_velocity)

        # Land
        land_success = await self.bridge.land()

        if land_success:
            self._is_flying = False
            self._is_armed = False

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED if land_success else ExecutionStatus.FAILED,
            action="return",
            details={"reason": reason},
            error=None if land_success else "Landing failed",
        )

    async def _handle_goto(self, decision: dict) -> ExecutionResult:
        """Fly to a specific GPS position."""
        params = decision.get("parameters", {})
        position = params.get("position", {})

        lat = position.get("latitude")
        lon = position.get("longitude")
        alt = position.get("altitude_agl", self.config.default_altitude_agl)
        velocity = params.get("speed_ms", self.config.default_velocity)

        if lat is None or lon is None:
            return ExecutionResult(
                status=ExecutionStatus.FAILED, action="goto", error="No position provided"
            )

        lat, lon, alt, avoidance = self._apply_avoidance(lat, lon, alt, "waypoint")

        # Ensure flying
        if not self._is_flying:
            takeoff_result = await self._handle_takeoff({"parameters": {"altitude_agl": alt}})
            if takeoff_result.status != ExecutionStatus.COMPLETED:
                return takeoff_result

        # Convert GPS to NED
        north, east, down = self.geo_ref.gps_to_ned(lat, lon, self.geo_ref.altitude + alt)

        logger.info(f"Flying to ({lat:.6f}, {lon:.6f}) at {velocity} m/s")

        # Use obstacle-aware movement if available
        if hasattr(self.bridge, "move_to_position_with_obstacle_avoidance"):
            success = await self.bridge.move_to_position_with_obstacle_avoidance(
                north,
                east,
                down,
                velocity=velocity,
                obstacle_distance_m=15.0,
                avoidance_step_m=10.0,
            )
        else:
            success = await self.bridge.move_to_position(north, east, down, velocity)

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED,
            action="goto",
            details={
                "position_gps": {"latitude": lat, "longitude": lon, "altitude_agl": alt},
                "position_ned": {"north": north, "east": east, "down": down},
                "velocity": velocity,
                "avoidance": avoidance,
            },
            error=None if success else "Move to position failed",
        )

    async def _handle_orbit(self, decision: dict) -> ExecutionResult:
        """Orbit around current or specified position."""
        params = decision.get("parameters", {})

        # Use current position if not specified
        state = await self.bridge.get_synchronized_state()
        if state and state.pose:
            center_x = params.get("center_x", state.pose.position.x)
            center_y = params.get("center_y", state.pose.position.y)
            center_z = params.get("center_z", state.pose.position.z)
        else:
            center_x = params.get("center_x", 0)
            center_y = params.get("center_y", 0)
            center_z = params.get("center_z", -self.config.default_altitude_agl)

        radius = params.get("radius_m", self.config.inspection_orbit_radius)
        duration = params.get("duration_s", self.config.inspection_dwell_time)

        success = await self.bridge.orbit(
            center_x, center_y, center_z, radius=radius, duration=duration
        )

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED,
            action="orbit",
            details={"radius_m": radius, "duration_s": duration},
            error=None if success else "Orbit failed",
        )

    async def _handle_land(self, decision: dict) -> ExecutionResult:
        """Land at current position."""
        logger.info("Landing")

        success = await self.bridge.land()

        if success:
            self._is_flying = False
            self._is_armed = False

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED,
            action="land",
            error=None if success else "Landing failed",
        )

    async def _handle_rtl(self, decision: dict) -> ExecutionResult:
        """Return to launch and land."""
        return await self._handle_return(decision)

    async def _handle_wait(self, decision: dict) -> ExecutionResult:
        """Hold position and wait."""
        params = decision.get("parameters", {})
        duration = params.get("duration_s", 5.0)

        logger.info(f"Waiting/hovering for {duration}s")

        if self._is_flying:
            await self.bridge.hover()

        await asyncio.sleep(duration)

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED, action="wait", details={"duration_s": duration}
        )

    async def _handle_abort(self, decision: dict) -> ExecutionResult:
        """Emergency abort - land immediately."""
        reason = decision.get("reason") or decision.get("reasoning", "abort requested")

        logger.warning(f"ABORT: {reason}")

        # Emergency land
        success = await self.bridge.land()

        if success:
            self._is_flying = False
            self._is_armed = False

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED,
            action="abort",
            details={"reason": reason},
            error=None if success else "Emergency landing failed",
        )

    async def _handle_recharge(self, decision: dict) -> ExecutionResult:
        """Return to dock and recharge (simulated)."""
        # First return home
        return_result = await self._handle_return(decision)

        if return_result.status != ExecutionStatus.COMPLETED:
            return return_result

        # Simulate recharge time
        logger.info("Simulating recharge...")
        await asyncio.sleep(5.0)

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED, action="recharge", details={"recharged": True}
        )

    async def _handle_none(self, decision: dict) -> ExecutionResult:
        """No action required."""
        return ExecutionResult(status=ExecutionStatus.COMPLETED, action="none")

    async def _handle_unknown(self, decision: dict) -> ExecutionResult:
        """Handle unknown action types."""
        action = decision.get("action", "unknown")
        logger.warning(f"Unknown action type: {action}")

        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            action=str(action),
            error=f"Unknown action type: {action}",
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_target(
        self, target_asset: dict | None, position: dict | None, params: dict
    ) -> tuple[float | None, float | None, float, float, float, str]:
        """Extract target position from various decision formats.

        Returns:
            Tuple of (lat, lon, alt, orbit_radius, dwell_time, asset_id)
        """
        lat = None
        lon = None
        alt = self.config.default_altitude_agl
        orbit_radius = self.config.inspection_orbit_radius
        dwell_time = self.config.inspection_dwell_time
        asset_id = "unknown"

        if target_asset:
            # Try to get position from target_asset
            if "latitude" in target_asset:
                lat = target_asset["latitude"]
                lon = target_asset.get("longitude")
            elif "position" in target_asset:
                pos = target_asset["position"]
                lat = pos.get("latitude")
                lon = pos.get("longitude")

            alt = target_asset.get(
                "inspection_altitude_agl",
                target_asset.get("altitude_agl", self.config.default_altitude_agl),
            )
            orbit_radius = target_asset.get("orbit_radius_m", self.config.inspection_orbit_radius)
            dwell_time = target_asset.get("dwell_time_s", self.config.inspection_dwell_time)
            asset_id = target_asset.get("asset_id", target_asset.get("name", "unknown"))

        elif position:
            lat = position.get("latitude")
            lon = position.get("longitude")
            alt = position.get("altitude_agl", self.config.default_altitude_agl)
            orbit_radius = params.get("orbit_radius_m", self.config.inspection_orbit_radius)
            dwell_time = params.get("dwell_time_s", self.config.inspection_dwell_time)
            asset_id = params.get("asset_id", "unknown")

        return lat, lon, alt, orbit_radius, dwell_time, asset_id

    def _apply_avoidance(
        self,
        lat: float,
        lon: float,
        alt: float,
        asset_id: str,
    ) -> tuple[float, float, float, dict[str, Any] | None]:
        if not self._avoid_zones:
            return lat, lon, alt, None

        avoidance_events: list[dict[str, Any]] = []
        target_n, target_e, _ = self.geo_ref.gps_to_ned(lat, lon, self.geo_ref.altitude + alt)
        adjusted = False

        for zone in self._avoid_zones:
            zone_lat = zone.get("latitude")
            zone_lon = zone.get("longitude")
            radius = float(zone.get("radius_m", 0.0))
            height = float(zone.get("height_m", 0.0))
            if height <= 0 and isinstance(zone.get("bbox"), dict):
                bbox = zone.get("bbox") or {}
                height = abs(float(bbox.get("max_z", 0.0)) - float(bbox.get("min_z", 0.0)))
            if radius <= 0:
                continue

            zone_n = None
            zone_e = None
            if zone_lat is not None and zone_lon is not None:
                zone_n, zone_e, _ = self.geo_ref.gps_to_ned(
                    zone_lat, zone_lon, self.geo_ref.altitude + height
                )
            elif zone.get("x_ned") is not None and zone.get("y_ned") is not None:
                zone_n = float(zone.get("x_ned", 0.0))
                zone_e = float(zone.get("y_ned", 0.0))

            if zone_n is None or zone_e is None:
                continue

            dx = target_n - zone_n
            dy = target_e - zone_e
            dist = math.hypot(dx, dy)
            buffer_m = float(zone.get("buffer_m", self._avoidance_buffer_m))
            min_clearance = radius + buffer_m
            if dist < min_clearance:
                if dist < 0.1:
                    dx, dy, dist = 1.0, 0.0, 1.0
                scale = min_clearance / dist
                target_n = zone_n + dx * scale
                target_e = zone_e + dy * scale
                adjusted = True
                avoidance_events.append({
                    "zone_id": zone.get("asset_id"),
                    "asset_type": zone.get("asset_type"),
                    "radius_m": radius,
                    "buffer_m": buffer_m,
                })

            required_alt = height + buffer_m
            if alt < required_alt:
                alt = required_alt
                adjusted = True
                avoidance_events.append({
                    "zone_id": zone.get("asset_id"),
                    "asset_type": zone.get("asset_type"),
                    "raised_altitude_m": alt,
                })

        if adjusted:
            lat, lon, _ = self.geo_ref.ned_to_gps(target_n, target_e, self.geo_ref.altitude + alt)
            logger.info(
                "avoidance_adjusted_target",
                asset_id=asset_id,
                adjusted_alt=alt,
                events=len(avoidance_events),
            )
            return lat, lon, alt, {"events": avoidance_events}

        return lat, lon, alt, None
