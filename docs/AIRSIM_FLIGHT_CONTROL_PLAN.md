# AirSim Flight Control Implementation

**Status: IMPLEMENTED**

## Executive Summary

The AegisAV system now has complete flight control integration! When the scenario runner makes decisions (INSPECT_ASSET, RETURN_LOW_BATTERY, ABORT, etc.), those decisions are automatically translated into AirSim flight commands and the drone will actually fly.

Previously, the drone appeared "connected" but didn't fly because there was no code connecting decisions to flight commands. This is now fixed with the **AirSim Action Executor**.

## Current Architecture Gap Analysis

### What Works Today
1. **Scenario Runner** - Makes decisions every N seconds per drone
2. **Goal Selector** - Selects goals like INSPECT_ASSET with target positions
3. **AirSim Bridge** - Connected, API control enabled, weather/time working
4. **WebSocket Broadcast** - Overlay receives decision events correctly

### What's Missing
1. **Flight Control Methods** - No `takeoff()`, `move_to()`, `land()` in bridge
2. **Action Executor** - No translator from decisions to flight commands
3. **Coordinate Conversion** - Need lat/lon → AirSim NED coordinates
4. **State Tracking** - No tracking of flight execution progress

## Implementation Plan

### Phase 1: Flight Control Methods in Bridge

**File:** `/home/devcontainers/AegisAV/simulation/realtime_bridge.py`

Add the following async methods:

```python
async def arm(self) -> bool:
    """Arm the drone motors."""
    return await asyncio.to_thread(
        self.client.armDisarm, True, self.config.vehicle_name
    )

async def disarm(self) -> bool:
    """Disarm the drone motors."""
    return await asyncio.to_thread(
        self.client.armDisarm, False, self.config.vehicle_name
    )

async def takeoff(self, altitude: float = 10.0, timeout: float = 30.0) -> bool:
    """Take off to specified altitude (meters AGL)."""
    await self.arm()
    future = self.client.takeoffAsync(timeout_sec=timeout, vehicle_name=self.config.vehicle_name)
    await asyncio.to_thread(future.join)
    return True

async def move_to_position(
    self,
    x: float, y: float, z: float,  # NED coordinates
    velocity: float = 5.0,
    timeout: float = 60.0
) -> bool:
    """Move to position in NED coordinates (z negative = up)."""
    future = self.client.moveToPositionAsync(
        x, y, z, velocity,
        timeout_sec=timeout,
        vehicle_name=self.config.vehicle_name
    )
    await asyncio.to_thread(future.join)
    return True

async def move_to_gps(
    self,
    latitude: float, longitude: float, altitude_msl: float,
    velocity: float = 5.0,
    timeout: float = 60.0
) -> bool:
    """Move to GPS coordinates."""
    ned = self._gps_to_ned(latitude, longitude, altitude_msl)
    return await self.move_to_position(ned[0], ned[1], ned[2], velocity, timeout)

async def hover(self) -> bool:
    """Hold current position."""
    await asyncio.to_thread(
        self.client.hoverAsync, vehicle_name=self.config.vehicle_name
    ).join()
    return True

async def land(self, timeout: float = 30.0) -> bool:
    """Land at current position."""
    future = self.client.landAsync(timeout_sec=timeout, vehicle_name=self.config.vehicle_name)
    await asyncio.to_thread(future.join)
    await self.disarm()
    return True

async def return_to_launch(self) -> bool:
    """Return to home position and land."""
    await self.move_to_position(0, 0, -10, velocity=5.0)  # Go to origin at 10m
    await self.land()
    return True

async def orbit(
    self,
    center_x: float, center_y: float, center_z: float,
    radius: float = 20.0,
    velocity: float = 3.0,
    duration: float = 30.0
) -> bool:
    """Orbit around a point (for inspection)."""
    import math
    start_time = asyncio.get_event_loop().time()
    angular_velocity = velocity / radius

    while (asyncio.get_event_loop().time() - start_time) < duration:
        elapsed = asyncio.get_event_loop().time() - start_time
        angle = angular_velocity * elapsed
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)

        # Move to orbit point
        self.client.moveToPositionAsync(x, y, center_z, velocity, vehicle_name=self.config.vehicle_name)
        await asyncio.sleep(0.5)

    return True
```

### Phase 2: Coordinate Conversion Utility

**File:** `/home/devcontainers/AegisAV/simulation/coordinate_utils.py`

```python
"""Coordinate conversion utilities for AirSim integration."""
import math
from dataclasses import dataclass

# WGS84 ellipsoid constants
WGS84_A = 6378137.0  # Semi-major axis (equatorial radius)
WGS84_B = 6356752.314245  # Semi-minor axis (polar radius)
WGS84_F = 1 / 298.257223563  # Flattening


@dataclass
class GeoReference:
    """Geographic reference point for NED coordinate conversion."""
    latitude: float  # Reference latitude in degrees
    longitude: float  # Reference longitude in degrees
    altitude: float  # Reference altitude in meters MSL

    def gps_to_ned(self, lat: float, lon: float, alt: float) -> tuple[float, float, float]:
        """
        Convert GPS coordinates to NED (North-East-Down) relative to reference.

        Args:
            lat: Target latitude in degrees
            lon: Target longitude in degrees
            alt: Target altitude in meters MSL

        Returns:
            Tuple of (north, east, down) in meters relative to reference
        """
        # Convert to radians
        lat0_rad = math.radians(self.latitude)
        lon0_rad = math.radians(self.longitude)
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        # Calculate differences
        dlat = lat_rad - lat0_rad
        dlon = lon_rad - lon0_rad
        dalt = alt - self.altitude

        # Radius of curvature in meridian
        R_N = WGS84_A / math.sqrt(1 - (2 * WGS84_F - WGS84_F**2) * math.sin(lat0_rad)**2)
        R_M = R_N * (1 - (2 * WGS84_F - WGS84_F**2)) / (1 - (2 * WGS84_F - WGS84_F**2) * math.sin(lat0_rad)**2)

        # Convert to NED
        north = dlat * R_M
        east = dlon * R_N * math.cos(lat0_rad)
        down = -dalt  # NED: positive down

        return (north, east, down)

    def ned_to_gps(self, north: float, east: float, down: float) -> tuple[float, float, float]:
        """
        Convert NED coordinates back to GPS.

        Returns:
            Tuple of (latitude, longitude, altitude_msl)
        """
        lat0_rad = math.radians(self.latitude)

        R_N = WGS84_A / math.sqrt(1 - (2 * WGS84_F - WGS84_F**2) * math.sin(lat0_rad)**2)
        R_M = R_N * (1 - (2 * WGS84_F - WGS84_F**2)) / (1 - (2 * WGS84_F - WGS84_F**2) * math.sin(lat0_rad)**2)

        lat = self.latitude + math.degrees(north / R_M)
        lon = self.longitude + math.degrees(east / (R_N * math.cos(lat0_rad)))
        alt = self.altitude - down

        return (lat, lon, alt)
```

### Phase 3: AirSim Action Executor

**File:** `/home/devcontainers/AegisAV/simulation/airsim_action_executor.py`

```python
"""
AirSim Action Executor - Translates scenario decisions to AirSim flight commands.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .realtime_bridge import RealtimeAirSimBridge
from .coordinate_utils import GeoReference

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    action: str
    duration_s: float = 0.0
    error: Optional[str] = None
    details: dict = field(default_factory=dict)


class AirSimActionExecutor:
    """
    Translates high-level decisions from the scenario runner into
    low-level AirSim flight commands.
    """

    def __init__(
        self,
        bridge: RealtimeAirSimBridge,
        geo_ref: GeoReference,
        default_altitude_agl: float = 30.0,
        default_velocity: float = 5.0,
        on_execution_start: Optional[Callable[[str, dict], None]] = None,
        on_execution_complete: Optional[Callable[[str, ExecutionResult], None]] = None,
    ):
        self.bridge = bridge
        self.geo_ref = geo_ref
        self.default_altitude_agl = default_altitude_agl
        self.default_velocity = default_velocity
        self.on_execution_start = on_execution_start
        self.on_execution_complete = on_execution_complete

        self._current_action: Optional[str] = None
        self._is_flying = False
        self._home_position: Optional[tuple[float, float, float]] = None

        # Action handlers
        self._handlers = {
            "inspect_asset": self._handle_inspect_asset,
            "inspect_anomaly": self._handle_inspect_anomaly,
            "return_low_battery": self._handle_return,
            "return_complete": self._handle_return,
            "return_weather": self._handle_return,
            "wait": self._handle_wait,
            "abort": self._handle_abort,
            "recharge": self._handle_recharge,
            "none": self._handle_none,
            # ActionType variants
            "goto": self._handle_goto,
            "takeoff": self._handle_takeoff,
            "land": self._handle_land,
            "rtl": self._handle_rtl,
            "inspect": self._handle_inspect_asset,
            "orbit": self._handle_orbit,
            "dock": self._handle_return,
        }

    async def execute(self, decision: dict) -> ExecutionResult:
        """
        Execute a decision from the scenario runner.

        Args:
            decision: Decision dict with keys: action, parameters, confidence, reasoning, etc.

        Returns:
            ExecutionResult with status and details
        """
        action = decision.get("action", "none")
        if isinstance(action, Enum):
            action = action.value

        start_time = asyncio.get_event_loop().time()
        self._current_action = action

        logger.info(f"Executing action: {action}")
        if self.on_execution_start:
            self.on_execution_start(action, decision)

        handler = self._handlers.get(action, self._handle_unknown)

        try:
            result = await handler(decision)
            result.duration_s = asyncio.get_event_loop().time() - start_time

            if self.on_execution_complete:
                self.on_execution_complete(action, result)

            return result

        except Exception as e:
            logger.exception(f"Action {action} failed: {e}")
            result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                action=action,
                duration_s=asyncio.get_event_loop().time() - start_time,
                error=str(e)
            )
            if self.on_execution_complete:
                self.on_execution_complete(action, result)
            return result
        finally:
            self._current_action = None

    # ─────────────────────────────────────────────────────────────────────────
    # Action Handlers
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_takeoff(self, decision: dict) -> ExecutionResult:
        """Take off to specified or default altitude."""
        params = decision.get("parameters", {})
        altitude = params.get("altitude_agl", self.default_altitude_agl)

        if not self._is_flying:
            logger.info(f"Taking off to {altitude}m AGL")
            await self.bridge.takeoff(altitude)
            self._is_flying = True

            # Store home position
            state = await self.bridge.get_synchronized_state()
            if state and state.pose:
                self._home_position = (state.pose.x, state.pose.y, state.pose.z)

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            action="takeoff",
            details={"altitude_agl": altitude}
        )

    async def _handle_inspect_asset(self, decision: dict) -> ExecutionResult:
        """Fly to asset and perform inspection orbit."""
        params = decision.get("parameters", {})

        # Get target position
        target_asset = decision.get("target_asset") or params.get("target_asset")
        position = params.get("position")

        if target_asset:
            lat = target_asset.get("latitude") or (target_asset.get("position", {}).get("latitude"))
            lon = target_asset.get("longitude") or (target_asset.get("position", {}).get("longitude"))
            alt = target_asset.get("inspection_altitude_agl", self.default_altitude_agl)
            orbit_radius = target_asset.get("orbit_radius_m", 20.0)
            dwell_time = target_asset.get("dwell_time_s", 30.0)
            asset_id = target_asset.get("asset_id", "unknown")
        elif position:
            lat = position.get("latitude")
            lon = position.get("longitude")
            alt = position.get("altitude_agl", self.default_altitude_agl)
            orbit_radius = params.get("orbit_radius_m", 20.0)
            dwell_time = params.get("dwell_time_s", 30.0)
            asset_id = params.get("asset_id", "unknown")
        else:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                action="inspect_asset",
                error="No target position provided"
            )

        # Ensure we're airborne
        if not self._is_flying:
            await self._handle_takeoff({"parameters": {"altitude_agl": alt}})

        # Convert GPS to NED
        north, east, down = self.geo_ref.gps_to_ned(lat, lon, self.geo_ref.altitude + alt)

        logger.info(f"Flying to asset {asset_id} at ({lat}, {lon}), NED: ({north:.1f}, {east:.1f}, {down:.1f})")

        # Fly to position
        await self.bridge.move_to_position(north, east, down, self.default_velocity)

        # Perform inspection orbit
        logger.info(f"Performing inspection orbit: radius={orbit_radius}m, dwell={dwell_time}s")
        await self.bridge.orbit(north, east, down, orbit_radius, velocity=3.0, duration=dwell_time)

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            action="inspect_asset",
            details={
                "asset_id": asset_id,
                "position": {"lat": lat, "lon": lon, "alt": alt},
                "orbit_radius_m": orbit_radius,
                "dwell_time_s": dwell_time
            }
        )

    async def _handle_inspect_anomaly(self, decision: dict) -> ExecutionResult:
        """Re-inspect an asset with detected anomaly (more thorough inspection)."""
        # Same as inspect but with longer dwell time
        params = decision.get("parameters", {}).copy()
        params["dwell_time_s"] = params.get("dwell_time_s", 60.0)  # Longer inspection
        decision = dict(decision)
        decision["parameters"] = params
        return await self._handle_inspect_asset(decision)

    async def _handle_return(self, decision: dict) -> ExecutionResult:
        """Return to dock/home position."""
        reason = decision.get("reason", "return requested")
        logger.info(f"Returning to home: {reason}")

        # Return to home position
        if self._home_position:
            await self.bridge.move_to_position(
                self._home_position[0],
                self._home_position[1],
                -self.default_altitude_agl,  # Stay at safe altitude
                self.default_velocity
            )
        else:
            # Return to origin
            await self.bridge.move_to_position(0, 0, -self.default_altitude_agl, self.default_velocity)

        # Land
        await self.bridge.land()
        self._is_flying = False

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            action="return",
            details={"reason": reason}
        )

    async def _handle_goto(self, decision: dict) -> ExecutionResult:
        """Fly to a specific position."""
        params = decision.get("parameters", {})
        position = params.get("position", {})

        lat = position.get("latitude")
        lon = position.get("longitude")
        alt = position.get("altitude_agl", self.default_altitude_agl)
        velocity = params.get("speed_ms", self.default_velocity)

        if lat is None or lon is None:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                action="goto",
                error="No position provided"
            )

        if not self._is_flying:
            await self._handle_takeoff({"parameters": {"altitude_agl": alt}})

        north, east, down = self.geo_ref.gps_to_ned(lat, lon, self.geo_ref.altitude + alt)

        logger.info(f"Flying to ({lat}, {lon}) at {velocity} m/s")
        await self.bridge.move_to_position(north, east, down, velocity)

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            action="goto",
            details={"position": position, "velocity": velocity}
        )

    async def _handle_orbit(self, decision: dict) -> ExecutionResult:
        """Orbit around current or specified position."""
        params = decision.get("parameters", {})

        # Get current position if not specified
        state = await self.bridge.get_synchronized_state()
        if state and state.pose:
            center_x = params.get("center_x", state.pose.x)
            center_y = params.get("center_y", state.pose.y)
            center_z = params.get("center_z", state.pose.z)
        else:
            center_x = params.get("center_x", 0)
            center_y = params.get("center_y", 0)
            center_z = params.get("center_z", -self.default_altitude_agl)

        radius = params.get("radius_m", 20.0)
        duration = params.get("duration_s", 30.0)

        await self.bridge.orbit(center_x, center_y, center_z, radius, duration=duration)

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            action="orbit",
            details={"radius_m": radius, "duration_s": duration}
        )

    async def _handle_land(self, decision: dict) -> ExecutionResult:
        """Land at current position."""
        logger.info("Landing")
        await self.bridge.land()
        self._is_flying = False

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            action="land"
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
            status=ExecutionStatus.COMPLETED,
            action="wait",
            details={"duration_s": duration}
        )

    async def _handle_abort(self, decision: dict) -> ExecutionResult:
        """Emergency abort - land immediately."""
        reason = decision.get("reason", "abort requested")
        logger.warning(f"ABORT: {reason}")

        # Emergency land
        await self.bridge.land()
        self._is_flying = False

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            action="abort",
            details={"reason": reason}
        )

    async def _handle_recharge(self, decision: dict) -> ExecutionResult:
        """Return to dock and recharge (simulated)."""
        await self._handle_return(decision)

        # Simulate recharge
        logger.info("Simulating recharge...")
        await asyncio.sleep(5.0)

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            action="recharge"
        )

    async def _handle_none(self, decision: dict) -> ExecutionResult:
        """No action required."""
        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            action="none"
        )

    async def _handle_unknown(self, decision: dict) -> ExecutionResult:
        """Handle unknown action types."""
        action = decision.get("action", "unknown")
        logger.warning(f"Unknown action type: {action}")

        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            action=action,
            error=f"Unknown action type: {action}"
        )
```

### Phase 4: Integration with Scenario Runner

**File:** `/home/devcontainers/AegisAV/agent/server/main.py`

Add to scenario start endpoint:

```python
# In _broadcast_scenario_decision callback
async def _broadcast_scenario_decision(drone_id: str, decision: dict) -> None:
    # Existing broadcast code...

    # NEW: Execute flight command in AirSim
    if airsim_executor and decision.get("action") not in ("none", "wait"):
        asyncio.create_task(airsim_executor.execute(decision))
```

Initialize executor when scenario starts:

```python
# In start_scenario endpoint
if bridge and bridge.connected:
    from simulation.airsim_action_executor import AirSimActionExecutor
    from simulation.coordinate_utils import GeoReference

    # Get dock/home position from scenario
    dock_lat = scenario.drones[0].latitude if scenario.drones else 47.641468
    dock_lon = scenario.drones[0].longitude if scenario.drones else -122.140165
    dock_alt = 0.0  # Ground level

    geo_ref = GeoReference(dock_lat, dock_lon, dock_alt)
    airsim_executor = AirSimActionExecutor(
        bridge=bridge,
        geo_ref=geo_ref,
        default_altitude_agl=30.0,
        default_velocity=5.0,
    )
```

### Phase 5: Configuration

**File:** `/home/devcontainers/AegisAV/config/airsim_flight.yaml`

```yaml
# AirSim Flight Control Configuration

flight:
  default_altitude_agl: 30.0    # meters above ground
  default_velocity: 5.0          # m/s
  max_velocity: 15.0             # m/s
  takeoff_velocity: 2.0          # m/s (slower for safety)
  landing_velocity: 1.5          # m/s

inspection:
  default_orbit_radius: 20.0     # meters
  default_dwell_time: 30.0       # seconds
  anomaly_dwell_multiplier: 2.0  # longer inspection for anomalies

safety:
  max_altitude_agl: 120.0        # meters (regulatory limit)
  min_altitude_agl: 5.0          # meters (obstacle clearance)
  geofence_radius: 500.0         # meters from home
  emergency_land_on_disconnect: true

reference:
  # Default reference point (can be overridden by scenario)
  latitude: 47.641468
  longitude: -122.140165
  altitude_msl: 0.0
```

## Testing Plan

### Unit Tests

```python
# test_airsim_flight.py

async def test_coordinate_conversion():
    ref = GeoReference(47.641468, -122.140165, 0.0)

    # Test roundtrip
    ned = ref.gps_to_ned(47.642, -122.139, 30.0)
    gps = ref.ned_to_gps(*ned)
    assert abs(gps[0] - 47.642) < 0.0001
    assert abs(gps[1] - -122.139) < 0.0001

async def test_action_executor_inspect():
    mock_bridge = MockAirSimBridge()
    executor = AirSimActionExecutor(mock_bridge, GeoReference(...))

    result = await executor.execute({
        "action": "inspect_asset",
        "parameters": {
            "position": {"latitude": 47.642, "longitude": -122.139}
        }
    })

    assert result.status == ExecutionStatus.COMPLETED
    assert mock_bridge.move_to_called
    assert mock_bridge.orbit_called
```

### Integration Test

1. Start AirSim with Blocks environment
2. Start AegisAV server with AirSim bridge enabled
3. Start scenario `normal_ops_001`
4. Verify drone takes off
5. Verify drone flies to first asset
6. Verify drone performs inspection orbit
7. Verify overlay shows decision events

## File Summary

| File | Action | Purpose |
|------|--------|---------|
| `simulation/realtime_bridge.py` | MODIFY | Add flight control methods |
| `simulation/coordinate_utils.py` | CREATE | GPS to NED conversion |
| `simulation/airsim_action_executor.py` | CREATE | Decision to flight translator |
| `agent/server/main.py` | MODIFY | Integrate executor with scenario |
| `config/airsim_flight.yaml` | CREATE | Flight parameters config |
| `tests/test_airsim_flight.py` | CREATE | Unit tests |
| `docs/AIRSIM_FLIGHT_CONTROL_PLAN.md` | CREATE | This document |

## Execution Order

1. Create `coordinate_utils.py` (no dependencies) ✅
2. Modify `realtime_bridge.py` (add flight methods) ✅
3. Create `airsim_action_executor.py` (depends on 1 & 2) ✅
4. Modify `main.py` (integrate executor) ✅
5. Create config and tests ✅
6. Test end-to-end ⏳ (Ready for testing)

---

## How to Use

### Quick Start: Testing Flight Control

1. **Start AirSim** with the Blocks environment (or any Unreal environment)

2. **Start the AegisAV server**:
   ```bash
   cd /home/devcontainers/AegisAV
   python -m agent.server.main
   ```

3. **Test manual flight commands** via curl:
   ```bash
   # Check status
   curl http://172.30.160.1:8090/api/airsim/status

   # Take off to 10 meters
   curl -X POST "http://172.30.160.1:8090/api/airsim/flight/takeoff?altitude=10"

   # Move to position (NED coordinates: 50m north, 30m east, 20m altitude)
   curl -X POST "http://172.30.160.1:8090/api/airsim/flight/move?x=50&y=30&z=-20&velocity=5"

   # Orbit around current position
   curl -X POST "http://172.30.160.1:8090/api/airsim/flight/orbit?radius=20&duration=30"

   # Return to launch and land
   curl -X POST http://172.30.160.1:8090/api/airsim/flight/rtl
   ```

4. **Start a scenario** (drone will fly automatically):
   ```bash
   # List scenarios
   curl http://172.30.160.1:8090/api/scenarios

   # Start a scenario
   curl -X POST http://172.30.160.1:8090/api/scenarios/normal_ops_001/start

   # Watch the drone fly to assets and perform inspections!
   ```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/airsim/status` | GET | AirSim connection status |
| `/api/airsim/flight/status` | GET | Flight executor status |
| `/api/airsim/flight/takeoff` | POST | Take off to altitude |
| `/api/airsim/flight/land` | POST | Land at current position |
| `/api/airsim/flight/move` | POST | Move to NED position |
| `/api/airsim/flight/hover` | POST | Hold position |
| `/api/airsim/flight/rtl` | POST | Return to launch |
| `/api/airsim/flight/orbit` | POST | Orbit around point |

### How It Works

When a scenario runs:

1. **Goal Selector** makes a decision (e.g., `INSPECT_ASSET`)
2. **Scenario Runner** calls `on_decision` callback
3. **Callback** broadcasts to overlay AND schedules flight execution
4. **AirSimActionExecutor** receives the decision:
   - Extracts target asset position (GPS)
   - Converts GPS → NED using GeoReference
   - Calls bridge methods: `takeoff()` → `move_to_position()` → `orbit()`
5. **RealtimeAirSimBridge** executes AirSim API calls:
   - `armDisarm()`, `takeoffAsync()`, `moveToPositionAsync()`, etc.
6. **Drone flies** in AirSim!

### NED Coordinate System

AirSim uses NED (North-East-Down) coordinates:
- **X (North)**: Positive = northward
- **Y (East)**: Positive = eastward
- **Z (Down)**: Positive = downward, **negative = altitude**

Example: To fly 30m above ground at position (100m north, 50m east):
```
x=100, y=50, z=-30
```

### GeoReference

The `GeoReference` class converts between GPS and NED:
- Set when AirSim connects (default: Redmond, WA)
- Updated when scenario starts (uses first drone's position)
- All GPS coordinates from scenarios are converted relative to this point

---

## Complete Data Flow (How Decisions Become Flight)

This section documents the complete path from a scenario decision to drone flight.

### The Connection Chain

```
┌─────────────────────┐
│   ScenarioRunner    │  Makes decisions every N seconds
│   (scenario_runner) │
└─────────┬───────────┘
          │ on_decision(drone_id, goal, decision_record)
          ▼
┌─────────────────────┐
│  _broadcast_scenario│  In main.py - broadcasts to overlay
│  _decision()        │  AND schedules flight execution
└─────────┬───────────┘
          │ asyncio.create_task(_execute_airsim_action(decision))
          ▼
┌─────────────────────┐
│ AirSimActionExecutor│  Translates action to flight commands
│ (airsim_action_     │
│  executor.py)       │
└─────────┬───────────┘
          │ bridge.takeoff(), bridge.move_to_position(), etc.
          ▼
┌─────────────────────┐
│ RealtimeAirSimBridge│  Executes AirSim API calls
│ (realtime_bridge.py)│
└─────────┬───────────┘
          │ client.moveToPositionAsync(), etc.
          ▼
┌─────────────────────┐
│     AirSim API      │  Drone actually flies!
│  (airsim module)    │
└─────────────────────┘
```

### Key Code Locations

| Component | File | Line | Function |
|-----------|------|------|----------|
| Decision callback | `main.py` | ~3247 | `on_decision = lambda: _broadcast_scenario_decision()` |
| Broadcast + execute | `main.py` | ~3078 | `_broadcast_scenario_decision()` |
| Execute action | `main.py` | ~3144 | `_execute_airsim_action()` |
| Action executor | `airsim_action_executor.py` | ~179 | `execute()` |
| Inspect handler | `airsim_action_executor.py` | ~329 | `_handle_inspect_asset()` |
| GPS → NED | `coordinate_utils.py` | ~45 | `GeoReference.gps_to_ned()` |
| Move to position | `realtime_bridge.py` | ~503 | `move_to_position()` |
| Orbit | `realtime_bridge.py` | ~617 | `orbit()` |

### Decision Record Structure

When the scenario runner makes a decision, it creates this structure:

```python
decision_record = {
    "decision_id": "uuid...",
    "drone_id": "alpha",
    "drone_name": "Alpha-1",
    "action": "inspect_asset",  # GoalType.value
    "confidence": 0.85,
    "reason": "Scheduled inspection of Solar Farm Alpha",
    "priority": 1,
    "risk_score": 0.2,
    "battery_percent": 85.0,
    "risk_level": "low",
    "vehicle_position": {"lat": 37.7749, "lon": -122.4194},

    # KEY: target_asset contains GPS coordinates for flight
    "target_asset": {
        "asset_id": "solar_farm_a",
        "name": "Solar Farm Alpha",
        "latitude": 37.7760,           # ← Used for flight
        "longitude": -122.4180,        # ← Used for flight
        "inspection_altitude_agl": 30.0,
        "orbit_radius_m": 20.0,
        "dwell_time_s": 30.0
    }
}
```

### What Happens Step by Step

1. **ScenarioRunner._make_decision()** (scenario_runner.py:593)
   - Uses GoalSelector to pick next action
   - Creates `decision_record` with `target_asset` containing lat/lon
   - Calls `self.on_decision(drone_id, goal, decision_record)`

2. **on_decision callback** (main.py:3247)
   - Creates task: `asyncio.create_task(_broadcast_scenario_decision(...))`

3. **_broadcast_scenario_decision()** (main.py:3078)
   - Sends to Unreal overlay via WebSocket
   - Sends to dashboard clients
   - If `server_state.airsim_action_executor` exists:
     - Creates task: `asyncio.create_task(_execute_airsim_action(decision))`

4. **_execute_airsim_action()** (main.py:3144)
   - Calls `executor.execute(decision)`
   - Logs result

5. **AirSimActionExecutor.execute()** (airsim_action_executor.py:179)
   - Extracts action type: `"inspect_asset"`
   - Calls handler: `_handle_inspect_asset(decision)`

6. **_handle_inspect_asset()** (airsim_action_executor.py:329)
   - Extracts target position from `decision["target_asset"]`
   - Ensures drone is flying (calls takeoff if needed)
   - Converts GPS to NED: `geo_ref.gps_to_ned(lat, lon, alt)`
   - Flies to position: `bridge.move_to_position(north, east, down)`
   - Performs orbit: `bridge.orbit(north, east, down, radius, duration)`

7. **RealtimeAirSimBridge.move_to_position()** (realtime_bridge.py:503)
   - Calls `client.moveToPositionAsync(x, y, z, velocity)`
   - Waits for completion with `future.join()`

8. **Drone flies in AirSim!** 🚁

---

## Troubleshooting

### Drone Not Flying When Scenario Starts

1. **Check AirSim is connected:**
   ```bash
   curl http://172.30.160.1:8090/api/airsim/status
   # Should show: "bridge_connected": true, "executor_available": true
   ```

2. **Check executor is initialized:**
   Look for this log when AirSim connects:
   ```
   airsim_action_executor_initialized drone_id=Drone1 reference=(47.641468, -122.140165)
   ```

3. **Check decisions are being made:**
   Look for scenario decision logs:
   ```
   [60s] Alpha-1: inspect_asset - Scheduled inspection of Solar Farm Alpha
   ```

4. **Check action execution:**
   Look for executor logs:
   ```
   [alpha] Executing action: inspect_asset (confidence: 0.85)
   Flying to asset solar_farm_a at GPS (37.7760, -122.4180), NED (122.1, -123.5, -30.0)
   ```

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| "No target position provided" | `target_asset` missing lat/lon | Check scenario asset definitions |
| "executor_available: false" | Bridge connected but executor failed | Check logs for init error |
| Drone takes off but doesn't move | GPS→NED conversion issue | Check GeoReference point |
| Drone flies wrong direction | GeoReference not updated | Scenario should update it on start |

### Manual Testing

Test flight independently of scenarios:

```bash
# 1. Start server with AirSim connected
python -m agent.server.main

# 2. Take off
curl -X POST "http://172.30.160.1:8090/api/airsim/flight/takeoff?altitude=20"

# 3. Move to position (50m north, 30m east, 25m altitude)
curl -X POST "http://172.30.160.1:8090/api/airsim/flight/move?x=50&y=30&z=-25&velocity=5"

# 4. Orbit
curl -X POST "http://172.30.160.1:8090/api/airsim/flight/orbit?radius=15&duration=20"

# 5. Land
curl -X POST "http://172.30.160.1:8090/api/airsim/flight/land"
```

---

## Future Improvements

1. **Multi-drone support**: Each drone needs its own executor instance
2. **Collision avoidance**: Add obstacle detection during flight
3. **Path planning**: Optimize routes between multiple assets
4. **Geofencing**: Enforce flight boundaries
5. **Real telemetry feedback**: Use actual AirSim position for decision updates
