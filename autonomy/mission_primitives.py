"""
Mission Primitives

High-level mission actions built on top of MAVLink commands.
These primitives provide reliable, monitored execution of common flight operations.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from autonomy.mavlink_interface import MAVLinkInterface
from autonomy.vehicle_state import Position

logger = logging.getLogger(__name__)


class PrimitiveResult(Enum):
    """Result of executing a mission primitive."""
    
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ABORTED = "aborted"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"


@dataclass
class PrimitiveConfig:
    """Configuration for mission primitive execution."""
    
    position_tolerance_m: float = 2.0
    altitude_tolerance_m: float = 1.0
    heading_tolerance_deg: float = 10.0
    default_timeout_s: float = 120.0
    poll_interval_s: float = 0.5


class MissionPrimitives:
    """
    High-level mission primitives for common flight operations.
    
    Each primitive:
    - Sends appropriate MAVLink commands
    - Monitors progress toward completion
    - Returns result indicating success/failure
    - Can be cancelled via abort flag
    
    Example:
        primitives = MissionPrimitives(mavlink_interface)
        
        result = await primitives.takeoff(altitude=10.0)
        if result == PrimitiveResult.SUCCESS:
            result = await primitives.goto(target_position)
    """
    
    def __init__(
        self,
        mavlink: MAVLinkInterface,
        config: Optional[PrimitiveConfig] = None,
    ):
        self.mavlink = mavlink
        self.config = config or PrimitiveConfig()
        self._abort_requested = False
    
    def request_abort(self) -> None:
        """Request abort of current primitive."""
        self._abort_requested = True
    
    def clear_abort(self) -> None:
        """Clear abort flag for next primitive."""
        self._abort_requested = False
    
    async def arm_and_takeoff(
        self,
        altitude: float,
        timeout: Optional[float] = None,
    ) -> PrimitiveResult:
        """
        Arm the vehicle and take off to specified altitude.
        
        Args:
            altitude: Target altitude in meters AGL
            timeout: Maximum time to wait for takeoff completion
            
        Returns:
            PrimitiveResult indicating success or failure reason
        """
        timeout = timeout or self.config.default_timeout_s
        self.clear_abort()
        
        logger.info(f"Arm and takeoff to {altitude}m")
        
        # Set GUIDED mode
        if not await self.mavlink.set_mode("GUIDED"):
            logger.error("Failed to set GUIDED mode")
            return PrimitiveResult.FAILED
        
        await asyncio.sleep(1.0)  # Allow mode change
        
        # Arm
        if not await self.mavlink.arm():
            logger.error("Failed to arm")
            return PrimitiveResult.FAILED
        
        await asyncio.sleep(2.0)  # Allow arming
        
        # Takeoff
        if not await self.mavlink.takeoff(altitude):
            logger.error("Failed to send takeoff command")
            return PrimitiveResult.FAILED
        
        # Wait for altitude
        return await self._wait_for_altitude(
            target_agl=altitude,
            timeout=timeout,
        )
    
    async def goto(
        self,
        target: Position,
        speed: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> PrimitiveResult:
        """
        Fly to target position.
        
        Args:
            target: Target position (lat, lon, alt MSL)
            speed: Optional ground speed in m/s
            timeout: Maximum time to wait for arrival
            
        Returns:
            PrimitiveResult indicating success or failure reason
        """
        timeout = timeout or self.config.default_timeout_s
        self.clear_abort()
        
        logger.info(f"Goto ({target.latitude:.6f}, {target.longitude:.6f}, {target.altitude_msl:.1f}m)")
        
        if not await self.mavlink.goto(
            target.latitude,
            target.longitude,
            target.altitude_msl,
            speed,
        ):
            return PrimitiveResult.FAILED
        
        return await self._wait_for_position(target, timeout)
    
    async def orbit(
        self,
        center: Position,
        radius: float,
        altitude_agl: float,
        orbits: int = 1,
        speed: float = 5.0,
        clockwise: bool = True,
        timeout: Optional[float] = None,
    ) -> PrimitiveResult:
        """
        Orbit around a center point.
        
        Args:
            center: Center position to orbit around
            radius: Orbit radius in meters
            altitude_agl: Altitude above ground level
            orbits: Number of complete orbits
            speed: Tangential speed in m/s
            clockwise: Direction of orbit
            timeout: Maximum time for all orbits
            
        Returns:
            PrimitiveResult indicating success or failure
        """
        import math
        
        timeout = timeout or self.config.default_timeout_s * orbits
        self.clear_abort()
        
        logger.info(f"Orbit around ({center.latitude:.6f}, {center.longitude:.6f}), r={radius}m, {orbits} orbits")
        
        # Calculate waypoints around orbit
        num_waypoints = 8 * orbits
        angle_step = (2 * math.pi) / 8
        if not clockwise:
            angle_step = -angle_step
        
        for i in range(num_waypoints):
            if self._abort_requested:
                logger.info("Orbit aborted")
                return PrimitiveResult.ABORTED
            
            angle = i * angle_step
            
            # Calculate position on orbit (rough approximation)
            # Note: For production, use proper geodetic calculations
            lat_offset = (radius * math.cos(angle)) / 111320  # degrees
            lon_offset = (radius * math.sin(angle)) / (111320 * math.cos(math.radians(center.latitude)))
            
            waypoint = Position(
                latitude=center.latitude + lat_offset,
                longitude=center.longitude + lon_offset,
                altitude_msl=center.altitude_msl + altitude_agl,
            )
            
            result = await self.goto(waypoint, speed, timeout / num_waypoints)
            if result != PrimitiveResult.SUCCESS:
                return result
        
        return PrimitiveResult.SUCCESS
    
    async def land(
        self,
        timeout: Optional[float] = None,
    ) -> PrimitiveResult:
        """
        Land at current position.
        
        Args:
            timeout: Maximum time to wait for landing
            
        Returns:
            PrimitiveResult indicating success or failure
        """
        timeout = timeout or self.config.default_timeout_s
        self.clear_abort()
        
        logger.info("Landing")
        
        if not await self.mavlink.land():
            return PrimitiveResult.FAILED
        
        return await self._wait_for_disarm(timeout)
    
    async def return_to_launch(
        self,
        timeout: Optional[float] = None,
    ) -> PrimitiveResult:
        """
        Return to launch point and land.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            PrimitiveResult indicating success or failure
        """
        timeout = timeout or self.config.default_timeout_s * 2
        self.clear_abort()
        
        logger.info("Return to launch")
        
        if not await self.mavlink.return_to_launch():
            return PrimitiveResult.FAILED
        
        return await self._wait_for_disarm(timeout)
    
    async def dock(
        self,
        dock_position: Position,
        approach_altitude: float = 10.0,
        approach_speed: float = 2.0,
        landing_speed: float = 0.5,
        timeout: Optional[float] = None,
    ) -> PrimitiveResult:
        """
        Perform precision landing at dock position.
        
        This is a multi-phase operation:
        1. Fly to approach point above dock
        2. Descend slowly to landing
        3. Disarm
        
        Args:
            dock_position: Position of the dock
            approach_altitude: Altitude for approach in meters AGL
            approach_speed: Speed during approach
            landing_speed: Speed during final descent
            timeout: Maximum time for entire operation
            
        Returns:
            PrimitiveResult indicating success or failure
        """
        timeout = timeout or self.config.default_timeout_s * 2
        self.clear_abort()
        
        logger.info(f"Docking at ({dock_position.latitude:.6f}, {dock_position.longitude:.6f})")
        
        # Phase 1: Approach point
        approach_position = Position(
            latitude=dock_position.latitude,
            longitude=dock_position.longitude,
            altitude_msl=dock_position.altitude_msl + approach_altitude,
        )
        
        result = await self.goto(approach_position, approach_speed, timeout / 2)
        if result != PrimitiveResult.SUCCESS:
            return result
        
        # Phase 2: Land
        return await self.land(timeout / 2)
    
    async def _wait_for_altitude(
        self,
        target_agl: float,
        timeout: float,
    ) -> PrimitiveResult:
        """Wait for vehicle to reach target altitude."""
        elapsed = 0.0
        
        while elapsed < timeout:
            if self._abort_requested:
                return PrimitiveResult.ABORTED
            
            state = self.mavlink.get_current_state()
            if state:
                current_agl = state.position.altitude_agl
                if abs(current_agl - target_agl) < self.config.altitude_tolerance_m:
                    logger.info(f"Reached target altitude: {current_agl:.1f}m")
                    return PrimitiveResult.SUCCESS
            
            await asyncio.sleep(self.config.poll_interval_s)
            elapsed += self.config.poll_interval_s
        
        return PrimitiveResult.TIMEOUT
    
    async def _wait_for_position(
        self,
        target: Position,
        timeout: float,
    ) -> PrimitiveResult:
        """Wait for vehicle to reach target position."""
        elapsed = 0.0
        
        while elapsed < timeout:
            if self._abort_requested:
                return PrimitiveResult.ABORTED
            
            state = self.mavlink.get_current_state()
            if state:
                distance = state.position.distance_to(target)
                if distance < self.config.position_tolerance_m:
                    logger.info(f"Reached target position, distance: {distance:.1f}m")
                    return PrimitiveResult.SUCCESS
            
            await asyncio.sleep(self.config.poll_interval_s)
            elapsed += self.config.poll_interval_s
        
        return PrimitiveResult.TIMEOUT
    
    async def _wait_for_disarm(
        self,
        timeout: float,
    ) -> PrimitiveResult:
        """Wait for vehicle to disarm (indicating landed)."""
        elapsed = 0.0
        
        while elapsed < timeout:
            if self._abort_requested:
                return PrimitiveResult.ABORTED
            
            state = self.mavlink.get_current_state()
            if state and not state.armed:
                logger.info("Vehicle disarmed")
                return PrimitiveResult.SUCCESS
            
            await asyncio.sleep(self.config.poll_interval_s)
            elapsed += self.config.poll_interval_s
        
        return PrimitiveResult.TIMEOUT
