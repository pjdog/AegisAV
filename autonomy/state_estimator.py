"""State Estimator with GPS/Visual Fusion.

Provides robust position estimation by fusing GPS and visual odometry data.
Automatically switches between localization modes based on sensor availability.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel

from autonomy.vehicle_state import Position, VehicleState, Velocity

logger = logging.getLogger(__name__)


class LocalizationMode(Enum):
    """Current localization mode based on sensor availability."""

    GPS_PRIMARY = "gps_primary"  # GPS available and good quality
    VISUAL_PRIMARY = "visual_primary"  # GPS unavailable, using visual odometry
    FUSED = "fused"  # Both available, using weighted fusion
    DEGRADED = "degraded"  # Neither reliable, using last known position
    INITIALIZING = "initializing"  # Waiting for first valid data


class GPSQuality(Enum):
    """GPS signal quality classification."""

    EXCELLENT = "excellent"  # RTK fix, HDOP < 1.0
    GOOD = "good"  # 3D fix, HDOP < 2.0, 6+ sats
    MARGINAL = "marginal"  # 3D fix, HDOP < 5.0
    POOR = "poor"  # 2D fix or HDOP >= 5.0
    UNAVAILABLE = "unavailable"  # No fix


@dataclass
class GPSQualityThresholds:
    """Thresholds for GPS quality assessment."""

    excellent_hdop: float = 1.0
    good_hdop: float = 2.0
    marginal_hdop: float = 5.0
    min_satellites: int = 6
    min_fix_type: int = 3  # 3D fix required
    timeout_s: float = 2.0  # GPS timeout in seconds


@dataclass
class VisualOdometryInput:
    """Input from visual odometry system."""

    timestamp: datetime
    delta_north: float  # Change in north since last update (meters)
    delta_east: float  # Change in east (meters)
    delta_down: float  # Change in down (meters)
    confidence: float  # 0.0 to 1.0
    features_tracked: int = 0
    is_valid: bool = True


class EstimatedState(BaseModel):
    """Fused state estimate with uncertainty."""

    timestamp: datetime
    mode: str  # LocalizationMode value

    # Position estimate
    north: float
    east: float
    down: float

    # Velocity estimate
    velocity_north: float = 0.0
    velocity_east: float = 0.0
    velocity_down: float = 0.0

    # Uncertainty (1-sigma)
    position_uncertainty_m: float = 0.0
    velocity_uncertainty_ms: float = 0.0

    # GPS position (if available)
    latitude: float | None = None
    longitude: float | None = None
    altitude_msl: float | None = None

    # Quality metrics
    gps_quality: str = "unavailable"
    visual_confidence: float = 0.0
    fusion_weight_gps: float = 0.0  # 0 = visual only, 1 = GPS only


@dataclass
class StateEstimatorConfig:
    """Configuration for state estimator."""

    # GPS quality thresholds
    gps_thresholds: GPSQualityThresholds = field(default_factory=GPSQualityThresholds)

    # Fusion parameters
    gps_noise_std_m: float = 2.0  # GPS position noise std dev
    visual_noise_std_m: float = 0.1  # Visual odometry noise per update
    visual_drift_rate: float = 0.01  # Drift accumulation per second

    # EKF parameters
    process_noise_position: float = 0.1
    process_noise_velocity: float = 0.5

    # Mode switching
    mode_switch_delay_s: float = 1.0  # Delay before switching modes
    degraded_timeout_s: float = 5.0  # Time before entering degraded mode


class StateEstimator:
    """Fuses GPS and visual odometry for robust position estimation.

    Supports multiple localization modes:
    - GPS_PRIMARY: Use GPS when good quality is available
    - VISUAL_PRIMARY: Use visual odometry when GPS is unavailable
    - FUSED: Combine both sources using EKF when both available
    - DEGRADED: Hold last known position when neither is reliable

    Example:
        estimator = StateEstimator(geo_ref, config)

        # Update with GPS
        estimator.update_gps(vehicle_state)

        # Update with visual odometry
        estimator.update_visual(visual_delta)

        # Get fused estimate
        estimate = estimator.get_estimate()
    """

    def __init__(
        self,
        config: StateEstimatorConfig | None = None,
    ) -> None:
        """Initialize state estimator.

        Args:
            config: Estimator configuration
        """
        self._config = config or StateEstimatorConfig()
        self._mode = LocalizationMode.INITIALIZING

        # State vector [north, east, down, vn, ve, vd]
        self._state = np.zeros(6)

        # Covariance matrix
        self._covariance = np.eye(6) * 100.0  # High initial uncertainty

        # Latest sensor data
        self._last_gps_update: datetime | None = None
        self._last_visual_update: datetime | None = None
        self._last_gps_state: VehicleState | None = None
        self._last_gps_quality = GPSQuality.UNAVAILABLE

        # Visual odometry accumulation
        self._visual_position = np.zeros(3)  # Accumulated VO position
        self._visual_uncertainty = 0.0

        # Reference position for NED
        self._reference_set = False
        self._reference_lat = 0.0
        self._reference_lon = 0.0
        self._reference_alt = 0.0

        # Mode switching state
        self._mode_switch_time: datetime | None = None
        self._pending_mode: LocalizationMode | None = None

        # Timing
        self._last_predict_time: float | None = None

        logger.info("StateEstimator initialized in INITIALIZING mode")

    @property
    def mode(self) -> LocalizationMode:
        """Current localization mode."""
        return self._mode

    @property
    def is_healthy(self) -> bool:
        """Check if estimator is providing reliable estimates."""
        return self._mode in (
            LocalizationMode.GPS_PRIMARY,
            LocalizationMode.VISUAL_PRIMARY,
            LocalizationMode.FUSED,
        )

    def set_reference(self, latitude: float, longitude: float, altitude_msl: float) -> None:
        """Set the reference point for NED coordinates.

        Args:
            latitude: Reference latitude
            longitude: Reference longitude
            altitude_msl: Reference altitude in meters MSL
        """
        self._reference_lat = latitude
        self._reference_lon = longitude
        self._reference_alt = altitude_msl
        self._reference_set = True
        logger.info(f"Reference set: ({latitude:.6f}, {longitude:.6f}, {altitude_msl:.1f}m)")

    def update_gps(self, state: VehicleState) -> None:
        """Update with GPS data from vehicle state.

        Args:
            state: Vehicle state containing GPS position
        """
        now = datetime.now()

        # Assess GPS quality
        quality = self._assess_gps_quality(state)
        self._last_gps_quality = quality
        self._last_gps_state = state

        if quality == GPSQuality.UNAVAILABLE:
            return

        # Set reference if not set
        if not self._reference_set:
            self.set_reference(
                state.position.latitude,
                state.position.longitude,
                state.position.altitude_msl,
            )

        # Convert GPS to NED
        gps_ned = self._gps_to_ned(
            state.position.latitude,
            state.position.longitude,
            state.position.altitude_msl,
        )

        # Get measurement noise based on quality
        noise_std = self._get_gps_noise(quality)

        # Update state based on mode
        if self._mode == LocalizationMode.INITIALIZING:
            # Initialize state from GPS
            self._state[:3] = gps_ned
            if state.velocity:
                self._state[3] = state.velocity.north
                self._state[4] = state.velocity.east
                self._state[5] = state.velocity.down
            self._covariance = np.eye(6) * noise_std**2
            self._request_mode_switch(LocalizationMode.GPS_PRIMARY)

        elif self._mode in (LocalizationMode.GPS_PRIMARY, LocalizationMode.FUSED):
            # EKF update with GPS measurement
            self._ekf_update_position(gps_ned, noise_std)

            if state.velocity:
                vel_measurement = np.array([
                    state.velocity.north,
                    state.velocity.east,
                    state.velocity.down,
                ])
                self._ekf_update_velocity(vel_measurement, noise_std * 0.5)

        elif self._mode == LocalizationMode.VISUAL_PRIMARY:
            # GPS recovered, consider switching to FUSED
            if quality in (GPSQuality.EXCELLENT, GPSQuality.GOOD):
                self._request_mode_switch(LocalizationMode.FUSED)

        elif self._mode == LocalizationMode.DEGRADED:
            # GPS recovered, switch to GPS_PRIMARY
            self._request_mode_switch(LocalizationMode.GPS_PRIMARY)

        self._last_gps_update = now
        self._update_mode()

    def update_visual(self, visual: VisualOdometryInput) -> None:
        """Update with visual odometry data.

        Args:
            visual: Visual odometry delta measurement
        """
        if not visual.is_valid or visual.confidence < 0.1:
            return

        now = datetime.now()

        # Accumulate visual odometry
        delta = np.array([visual.delta_north, visual.delta_east, visual.delta_down])
        self._visual_position += delta

        # Accumulate uncertainty based on confidence
        delta_uncertainty = (1.0 - visual.confidence) * self._config.visual_noise_std_m
        self._visual_uncertainty += delta_uncertainty

        # Update state based on mode
        if self._mode == LocalizationMode.INITIALIZING:
            # Can't initialize from VO alone - need GPS first
            pass

        elif self._mode == LocalizationMode.VISUAL_PRIMARY:
            # Use VO as primary
            self._state[:3] += delta
            self._covariance[:3, :3] += np.eye(3) * delta_uncertainty**2

        elif self._mode == LocalizationMode.FUSED:
            # EKF update with visual measurement as delta
            # This is a simplified approach - full VO would use features
            self._ekf_update_position(
                self._state[:3] + delta,
                self._config.visual_noise_std_m / max(0.1, visual.confidence),
            )

        elif self._mode == LocalizationMode.GPS_PRIMARY:
            # Check if GPS is stale, switch to visual
            if self._is_gps_stale():
                self._request_mode_switch(LocalizationMode.VISUAL_PRIMARY)

        self._last_visual_update = now
        self._update_mode()

    def predict(self, dt: float | None = None) -> None:
        """Predict state forward in time.

        Args:
            dt: Time step in seconds (auto-computed if None)
        """
        current_time = time.time()

        if dt is None:
            if self._last_predict_time is None:
                dt = 0.0
            else:
                dt = current_time - self._last_predict_time
        self._last_predict_time = current_time

        if dt <= 0:
            return

        # State transition (constant velocity model)
        # x_{k+1} = x_k + v_k * dt
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        # Process noise
        q_pos = self._config.process_noise_position * dt
        q_vel = self._config.process_noise_velocity * dt
        Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

        # Predict
        self._state = F @ self._state
        self._covariance = F @ self._covariance @ F.T + Q

        # Add drift for visual-only mode
        if self._mode == LocalizationMode.VISUAL_PRIMARY:
            drift = self._config.visual_drift_rate * dt
            self._covariance[:3, :3] += np.eye(3) * drift**2

    def get_estimate(self) -> EstimatedState:
        """Get current fused state estimate.

        Returns:
            EstimatedState with position, velocity, and uncertainty
        """
        # Run prediction step
        self.predict()

        # Calculate uncertainty (trace of position covariance)
        pos_uncertainty = math.sqrt(np.trace(self._covariance[:3, :3]) / 3)
        vel_uncertainty = math.sqrt(np.trace(self._covariance[3:6, 3:6]) / 3)

        # Convert NED to GPS if reference is set
        lat, lon, alt = None, None, None
        if self._reference_set:
            lat, lon, alt = self._ned_to_gps(
                self._state[0], self._state[1], self._state[2]
            )

        # Calculate fusion weight
        fusion_weight = self._calculate_fusion_weight()

        return EstimatedState(
            timestamp=datetime.now(),
            mode=self._mode.value,
            north=float(self._state[0]),
            east=float(self._state[1]),
            down=float(self._state[2]),
            velocity_north=float(self._state[3]),
            velocity_east=float(self._state[4]),
            velocity_down=float(self._state[5]),
            position_uncertainty_m=pos_uncertainty,
            velocity_uncertainty_ms=vel_uncertainty,
            latitude=lat,
            longitude=lon,
            altitude_msl=alt,
            gps_quality=self._last_gps_quality.value,
            visual_confidence=0.0,  # TODO: track from VO updates
            fusion_weight_gps=fusion_weight,
        )

    def get_position_ned(self) -> tuple[float, float, float]:
        """Get current NED position estimate.

        Returns:
            Tuple of (north, east, down)
        """
        return (float(self._state[0]), float(self._state[1]), float(self._state[2]))

    def get_velocity_ned(self) -> tuple[float, float, float]:
        """Get current NED velocity estimate.

        Returns:
            Tuple of (vn, ve, vd)
        """
        return (float(self._state[3]), float(self._state[4]), float(self._state[5]))

    def reset(self) -> None:
        """Reset estimator to initial state."""
        self._mode = LocalizationMode.INITIALIZING
        self._state = np.zeros(6)
        self._covariance = np.eye(6) * 100.0
        self._last_gps_update = None
        self._last_visual_update = None
        self._visual_position = np.zeros(3)
        self._visual_uncertainty = 0.0
        logger.info("StateEstimator reset")

    def _assess_gps_quality(self, state: VehicleState) -> GPSQuality:
        """Assess GPS quality from vehicle state."""
        if not state.gps or not state.gps.has_fix:
            return GPSQuality.UNAVAILABLE

        gps = state.gps
        thresholds = self._config.gps_thresholds

        if gps.fix_type < thresholds.min_fix_type:
            return GPSQuality.POOR

        if gps.hdop < thresholds.excellent_hdop and gps.satellites_visible >= 8:
            return GPSQuality.EXCELLENT
        elif gps.hdop < thresholds.good_hdop and gps.satellites_visible >= thresholds.min_satellites:
            return GPSQuality.GOOD
        elif gps.hdop < thresholds.marginal_hdop:
            return GPSQuality.MARGINAL
        else:
            return GPSQuality.POOR

    def _get_gps_noise(self, quality: GPSQuality) -> float:
        """Get GPS measurement noise based on quality."""
        noise_map = {
            GPSQuality.EXCELLENT: 0.5,
            GPSQuality.GOOD: self._config.gps_noise_std_m,
            GPSQuality.MARGINAL: self._config.gps_noise_std_m * 2,
            GPSQuality.POOR: self._config.gps_noise_std_m * 5,
            GPSQuality.UNAVAILABLE: 100.0,
        }
        return noise_map[quality]

    def _is_gps_stale(self) -> bool:
        """Check if GPS data is stale."""
        if self._last_gps_update is None:
            return True
        elapsed = (datetime.now() - self._last_gps_update).total_seconds()
        return elapsed > self._config.gps_thresholds.timeout_s

    def _gps_to_ned(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """Convert GPS to NED relative to reference."""
        # Simplified conversion (for small distances)
        R = 6371000  # Earth radius in meters

        dlat = math.radians(lat - self._reference_lat)
        dlon = math.radians(lon - self._reference_lon)

        north = dlat * R
        east = dlon * R * math.cos(math.radians(self._reference_lat))
        down = -(alt - self._reference_alt)

        return np.array([north, east, down])

    def _ned_to_gps(self, north: float, east: float, down: float) -> tuple[float, float, float]:
        """Convert NED to GPS."""
        R = 6371000

        lat = self._reference_lat + math.degrees(north / R)
        lon = self._reference_lon + math.degrees(east / (R * math.cos(math.radians(self._reference_lat))))
        alt = self._reference_alt - down

        return (lat, lon, alt)

    def _ekf_update_position(self, measurement: np.ndarray, noise_std: float) -> None:
        """EKF update step for position measurement."""
        # Measurement matrix (observe position only)
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)

        # Measurement noise
        R = np.eye(3) * noise_std**2

        # Innovation
        y = measurement - H @ self._state

        # Innovation covariance
        S = H @ self._covariance @ H.T + R

        # Kalman gain
        K = self._covariance @ H.T @ np.linalg.inv(S)

        # State update
        self._state = self._state + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(6) - K @ H
        self._covariance = I_KH @ self._covariance @ I_KH.T + K @ R @ K.T

    def _ekf_update_velocity(self, measurement: np.ndarray, noise_std: float) -> None:
        """EKF update step for velocity measurement."""
        H = np.zeros((3, 6))
        H[:3, 3:6] = np.eye(3)

        R = np.eye(3) * noise_std**2

        y = measurement - H @ self._state
        S = H @ self._covariance @ H.T + R
        K = self._covariance @ H.T @ np.linalg.inv(S)

        self._state = self._state + K @ y
        I_KH = np.eye(6) - K @ H
        self._covariance = I_KH @ self._covariance @ I_KH.T + K @ R @ K.T

    def _request_mode_switch(self, new_mode: LocalizationMode) -> None:
        """Request a mode switch (with delay for stability)."""
        if new_mode == self._mode:
            self._pending_mode = None
            self._mode_switch_time = None
            return

        if self._pending_mode != new_mode:
            self._pending_mode = new_mode
            self._mode_switch_time = datetime.now()

    def _update_mode(self) -> None:
        """Process pending mode switch."""
        if self._pending_mode is None:
            return

        now = datetime.now()
        if self._mode_switch_time:
            elapsed = (now - self._mode_switch_time).total_seconds()
            if elapsed >= self._config.mode_switch_delay_s:
                logger.info(f"Mode switch: {self._mode.value} -> {self._pending_mode.value}")
                self._mode = self._pending_mode
                self._pending_mode = None
                self._mode_switch_time = None

        # Check for degraded mode
        if self._mode not in (LocalizationMode.INITIALIZING, LocalizationMode.DEGRADED):
            gps_stale = self._is_gps_stale()
            visual_stale = (
                self._last_visual_update is None
                or (now - self._last_visual_update).total_seconds() > self._config.degraded_timeout_s
            )

            if gps_stale and visual_stale:
                logger.warning("Both GPS and visual are stale - entering DEGRADED mode")
                self._mode = LocalizationMode.DEGRADED

    def _calculate_fusion_weight(self) -> float:
        """Calculate GPS weight in fusion (0 = visual only, 1 = GPS only)."""
        if self._mode == LocalizationMode.GPS_PRIMARY:
            return 1.0
        elif self._mode == LocalizationMode.VISUAL_PRIMARY:
            return 0.0
        elif self._mode == LocalizationMode.FUSED:
            # Weight based on GPS quality
            quality_weights = {
                GPSQuality.EXCELLENT: 0.9,
                GPSQuality.GOOD: 0.7,
                GPSQuality.MARGINAL: 0.4,
                GPSQuality.POOR: 0.2,
                GPSQuality.UNAVAILABLE: 0.0,
            }
            return quality_weights.get(self._last_gps_quality, 0.0)
        else:
            return 0.5  # Unknown - equal weight
