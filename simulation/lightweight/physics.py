"""Quadrotor Physics Engine.

Realistic drone physics simulation using NumPy.
Implements:
- Rigid body dynamics
- Motor thrust and torque
- Aerodynamic drag
- IMU sensor simulation with noise
- Battery discharge model
- Wind disturbances
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class DroneConfig:
    """Physical configuration for a quadrotor drone."""

    # Mass and inertia
    mass_kg: float = 1.5  # Total mass
    arm_length_m: float = 0.25  # Motor arm length
    inertia_xx: float = 0.0082  # Moment of inertia (roll)
    inertia_yy: float = 0.0082  # Moment of inertia (pitch)
    inertia_zz: float = 0.0149  # Moment of inertia (yaw)

    # Motor properties
    max_thrust_n: float = 10.0  # Max thrust per motor
    motor_time_constant_s: float = 0.02  # Motor response time
    thrust_to_torque: float = 0.016  # Thrust to torque coefficient

    # Aerodynamics
    drag_coefficient: float = 0.1  # Linear drag
    drag_coefficient_sq: float = 0.01  # Quadratic drag

    # Battery
    battery_capacity_mah: float = 5000.0
    battery_voltage_full: float = 25.2  # 6S LiPo full
    battery_voltage_empty: float = 21.0  # 6S LiPo empty
    hover_current_a: float = 15.0  # Current at hover
    idle_current_a: float = 0.5  # Idle current

    # Sensors
    imu_accel_noise_std: float = 0.1  # m/s²
    imu_gyro_noise_std: float = 0.01  # rad/s
    imu_accel_bias: float = 0.02  # m/s²
    imu_gyro_bias: float = 0.001  # rad/s
    gps_noise_std_m: float = 1.5  # GPS position noise

    # Limits
    max_velocity_ms: float = 20.0
    max_tilt_rad: float = 0.7  # ~40 degrees
    max_yaw_rate_rads: float = 3.14  # 180 deg/s


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    gravity_ms2: float = 9.81
    air_density_kgm3: float = 1.225
    ground_level_m: float = 0.0

    # Wind
    wind_speed_ms: float = 0.0
    wind_direction_rad: float = 0.0  # Direction wind is coming FROM
    wind_gust_intensity: float = 0.0  # 0-1
    wind_turbulence: float = 0.0  # 0-1


@dataclass
class WindModel:
    """Dynamic wind model with gusts and turbulence."""

    base_speed_ms: float = 0.0
    base_direction_rad: float = 0.0
    gust_intensity: float = 0.0
    turbulence: float = 0.0

    # Internal state
    _gust_timer: float = 0.0
    _current_gust: float = 0.0
    _turbulence_state: NDArray = field(default_factory=lambda: np.zeros(3))

    def update(self, dt: float) -> NDArray:
        """Update wind model and return wind velocity vector (NED)."""
        # Base wind
        wind_n = -self.base_speed_ms * math.cos(self.base_direction_rad)
        wind_e = -self.base_speed_ms * math.sin(self.base_direction_rad)

        # Gusts (occasional stronger bursts)
        self._gust_timer += dt
        if self._gust_timer > np.random.exponential(5.0):  # Avg 5s between gusts
            self._current_gust = (
                self.gust_intensity * self.base_speed_ms * np.random.uniform(0.5, 1.5)
            )
            self._gust_timer = 0.0

        self._current_gust *= 0.95  # Decay gust

        # Turbulence (high-frequency noise)
        if self.turbulence > 0:
            noise = np.random.randn(3) * self.turbulence * 2.0
            self._turbulence_state = 0.9 * self._turbulence_state + 0.1 * noise

        total_wind = np.array([
            wind_n
            + self._current_gust * math.cos(self.base_direction_rad)
            + self._turbulence_state[0],
            wind_e
            + self._current_gust * math.sin(self.base_direction_rad)
            + self._turbulence_state[1],
            self._turbulence_state[2] * 0.3,  # Less vertical turbulence
        ])

        return total_wind


@dataclass
class DroneState:
    """Complete drone state."""

    # Position (NED frame, meters)
    position: NDArray = field(default_factory=lambda: np.zeros(3))

    # Velocity (NED frame, m/s)
    velocity: NDArray = field(default_factory=lambda: np.zeros(3))

    # Attitude (quaternion: w, x, y, z)
    quaternion: NDArray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    # Angular velocity (body frame, rad/s)
    angular_velocity: NDArray = field(default_factory=lambda: np.zeros(3))

    # Motor states (0-1 throttle for each motor)
    motor_speeds: NDArray = field(default_factory=lambda: np.zeros(4))

    # Battery
    battery_remaining_mah: float = 5000.0
    battery_voltage: float = 25.2

    # Flags
    armed: bool = False
    in_air: bool = False
    crashed: bool = False

    # Timestamps
    sim_time_s: float = 0.0


@dataclass
class IMUReading:
    """Simulated IMU sensor reading."""

    timestamp_s: float
    accelerometer: NDArray  # m/s² (body frame)
    gyroscope: NDArray  # rad/s (body frame)
    magnetometer: NDArray  # normalized (body frame)


@dataclass
class MotorCommand:
    """Command to the motors."""

    throttle: NDArray  # 0-1 for each motor (4 motors)
    armed: bool = True


class DronePhysics:
    """Physics simulation for a single drone."""

    def __init__(
        self,
        config: DroneConfig | None = None,
        env_config: EnvironmentConfig | None = None,
        initial_position: NDArray | None = None,
    ) -> None:
        """Initialize drone physics."""
        self.config = config or DroneConfig()
        self.env = env_config or EnvironmentConfig()

        # Initialize state
        self.state = DroneState()
        if initial_position is not None:
            self.state.position = initial_position.copy()

        self.state.battery_remaining_mah = self.config.battery_capacity_mah

        # Wind model
        self.wind = WindModel(
            base_speed_ms=self.env.wind_speed_ms,
            base_direction_rad=self.env.wind_direction_rad,
            gust_intensity=self.env.wind_gust_intensity,
            turbulence=self.env.wind_turbulence,
        )

        # IMU bias (constant per session)
        self._accel_bias = np.random.randn(3) * self.config.imu_accel_bias
        self._gyro_bias = np.random.randn(3) * self.config.imu_gyro_bias

        # Inertia tensor
        self._inertia = np.diag([
            self.config.inertia_xx,
            self.config.inertia_yy,
            self.config.inertia_zz,
        ])
        self._inertia_inv = np.linalg.inv(self._inertia)

    def step(self, command: MotorCommand, dt: float) -> None:
        """Advance simulation by dt seconds."""
        if self.state.crashed:
            return

        # Update motor speeds (with time constant)
        target_speeds = command.throttle if command.armed else np.zeros(4)
        alpha = 1.0 - math.exp(-dt / self.config.motor_time_constant_s)
        self.state.motor_speeds = self.state.motor_speeds * (1 - alpha) + target_speeds * alpha

        # Compute forces and torques
        forces, torques = self._compute_forces_torques()

        # Update state using RK4 integration
        self._integrate_rk4(forces, torques, dt)

        # Ground collision
        if self.state.position[2] > -self.env.ground_level_m:
            self.state.position[2] = -self.env.ground_level_m
            if self.state.velocity[2] > 0:
                # Hard landing detection
                if self.state.velocity[2] > 3.0:
                    self.state.crashed = True
                self.state.velocity[2] = 0
                self.state.in_air = False
        else:
            self.state.in_air = True

        # Update battery
        self._update_battery(dt)

        # Update armed state
        self.state.armed = command.armed

        # Update sim time
        self.state.sim_time_s += dt

    def _compute_forces_torques(self) -> tuple[NDArray, NDArray]:
        """Compute total forces and torques on the drone."""
        # Motor thrusts
        thrusts = self.state.motor_speeds * self.config.max_thrust_n
        total_thrust = np.sum(thrusts)

        # Thrust vector in body frame (up is -Z in NED)
        thrust_body = np.array([0, 0, -total_thrust])

        # Rotate to world frame
        R = self._quaternion_to_rotation_matrix(self.state.quaternion)
        thrust_world = R @ thrust_body

        # Gravity (world frame)
        gravity = np.array([0, 0, self.config.mass_kg * self.env.gravity_ms2])

        # Aerodynamic drag (world frame)
        wind_velocity = self.wind.update(0.01)  # Get current wind
        relative_velocity = self.state.velocity - wind_velocity
        speed = np.linalg.norm(relative_velocity)
        if speed > 0.01:
            drag_dir = -relative_velocity / speed
            drag_mag = (
                self.config.drag_coefficient * speed + self.config.drag_coefficient_sq * speed**2
            )
            drag = drag_dir * drag_mag
        else:
            drag = np.zeros(3)

        # Total force
        forces = thrust_world + gravity + drag

        # Motor torques (body frame)
        # Assuming X configuration:
        #   Motor 0: front-right (+x, +y) -> CW
        #   Motor 1: rear-left (-x, -y) -> CW
        #   Motor 2: front-left (+x, -y) -> CCW
        #   Motor 3: rear-right (-x, +y) -> CCW
        L = self.config.arm_length_m
        k = self.config.thrust_to_torque

        # Roll torque (positive = right side down)
        roll_torque = L * (thrusts[2] + thrusts[0] - thrusts[1] - thrusts[3]) * 0.707

        # Pitch torque (positive = nose down)
        pitch_torque = L * (thrusts[0] + thrusts[2] - thrusts[1] - thrusts[3]) * 0.707

        # Yaw torque (from motor reaction)
        yaw_torque = k * (thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3])

        torques = np.array([roll_torque, pitch_torque, yaw_torque])

        return forces, torques

    def _integrate_rk4(self, forces: NDArray, torques: NDArray, dt: float) -> None:
        """Integrate equations of motion using RK4."""
        # For simplicity, using semi-implicit Euler here
        # (RK4 would be more accurate but more complex)

        # Linear acceleration
        accel = forces / self.config.mass_kg

        # Angular acceleration
        omega = self.state.angular_velocity
        # Euler's rotation equations: I * omega_dot = torques - omega x (I * omega)
        omega_dot = self._inertia_inv @ (torques - np.cross(omega, self._inertia @ omega))

        # Update velocities
        self.state.velocity += accel * dt
        self.state.angular_velocity += omega_dot * dt

        # Clamp velocities
        speed = np.linalg.norm(self.state.velocity)
        if speed > self.config.max_velocity_ms:
            self.state.velocity *= self.config.max_velocity_ms / speed

        # Update position
        self.state.position += self.state.velocity * dt

        # Update quaternion
        self._integrate_quaternion(dt)

    def _integrate_quaternion(self, dt: float) -> None:
        """Integrate quaternion using angular velocity."""
        q = self.state.quaternion
        omega = self.state.angular_velocity

        # Quaternion derivative: q_dot = 0.5 * q * omega_quat
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * self._quaternion_multiply(q, omega_quat)

        # Integrate
        q_new = q + q_dot * dt

        # Normalize
        self.state.quaternion = q_new / np.linalg.norm(q_new)

    def _update_battery(self, dt: float) -> None:
        """Update battery state."""
        # Current draw based on motor power
        avg_throttle = np.mean(self.state.motor_speeds)
        current = (
            self.config.idle_current_a
            + avg_throttle * (self.config.hover_current_a - self.config.idle_current_a) * 2
        )

        # Discharge
        mah_used = current * (dt / 3600) * 1000
        self.state.battery_remaining_mah -= mah_used

        # Voltage (simple linear model)
        soc = self.state.battery_remaining_mah / self.config.battery_capacity_mah
        self.state.battery_voltage = self.config.battery_voltage_empty + soc * (
            self.config.battery_voltage_full - self.config.battery_voltage_empty
        )

    def get_imu_reading(self) -> IMUReading:
        """Get simulated IMU reading with noise."""
        # True accelerometer (body frame) = R^T * (accel - gravity)
        R = self._quaternion_to_rotation_matrix(self.state.quaternion)
        gravity_world = np.array([0, 0, self.env.gravity_ms2])

        # Compute acceleration from forces
        forces, _ = self._compute_forces_torques()
        accel_world = forces / self.config.mass_kg

        # Transform to body frame
        accel_body = R.T @ (accel_world - gravity_world)

        # Add noise and bias
        accel_noisy = (
            accel_body + self._accel_bias + np.random.randn(3) * self.config.imu_accel_noise_std
        )

        # Gyroscope (body frame)
        gyro_noisy = (
            self.state.angular_velocity
            + self._gyro_bias
            + np.random.randn(3) * self.config.imu_gyro_noise_std
        )

        # Simple magnetometer (assuming magnetic north = world X)
        mag_world = np.array([1.0, 0.0, 0.3])  # Inclination
        mag_body = R.T @ mag_world
        mag_body /= np.linalg.norm(mag_body)

        return IMUReading(
            timestamp_s=self.state.sim_time_s,
            accelerometer=accel_noisy,
            gyroscope=gyro_noisy,
            magnetometer=mag_body,
        )

    def get_euler_angles(self) -> tuple[float, float, float]:
        """Get roll, pitch, yaw from quaternion."""
        q = self.state.quaternion
        w, x, y, z = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def get_battery_percent(self) -> float:
        """Get battery state of charge as percentage."""
        return (self.state.battery_remaining_mah / self.config.battery_capacity_mah) * 100

    @staticmethod
    def _quaternion_to_rotation_matrix(q: NDArray) -> NDArray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ])

    @staticmethod
    def _quaternion_multiply(q1: NDArray, q2: NDArray) -> NDArray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])


class SimpleFlightController:
    """Simple PID-based flight controller."""

    def __init__(self, config: DroneConfig) -> None:
        """Initialize flight controller."""
        self.config = config

        # PID gains
        self.kp_pos = np.array([2.0, 2.0, 3.0])  # Position
        self.kd_pos = np.array([1.5, 1.5, 2.0])  # Velocity
        self.kp_att = np.array([8.0, 8.0, 4.0])  # Attitude
        self.kd_att = np.array([2.0, 2.0, 1.0])  # Angular rate

        # Hover throttle
        self.hover_throttle = 0.5

    def compute_command(
        self,
        current_state: DroneState,
        target_position: NDArray,
        target_yaw: float = 0.0,
    ) -> MotorCommand:
        """Compute motor commands to reach target position."""
        # Position error
        pos_error = target_position - current_state.position

        # Velocity command (PD on position)
        vel_cmd = self.kp_pos * pos_error - self.kd_pos * current_state.velocity

        # Clamp velocity command
        vel_mag = np.linalg.norm(vel_cmd[:2])
        if vel_mag > 5.0:
            vel_cmd[:2] *= 5.0 / vel_mag

        # Desired thrust direction
        thrust_dir = vel_cmd + np.array([0, 0, -9.81])  # Add gravity compensation
        thrust_mag = np.linalg.norm(thrust_dir)
        thrust_dir_norm = thrust_dir / thrust_mag if thrust_mag > 0.01 else np.array([0, 0, -1])

        # Desired attitude from thrust direction
        desired_roll = math.asin(-thrust_dir_norm[1])
        desired_pitch = math.atan2(thrust_dir_norm[0], -thrust_dir_norm[2])

        # Current attitude
        roll, pitch, yaw = self._quaternion_to_euler(current_state.quaternion)

        # Attitude error
        roll_error = desired_roll - roll
        pitch_error = desired_pitch - pitch
        yaw_error = self._wrap_angle(target_yaw - yaw)

        att_error = np.array([roll_error, pitch_error, yaw_error])

        # Rate command
        rate_cmd = self.kp_att * att_error - self.kd_att * current_state.angular_velocity

        # Thrust command (normalized)
        base_throttle = self.hover_throttle + 0.1 * (
            -pos_error[2] - current_state.velocity[2] * 0.5
        )
        base_throttle = np.clip(base_throttle, 0.1, 0.9)

        # Mix to motors (simplified mixer)
        # Assuming X configuration
        motor_throttles = np.array([
            base_throttle - rate_cmd[0] + rate_cmd[1] + rate_cmd[2],  # Front-right
            base_throttle + rate_cmd[0] - rate_cmd[1] + rate_cmd[2],  # Rear-left
            base_throttle - rate_cmd[0] - rate_cmd[1] - rate_cmd[2],  # Front-left
            base_throttle + rate_cmd[0] + rate_cmd[1] - rate_cmd[2],  # Rear-right
        ])

        # Clamp throttles
        motor_throttles = np.clip(motor_throttles, 0.0, 1.0)

        return MotorCommand(throttle=motor_throttles, armed=True)

    @staticmethod
    def _quaternion_to_euler(q: NDArray) -> tuple[float, float, float]:
        """Convert quaternion to Euler angles."""
        w, x, y, z = q
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(np.clip(sinp, -1, 1))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return roll, pitch, yaw

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
