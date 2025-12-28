#!/usr/bin/env python3
"""
AegisAV Simulation Scenario Orchestrator
Uses pymavlink to dynamically manipulate SITL environment parameters.
"""

import argparse
import logging
import time

from pymavlink import mavutil

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SCENARIO] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SimOrchestrator:
    def __init__(self, connection_string: str):
        logger.info(f"Connecting to SITL at {connection_string}...")
        self.master = mavutil.mavlink_connection(connection_string)
        self.master.wait_heartbeat()
        logger.info("Connected to vehicle.")

    def set_param(self, param_name: str, value: float):
        """Set a MAVLink parameter in SITL."""
        logger.info(f"Setting parameter {param_name} to {value}")
        self.master.mav.param_set_send(
            self.master.target_system,
            self.master.target_component,
            param_name.encode("utf-8"),
            value,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )

    def set_wind(
        self, speed: float, direction: float | None = None, turbulence: float | None = None
    ):
        """Set wind conditions."""
        self.set_param("SIM_WIND_SPD", speed)
        if direction is not None:
            self.set_param("SIM_WIND_DIR", direction)
        if turbulence is not None:
            self.set_param("SIM_WIND_TURB", turbulence)

    def set_gps_accuracy(self, horizontal_acc: float):
        """Set GPS horizontal accuracy (meters)."""
        self.set_param("SIM_GPS_ACC", horizontal_acc)

    def set_battery(self, voltage: float | None = None, _pct: float | None = None):
        """Set battery state (if supported by SITL backend)."""
        if voltage is not None:
            self.set_param("SIM_BATT_VOLTAGE", voltage)
        # Note: Percentage is usually derived from voltage in SITL

    def run_gust_scenario(self, duration_s: int, peak_speed: float):
        """Simulate a temporary wind gust."""
        original_speed = 5.0  # Assume default
        logger.info(
            f"Starting gust scenario: {original_speed}m/s -> {peak_speed}m/s for {duration_s}s"
        )

        # Ramp up
        self.set_wind(peak_speed)
        time.sleep(duration_s)

        # Ramp down
        self.set_wind(original_speed)
        logger.info("Gust scenario complete.")

    def run_gps_glitch(self, duration_s: int):
        """Simulate a temporary GPS glitch."""
        logger.info(f"Injecting GPS glitch (100m error) for {duration_s}s")
        self.set_gps_accuracy(100.0)
        time.sleep(duration_s)
        self.set_gps_accuracy(2.5)  # Restore default
        logger.info("GPS glitch cleared.")


def main():
    parser = argparse.ArgumentParser(description="AegisAV SITL Scenario Orchestrator")
    parser.add_argument(
        "--connect", default="udp:127.0.0.1:14550", help="MAVLink connection string"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Wind command
    wind_parser = subparsers.add_parser("wind", help="Set wind conditions")
    wind_parser.add_argument("speed", type=float, help="Wind speed in m/s")
    wind_parser.add_argument("--dir", type=float, help="Wind direction in degrees")
    wind_parser.add_argument("--turb", type=float, help="Turbulence factor")

    # GPS command
    gps_parser = subparsers.add_parser("gps", help="Set GPS accuracy")
    gps_parser.add_argument("acc", type=float, help="Horizontal accuracy in meters")

    # Scenario command
    scenario_parser = subparsers.add_parser("scenario", help="Run pre-defined scenarios")
    scenario_parser.add_argument("name", choices=["gust", "glitch"], help="Scenario name")
    scenario_parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    scenario_parser.add_argument(
        "--val", type=float, help="Value for the scenario (e.g., peak wind speed)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    orch = SimOrchestrator(args.connect)

    if args.command == "wind":
        orch.set_wind(args.speed, args.dir, args.turb)
    elif args.command == "gps":
        orch.set_gps_accuracy(args.acc)
    elif args.command == "scenario":
        if args.name == "gust":
            orch.run_gust_scenario(args.duration, args.val or 15.0)
        elif args.name == "glitch":
            orch.run_gps_glitch(args.duration)


if __name__ == "__main__":
    main()
