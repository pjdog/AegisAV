"""
Coordinate conversion utilities for AirSim integration.

Converts between GPS (WGS84 lat/lon) and AirSim NED (North-East-Down) coordinates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# WGS84 ellipsoid constants
WGS84_A = 6378137.0  # Semi-major axis (equatorial radius) in meters
WGS84_B = 6356752.314245  # Semi-minor axis (polar radius) in meters
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # First eccentricity squared


@dataclass
class GeoReference:
    """
    Geographic reference point for NED coordinate conversion.

    AirSim uses a local NED (North-East-Down) coordinate system where:
    - X is North (positive = northward)
    - Y is East (positive = eastward)
    - Z is Down (positive = downward, so negative Z = altitude)

    This class converts between GPS coordinates and NED relative to a reference point.

    Example:
        >>> ref = GeoReference(latitude=47.641468, longitude=-122.140165, altitude=0.0)
        >>> north, east, down = ref.gps_to_ned(47.642, -122.139, 30.0)
        >>> print(f"NED: ({north:.1f}, {east:.1f}, {down:.1f})")
        NED: (59.2, 83.5, -30.0)
    """
    latitude: float  # Reference latitude in degrees
    longitude: float  # Reference longitude in degrees
    altitude: float  # Reference altitude in meters MSL (Mean Sea Level)

    def gps_to_ned(
        self,
        lat: float,
        lon: float,
        alt: float
    ) -> tuple[float, float, float]:
        """
        Convert GPS coordinates to NED (North-East-Down) relative to reference.

        Args:
            lat: Target latitude in degrees
            lon: Target longitude in degrees
            alt: Target altitude in meters MSL

        Returns:
            Tuple of (north, east, down) in meters relative to reference point.
            - north: Positive = northward from reference
            - east: Positive = eastward from reference
            - down: Positive = below reference altitude (negative = above)
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

        # Radius of curvature in the prime vertical (N)
        sin_lat0 = math.sin(lat0_rad)
        R_N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat0**2)

        # Radius of curvature in the meridian (M)
        R_M = R_N * (1 - WGS84_E2) / (1 - WGS84_E2 * sin_lat0**2)

        # Convert to NED
        north = dlat * R_M
        east = dlon * R_N * math.cos(lat0_rad)
        down = -dalt  # NED: positive down, so negative altitude difference

        return (north, east, down)

    def ned_to_gps(
        self,
        north: float,
        east: float,
        down: float
    ) -> tuple[float, float, float]:
        """
        Convert NED coordinates back to GPS.

        Args:
            north: Meters north of reference (positive = north)
            east: Meters east of reference (positive = east)
            down: Meters below reference (positive = down, negative = up)

        Returns:
            Tuple of (latitude, longitude, altitude_msl) in degrees and meters
        """
        lat0_rad = math.radians(self.latitude)
        sin_lat0 = math.sin(lat0_rad)

        # Radius of curvature
        R_N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat0**2)
        R_M = R_N * (1 - WGS84_E2) / (1 - WGS84_E2 * sin_lat0**2)

        # Convert back to GPS
        lat = self.latitude + math.degrees(north / R_M)
        lon = self.longitude + math.degrees(east / (R_N * math.cos(lat0_rad)))
        alt = self.altitude - down  # Negate down to get altitude

        return (lat, lon, alt)

    def distance_ned(
        self,
        ned1: tuple[float, float, float],
        ned2: tuple[float, float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two NED positions.

        Args:
            ned1: First position (north, east, down)
            ned2: Second position (north, east, down)

        Returns:
            Distance in meters
        """
        return math.sqrt(
            (ned2[0] - ned1[0])**2 +
            (ned2[1] - ned1[1])**2 +
            (ned2[2] - ned1[2])**2
        )

    def bearing_to(
        self,
        from_ned: tuple[float, float, float],
        to_ned: tuple[float, float, float]
    ) -> float:
        """
        Calculate bearing from one NED position to another.

        Args:
            from_ned: Starting position (north, east, down)
            to_ned: Target position (north, east, down)

        Returns:
            Bearing in degrees (0 = north, 90 = east, 180 = south, 270 = west)
        """
        dn = to_ned[0] - from_ned[0]
        de = to_ned[1] - from_ned[1]

        bearing = math.degrees(math.atan2(de, dn))
        if bearing < 0:
            bearing += 360
        return bearing


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate great-circle distance between two GPS points using Haversine formula.

    Args:
        lat1, lon1: First point in degrees
        lat2, lon2: Second point in degrees

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth's mean radius in meters

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2)**2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def initial_bearing(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate initial bearing from point 1 to point 2.

    Args:
        lat1, lon1: Starting point in degrees
        lat2, lon2: Destination point in degrees

    Returns:
        Initial bearing in degrees (0-360)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360
