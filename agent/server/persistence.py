"""Redis Persistence Layer.

Provides persistent storage for AegisAV using Redis.
Stores assets, anomalies, inspection history, and mission state.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Try to import redis
try:
    import redis.asyncio as redis
    from redis.asyncio import Redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    Redis = None

T = TypeVar("T", bound=BaseModel)


class RedisConfig(BaseModel):
    """Redis connection configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    decode_responses: bool = True
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0

    # Key prefixes for organization
    prefix: str = "aegis"

    # TTL settings (in seconds)
    telemetry_ttl: int = 3600  # 1 hour for telemetry data
    detection_ttl: int = 86400 * 7  # 7 days for detections
    asset_ttl: int = 0  # No expiry for assets (0 = permanent)
    anomaly_ttl: int = 86400 * 30  # 30 days for anomalies
    mission_ttl: int = 86400 * 90  # 90 days for mission history


class RedisStore:
    """Redis-based persistence layer for AegisAV.

    Provides async methods for storing and retrieving:
    - Assets (infrastructure being monitored)
    - Anomalies (detected issues)
    - Detections (raw detection results)
    - Vehicle telemetry
    - Mission history
    - System state

    Example:
        store = RedisStore(RedisConfig())
        await store.connect()

        # Store an asset
        await store.set_asset("asset_001", asset_data)

        # Get all assets
        assets = await store.get_all_assets()

        await store.disconnect()
    """

    def __init__(self, config: RedisConfig | None = None) -> None:
        """Initialize the RedisStore.

        Args:
            config: Redis configuration. Uses defaults if None.
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package not installed. Install with: pip install redis"
            )

        self.config = config or RedisConfig()
        self._client: Redis | None = None
        self._connected = False
        self.logger = logger

    async def connect(self) -> bool:
        """Connect to Redis server.

        Returns:
            True if connection successful
        """
        try:
            self.logger.info(
                f"Connecting to Redis at {self.config.host}:{self.config.port}"
            )

            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=self.config.decode_responses,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
            )

            # Test connection
            await self._client.ping()
            self._connected = True
            self.logger.info("Connected to Redis successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from Redis.

        Closes the connection and resets the connected state.
        """
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        self.logger.info("Disconnected from Redis")

    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected and self._client is not None

    def _key(self, *parts: str) -> str:
        """Build a Redis key with prefix.

        Args:
            *parts: Key path components to join.

        Returns:
            Full Redis key with prefix.
        """
        return f"{self.config.prefix}:{':'.join(parts)}"

    # ==================== Asset Operations ====================

    async def set_asset(self, asset_id: str, asset: BaseModel | dict) -> bool:
        """Store an asset.

        Args:
            asset_id: Unique asset identifier
            asset: Asset data (Pydantic model or dict)

        Returns:
            True if successful
        """
        if not self.is_connected:
            return False

        try:
            key = self._key("asset", asset_id)
            data = asset.model_dump_json() if isinstance(asset, BaseModel) else json.dumps(asset)

            if self.config.asset_ttl > 0:
                await self._client.setex(key, self.config.asset_ttl, data)
            else:
                await self._client.set(key, data)

            # Add to asset index
            await self._client.sadd(self._key("assets"), asset_id)
            return True

        except Exception as e:
            self.logger.error(f"Failed to set asset {asset_id}: {e}")
            return False

    async def get_asset(self, asset_id: str) -> dict | None:
        """Get an asset by ID.

        Args:
            asset_id: Asset identifier

        Returns:
            Asset data dict or None if not found
        """
        if not self.is_connected:
            return None

        try:
            key = self._key("asset", asset_id)
            data = await self._client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            self.logger.error(f"Failed to get asset {asset_id}: {e}")
            return None

    async def get_all_assets(self) -> list[dict]:
        """Get all stored assets.

        Returns:
            List of asset data dicts
        """
        if not self.is_connected:
            return []

        try:
            asset_ids = await self._client.smembers(self._key("assets"))
            assets = []

            for asset_id in asset_ids:
                asset = await self.get_asset(asset_id)
                if asset:
                    assets.append(asset)

            return assets

        except Exception as e:
            self.logger.error(f"Failed to get all assets: {e}")
            return []

    async def delete_asset(self, asset_id: str) -> bool:
        """Delete an asset.

        Args:
            asset_id: Asset identifier to delete.

        Returns:
            True if successful.
        """
        if not self.is_connected:
            return False

        try:
            key = self._key("asset", asset_id)
            await self._client.delete(key)
            await self._client.srem(self._key("assets"), asset_id)
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete asset {asset_id}: {e}")
            return False

    # ==================== Anomaly Operations ====================

    async def add_anomaly(self, anomaly_id: str, anomaly: BaseModel | dict) -> bool:
        """Store an anomaly.

        Args:
            anomaly_id: Unique anomaly identifier
            anomaly: Anomaly data

        Returns:
            True if successful
        """
        if not self.is_connected:
            return False

        try:
            key = self._key("anomaly", anomaly_id)
            data = anomaly.model_dump_json() if isinstance(anomaly, BaseModel) else json.dumps(anomaly)

            if self.config.anomaly_ttl > 0:
                await self._client.setex(key, self.config.anomaly_ttl, data)
            else:
                await self._client.set(key, data)

            # Add to anomaly index with timestamp score for sorting
            timestamp = datetime.now().timestamp()
            await self._client.zadd(self._key("anomalies"), {anomaly_id: timestamp})

            # Track by asset if asset_id present
            anomaly_dict = anomaly.model_dump() if isinstance(anomaly, BaseModel) else anomaly
            if "asset_id" in anomaly_dict:
                await self._client.sadd(
                    self._key("anomalies", "asset", anomaly_dict["asset_id"]),
                    anomaly_id
                )

            return True

        except Exception as e:
            self.logger.error(f"Failed to add anomaly {anomaly_id}: {e}")
            return False

    async def get_anomaly(self, anomaly_id: str) -> dict | None:
        """Get an anomaly by ID.

        Args:
            anomaly_id: Anomaly identifier.

        Returns:
            Anomaly data dict or None if not found.
        """
        if not self.is_connected:
            return None

        try:
            key = self._key("anomaly", anomaly_id)
            data = await self._client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            self.logger.error(f"Failed to get anomaly {anomaly_id}: {e}")
            return None

    async def get_recent_anomalies(self, limit: int = 100) -> list[dict]:
        """Get recent anomalies, sorted by timestamp (newest first).

        Args:
            limit: Maximum number to return

        Returns:
            List of anomaly dicts
        """
        if not self.is_connected:
            return []

        try:
            # Get anomaly IDs sorted by score (timestamp) descending
            anomaly_ids = await self._client.zrevrange(
                self._key("anomalies"), 0, limit - 1
            )

            anomalies = []
            for anomaly_id in anomaly_ids:
                anomaly = await self.get_anomaly(anomaly_id)
                if anomaly:
                    anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            self.logger.error(f"Failed to get recent anomalies: {e}")
            return []

    async def get_anomalies_for_asset(self, asset_id: str) -> list[dict]:
        """Get all anomalies for a specific asset.

        Args:
            asset_id: Asset identifier.

        Returns:
            List of anomaly dicts for the asset.
        """
        if not self.is_connected:
            return []

        try:
            anomaly_ids = await self._client.smembers(
                self._key("anomalies", "asset", asset_id)
            )

            anomalies = []
            for anomaly_id in anomaly_ids:
                anomaly = await self.get_anomaly(anomaly_id)
                if anomaly:
                    anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            self.logger.error(f"Failed to get anomalies for asset {asset_id}: {e}")
            return []

    async def resolve_anomaly(self, anomaly_id: str) -> bool:
        """Mark an anomaly as resolved.

        Args:
            anomaly_id: Anomaly identifier to resolve.

        Returns:
            True if successful.
        """
        if not self.is_connected:
            return False

        try:
            anomaly = await self.get_anomaly(anomaly_id)
            if anomaly:
                anomaly["resolved"] = True
                anomaly["resolved_at"] = datetime.now().isoformat()
                return await self.add_anomaly(anomaly_id, anomaly)
            return False
        except Exception as e:
            self.logger.error(f"Failed to resolve anomaly {anomaly_id}: {e}")
            return False

    # ==================== Detection Operations ====================

    async def add_detection(
        self,
        asset_id: str,
        detection: BaseModel | dict,
        timestamp: datetime | None = None
    ) -> bool:
        """Store a detection result.

        Args:
            asset_id: Asset the detection is for
            detection: Detection data
            timestamp: Detection timestamp (defaults to now)

        Returns:
            True if successful
        """
        if not self.is_connected:
            return False

        try:
            ts = timestamp or datetime.now()
            detection_id = f"{asset_id}:{ts.timestamp()}"
            key = self._key("detection", detection_id)

            data = detection.model_dump_json() if isinstance(detection, BaseModel) else json.dumps(detection)

            if self.config.detection_ttl > 0:
                await self._client.setex(key, self.config.detection_ttl, data)
            else:
                await self._client.set(key, data)

            # Add to detection list for asset (sorted by timestamp)
            await self._client.zadd(
                self._key("detections", asset_id),
                {detection_id: ts.timestamp()}
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to add detection for {asset_id}: {e}")
            return False

    async def get_detections_for_asset(
        self,
        asset_id: str,
        limit: int = 50,
        since: datetime | None = None
    ) -> list[dict]:
        """Get detections for an asset.

        Args:
            asset_id: Asset identifier
            limit: Maximum number to return
            since: Only return detections after this time

        Returns:
            List of detection dicts (newest first)
        """
        if not self.is_connected:
            return []

        try:
            min_score = since.timestamp() if since else "-inf"

            detection_ids = await self._client.zrevrangebyscore(
                self._key("detections", asset_id),
                "+inf",
                min_score,
                start=0,
                num=limit
            )

            detections = []
            for detection_id in detection_ids:
                key = self._key("detection", detection_id)
                data = await self._client.get(key)
                if data:
                    detections.append(json.loads(data))

            return detections

        except Exception as e:
            self.logger.error(f"Failed to get detections for {asset_id}: {e}")
            return []

    # ==================== Telemetry Operations ====================

    async def add_telemetry(self, vehicle_id: str, telemetry: dict) -> bool:
        """Store vehicle telemetry (time-series data).

        Uses a sorted set for efficient time-range queries.

        Args:
            vehicle_id: Vehicle identifier
            telemetry: Telemetry data dict

        Returns:
            True if successful
        """
        if not self.is_connected:
            return False

        try:
            timestamp = datetime.now()
            telemetry["timestamp"] = timestamp.isoformat()

            key = self._key("telemetry", vehicle_id)
            data = json.dumps(telemetry)

            # Add to sorted set with timestamp as score
            await self._client.zadd(key, {data: timestamp.timestamp()})

            # Trim old entries (keep last hour)
            cutoff = (timestamp - timedelta(seconds=self.config.telemetry_ttl)).timestamp()
            await self._client.zremrangebyscore(key, "-inf", cutoff)

            return True

        except Exception as e:
            self.logger.error(f"Failed to add telemetry for {vehicle_id}: {e}")
            return False

    async def get_telemetry(
        self,
        vehicle_id: str,
        since: datetime | None = None,
        limit: int = 100
    ) -> list[dict]:
        """Get telemetry for a vehicle.

        Args:
            vehicle_id: Vehicle identifier
            since: Only return telemetry after this time
            limit: Maximum number to return

        Returns:
            List of telemetry dicts (newest first)
        """
        if not self.is_connected:
            return []

        try:
            min_score = since.timestamp() if since else "-inf"

            key = self._key("telemetry", vehicle_id)
            entries = await self._client.zrevrangebyscore(
                key, "+inf", min_score, start=0, num=limit
            )

            return [json.loads(entry) for entry in entries]

        except Exception as e:
            self.logger.error(f"Failed to get telemetry for {vehicle_id}: {e}")
            return []

    async def get_latest_telemetry(self, vehicle_id: str) -> dict | None:
        """Get the most recent telemetry for a vehicle.

        Args:
            vehicle_id: Vehicle identifier.

        Returns:
            Latest telemetry dict or None if not available.
        """
        telemetry = await self.get_telemetry(vehicle_id, limit=1)
        return telemetry[0] if telemetry else None

    # ==================== Mission Operations ====================

    async def save_mission(self, mission_id: str, mission: dict) -> bool:
        """Save a mission record.

        Args:
            mission_id: Mission identifier
            mission: Mission data

        Returns:
            True if successful
        """
        if not self.is_connected:
            return False

        try:
            key = self._key("mission", mission_id)

            if self.config.mission_ttl > 0:
                await self._client.setex(key, self.config.mission_ttl, json.dumps(mission))
            else:
                await self._client.set(key, json.dumps(mission))

            # Add to mission index
            timestamp = datetime.now().timestamp()
            await self._client.zadd(self._key("missions"), {mission_id: timestamp})

            return True

        except Exception as e:
            self.logger.error(f"Failed to save mission {mission_id}: {e}")
            return False

    async def get_mission(self, mission_id: str) -> dict | None:
        """Get a mission by ID.

        Args:
            mission_id: Mission identifier.

        Returns:
            Mission data dict or None if not found.
        """
        if not self.is_connected:
            return None

        try:
            key = self._key("mission", mission_id)
            data = await self._client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            self.logger.error(f"Failed to get mission {mission_id}: {e}")
            return None

    async def get_recent_missions(self, limit: int = 20) -> list[dict]:
        """Get recent missions.

        Args:
            limit: Maximum number of missions to return.

        Returns:
            List of mission dicts (newest first).
        """
        if not self.is_connected:
            return []

        try:
            mission_ids = await self._client.zrevrange(
                self._key("missions"), 0, limit - 1
            )

            missions = []
            for mission_id in mission_ids:
                mission = await self.get_mission(mission_id)
                if mission:
                    missions.append(mission)

            return missions

        except Exception as e:
            self.logger.error(f"Failed to get recent missions: {e}")
            return []

    # ==================== System State Operations ====================

    async def set_state(self, key: str, value: object) -> bool:
        """Set a system state value.

        Args:
            key: State key name.
            value: Value to store (will be JSON serialized).

        Returns:
            True if successful.
        """
        if not self.is_connected:
            return False

        try:
            state_key = self._key("state", key)
            data = json.dumps(value) if not isinstance(value, str) else value
            await self._client.set(state_key, data)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set state {key}: {e}")
            return False

    async def get_state(self, key: str, default: object = None) -> object:
        """Get a system state value.

        Args:
            key: State key name.
            default: Default value if key not found.

        Returns:
            Stored value or default.
        """
        if not self.is_connected:
            return default

        try:
            state_key = self._key("state", key)
            data = await self._client.get(state_key)
            if data is None:
                return default
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        except Exception as e:
            self.logger.error(f"Failed to get state {key}: {e}")
            return default

    # ==================== Utility Operations ====================

    async def clear_all(self) -> bool:
        """Clear all AegisAV data from Redis.

        WARNING: This deletes all stored data!

        Returns:
            True if successful
        """
        if not self.is_connected:
            return False

        try:
            # Find all keys with our prefix
            pattern = f"{self.config.prefix}:*"
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break

            self.logger.warning(f"Cleared {deleted} keys from Redis")
            return True

        except Exception as e:
            self.logger.error(f"Failed to clear Redis data: {e}")
            return False

    async def get_stats(self) -> dict:
        """Get storage statistics.

        Returns:
            Dictionary with connection status and counts.
        """
        if not self.is_connected:
            return {"connected": False}

        try:
            info = await self._client.info()

            # Count our keys
            asset_count = await self._client.scard(self._key("assets"))
            anomaly_count = await self._client.zcard(self._key("anomalies"))
            mission_count = await self._client.zcard(self._key("missions"))

            return {
                "connected": True,
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "asset_count": asset_count,
                "anomaly_count": anomaly_count,
                "mission_count": mission_count,
            }

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"connected": True, "error": str(e)}


# In-memory fallback for when Redis is not available
class InMemoryStore:
    """In-memory fallback store when Redis is not available.

    Provides the same interface as RedisStore but stores data in memory.
    Data is lost when the process exits.
    """

    def __init__(self, config: RedisConfig | None = None) -> None:
        """Initialize the InMemoryStore.

        Args:
            config: Configuration (used for compatibility, not required).
        """
        self.config = config or RedisConfig()
        self._assets: dict[str, dict] = {}
        self._anomalies: dict[str, dict] = {}
        self._detections: dict[str, list[dict]] = {}
        self._telemetry: dict[str, list[dict]] = {}
        self._missions: dict[str, dict] = {}
        self._state: dict[str, Any] = {}
        self._connected = False
        self.logger = logger

    async def connect(self) -> bool:
        """Simulate connection.

        Returns:
            True always, as in-memory is always available.
        """
        self.logger.warning(
            "Using in-memory store - data will not persist across restarts"
        )
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Simulate disconnection.

        Resets the connected state.
        """
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if the store is connected."""
        return self._connected

    async def set_asset(self, asset_id: str, asset: BaseModel | dict) -> bool:
        """Store an asset in memory.

        Args:
            asset_id: Unique asset identifier.
            asset: Asset data.

        Returns:
            True if successful.
        """
        data = asset.model_dump() if isinstance(asset, BaseModel) else asset
        self._assets[asset_id] = data
        return True

    async def get_asset(self, asset_id: str) -> dict | None:
        """Get an asset by ID.

        Args:
            asset_id: Asset identifier.

        Returns:
            Asset data dict or None if not found.
        """
        return self._assets.get(asset_id)

    async def get_all_assets(self) -> list[dict]:
        """Get all stored assets.

        Returns:
            List of asset data dicts.
        """
        return list(self._assets.values())

    async def delete_asset(self, asset_id: str) -> bool:
        """Delete an asset.

        Args:
            asset_id: Asset identifier to delete.

        Returns:
            True if successful.
        """
        self._assets.pop(asset_id, None)
        return True

    async def add_anomaly(self, anomaly_id: str, anomaly: BaseModel | dict) -> bool:
        """Store an anomaly in memory.

        Args:
            anomaly_id: Unique anomaly identifier.
            anomaly: Anomaly data.

        Returns:
            True if successful.
        """
        data = anomaly.model_dump() if isinstance(anomaly, BaseModel) else anomaly
        self._anomalies[anomaly_id] = data
        return True

    async def get_anomaly(self, anomaly_id: str) -> dict | None:
        """Get an anomaly by ID.

        Args:
            anomaly_id: Anomaly identifier.

        Returns:
            Anomaly data dict or None if not found.
        """
        return self._anomalies.get(anomaly_id)

    async def get_recent_anomalies(self, limit: int = 100) -> list[dict]:
        """Get recent anomalies.

        Args:
            limit: Maximum number to return.

        Returns:
            List of anomaly dicts.
        """
        return list(self._anomalies.values())[:limit]

    async def get_anomalies_for_asset(self, asset_id: str) -> list[dict]:
        """Get all anomalies for a specific asset.

        Args:
            asset_id: Asset identifier.

        Returns:
            List of anomaly dicts for the asset.
        """
        return [a for a in self._anomalies.values() if a.get("asset_id") == asset_id]

    async def resolve_anomaly(self, anomaly_id: str) -> bool:
        """Mark an anomaly as resolved.

        Args:
            anomaly_id: Anomaly identifier to resolve.

        Returns:
            True if found and resolved, False otherwise.
        """
        if anomaly_id in self._anomalies:
            self._anomalies[anomaly_id]["resolved"] = True
            return True
        return False

    async def add_detection(
        self, asset_id: str, detection: BaseModel | dict, _timestamp: datetime | None = None
    ) -> bool:
        """Add detection. Note: timestamp is ignored in memory store."""
        data = detection.model_dump() if isinstance(detection, BaseModel) else detection
        if asset_id not in self._detections:
            self._detections[asset_id] = []
        self._detections[asset_id].append(data)
        return True

    async def get_detections_for_asset(
        self, asset_id: str, limit: int = 50, _since: datetime | None = None
    ) -> list[dict]:
        """Get detections. Note: since filter is ignored in memory store."""
        return self._detections.get(asset_id, [])[:limit]

    async def add_telemetry(self, vehicle_id: str, telemetry: dict) -> bool:
        """Store vehicle telemetry.

        Args:
            vehicle_id: Vehicle identifier.
            telemetry: Telemetry data dict.

        Returns:
            True if successful.
        """
        if vehicle_id not in self._telemetry:
            self._telemetry[vehicle_id] = []
        telemetry["timestamp"] = datetime.now().isoformat()
        self._telemetry[vehicle_id].append(telemetry)
        # Keep only last 1000 entries
        self._telemetry[vehicle_id] = self._telemetry[vehicle_id][-1000:]
        return True

    async def get_telemetry(
        self, vehicle_id: str, _since: datetime | None = None, limit: int = 100
    ) -> list[dict]:
        """Get telemetry. Note: since filter is ignored in memory store."""
        return self._telemetry.get(vehicle_id, [])[-limit:]

    async def get_latest_telemetry(self, vehicle_id: str) -> dict | None:
        """Get the most recent telemetry for a vehicle.

        Args:
            vehicle_id: Vehicle identifier.

        Returns:
            Latest telemetry dict or None if not available.
        """
        entries = self._telemetry.get(vehicle_id, [])
        return entries[-1] if entries else None

    async def save_mission(self, mission_id: str, mission: dict) -> bool:
        """Save a mission record.

        Args:
            mission_id: Mission identifier.
            mission: Mission data dict.

        Returns:
            True if successful.
        """
        self._missions[mission_id] = mission
        return True

    async def get_mission(self, mission_id: str) -> dict | None:
        """Get a mission by ID.

        Args:
            mission_id: Mission identifier.

        Returns:
            Mission data dict or None if not found.
        """
        return self._missions.get(mission_id)

    async def get_recent_missions(self, limit: int = 20) -> list[dict]:
        """Get recent missions.

        Args:
            limit: Maximum number to return.

        Returns:
            List of mission dicts.
        """
        return list(self._missions.values())[:limit]

    async def set_state(self, key: str, value: object) -> bool:
        """Set a system state value.

        Args:
            key: State key name.
            value: Value to store.

        Returns:
            True if successful.
        """
        self._state[key] = value
        return True

    async def get_state(self, key: str, default: object = None) -> object:
        """Get a system state value.

        Args:
            key: State key name.
            default: Default value if key not found.

        Returns:
            Stored value or default.
        """
        return self._state.get(key, default)

    async def clear_all(self) -> bool:
        """Clear all stored data.

        Returns:
            True if successful.
        """
        self._assets.clear()
        self._anomalies.clear()
        self._detections.clear()
        self._telemetry.clear()
        self._missions.clear()
        self._state.clear()
        return True

    async def get_stats(self) -> dict:
        """Get storage statistics.

        Returns:
            Dictionary with connection status and counts.
        """
        return {
            "connected": True,
            "type": "in_memory",
            "asset_count": len(self._assets),
            "anomaly_count": len(self._anomalies),
            "mission_count": len(self._missions),
        }


def create_store(config: RedisConfig | None = None) -> RedisStore | InMemoryStore:
    """Factory function to create appropriate store.

    Uses Redis if available, falls back to in-memory.

    Args:
        config: Redis configuration

    Returns:
        RedisStore or InMemoryStore instance
    """
    if REDIS_AVAILABLE:
        return RedisStore(config)
    else:
        logger.warning(
            "redis package not available, using in-memory store. "
            "Install with: pip install redis"
        )
        return InMemoryStore(config)
