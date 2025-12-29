"""
Tests for persistence layer (InMemoryStore and RedisStore with mocks).
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from agent.server.persistence import (
    REDIS_AVAILABLE,
    InMemoryStore,
    RedisConfig,
    RedisStore,
    create_store,
)


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""
    id: str
    name: str
    value: float


class SampleAnomalyModel(BaseModel):
    """Sample Pydantic model for anomaly testing."""
    id: str
    asset_id: str
    severity: float


class TestRedisConfig:
    """Test RedisConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.prefix == "aegis"
        assert config.telemetry_ttl == 3600
        assert config.detection_ttl == 86400 * 7
        assert config.asset_ttl == 0
        assert config.anomaly_ttl == 86400 * 30
        assert config.mission_ttl == 86400 * 90

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RedisConfig(
            host="redis.example.com",
            port=6380,
            password="secret",
            prefix="custom",
        )
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.password == "secret"
        assert config.prefix == "custom"


class TestInMemoryStore:
    """Test InMemoryStore implementation."""

    @pytest.fixture
    async def store(self):
        """Create and connect an in-memory store."""
        store = InMemoryStore()
        await store.connect()
        yield store
        await store.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_connect_disconnect(self):
        """Test connect and disconnect."""
        store = InMemoryStore()
        assert store.is_connected is False

        result = await store.connect()
        assert result is True
        assert store.is_connected is True

        await store.disconnect()
        assert store.is_connected is False

    @pytest.mark.asyncio
    async def test_asset_operations(self, store):
        """Test asset CRUD operations."""
        # Set asset as dict
        asset_data = {"id": "asset_001", "name": "Solar Panel 1", "lat": 37.7749}
        result = await store.set_asset("asset_001", asset_data)
        assert result is True

        # Get asset
        asset = await store.get_asset("asset_001")
        assert asset is not None
        assert asset["id"] == "asset_001"
        assert asset["name"] == "Solar Panel 1"

        # Get non-existent asset
        missing = await store.get_asset("nonexistent")
        assert missing is None

        # Get all assets
        all_assets = await store.get_all_assets()
        assert len(all_assets) == 1
        assert all_assets[0]["id"] == "asset_001"

        # Delete asset
        result = await store.delete_asset("asset_001")
        assert result is True
        assert await store.get_asset("asset_001") is None

    @pytest.mark.asyncio
    async def test_asset_with_pydantic_model(self, store):
        """Test asset operations with Pydantic model."""
        model = SampleModel(id="model_001", name="Test Model", value=42.5)
        result = await store.set_asset("model_001", model)
        assert result is True

        asset = await store.get_asset("model_001")
        assert asset is not None
        assert asset["id"] == "model_001"
        assert asset["value"] == 42.5

    @pytest.mark.asyncio
    async def test_anomaly_operations(self, store):
        """Test anomaly operations."""
        # Add anomaly
        anomaly_data = {
            "id": "anomaly_001",
            "asset_id": "asset_001",
            "severity": 0.8,
            "description": "Crack detected",
        }
        result = await store.add_anomaly("anomaly_001", anomaly_data)
        assert result is True

        # Get anomaly
        anomaly = await store.get_anomaly("anomaly_001")
        assert anomaly is not None
        assert anomaly["severity"] == 0.8

        # Get non-existent anomaly
        missing = await store.get_anomaly("nonexistent")
        assert missing is None

        # Get recent anomalies
        recent = await store.get_recent_anomalies(limit=10)
        assert len(recent) == 1
        assert recent[0]["id"] == "anomaly_001"

        # Get anomalies for asset
        asset_anomalies = await store.get_anomalies_for_asset("asset_001")
        assert len(asset_anomalies) == 1

        # Resolve anomaly
        result = await store.resolve_anomaly("anomaly_001")
        assert result is True
        resolved = await store.get_anomaly("anomaly_001")
        assert resolved["resolved"] is True

        # Resolve non-existent anomaly
        result = await store.resolve_anomaly("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_detection_operations(self, store):
        """Test detection operations."""
        # Add detection
        detection_data = {
            "id": "detection_001",
            "class": "crack",
            "confidence": 0.95,
            "bbox": [100, 100, 200, 200],
        }
        result = await store.add_detection("asset_001", detection_data)
        assert result is True

        # Add more detections
        await store.add_detection("asset_001", {"id": "detection_002", "class": "rust"})
        await store.add_detection("asset_002", {"id": "detection_003", "class": "debris"})

        # Get detections for asset
        detections = await store.get_detections_for_asset("asset_001")
        assert len(detections) == 2
        assert detections[0]["id"] == "detection_001"

        # Limit detections
        limited = await store.get_detections_for_asset("asset_001", limit=1)
        assert len(limited) == 1

        # Get detections for different asset
        other = await store.get_detections_for_asset("asset_002")
        assert len(other) == 1
        assert other[0]["class"] == "debris"

    @pytest.mark.asyncio
    async def test_telemetry_operations(self, store):
        """Test telemetry operations."""
        # Add telemetry
        telemetry = {"battery": 80, "altitude": 100, "speed": 5.0}
        result = await store.add_telemetry("vehicle_001", telemetry)
        assert result is True

        # Get telemetry
        entries = await store.get_telemetry("vehicle_001")
        assert len(entries) == 1
        assert entries[0]["battery"] == 80
        assert "timestamp" in entries[0]

        # Get latest telemetry
        latest = await store.get_latest_telemetry("vehicle_001")
        assert latest is not None
        assert latest["battery"] == 80

        # Add more telemetry
        for i in range(5):
            await store.add_telemetry("vehicle_001", {"battery": 80 - i})

        # Test limit
        limited = await store.get_telemetry("vehicle_001", limit=3)
        assert len(limited) == 3

        # Get telemetry for non-existent vehicle
        empty = await store.get_telemetry("nonexistent")
        assert empty == []

        # Get latest for non-existent vehicle
        missing = await store.get_latest_telemetry("nonexistent")
        assert missing is None

    @pytest.mark.asyncio
    async def test_mission_operations(self, store):
        """Test mission operations."""
        # Save mission
        mission = {
            "id": "mission_001",
            "name": "Inspection Run 1",
            "status": "completed",
            "duration": 3600,
        }
        result = await store.save_mission("mission_001", mission)
        assert result is True

        # Get mission
        retrieved = await store.get_mission("mission_001")
        assert retrieved is not None
        assert retrieved["name"] == "Inspection Run 1"

        # Get non-existent mission
        missing = await store.get_mission("nonexistent")
        assert missing is None

        # Save more missions
        await store.save_mission("mission_002", {"id": "mission_002", "name": "Run 2"})
        await store.save_mission("mission_003", {"id": "mission_003", "name": "Run 3"})

        # Get recent missions
        recent = await store.get_recent_missions(limit=2)
        assert len(recent) == 2

    @pytest.mark.asyncio
    async def test_state_operations(self, store):
        """Test state key-value operations."""
        # Set state
        result = await store.set_state("last_decision", "INSPECT")
        assert result is True

        # Get state
        value = await store.get_state("last_decision")
        assert value == "INSPECT"

        # Get with default
        default_value = await store.get_state("nonexistent", default="default")
        assert default_value == "default"

        # Get without default
        none_value = await store.get_state("nonexistent")
        assert none_value is None

    @pytest.mark.asyncio
    async def test_clear_all(self, store):
        """Test clearing all data."""
        # Add some data
        await store.set_asset("asset_001", {"id": "asset_001"})
        await store.add_anomaly("anomaly_001", {"id": "anomaly_001"})
        await store.save_mission("mission_001", {"id": "mission_001"})
        await store.set_state("key", "value")

        # Clear all
        result = await store.clear_all()
        assert result is True

        # Verify cleared
        assert await store.get_asset("asset_001") is None
        assert await store.get_anomaly("anomaly_001") is None
        assert await store.get_mission("mission_001") is None
        assert await store.get_state("key") is None

    @pytest.mark.asyncio
    async def test_get_stats(self, store):
        """Test getting stats."""
        # Add some data
        await store.set_asset("asset_001", {"id": "asset_001"})
        await store.set_asset("asset_002", {"id": "asset_002"})
        await store.add_anomaly("anomaly_001", {"id": "anomaly_001"})
        await store.save_mission("mission_001", {"id": "mission_001"})

        stats = await store.get_stats()
        assert stats["connected"] is True
        assert stats["type"] == "in_memory"
        assert stats["asset_count"] == 2
        assert stats["anomaly_count"] == 1
        assert stats["mission_count"] == 1


class TestCreateStore:
    """Test create_store factory function."""

    @pytest.mark.allow_error_logs
    def test_create_store_returns_store(self):
        """Test that create_store returns a store instance."""
        store = create_store()
        # Should return either RedisStore or InMemoryStore
        assert hasattr(store, "connect")
        assert hasattr(store, "disconnect")
        assert hasattr(store, "set_asset")
        assert hasattr(store, "get_asset")

    @pytest.mark.allow_error_logs
    def test_create_store_with_config(self):
        """Test create_store with custom config."""
        config = RedisConfig(host="custom", port=6380)
        store = create_store(config)
        assert store.config.host == "custom"
        assert store.config.port == 6380


class TestRedisConfigExtended:
    """Extended tests for RedisConfig model."""

    def test_socket_timeout_values(self):
        """Test socket timeout configuration values."""
        config = RedisConfig(
            socket_timeout=10.0,
            socket_connect_timeout=15.0,
        )
        assert config.socket_timeout == 10.0
        assert config.socket_connect_timeout == 15.0

    def test_decode_responses_default(self):
        """Test decode_responses default is True."""
        config = RedisConfig()
        assert config.decode_responses is True

    def test_all_ttl_values(self):
        """Test all TTL values can be customized."""
        config = RedisConfig(
            telemetry_ttl=7200,
            detection_ttl=86400 * 14,
            asset_ttl=86400,
            anomaly_ttl=86400 * 60,
            mission_ttl=86400 * 180,
        )
        assert config.telemetry_ttl == 7200
        assert config.detection_ttl == 86400 * 14
        assert config.asset_ttl == 86400
        assert config.anomaly_ttl == 86400 * 60
        assert config.mission_ttl == 86400 * 180


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis package not installed")
class TestRedisStore:
    """Test RedisStore implementation with mocks."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.ping = AsyncMock(return_value=True)
        mock.aclose = AsyncMock()
        mock.set = AsyncMock()
        mock.setex = AsyncMock()
        mock.get = AsyncMock(return_value=None)
        mock.delete = AsyncMock()
        mock.sadd = AsyncMock()
        mock.srem = AsyncMock()
        mock.smembers = AsyncMock(return_value=set())
        mock.scard = AsyncMock(return_value=0)
        mock.zadd = AsyncMock()
        mock.zrevrange = AsyncMock(return_value=[])
        mock.zrevrangebyscore = AsyncMock(return_value=[])
        mock.zremrangebyscore = AsyncMock()
        mock.zcard = AsyncMock(return_value=0)
        mock.scan = AsyncMock(return_value=(0, []))
        mock.info = AsyncMock(return_value={
            "redis_version": "7.0.0",
            "used_memory_human": "1M",
            "connected_clients": 1,
        })
        return mock

    @pytest.fixture
    def redis_store(self, mock_redis_client):
        """Create a RedisStore with mocked client."""
        store = RedisStore()
        store._client = mock_redis_client
        store._connected = True
        return store

    def test_init_with_default_config(self):
        """Test RedisStore initialization with default config."""
        store = RedisStore()
        assert store.config.host == "localhost"
        assert store.config.port == 6379
        assert store._connected is False
        assert store._client is None

    def test_init_with_custom_config(self):
        """Test RedisStore initialization with custom config."""
        config = RedisConfig(host="redis.local", port=6380, password="secret")
        store = RedisStore(config)
        assert store.config.host == "redis.local"
        assert store.config.port == 6380
        assert store.config.password == "secret"

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_redis_client):
        """Test successful connection."""
        with patch("agent.server.persistence.redis.Redis", return_value=mock_redis_client):
            store = RedisStore()
            result = await store.connect()
            assert result is True
            assert store.is_connected is True

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_connect_failure(self, mock_redis_client):
        """Test connection failure."""
        mock_redis_client.ping.side_effect = Exception("Connection refused")
        with patch("agent.server.persistence.redis.Redis", return_value=mock_redis_client):
            store = RedisStore()
            result = await store.connect()
            assert result is False
            assert store.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, redis_store, mock_redis_client):
        """Test disconnection."""
        await redis_store.disconnect()
        mock_redis_client.aclose.assert_called_once()
        assert redis_store.is_connected is False
        assert redis_store._client is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """Test disconnect when already disconnected."""
        store = RedisStore()
        await store.disconnect()  # Should not raise
        assert store.is_connected is False

    def test_is_connected_property(self, redis_store):
        """Test is_connected property."""
        assert redis_store.is_connected is True
        redis_store._connected = False
        assert redis_store.is_connected is False
        redis_store._connected = True
        redis_store._client = None
        assert redis_store.is_connected is False

    def test_key_method(self, redis_store):
        """Test _key method for building Redis keys."""
        key = redis_store._key("asset", "asset_001")
        assert key == "aegis:asset:asset_001"

        key = redis_store._key("anomaly", "asset", "asset_001")
        assert key == "aegis:anomaly:asset:asset_001"

    # Asset Operations Tests
    @pytest.mark.asyncio
    async def test_set_asset_dict(self, redis_store, mock_redis_client):
        """Test setting an asset from a dictionary."""
        asset_data = {"id": "asset_001", "name": "Solar Panel 1"}
        result = await redis_store.set_asset("asset_001", asset_data)
        assert result is True
        mock_redis_client.set.assert_called_once()
        mock_redis_client.sadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_asset_with_ttl(self, redis_store, mock_redis_client):
        """Test setting an asset with TTL."""
        redis_store.config.asset_ttl = 3600
        asset_data = {"id": "asset_001", "name": "Solar Panel 1"}
        result = await redis_store.set_asset("asset_001", asset_data)
        assert result is True
        mock_redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_asset_pydantic(self, redis_store, mock_redis_client):
        """Test setting an asset from a Pydantic model."""
        model = SampleModel(id="model_001", name="Test Model", value=42.5)
        result = await redis_store.set_asset("model_001", model)
        assert result is True

    @pytest.mark.asyncio
    async def test_set_asset_not_connected(self, redis_store):
        """Test set_asset when not connected."""
        redis_store._connected = False
        result = await redis_store.set_asset("asset_001", {"id": "asset_001"})
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_set_asset_exception(self, redis_store, mock_redis_client):
        """Test set_asset handling exception."""
        mock_redis_client.set.side_effect = Exception("Redis error")
        result = await redis_store.set_asset("asset_001", {"id": "asset_001"})
        assert result is False

    @pytest.mark.asyncio
    async def test_get_asset_success(self, redis_store, mock_redis_client):
        """Test getting an asset successfully."""
        mock_redis_client.get.return_value = '{"id": "asset_001", "name": "Solar Panel 1"}'
        result = await redis_store.get_asset("asset_001")
        assert result is not None
        assert result["id"] == "asset_001"
        assert result["name"] == "Solar Panel 1"

    @pytest.mark.asyncio
    async def test_get_asset_not_found(self, redis_store, mock_redis_client):
        """Test getting a non-existent asset."""
        mock_redis_client.get.return_value = None
        result = await redis_store.get_asset("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_asset_not_connected(self, redis_store):
        """Test get_asset when not connected."""
        redis_store._connected = False
        result = await redis_store.get_asset("asset_001")
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_get_asset_exception(self, redis_store, mock_redis_client):
        """Test get_asset handling exception."""
        mock_redis_client.get.side_effect = Exception("Redis error")
        result = await redis_store.get_asset("asset_001")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_assets(self, redis_store, mock_redis_client):
        """Test getting all assets."""
        mock_redis_client.smembers.return_value = {"asset_001", "asset_002"}
        mock_redis_client.get.side_effect = [
            '{"id": "asset_001", "name": "Panel 1"}',
            '{"id": "asset_002", "name": "Panel 2"}',
        ]
        result = await redis_store.get_all_assets()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_all_assets_not_connected(self, redis_store):
        """Test get_all_assets when not connected."""
        redis_store._connected = False
        result = await redis_store.get_all_assets()
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_get_all_assets_exception(self, redis_store, mock_redis_client):
        """Test get_all_assets handling exception."""
        mock_redis_client.smembers.side_effect = Exception("Redis error")
        result = await redis_store.get_all_assets()
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_asset(self, redis_store, mock_redis_client):
        """Test deleting an asset."""
        result = await redis_store.delete_asset("asset_001")
        assert result is True
        mock_redis_client.delete.assert_called_once()
        mock_redis_client.srem.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_asset_not_connected(self, redis_store):
        """Test delete_asset when not connected."""
        redis_store._connected = False
        result = await redis_store.delete_asset("asset_001")
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_delete_asset_exception(self, redis_store, mock_redis_client):
        """Test delete_asset handling exception."""
        mock_redis_client.delete.side_effect = Exception("Redis error")
        result = await redis_store.delete_asset("asset_001")
        assert result is False

    # Anomaly Operations Tests
    @pytest.mark.asyncio
    async def test_add_anomaly_dict(self, redis_store, mock_redis_client):
        """Test adding an anomaly from a dictionary."""
        anomaly_data = {"id": "anomaly_001", "asset_id": "asset_001", "severity": 0.8}
        result = await redis_store.add_anomaly("anomaly_001", anomaly_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_add_anomaly_pydantic(self, redis_store, mock_redis_client):
        """Test adding an anomaly from a Pydantic model."""
        model = SampleAnomalyModel(id="anomaly_001", asset_id="asset_001", severity=0.8)
        result = await redis_store.add_anomaly("anomaly_001", model)
        assert result is True
        # Should also add to asset index
        assert mock_redis_client.sadd.call_count >= 1

    @pytest.mark.asyncio
    async def test_add_anomaly_with_ttl(self, redis_store, mock_redis_client):
        """Test adding an anomaly with TTL."""
        redis_store.config.anomaly_ttl = 3600
        anomaly_data = {"id": "anomaly_001", "severity": 0.8}
        result = await redis_store.add_anomaly("anomaly_001", anomaly_data)
        assert result is True
        mock_redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_anomaly_no_ttl(self, redis_store, mock_redis_client):
        """Test adding an anomaly without TTL."""
        redis_store.config.anomaly_ttl = 0
        anomaly_data = {"id": "anomaly_001", "severity": 0.8}
        result = await redis_store.add_anomaly("anomaly_001", anomaly_data)
        assert result is True
        mock_redis_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_anomaly_not_connected(self, redis_store):
        """Test add_anomaly when not connected."""
        redis_store._connected = False
        result = await redis_store.add_anomaly("anomaly_001", {"id": "anomaly_001"})
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_add_anomaly_exception(self, redis_store, mock_redis_client):
        """Test add_anomaly handling exception."""
        mock_redis_client.setex.side_effect = Exception("Redis error")
        result = await redis_store.add_anomaly("anomaly_001", {"id": "anomaly_001"})
        assert result is False

    @pytest.mark.asyncio
    async def test_get_anomaly(self, redis_store, mock_redis_client):
        """Test getting an anomaly."""
        mock_redis_client.get.return_value = '{"id": "anomaly_001", "severity": 0.8}'
        result = await redis_store.get_anomaly("anomaly_001")
        assert result is not None
        assert result["severity"] == 0.8

    @pytest.mark.asyncio
    async def test_get_anomaly_not_connected(self, redis_store):
        """Test get_anomaly when not connected."""
        redis_store._connected = False
        result = await redis_store.get_anomaly("anomaly_001")
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_get_anomaly_exception(self, redis_store, mock_redis_client):
        """Test get_anomaly handling exception."""
        mock_redis_client.get.side_effect = Exception("Redis error")
        result = await redis_store.get_anomaly("anomaly_001")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_recent_anomalies(self, redis_store, mock_redis_client):
        """Test getting recent anomalies."""
        mock_redis_client.zrevrange.return_value = ["anomaly_001", "anomaly_002"]
        mock_redis_client.get.side_effect = [
            '{"id": "anomaly_001", "severity": 0.8}',
            '{"id": "anomaly_002", "severity": 0.5}',
        ]
        result = await redis_store.get_recent_anomalies(limit=10)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_recent_anomalies_not_connected(self, redis_store):
        """Test get_recent_anomalies when not connected."""
        redis_store._connected = False
        result = await redis_store.get_recent_anomalies()
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_get_recent_anomalies_exception(self, redis_store, mock_redis_client):
        """Test get_recent_anomalies handling exception."""
        mock_redis_client.zrevrange.side_effect = Exception("Redis error")
        result = await redis_store.get_recent_anomalies()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_anomalies_for_asset(self, redis_store, mock_redis_client):
        """Test getting anomalies for a specific asset."""
        mock_redis_client.smembers.return_value = {"anomaly_001"}
        mock_redis_client.get.return_value = '{"id": "anomaly_001", "asset_id": "asset_001"}'
        result = await redis_store.get_anomalies_for_asset("asset_001")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_anomalies_for_asset_not_connected(self, redis_store):
        """Test get_anomalies_for_asset when not connected."""
        redis_store._connected = False
        result = await redis_store.get_anomalies_for_asset("asset_001")
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_get_anomalies_for_asset_exception(self, redis_store, mock_redis_client):
        """Test get_anomalies_for_asset handling exception."""
        mock_redis_client.smembers.side_effect = Exception("Redis error")
        result = await redis_store.get_anomalies_for_asset("asset_001")
        assert result == []

    @pytest.mark.asyncio
    async def test_resolve_anomaly_success(self, redis_store, mock_redis_client):
        """Test resolving an anomaly successfully."""
        mock_redis_client.get.return_value = '{"id": "anomaly_001", "resolved": false}'
        result = await redis_store.resolve_anomaly("anomaly_001")
        assert result is True

    @pytest.mark.asyncio
    async def test_resolve_anomaly_not_found(self, redis_store, mock_redis_client):
        """Test resolving a non-existent anomaly."""
        mock_redis_client.get.return_value = None
        result = await redis_store.resolve_anomaly("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_resolve_anomaly_not_connected(self, redis_store):
        """Test resolve_anomaly when not connected."""
        redis_store._connected = False
        result = await redis_store.resolve_anomaly("anomaly_001")
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_resolve_anomaly_exception(self, redis_store, mock_redis_client):
        """Test resolve_anomaly handling exception."""
        mock_redis_client.get.side_effect = Exception("Redis error")
        result = await redis_store.resolve_anomaly("anomaly_001")
        assert result is False

    # Detection Operations Tests
    @pytest.mark.asyncio
    async def test_add_detection(self, redis_store, mock_redis_client):
        """Test adding a detection."""
        detection = {"class": "crack", "confidence": 0.95}
        result = await redis_store.add_detection("asset_001", detection)
        assert result is True

    @pytest.mark.asyncio
    async def test_add_detection_with_timestamp(self, redis_store, mock_redis_client):
        """Test adding a detection with explicit timestamp."""
        detection = {"class": "crack", "confidence": 0.95}
        ts = datetime.now()
        result = await redis_store.add_detection("asset_001", detection, timestamp=ts)
        assert result is True

    @pytest.mark.asyncio
    async def test_add_detection_with_ttl(self, redis_store, mock_redis_client):
        """Test adding a detection with TTL."""
        redis_store.config.detection_ttl = 3600
        detection = {"class": "crack", "confidence": 0.95}
        result = await redis_store.add_detection("asset_001", detection)
        assert result is True
        mock_redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_detection_no_ttl(self, redis_store, mock_redis_client):
        """Test adding a detection without TTL."""
        redis_store.config.detection_ttl = 0
        detection = {"class": "crack", "confidence": 0.95}
        result = await redis_store.add_detection("asset_001", detection)
        assert result is True
        mock_redis_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_detection_pydantic(self, redis_store, mock_redis_client):
        """Test adding a detection from Pydantic model."""
        model = SampleModel(id="det_001", name="Crack", value=0.95)
        result = await redis_store.add_detection("asset_001", model)
        assert result is True

    @pytest.mark.asyncio
    async def test_add_detection_not_connected(self, redis_store):
        """Test add_detection when not connected."""
        redis_store._connected = False
        result = await redis_store.add_detection("asset_001", {"class": "crack"})
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_add_detection_exception(self, redis_store, mock_redis_client):
        """Test add_detection handling exception."""
        mock_redis_client.setex.side_effect = Exception("Redis error")
        result = await redis_store.add_detection("asset_001", {"class": "crack"})
        assert result is False

    @pytest.mark.asyncio
    async def test_get_detections_for_asset(self, redis_store, mock_redis_client):
        """Test getting detections for an asset."""
        mock_redis_client.zrevrangebyscore.return_value = ["asset_001:12345.0"]
        mock_redis_client.get.return_value = '{"class": "crack", "confidence": 0.95}'
        result = await redis_store.get_detections_for_asset("asset_001", limit=50)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_detections_for_asset_with_since(self, redis_store, mock_redis_client):
        """Test getting detections with since filter."""
        since = datetime.now() - timedelta(hours=1)
        mock_redis_client.zrevrangebyscore.return_value = []
        result = await redis_store.get_detections_for_asset("asset_001", since=since)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_detections_for_asset_not_connected(self, redis_store):
        """Test get_detections_for_asset when not connected."""
        redis_store._connected = False
        result = await redis_store.get_detections_for_asset("asset_001")
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_get_detections_for_asset_exception(self, redis_store, mock_redis_client):
        """Test get_detections_for_asset handling exception."""
        mock_redis_client.zrevrangebyscore.side_effect = Exception("Redis error")
        result = await redis_store.get_detections_for_asset("asset_001")
        assert result == []

    # Telemetry Operations Tests
    @pytest.mark.asyncio
    async def test_add_telemetry(self, redis_store, mock_redis_client):
        """Test adding telemetry."""
        telemetry = {"battery": 80, "altitude": 100}
        result = await redis_store.add_telemetry("vehicle_001", telemetry)
        assert result is True
        mock_redis_client.zadd.assert_called_once()
        mock_redis_client.zremrangebyscore.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_telemetry_not_connected(self, redis_store):
        """Test add_telemetry when not connected."""
        redis_store._connected = False
        result = await redis_store.add_telemetry("vehicle_001", {"battery": 80})
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_add_telemetry_exception(self, redis_store, mock_redis_client):
        """Test add_telemetry handling exception."""
        mock_redis_client.zadd.side_effect = Exception("Redis error")
        result = await redis_store.add_telemetry("vehicle_001", {"battery": 80})
        assert result is False

    @pytest.mark.asyncio
    async def test_get_telemetry(self, redis_store, mock_redis_client):
        """Test getting telemetry."""
        mock_redis_client.zrevrangebyscore.return_value = ['{"battery": 80, "timestamp": "2023-01-01T00:00:00"}']
        result = await redis_store.get_telemetry("vehicle_001", limit=10)
        assert len(result) == 1
        assert result[0]["battery"] == 80

    @pytest.mark.asyncio
    async def test_get_telemetry_with_since(self, redis_store, mock_redis_client):
        """Test getting telemetry with since filter."""
        since = datetime.now() - timedelta(hours=1)
        mock_redis_client.zrevrangebyscore.return_value = []
        result = await redis_store.get_telemetry("vehicle_001", since=since)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_telemetry_not_connected(self, redis_store):
        """Test get_telemetry when not connected."""
        redis_store._connected = False
        result = await redis_store.get_telemetry("vehicle_001")
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_get_telemetry_exception(self, redis_store, mock_redis_client):
        """Test get_telemetry handling exception."""
        mock_redis_client.zrevrangebyscore.side_effect = Exception("Redis error")
        result = await redis_store.get_telemetry("vehicle_001")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_latest_telemetry(self, redis_store, mock_redis_client):
        """Test getting latest telemetry."""
        mock_redis_client.zrevrangebyscore.return_value = ['{"battery": 80}']
        result = await redis_store.get_latest_telemetry("vehicle_001")
        assert result is not None
        assert result["battery"] == 80

    @pytest.mark.asyncio
    async def test_get_latest_telemetry_empty(self, redis_store, mock_redis_client):
        """Test getting latest telemetry when none exists."""
        mock_redis_client.zrevrangebyscore.return_value = []
        result = await redis_store.get_latest_telemetry("vehicle_001")
        assert result is None

    # Mission Operations Tests
    @pytest.mark.asyncio
    async def test_save_mission(self, redis_store, mock_redis_client):
        """Test saving a mission."""
        mission = {"id": "mission_001", "name": "Inspection Run"}
        result = await redis_store.save_mission("mission_001", mission)
        assert result is True

    @pytest.mark.asyncio
    async def test_save_mission_with_ttl(self, redis_store, mock_redis_client):
        """Test saving a mission with TTL."""
        redis_store.config.mission_ttl = 3600
        mission = {"id": "mission_001", "name": "Inspection Run"}
        result = await redis_store.save_mission("mission_001", mission)
        assert result is True
        mock_redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_mission_no_ttl(self, redis_store, mock_redis_client):
        """Test saving a mission without TTL."""
        redis_store.config.mission_ttl = 0
        mission = {"id": "mission_001", "name": "Inspection Run"}
        result = await redis_store.save_mission("mission_001", mission)
        assert result is True
        mock_redis_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_mission_not_connected(self, redis_store):
        """Test save_mission when not connected."""
        redis_store._connected = False
        result = await redis_store.save_mission("mission_001", {"id": "mission_001"})
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_save_mission_exception(self, redis_store, mock_redis_client):
        """Test save_mission handling exception."""
        mock_redis_client.setex.side_effect = Exception("Redis error")
        result = await redis_store.save_mission("mission_001", {"id": "mission_001"})
        assert result is False

    @pytest.mark.asyncio
    async def test_get_mission(self, redis_store, mock_redis_client):
        """Test getting a mission."""
        mock_redis_client.get.return_value = '{"id": "mission_001", "name": "Run 1"}'
        result = await redis_store.get_mission("mission_001")
        assert result is not None
        assert result["name"] == "Run 1"

    @pytest.mark.asyncio
    async def test_get_mission_not_found(self, redis_store, mock_redis_client):
        """Test getting a non-existent mission."""
        mock_redis_client.get.return_value = None
        result = await redis_store.get_mission("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_mission_not_connected(self, redis_store):
        """Test get_mission when not connected."""
        redis_store._connected = False
        result = await redis_store.get_mission("mission_001")
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_get_mission_exception(self, redis_store, mock_redis_client):
        """Test get_mission handling exception."""
        mock_redis_client.get.side_effect = Exception("Redis error")
        result = await redis_store.get_mission("mission_001")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_recent_missions(self, redis_store, mock_redis_client):
        """Test getting recent missions."""
        mock_redis_client.zrevrange.return_value = ["mission_001", "mission_002"]
        mock_redis_client.get.side_effect = [
            '{"id": "mission_001", "name": "Run 1"}',
            '{"id": "mission_002", "name": "Run 2"}',
        ]
        result = await redis_store.get_recent_missions(limit=20)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_recent_missions_not_connected(self, redis_store):
        """Test get_recent_missions when not connected."""
        redis_store._connected = False
        result = await redis_store.get_recent_missions()
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_get_recent_missions_exception(self, redis_store, mock_redis_client):
        """Test get_recent_missions handling exception."""
        mock_redis_client.zrevrange.side_effect = Exception("Redis error")
        result = await redis_store.get_recent_missions()
        assert result == []

    # State Operations Tests
    @pytest.mark.asyncio
    async def test_set_state_string(self, redis_store, mock_redis_client):
        """Test setting a string state value."""
        result = await redis_store.set_state("last_decision", "INSPECT")
        assert result is True
        mock_redis_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_state_dict(self, redis_store, mock_redis_client):
        """Test setting a dict state value."""
        result = await redis_store.set_state("config", {"key": "value"})
        assert result is True

    @pytest.mark.asyncio
    async def test_set_state_not_connected(self, redis_store):
        """Test set_state when not connected."""
        redis_store._connected = False
        result = await redis_store.set_state("key", "value")
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_set_state_exception(self, redis_store, mock_redis_client):
        """Test set_state handling exception."""
        mock_redis_client.set.side_effect = Exception("Redis error")
        result = await redis_store.set_state("key", "value")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_state(self, redis_store, mock_redis_client):
        """Test getting a state value."""
        mock_redis_client.get.return_value = '"INSPECT"'
        result = await redis_store.get_state("last_decision")
        assert result == "INSPECT"

    @pytest.mark.asyncio
    async def test_get_state_dict(self, redis_store, mock_redis_client):
        """Test getting a dict state value."""
        mock_redis_client.get.return_value = '{"key": "value"}'
        result = await redis_store.get_state("config")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_state_with_default(self, redis_store, mock_redis_client):
        """Test getting a state value with default."""
        mock_redis_client.get.return_value = None
        result = await redis_store.get_state("nonexistent", default="default")
        assert result == "default"

    @pytest.mark.asyncio
    async def test_get_state_invalid_json(self, redis_store, mock_redis_client):
        """Test getting a state value with invalid JSON returns raw string."""
        mock_redis_client.get.return_value = "plain_string"
        result = await redis_store.get_state("key")
        assert result == "plain_string"

    @pytest.mark.asyncio
    async def test_get_state_not_connected(self, redis_store):
        """Test get_state when not connected returns default."""
        redis_store._connected = False
        result = await redis_store.get_state("key", default="default")
        assert result == "default"

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_get_state_exception(self, redis_store, mock_redis_client):
        """Test get_state handling exception."""
        mock_redis_client.get.side_effect = Exception("Redis error")
        result = await redis_store.get_state("key", default="default")
        assert result == "default"

    # Utility Operations Tests
    @pytest.mark.asyncio
    async def test_clear_all(self, redis_store, mock_redis_client):
        """Test clearing all data."""
        mock_redis_client.scan.side_effect = [
            (0, ["aegis:key1", "aegis:key2"]),
        ]
        result = await redis_store.clear_all()
        assert result is True
        mock_redis_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_all_multiple_batches(self, redis_store, mock_redis_client):
        """Test clearing all data with multiple scan batches."""
        mock_redis_client.scan.side_effect = [
            (1, ["aegis:key1"]),
            (0, ["aegis:key2"]),
        ]
        result = await redis_store.clear_all()
        assert result is True
        assert mock_redis_client.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_clear_all_no_keys(self, redis_store, mock_redis_client):
        """Test clearing all data when no keys exist."""
        mock_redis_client.scan.return_value = (0, [])
        result = await redis_store.clear_all()
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_all_not_connected(self, redis_store):
        """Test clear_all when not connected."""
        redis_store._connected = False
        result = await redis_store.clear_all()
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_clear_all_exception(self, redis_store, mock_redis_client):
        """Test clear_all handling exception."""
        mock_redis_client.scan.side_effect = Exception("Redis error")
        result = await redis_store.clear_all()
        assert result is False

    @pytest.mark.asyncio
    async def test_get_stats(self, redis_store, mock_redis_client):
        """Test getting stats."""
        mock_redis_client.scard.return_value = 5
        mock_redis_client.zcard.side_effect = [10, 3]
        result = await redis_store.get_stats()
        assert result["connected"] is True
        assert result["redis_version"] == "7.0.0"
        assert result["asset_count"] == 5
        assert result["anomaly_count"] == 10
        assert result["mission_count"] == 3

    @pytest.mark.asyncio
    async def test_get_stats_not_connected(self, redis_store):
        """Test get_stats when not connected."""
        redis_store._connected = False
        result = await redis_store.get_stats()
        assert result == {"connected": False}

    @pytest.mark.asyncio
    @pytest.mark.allow_error_logs
    async def test_get_stats_exception(self, redis_store, mock_redis_client):
        """Test get_stats handling exception."""
        mock_redis_client.info.side_effect = Exception("Redis error")
        result = await redis_store.get_stats()
        assert result["connected"] is True
        assert "error" in result


class TestInMemoryStoreExtended:
    """Extended tests for InMemoryStore."""

    @pytest.fixture
    async def store(self):
        """Create and connect an in-memory store."""
        store = InMemoryStore()
        await store.connect()
        yield store
        await store.disconnect()

    @pytest.mark.asyncio
    async def test_init_with_config(self):
        """Test InMemoryStore initialization with config."""
        config = RedisConfig(prefix="test", host="custom")
        store = InMemoryStore(config)
        assert store.config.prefix == "test"
        assert store.config.host == "custom"

    @pytest.mark.asyncio
    async def test_anomaly_with_pydantic(self, store):
        """Test anomaly operations with Pydantic model."""
        model = SampleAnomalyModel(id="anomaly_001", asset_id="asset_001", severity=0.8)
        result = await store.add_anomaly("anomaly_001", model)
        assert result is True

        anomaly = await store.get_anomaly("anomaly_001")
        assert anomaly is not None
        assert anomaly["asset_id"] == "asset_001"

    @pytest.mark.asyncio
    async def test_detection_with_pydantic(self, store):
        """Test detection operations with Pydantic model."""
        model = SampleModel(id="det_001", name="Crack", value=0.95)
        result = await store.add_detection("asset_001", model)
        assert result is True

        detections = await store.get_detections_for_asset("asset_001")
        assert len(detections) == 1
        assert detections[0]["value"] == 0.95

    @pytest.mark.asyncio
    async def test_telemetry_trimming(self, store):
        """Test telemetry is trimmed to last 1000 entries."""
        # Add more than 1000 entries
        for i in range(1100):
            await store.add_telemetry("vehicle_001", {"battery": i})

        # Should be trimmed to 1000
        entries = await store.get_telemetry("vehicle_001", limit=2000)
        assert len(entries) == 1000

    @pytest.mark.asyncio
    async def test_get_anomalies_for_different_assets(self, store):
        """Test getting anomalies for different assets."""
        await store.add_anomaly("a1", {"id": "a1", "asset_id": "asset_001"})
        await store.add_anomaly("a2", {"id": "a2", "asset_id": "asset_001"})
        await store.add_anomaly("a3", {"id": "a3", "asset_id": "asset_002"})

        asset1_anomalies = await store.get_anomalies_for_asset("asset_001")
        asset2_anomalies = await store.get_anomalies_for_asset("asset_002")

        assert len(asset1_anomalies) == 2
        assert len(asset2_anomalies) == 1

    @pytest.mark.asyncio
    async def test_state_with_complex_value(self, store):
        """Test state operations with complex values."""
        complex_value = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42,
        }
        await store.set_state("complex", complex_value)
        result = await store.get_state("complex")
        assert result == complex_value
