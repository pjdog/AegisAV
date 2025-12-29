"""
Tests for persistence layer (InMemoryStore).
"""

from datetime import datetime

import pytest
from pydantic import BaseModel

from agent.server.persistence import InMemoryStore, RedisConfig, create_store


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""
    id: str
    name: str
    value: float


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
