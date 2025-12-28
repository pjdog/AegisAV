"""
Tests for Simulated Camera

Validates simulated camera functionality including:
- Camera initialization and shutdown
- Image capture
- Probabilistic defect injection
- State management
"""

import tempfile
from pathlib import Path

import pytest

from vision.camera.simulated import DefectConfig, SimulatedCamera, SimulatedCameraConfig
from vision.data_models import CameraStatus, DetectionClass


class TestDefectConfig:
    """Tests for DefectConfig."""

    def test_defect_config_defaults(self):
        """Test default defect probabilities."""
        config = DefectConfig()

        assert config.crack_probability == 0.10
        assert config.corrosion_probability == 0.08
        assert config.severity_min == 0.3
        assert config.severity_max == 0.9

    def test_defect_config_custom(self):
        """Test custom defect probabilities."""
        config = DefectConfig(
            crack_probability=0.20,
            corrosion_probability=0.15,
            severity_min=0.4,
            severity_max=0.8,
        )

        assert config.crack_probability == 0.20
        assert config.corrosion_probability == 0.15
        assert config.severity_min == 0.4
        assert config.severity_max == 0.8

    def test_get_total_probability(self):
        """Test total defect probability calculation."""
        config = DefectConfig(
            crack_probability=0.10,
            corrosion_probability=0.08,
            structural_damage_probability=0.03,
            discoloration_probability=0.05,
            vegetation_probability=0.05,
            damage_probability=0.03,
        )

        total = config.get_total_probability()
        expected = 0.10 + 0.08 + 0.03 + 0.05 + 0.05 + 0.03
        assert total == pytest.approx(expected)

    def test_sample_defect_probabilistic(self):
        """Test probabilistic defect sampling."""
        # High probability config for testing (zero out other probabilities)
        config = DefectConfig(
            crack_probability=0.5,
            corrosion_probability=0.0,
            structural_damage_probability=0.0,
            discoloration_probability=0.0,
            vegetation_probability=0.0,
            damage_probability=0.0,
            severity_min=0.4,
            severity_max=0.8,
        )

        # Sample many times to check distribution
        samples = [config.sample_defect() for _ in range(100)]
        defects = [s for s in samples if s is not None]

        # Should get roughly 50% defects (allow variance)
        assert len(defects) > 30
        assert len(defects) < 70

        # Check severity ranges
        for defect_type, severity in defects:
            assert isinstance(defect_type, DetectionClass)
            assert 0.4 <= severity <= 0.8

    def test_sample_defect_no_defects(self):
        """Test sampling with zero probabilities."""
        config = DefectConfig(
            crack_probability=0.0,
            corrosion_probability=0.0,
            structural_damage_probability=0.0,
            discoloration_probability=0.0,
            vegetation_probability=0.0,
            damage_probability=0.0,
        )

        # Should always return None
        for _ in range(20):
            assert config.sample_defect() is None


class TestSimulatedCameraConfig:
    """Tests for SimulatedCameraConfig."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = SimulatedCameraConfig()

        assert config.resolution == (1920, 1080)
        assert config.capture_format == "RGB"
        assert config.enabled is True
        assert config.save_images is True
        assert isinstance(config.defect_config, DefectConfig)

    def test_config_custom(self):
        """Test custom configuration."""
        defect_config = DefectConfig(crack_probability=0.20)
        config = SimulatedCameraConfig(
            resolution=(3840, 2160),
            capture_format="RGBA",
            enabled=False,
            defect_config=defect_config,
            save_images=False,
        )

        assert config.resolution == (3840, 2160)
        assert config.capture_format == "RGBA"
        assert config.enabled is False
        assert config.save_images is False
        assert config.defect_config.crack_probability == 0.20


@pytest.mark.asyncio
class TestSimulatedCamera:
    """Tests for SimulatedCamera."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def camera_config(self, temp_dir):
        """Create camera config with temp directory."""
        return SimulatedCameraConfig(output_dir=temp_dir, save_images=True)

    @pytest.fixture
    def camera(self, camera_config):
        """Create simulated camera instance."""
        return SimulatedCamera(config=camera_config)

    async def test_camera_initialization(self, camera):
        """Test camera initialization."""
        assert camera._status == CameraStatus.OFFLINE

        success = await camera.initialize()

        assert success is True
        assert camera._status == CameraStatus.READY
        assert camera._capture_count == 0

    async def test_camera_state(self, camera):
        """Test getting camera state."""
        await camera.initialize()

        state = camera.get_state()

        assert state.status == CameraStatus.READY
        assert state.resolution == (1920, 1080)
        assert state.capture_format == "RGB"
        assert state.total_captures == 0
        assert state.error_message is None

    async def test_capture_success(self, camera):
        """Test successful image capture."""
        await camera.initialize()

        result = await camera.capture()

        assert result.success is True
        assert result.image_path is not None
        assert result.image_path.exists()
        assert result.camera_state.status == CameraStatus.READY
        assert result.error_message is None

    async def test_capture_without_initialization(self, camera):
        """Test capture fails if camera not initialized."""
        # Don't initialize
        result = await camera.capture()

        assert result.success is False
        assert result.error_message is not None
        assert "not ready" in result.error_message.lower()

    async def test_capture_increments_count(self, camera):
        """Test that capture count increments."""
        await camera.initialize()

        await camera.capture()
        assert camera._capture_count == 1

        await camera.capture()
        assert camera._capture_count == 2

        await camera.capture()
        assert camera._capture_count == 3

    async def test_capture_with_vehicle_state(self, camera):
        """Test capture with vehicle state metadata."""
        await camera.initialize()

        vehicle_state = {
            "position": {"latitude": 37.7749, "longitude": -122.4194},
            "altitude_msl": 150.0,
        }

        result = await camera.capture(vehicle_state=vehicle_state)

        assert result.success is True
        assert "vehicle_state" in result.metadata
        assert result.metadata["vehicle_state"] == vehicle_state

    async def test_capture_metadata(self, camera):
        """Test capture result metadata."""
        await camera.initialize()

        result = await camera.capture()

        assert result.success is True
        assert "simulated" in result.metadata
        assert result.metadata["simulated"] is True
        assert "capture_number" in result.metadata

    async def test_defect_injection(self):
        """Test probabilistic defect injection."""
        # Use high defect probability for reliable testing
        defect_config = DefectConfig(crack_probability=1.0, severity_min=0.5, severity_max=0.9)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimulatedCameraConfig(
                output_dir=Path(tmpdir), defect_config=defect_config, save_images=True
            )
            camera = SimulatedCamera(config=config)

            await camera.initialize()

            # Capture multiple times
            results = [await camera.capture() for _ in range(5)]

            # All should have defect injected
            for result in results:
                assert result.success is True
                assert "injected_defect" in result.metadata
                defect = result.metadata["injected_defect"]
                assert "type" in defect
                assert "severity" in defect
                assert 0.5 <= defect["severity"] <= 0.9

    async def test_no_defect_injection(self):
        """Test with zero defect probability."""
        # Zero probability
        defect_config = DefectConfig(
            crack_probability=0.0,
            corrosion_probability=0.0,
            structural_damage_probability=0.0,
            discoloration_probability=0.0,
            vegetation_probability=0.0,
            damage_probability=0.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimulatedCameraConfig(
                output_dir=Path(tmpdir), defect_config=defect_config, save_images=True
            )
            camera = SimulatedCamera(config=config)

            await camera.initialize()

            # Capture multiple times
            results = [await camera.capture() for _ in range(5)]

            # None should have defect injected
            for result in results:
                assert result.success is True
                assert "injected_defect" not in result.metadata

    async def test_capture_without_saving(self):
        """Test capture without saving images to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimulatedCameraConfig(output_dir=Path(tmpdir), save_images=False)
            camera = SimulatedCamera(config=config)

            await camera.initialize()
            result = await camera.capture()

            assert result.success is True
            # Image path is None when not saving
            assert result.image_path is None

    async def test_shutdown(self, camera):
        """Test camera shutdown."""
        await camera.initialize()
        await camera.capture()
        await camera.capture()

        assert camera._status == CameraStatus.READY
        assert camera._capture_count == 2

        await camera.shutdown()

        assert camera._status == CameraStatus.OFFLINE

    async def test_multiple_captures(self, camera):
        """Test multiple sequential captures."""
        await camera.initialize()

        # Capture 10 images
        results = []
        for _ in range(10):
            result = await camera.capture()
            results.append(result)

        # All should succeed
        assert all(r.success for r in results)
        assert camera._capture_count == 10

        # Each should have unique timestamp/filename
        paths = [r.image_path for r in results]
        assert len(set(paths)) == 10  # All unique
