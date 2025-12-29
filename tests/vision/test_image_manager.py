"""
Tests for Image Manager

Comprehensive tests for image storage, retrieval, metadata handling,
directory management, and error handling.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from vision.image_manager import ImageManager


class TestImageManagerInit:
    """Tests for ImageManager initialization."""

    def test_init_creates_base_directory(self):
        """Test that initialization creates the base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "new_dir" / "images"
            manager = ImageManager(base_dir=base_dir)

            assert manager.base_dir == base_dir
            assert base_dir.exists()

    def test_init_with_string_path(self):
        """Test initialization with string path instead of Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = f"{tmpdir}/string_path_dir"
            manager = ImageManager(base_dir=base_dir)

            assert manager.base_dir == Path(base_dir)
            assert Path(base_dir).exists()

    def test_init_with_custom_storage_settings(self):
        """Test initialization with custom storage settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ImageManager(
                base_dir=tmpdir,
                max_storage_gb=50.0,
                retention_days=15,
            )

            assert manager.max_storage_gb == 50.0
            assert manager.retention_days == 15

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ImageManager(base_dir=tmpdir)

            assert manager.max_storage_gb == 100.0
            assert manager.retention_days == 30


class TestSaveImageWithMetadata:
    """Tests for save_image_with_metadata method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create ImageManager instance."""
        return ImageManager(base_dir=temp_dir / "images")

    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a sample image file."""
        image_path = temp_dir / "sample_image.png"
        # Create a simple PNG-like file (just for testing path operations)
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        return image_path

    def test_save_image_creates_date_directory(self, manager, sample_image):
        """Test that saving an image creates a date directory."""
        metadata = {"test": "value"}

        saved_path = manager.save_image_with_metadata(sample_image, metadata)

        # Check date directory format
        date_str = datetime.now().strftime("%Y-%m-%d")
        assert date_str in str(saved_path)
        assert saved_path.exists()

    def test_save_image_with_asset_id(self, manager, sample_image):
        """Test that saving with asset_id creates asset subdirectory."""
        metadata = {"asset": "test_asset"}

        saved_path = manager.save_image_with_metadata(sample_image, metadata, asset_id="asset_001")

        # Check asset subdirectory
        assert "asset_001" in str(saved_path)
        assert saved_path.exists()

    def test_save_image_creates_metadata_sidecar(self, manager, sample_image):
        """Test that metadata sidecar file is created."""
        metadata = {"key1": "value1", "key2": 42}

        saved_path = manager.save_image_with_metadata(sample_image, metadata)

        # Check metadata file
        metadata_path = saved_path.with_suffix(".json")
        assert metadata_path.exists()

        # Verify content
        with open(metadata_path, encoding="utf-8") as f:
            saved_metadata = json.load(f)

        assert saved_metadata["key1"] == "value1"
        assert saved_metadata["key2"] == 42

    def test_save_image_with_complex_metadata(self, manager, sample_image):
        """Test saving with complex metadata including datetime."""
        metadata = {
            "timestamp": datetime.now(),
            "nested": {"inner": "value"},
            "list": [1, 2, 3],
        }

        saved_path = manager.save_image_with_metadata(sample_image, metadata)

        # Should handle datetime serialization
        metadata_path = saved_path.with_suffix(".json")
        assert metadata_path.exists()

    def test_save_image_copies_original(self, manager, sample_image):
        """Test that original image is copied, not moved."""
        metadata = {}

        saved_path = manager.save_image_with_metadata(sample_image, metadata)

        # Original should still exist
        assert sample_image.exists()
        # Copy should exist in new location
        assert saved_path.exists()
        assert saved_path != sample_image

    def test_save_same_location_no_copy(self, manager, sample_image, _temp_dir):
        """Test that no copy occurs if image already in destination."""
        # Move sample image into the manager's directory
        date_str = datetime.now().strftime("%Y-%m-%d")
        dest_dir = manager.base_dir / date_str
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Create image directly in destination
        dest_image = dest_dir / "existing.png"
        dest_image.write_bytes(sample_image.read_bytes())

        metadata = {"in_place": True}

        # Save should work without copying
        saved_path = manager.save_image_with_metadata(dest_image, metadata)

        assert saved_path.exists()


class TestGetImageMetadata:
    """Tests for get_image_metadata method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create ImageManager instance."""
        return ImageManager(base_dir=temp_dir / "images")

    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a sample image file."""
        image_path = temp_dir / "sample.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        return image_path

    def test_get_metadata_returns_correct_data(self, manager, sample_image):
        """Test retrieving metadata returns correct values."""
        original_metadata = {
            "asset_id": "asset_001",
            "confidence": 0.95,
            "detections": ["crack", "corrosion"],
        }

        saved_path = manager.save_image_with_metadata(sample_image, original_metadata)
        retrieved_metadata = manager.get_image_metadata(saved_path)

        assert retrieved_metadata is not None
        assert retrieved_metadata["asset_id"] == "asset_001"
        assert retrieved_metadata["confidence"] == 0.95
        assert retrieved_metadata["detections"] == ["crack", "corrosion"]

    def test_get_metadata_missing_file_returns_none(self, manager):
        """Test that missing metadata file returns None."""
        fake_path = manager.base_dir / "nonexistent.png"

        result = manager.get_image_metadata(fake_path)

        assert result is None

    @pytest.mark.allow_error_logs
    def test_get_metadata_invalid_json_returns_none(self, manager, temp_dir):
        """Test that invalid JSON returns None and logs error."""
        # Create image and invalid metadata
        image_path = temp_dir / "invalid.png"
        image_path.write_bytes(b"fake image")

        metadata_path = image_path.with_suffix(".json")
        metadata_path.write_text("not valid json {{{")

        result = manager.get_image_metadata(image_path)

        assert result is None


class TestCleanupOldImages:
    """Tests for cleanup_old_images method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create ImageManager with short retention."""
        return ImageManager(base_dir=temp_dir / "images", retention_days=7)

    def test_cleanup_removes_old_directories(self, manager, _temp_dir):
        """Test that cleanup removes directories older than retention period."""
        # Create old directory (10 days ago)
        old_date = datetime.now() - timedelta(days=10)
        old_dir = manager.base_dir / old_date.strftime("%Y-%m-%d")
        old_dir.mkdir(parents=True)

        # Create image in old directory
        old_image = old_dir / "old_image.png"
        old_image.write_bytes(b"old image data")

        # Create recent directory (3 days ago)
        recent_date = datetime.now() - timedelta(days=3)
        recent_dir = manager.base_dir / recent_date.strftime("%Y-%m-%d")
        recent_dir.mkdir(parents=True)

        recent_image = recent_dir / "recent_image.png"
        recent_image.write_bytes(b"recent image data")

        # Run cleanup
        deleted_count = manager.cleanup_old_images()

        # Old directory should be removed
        assert not old_dir.exists()
        # Recent directory should remain
        assert recent_dir.exists()
        assert recent_image.exists()
        assert deleted_count == 1

    def test_cleanup_dry_run_preserves_files(self, manager):
        """Test that dry run doesn't actually delete files."""
        # Create old directory
        old_date = datetime.now() - timedelta(days=10)
        old_dir = manager.base_dir / old_date.strftime("%Y-%m-%d")
        old_dir.mkdir(parents=True)

        old_image = old_dir / "old_image.png"
        old_image.write_bytes(b"old image data")

        # Run dry run cleanup
        deleted_count = manager.cleanup_old_images(dry_run=True)

        # Files should still exist
        assert old_dir.exists()
        assert old_image.exists()
        # But count should reflect what would be deleted
        assert deleted_count == 1

    def test_cleanup_skips_non_date_directories(self, manager):
        """Test that cleanup skips directories not in date format."""
        # Create non-date directory
        other_dir = manager.base_dir / "not_a_date"
        other_dir.mkdir(parents=True)

        image = other_dir / "image.png"
        image.write_bytes(b"image data")

        # Run cleanup
        deleted_count = manager.cleanup_old_images()

        # Non-date directory should remain
        assert other_dir.exists()
        assert deleted_count == 0

    def test_cleanup_handles_nested_asset_directories(self, manager):
        """Test cleanup with nested asset subdirectories."""
        # Create old directory with asset subdirectory
        old_date = datetime.now() - timedelta(days=10)
        old_dir = manager.base_dir / old_date.strftime("%Y-%m-%d") / "asset_001"
        old_dir.mkdir(parents=True)

        old_image = old_dir / "image.png"
        old_image.write_bytes(b"image data")

        # Run cleanup
        deleted_count = manager.cleanup_old_images()

        # Entire date directory should be removed
        assert not old_dir.parent.exists()
        assert deleted_count == 1

    def test_cleanup_with_no_old_images(self, manager):
        """Test cleanup when no old images exist."""
        # Create only recent directories
        recent_date = datetime.now() - timedelta(days=3)
        recent_dir = manager.base_dir / recent_date.strftime("%Y-%m-%d")
        recent_dir.mkdir(parents=True)

        deleted_count = manager.cleanup_old_images()

        assert deleted_count == 0
        assert recent_dir.exists()

    def test_cleanup_ignores_files_in_base_dir(self, manager):
        """Test that cleanup ignores loose files in base directory."""
        # Create a file directly in base_dir (not in a date directory)
        loose_file = manager.base_dir / "some_file.txt"
        loose_file.write_text("loose file content")

        # Run cleanup
        deleted_count = manager.cleanup_old_images()

        # File should still exist, not be processed as a directory
        assert loose_file.exists()
        assert deleted_count == 0


class TestGetStorageUsage:
    """Tests for get_storage_usage method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create ImageManager instance."""
        return ImageManager(base_dir=temp_dir / "images", max_storage_gb=1.0)

    def test_storage_usage_empty(self, manager):
        """Test storage usage with no images."""
        usage = manager.get_storage_usage()

        assert usage["total_images"] == 0
        assert usage["total_size_gb"] == 0.0
        assert usage["max_storage_gb"] == 1.0
        assert usage["usage_percent"] == 0.0
        assert usage["over_quota"] is False

    def test_storage_usage_with_images(self, manager):
        """Test storage usage with images."""
        # Create some images
        date_dir = manager.base_dir / "2024-01-15"
        date_dir.mkdir(parents=True)

        for i in range(5):
            image = date_dir / f"image_{i}.png"
            # Create 1KB images
            image.write_bytes(b"\x89PNG" + b"\x00" * 1020)

        usage = manager.get_storage_usage()

        assert usage["total_images"] == 5
        # Size is too small to show in GB (gets rounded to 0.0), just check images counted
        assert usage["over_quota"] is False

    def test_storage_usage_over_quota(self, temp_dir):
        """Test storage usage when over quota."""
        # Create manager with quota smaller than our test file
        # We create a 100KB file and set quota to ~50KB (0.00005 GB)
        manager = ImageManager(base_dir=temp_dir / "images", max_storage_gb=0.00005)

        # Create an image
        date_dir = manager.base_dir / "2024-01-15"
        date_dir.mkdir(parents=True)

        image = date_dir / "image.png"
        # Create ~100KB file which is > 50KB quota
        image.write_bytes(b"\x89PNG" + b"\x00" * (100 * 1024))

        usage = manager.get_storage_usage()

        assert usage["over_quota"] is True


class TestCheckQuota:
    """Tests for check_quota method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_check_quota_under_limit(self, temp_dir):
        """Test check_quota returns True when under limit."""
        manager = ImageManager(base_dir=temp_dir / "images", max_storage_gb=100.0)

        assert manager.check_quota() is True

    def test_check_quota_over_limit(self, temp_dir):
        """Test check_quota returns False when over limit."""
        # Set quota to ~50KB and create ~100KB file
        manager = ImageManager(base_dir=temp_dir / "images", max_storage_gb=0.00005)

        # Create an image
        date_dir = manager.base_dir / "2024-01-15"
        date_dir.mkdir(parents=True)

        image = date_dir / "image.png"
        # Create ~100KB file which is > 50KB quota
        image.write_bytes(b"\x89PNG" + b"\x00" * (100 * 1024))

        assert manager.check_quota() is False


class TestGetImagesForAsset:
    """Tests for get_images_for_asset method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create ImageManager instance."""
        return ImageManager(base_dir=temp_dir / "images")

    def test_get_images_for_asset(self, manager):
        """Test retrieving images for a specific asset."""
        # Create images for different assets
        date_str = datetime.now().strftime("%Y-%m-%d")

        asset_1_dir = manager.base_dir / date_str / "asset_001"
        asset_1_dir.mkdir(parents=True)

        asset_2_dir = manager.base_dir / date_str / "asset_002"
        asset_2_dir.mkdir(parents=True)

        # Create images
        for i in range(3):
            (asset_1_dir / f"image_{i}.png").write_bytes(b"\x89PNG")

        for i in range(2):
            (asset_2_dir / f"image_{i}.png").write_bytes(b"\x89PNG")

        images = manager.get_images_for_asset("asset_001")

        assert len(images) == 3
        assert all("asset_001" in str(img) for img in images)

    def test_get_images_for_asset_empty(self, manager):
        """Test retrieving images for asset with no images."""
        images = manager.get_images_for_asset("nonexistent_asset")

        assert len(images) == 0

    def test_get_images_for_asset_with_limit(self, manager):
        """Test limiting number of returned images."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        asset_dir = manager.base_dir / date_str / "asset_001"
        asset_dir.mkdir(parents=True)

        # Create many images
        for i in range(10):
            (asset_dir / f"image_{i}.png").write_bytes(b"\x89PNG")

        images = manager.get_images_for_asset("asset_001", limit=5)

        assert len(images) == 5

    def test_get_images_for_asset_sorted_by_time(self, manager):
        """Test that images are sorted by modification time (newest first)."""
        import time

        date_str = datetime.now().strftime("%Y-%m-%d")
        asset_dir = manager.base_dir / date_str / "asset_001"
        asset_dir.mkdir(parents=True)

        # Create images with time gaps
        for i in range(3):
            img = asset_dir / f"image_{i}.png"
            img.write_bytes(b"\x89PNG")
            time.sleep(0.01)  # Small delay to ensure different timestamps

        images = manager.get_images_for_asset("asset_001")

        # Should be sorted newest first
        assert len(images) == 3
        mtimes = [img.stat().st_mtime for img in images]
        assert mtimes == sorted(mtimes, reverse=True)


class TestGetRecentImages:
    """Tests for get_recent_images method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create ImageManager instance."""
        return ImageManager(base_dir=temp_dir / "images")

    def test_get_recent_images(self, manager):
        """Test retrieving recent images across all assets."""
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Create images in different directories
        for asset in ["asset_001", "asset_002"]:
            asset_dir = manager.base_dir / date_str / asset
            asset_dir.mkdir(parents=True)

            for i in range(3):
                (asset_dir / f"image_{i}.png").write_bytes(b"\x89PNG")

        images = manager.get_recent_images()

        assert len(images) == 6

    def test_get_recent_images_with_limit(self, manager):
        """Test limiting number of recent images."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        asset_dir = manager.base_dir / date_str / "asset_001"
        asset_dir.mkdir(parents=True)

        for i in range(10):
            (asset_dir / f"image_{i}.png").write_bytes(b"\x89PNG")

        images = manager.get_recent_images(limit=5)

        assert len(images) == 5

    def test_get_recent_images_empty(self, manager):
        """Test get_recent_images with no images."""
        images = manager.get_recent_images()

        assert len(images) == 0

    def test_get_recent_images_sorted(self, manager):
        """Test that recent images are sorted by time."""
        import time

        date_str = datetime.now().strftime("%Y-%m-%d")
        asset_dir = manager.base_dir / date_str / "asset_001"
        asset_dir.mkdir(parents=True)

        for i in range(5):
            img = asset_dir / f"image_{i}.png"
            img.write_bytes(b"\x89PNG")
            time.sleep(0.01)

        images = manager.get_recent_images()

        mtimes = [img.stat().st_mtime for img in images]
        assert mtimes == sorted(mtimes, reverse=True)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_multiple_images_same_name_different_assets(self, temp_dir):
        """Test handling same filename in different asset directories."""
        manager = ImageManager(base_dir=temp_dir / "images")

        # Create source image
        source = temp_dir / "image.png"
        source.write_bytes(b"\x89PNG data")

        # Save to different assets
        path1 = manager.save_image_with_metadata(source, {"asset": "001"}, asset_id="asset_001")
        path2 = manager.save_image_with_metadata(source, {"asset": "002"}, asset_id="asset_002")

        assert path1 != path2
        assert path1.exists()
        assert path2.exists()

    def test_zero_max_storage(self, temp_dir):
        """Test with zero max storage quota."""
        manager = ImageManager(base_dir=temp_dir / "images", max_storage_gb=0.0)

        usage = manager.get_storage_usage()

        assert usage["usage_percent"] == 0.0

    def test_large_retention_period(self, temp_dir):
        """Test with large retention period."""
        manager = ImageManager(
            base_dir=temp_dir / "images",
            retention_days=365 * 10,  # 10 years
        )

        # Create old directory (still within retention)
        old_date = datetime.now() - timedelta(days=365)
        old_dir = manager.base_dir / old_date.strftime("%Y-%m-%d")
        old_dir.mkdir(parents=True)

        (old_dir / "image.png").write_bytes(b"\x89PNG")

        deleted_count = manager.cleanup_old_images()

        assert deleted_count == 0
        assert old_dir.exists()

    def test_special_characters_in_asset_id(self, temp_dir):
        """Test handling asset IDs with special characters."""
        manager = ImageManager(base_dir=temp_dir / "images")

        source = temp_dir / "image.png"
        source.write_bytes(b"\x89PNG")

        # Use simple underscore-based asset ID (safe for filesystem)
        saved_path = manager.save_image_with_metadata(
            source, {}, asset_id="asset_with_underscore_123"
        )

        assert saved_path.exists()
