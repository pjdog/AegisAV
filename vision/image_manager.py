"""
Image Manager

Manages image storage, metadata, and lifecycle for vision subsystem.
Handles organized storage, cleanup, and metadata sidecar files.
"""

import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ImageManager:
    """
    Manages image storage and metadata.

    Features:
    - Organized directory structure (by date/asset)
    - Metadata sidecar files (.json)
    - Automatic cleanup of old images
    - Storage quota management
    """

    def __init__(
        self,
        base_dir: Path | str = "data/vision/images",
        max_storage_gb: float = 100.0,
        retention_days: int = 30,
    ):
        """
        Initialize image manager.

        Args:
            base_dir: Base directory for image storage
            max_storage_gb: Maximum storage in GB
            retention_days: Days to retain images
        """
        self.base_dir = Path(base_dir)
        self.max_storage_gb = max_storage_gb
        self.retention_days = retention_days
        self.logger = logger

        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_image_with_metadata(
        self,
        image_path: Path,
        metadata: dict[str, Any],
        asset_id: str | None = None,
    ) -> Path:
        """
        Save image with metadata sidecar file.

        Organizes images by date and optionally by asset.

        Args:
            image_path (Path): Path to source image (:class:`pathlib.Path`).
            metadata (dict[str, Any]): Metadata dictionary to save.
            asset_id (str | None): Optional asset ID for organization.

        Returns:
            Path: Saved image path (:class:`pathlib.Path`).
        """
        # Create organized path: base_dir/YYYY-MM-DD/asset_id/filename
        date_str = datetime.now().strftime("%Y-%m-%d")
        dest_dir = self.base_dir / date_str / asset_id if asset_id else self.base_dir / date_str

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy image
        dest_path = dest_dir / image_path.name
        if image_path != dest_path:
            shutil.copy2(image_path, dest_path)

        # Save metadata sidecar
        metadata_path = dest_path.with_suffix(".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.debug(f"Saved image with metadata: {dest_path}")
        return dest_path

    def get_image_metadata(self, image_path: Path) -> dict[str, Any] | None:
        """
        Load metadata for an image.

        Args:
            image_path (Path): Path to image (:class:`pathlib.Path`).

        Returns:
            dict[str, Any] | None: Metadata dictionary or None if not found.
        """
        metadata_path = image_path.with_suffix(".json")
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load metadata from {metadata_path}: {e}")
            return None

    def cleanup_old_images(self, dry_run: bool = False) -> int:
        """
        Remove images older than retention period.

        Args:
            dry_run (bool): If True, only report what would be deleted.

        Returns:
            int: Number of images deleted (or would be deleted).
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0

        for date_dir in self.base_dir.iterdir():
            if not date_dir.is_dir():
                continue

            try:
                # Parse directory name as date
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")

                if dir_date < cutoff_date:
                    if dry_run:
                        # Count images that would be deleted
                        image_count = sum(1 for _ in date_dir.rglob("*.png"))
                        deleted_count += image_count
                        self.logger.info(
                            f"[DRY RUN] Would delete {image_count} images from {date_dir}"
                        )
                    else:
                        # Actually delete
                        image_count = sum(1 for _ in date_dir.rglob("*.png"))
                        shutil.rmtree(date_dir)
                        deleted_count += image_count
                        self.logger.info(f"Deleted {image_count} images from {date_dir}")

            except ValueError:
                # Not a date directory, skip
                continue

        if deleted_count > 0 and not dry_run:
            self.logger.info(f"Cleanup complete: removed {deleted_count} old images")

        return deleted_count

    def get_storage_usage(self) -> dict[str, Any]:
        """
        Get current storage usage statistics.

        Returns:
            dict[str, Any]: Dictionary with usage info.
        """
        total_size_bytes = 0
        image_count = 0

        for image_path in self.base_dir.rglob("*.png"):
            total_size_bytes += image_path.stat().st_size
            image_count += 1

        total_size_gb = total_size_bytes / (1024**3)
        usage_percent = (
            (total_size_gb / self.max_storage_gb) * 100 if self.max_storage_gb > 0 else 0
        )

        return {
            "total_images": image_count,
            "total_size_gb": round(total_size_gb, 2),
            "max_storage_gb": self.max_storage_gb,
            "usage_percent": round(usage_percent, 2),
            "over_quota": total_size_gb > self.max_storage_gb,
        }

    def check_quota(self) -> bool:
        """
        Check if storage is under quota.

        Returns:
            bool: True if under quota, False if over quota.
        """
        usage = self.get_storage_usage()
        return not usage["over_quota"]

    def get_images_for_asset(self, asset_id: str, limit: int = 100) -> list[Path]:
        """
        Get all images for a specific asset.

        Args:
            asset_id (str): Asset ID.
            limit (int): Maximum number of images to return.

        Returns:
            list[Path]: Image paths (newest first).
        """
        images = []

        for asset_dir in self.base_dir.rglob(asset_id):
            if asset_dir.is_dir():
                for image_path in asset_dir.glob("*.png"):
                    images.append(image_path)

        # Sort by modification time (newest first)
        images.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return images[:limit]

    def get_recent_images(self, limit: int = 100) -> list[Path]:
        """
        Get most recent images across all assets.

        Args:
            limit (int): Maximum number of images.

        Returns:
            list[Path]: Image paths (newest first).
        """
        images = list(self.base_dir.rglob("*.png"))
        images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return images[:limit]
