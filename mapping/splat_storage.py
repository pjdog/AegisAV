"""Gaussian splat artifact storage with versioning.

Phase 3 Worker B: Implement splat artifact storage with versioning.

This module handles:
- Storage of Gaussian splat scene artifacts
- Version management for splat reconstructions
- Preview mesh/point cloud generation for planning
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SplatMetadata:
    """Metadata for a Gaussian splat scene."""

    # Identification
    run_id: str
    scene_id: str
    version: int = 1

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    training_started: str | None = None
    training_completed: str | None = None

    # Training info
    source_dataset: str | None = None
    keyframe_count: int = 0
    total_iterations: int = 0
    final_loss: float = 0.0

    # Scene bounds (in NED frame)
    bounds_min: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bounds_max: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Gaussian info
    gaussian_count: int = 0
    compressed_size_mb: float = 0.0

    # Quality metrics
    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = 0.0

    # Files
    model_file: str = "model.ply"
    preview_file: str = "preview.ply"
    config_file: str = "config.json"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "scene_id": self.scene_id,
            "version": self.version,
            "created_at": self.created_at,
            "training_started": self.training_started,
            "training_completed": self.training_completed,
            "source_dataset": self.source_dataset,
            "training": {
                "keyframe_count": self.keyframe_count,
                "total_iterations": self.total_iterations,
                "final_loss": self.final_loss,
            },
            "bounds": {
                "min": self.bounds_min,
                "max": self.bounds_max,
            },
            "gaussians": {
                "count": self.gaussian_count,
                "compressed_size_mb": self.compressed_size_mb,
            },
            "quality": {
                "psnr": self.psnr,
                "ssim": self.ssim,
                "lpips": self.lpips,
            },
            "files": {
                "model": self.model_file,
                "preview": self.preview_file,
                "config": self.config_file,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplatMetadata:
        """Create from dictionary."""
        training = data.get("training", {})
        bounds = data.get("bounds", {})
        gaussians = data.get("gaussians", {})
        quality = data.get("quality", {})
        files = data.get("files", {})

        return cls(
            run_id=data.get("run_id", "unknown"),
            scene_id=data.get("scene_id", "unknown"),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now().isoformat()),
            training_started=data.get("training_started"),
            training_completed=data.get("training_completed"),
            source_dataset=data.get("source_dataset"),
            keyframe_count=training.get("keyframe_count", 0),
            total_iterations=training.get("total_iterations", 0),
            final_loss=training.get("final_loss", 0.0),
            bounds_min=bounds.get("min", [0.0, 0.0, 0.0]),
            bounds_max=bounds.get("max", [0.0, 0.0, 0.0]),
            gaussian_count=gaussians.get("count", 0),
            compressed_size_mb=gaussians.get("compressed_size_mb", 0.0),
            psnr=quality.get("psnr", 0.0),
            ssim=quality.get("ssim", 0.0),
            lpips=quality.get("lpips", 0.0),
            model_file=files.get("model", "model.ply"),
            preview_file=files.get("preview", "preview.ply"),
            config_file=files.get("config", "config.json"),
        )


@dataclass
class SplatScene:
    """A versioned Gaussian splat scene."""

    metadata: SplatMetadata
    base_path: Path

    @property
    def scene_dir(self) -> Path:
        """Get the scene directory path."""
        return self.base_path / f"scene_{self.metadata.run_id}"

    @property
    def version_dir(self) -> Path:
        """Get the current version directory."""
        return self.scene_dir / f"v{self.metadata.version}"

    @property
    def model_path(self) -> Path:
        """Get the path to the main model file."""
        return self.version_dir / self.metadata.model_file

    @property
    def preview_path(self) -> Path:
        """Get the path to the preview point cloud."""
        return self.version_dir / self.metadata.preview_file

    def exists(self) -> bool:
        """Check if the scene files exist."""
        return self.version_dir.exists() and self.model_path.exists()


class SplatStorage:
    """Storage manager for Gaussian splat artifacts.

    Handles versioned storage of splat scenes with the following structure:
        splats/
            scene_{run_id}/
                v1/
                    model.ply
                    preview.ply
                    metadata.json
                    config.json
                v2/
                    ...
                latest -> v2/

    Usage:
        storage = SplatStorage(Path("data/splats"))

        # Store a new scene
        scene = storage.store_scene(
            run_id="flight_001",
            model_path=trained_model_path,
            preview_path=preview_ply_path,
            metadata=metadata,
        )

        # Load latest version
        scene = storage.get_scene("flight_001")

        # List all scenes
        scenes = storage.list_scenes()
    """

    def __init__(self, base_path: Path) -> None:
        """Initialize splat storage.

        Args:
            base_path: Root directory for splat storage.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_scene_dir(self, run_id: str) -> Path:
        """Get the directory for a scene."""
        return self.base_path / f"scene_{run_id}"

    def _get_next_version(self, run_id: str) -> int:
        """Get the next version number for a scene."""
        scene_dir = self._get_scene_dir(run_id)
        if not scene_dir.exists():
            return 1

        versions = []
        for path in scene_dir.iterdir():
            if path.is_dir() and path.name.startswith("v"):
                try:
                    versions.append(int(path.name[1:]))
                except ValueError:
                    pass

        return max(versions, default=0) + 1

    def _update_latest_link(self, scene_dir: Path, version: int) -> None:
        """Update the 'latest' symlink to point to a version."""
        latest_link = scene_dir / "latest"
        version_dir = scene_dir / f"v{version}"

        # Remove existing link
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create relative symlink
        latest_link.symlink_to(f"v{version}")

    def store_scene(
        self,
        run_id: str,
        model_path: Path,
        preview_path: Path | None = None,
        metadata: SplatMetadata | None = None,
        config: dict[str, Any] | None = None,
    ) -> SplatScene:
        """Store a new splat scene or version.

        Args:
            run_id: Unique run identifier.
            model_path: Path to the trained model file (.ply).
            preview_path: Path to preview point cloud (optional).
            metadata: Scene metadata.
            config: Training configuration.

        Returns:
            SplatScene instance.
        """
        version = self._get_next_version(run_id)
        scene_dir = self._get_scene_dir(run_id)
        version_dir = scene_dir / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata if not provided
        if metadata is None:
            metadata = SplatMetadata(run_id=run_id, scene_id=run_id, version=version)
        else:
            metadata.version = version

        # Copy model file
        model_dest = version_dir / metadata.model_file
        shutil.copy2(model_path, model_dest)

        # Copy preview if provided
        if preview_path and preview_path.exists():
            preview_dest = version_dir / metadata.preview_file
            shutil.copy2(preview_path, preview_dest)

        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Save config if provided
        if config:
            config_path = version_dir / metadata.config_file
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        # Update latest link
        self._update_latest_link(scene_dir, version)

        logger.info(
            "splat_scene_stored",
            run_id=run_id,
            version=version,
            path=str(version_dir),
        )

        return SplatScene(metadata=metadata, base_path=self.base_path)

    def get_scene(
        self,
        run_id: str,
        version: int | None = None,
    ) -> SplatScene | None:
        """Get a splat scene.

        Args:
            run_id: Scene run identifier.
            version: Specific version (None = latest).

        Returns:
            SplatScene if found, None otherwise.
        """
        scene_dir = self._get_scene_dir(run_id)
        if not scene_dir.exists():
            return None

        # Determine version directory
        if version is None:
            # Use latest link
            latest_link = scene_dir / "latest"
            if latest_link.is_symlink():
                version_dir = latest_link.resolve()
            else:
                # Find highest version
                versions = []
                for path in scene_dir.iterdir():
                    if path.is_dir() and path.name.startswith("v"):
                        try:
                            versions.append(int(path.name[1:]))
                        except ValueError:
                            pass
                if not versions:
                    return None
                version_dir = scene_dir / f"v{max(versions)}"
        else:
            version_dir = scene_dir / f"v{version}"

        if not version_dir.exists():
            return None

        # Load metadata
        metadata_path = version_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = SplatMetadata.from_dict(json.load(f))
        else:
            # Create minimal metadata
            metadata = SplatMetadata(
                run_id=run_id,
                scene_id=run_id,
                version=int(version_dir.name[1:]) if version_dir.name.startswith("v") else 1,
            )

        return SplatScene(metadata=metadata, base_path=self.base_path)

    def list_scenes(self) -> list[SplatScene]:
        """List all available scenes (latest versions only).

        Returns:
            List of SplatScene instances.
        """
        scenes = []

        for path in self.base_path.iterdir():
            if path.is_dir() and path.name.startswith("scene_"):
                run_id = path.name[6:]  # Remove "scene_" prefix
                scene = self.get_scene(run_id)
                if scene:
                    scenes.append(scene)

        return scenes

    def list_versions(self, run_id: str) -> list[int]:
        """List all versions for a scene.

        Args:
            run_id: Scene run identifier.

        Returns:
            List of version numbers.
        """
        scene_dir = self._get_scene_dir(run_id)
        if not scene_dir.exists():
            return []

        versions = []
        for path in scene_dir.iterdir():
            if path.is_dir() and path.name.startswith("v"):
                try:
                    versions.append(int(path.name[1:]))
                except ValueError:
                    pass

        return sorted(versions)

    def delete_scene(self, run_id: str, version: int | None = None) -> bool:
        """Delete a scene or specific version.

        Args:
            run_id: Scene run identifier.
            version: Specific version to delete (None = all versions).

        Returns:
            True if deleted successfully.
        """
        scene_dir = self._get_scene_dir(run_id)
        if not scene_dir.exists():
            return False

        if version is None:
            # Delete entire scene
            shutil.rmtree(scene_dir)
            logger.info("splat_scene_deleted", run_id=run_id)
        else:
            # Delete specific version
            version_dir = scene_dir / f"v{version}"
            if version_dir.exists():
                shutil.rmtree(version_dir)
                logger.info("splat_version_deleted", run_id=run_id, version=version)

                # Update latest link if needed
                remaining = self.list_versions(run_id)
                if remaining:
                    self._update_latest_link(scene_dir, max(remaining))
                else:
                    # No versions left, delete scene dir
                    shutil.rmtree(scene_dir)
            else:
                return False

        return True

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics.
        """
        scenes = self.list_scenes()
        total_size = 0
        version_counts = []

        for scene in scenes:
            versions = self.list_versions(scene.metadata.run_id)
            version_counts.append(len(versions))

            for v in versions:
                v_scene = self.get_scene(scene.metadata.run_id, v)
                if v_scene and v_scene.version_dir.exists():
                    for f in v_scene.version_dir.iterdir():
                        if f.is_file():
                            total_size += f.stat().st_size

        return {
            "scene_count": len(scenes),
            "total_versions": sum(version_counts),
            "total_size_mb": total_size / (1024 * 1024),
            "base_path": str(self.base_path),
        }
