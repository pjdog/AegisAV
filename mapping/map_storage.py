"""Versioned storage for fused navigation map artifacts."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CleanupResult:
    """Results from artifact cleanup."""

    deleted_versions: int = 0
    deleted_maps: int = 0
    freed_bytes: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def freed_mb(self) -> float:
        """Freed space in megabytes."""
        return self.freed_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deleted_versions": self.deleted_versions,
            "deleted_maps": self.deleted_maps,
            "freed_bytes": self.freed_bytes,
            "freed_mb": round(self.freed_mb, 2),
            "errors": self.errors,
            "dry_run": self.dry_run,
        }


@dataclass
class MapArtifactStoreConfig:
    """Configuration for fused map artifact storage."""

    base_dir: Path = Path("data/maps/fused")
    max_versions: int | None = None
    max_age_days: int | None = None
    keep_last: int = 3
    max_total_size_mb: float | None = None


class MapArtifactStore:
    """Store fused navigation maps with versioned metadata."""

    def __init__(self, config: MapArtifactStoreConfig | None = None) -> None:
        self.config = config or MapArtifactStoreConfig()

    @staticmethod
    def _sanitize(name: str) -> str:
        if not name:
            return "map"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    @staticmethod
    def _next_version(map_dir: Path) -> int:
        if not map_dir.exists():
            return 1
        versions = []
        for child in map_dir.iterdir():
            if child.is_dir() and child.name.startswith("v"):
                try:
                    versions.append(int(child.name[1:]))
                except ValueError:
                    continue
        return (max(versions) + 1) if versions else 1

    def store(self, navigation_map: dict[str, Any]) -> dict[str, Any]:
        """Persist a navigation map to a versioned directory."""
        base_dir = Path(self.config.base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        metadata = navigation_map.get("metadata", {})
        map_id = metadata.get("map_id") or navigation_map.get("scenario_id") or "map"
        map_key = self._sanitize(str(map_id))

        map_dir = base_dir / f"map_{map_key}"
        map_dir.mkdir(parents=True, exist_ok=True)

        version = self._next_version(map_dir)
        version_dir = map_dir / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        nav_path = version_dir / "navigation_map.json"
        nav_path.write_text(json.dumps(navigation_map, indent=2))

        summary = {
            "map_id": map_id,
            "map_key": map_key,
            "version": version,
            "scenario_id": navigation_map.get("scenario_id"),
            "source": navigation_map.get("source"),
            "generated_at": navigation_map.get("generated_at"),
            "stored_at": datetime.now().isoformat(),
            "obstacle_count": len(navigation_map.get("obstacles", [])),
            "tile_count": len(navigation_map.get("tiles") or []),
            "voxel_count": len(navigation_map.get("voxels") or []),
            "metadata": metadata,
            "path": str(version_dir),
        }

        summary_path = version_dir / "metadata.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        (map_dir / "latest.json").write_text(json.dumps(summary, indent=2))

        self.cleanup(map_dir)

        logger.info(
            "map_artifact_stored",
            map_id=map_id,
            version=version,
            path=str(version_dir),
        )

        return summary

    def list_maps(self) -> list[dict[str, Any]]:
        """List stored fused maps with latest metadata."""
        base_dir = Path(self.config.base_dir)
        if not base_dir.exists():
            return []

        maps: list[dict[str, Any]] = []
        for map_dir in sorted(base_dir.iterdir()):
            if not map_dir.is_dir() or not map_dir.name.startswith("map_"):
                continue
            latest_path = map_dir / "latest.json"
            if latest_path.exists():
                try:
                    with open(latest_path, encoding="utf-8") as f:
                        maps.append(json.load(f))
                    continue
                except Exception:
                    pass

            versions = []
            for child in map_dir.iterdir():
                if child.is_dir() and child.name.startswith("v"):
                    try:
                        versions.append(int(child.name[1:]))
                    except ValueError:
                        continue
            if not versions:
                continue

            latest_v = max(versions)
            summary_path = map_dir / f"v{latest_v}" / "metadata.json"
            if summary_path.exists():
                try:
                    with open(summary_path, encoding="utf-8") as f:
                        maps.append(json.load(f))
                    continue
                except Exception:
                    pass

        return maps

    def cleanup(self, map_dir: Path, dry_run: bool = False) -> CleanupResult:
        """Apply retention policy to a map directory.

        Args:
            map_dir: Path to the map directory to clean.
            dry_run: If True, only report what would be deleted.

        Returns:
            CleanupResult with details of what was (or would be) deleted.
        """
        result = CleanupResult(dry_run=dry_run)
        map_dir = Path(map_dir)
        if not map_dir.exists():
            return result

        versions = []
        for child in map_dir.iterdir():
            if child.is_dir() and child.name.startswith("v"):
                try:
                    versions.append(int(child.name[1:]))
                except ValueError:
                    continue

        if not versions:
            return result

        keep_last = max(1, int(self.config.keep_last))
        versions_sorted = sorted(versions)
        protected = set(versions_sorted[-keep_last:])

        now = datetime.now().timestamp()
        to_delete: list[Path] = []

        for v in versions_sorted:
            if v in protected:
                continue
            version_dir = map_dir / f"v{v}"
            if not version_dir.exists():
                continue

            should_delete = False

            # Check version count limit
            if self.config.max_versions and len(versions_sorted) > self.config.max_versions:
                # Only delete if we have more than max_versions
                remaining = len(versions_sorted) - len(to_delete)
                if remaining > self.config.max_versions:
                    should_delete = True

            # Check age limit
            if self.config.max_age_days:
                try:
                    age_s = now - version_dir.stat().st_mtime
                    if age_s > float(self.config.max_age_days) * 86400.0:
                        should_delete = True
                except OSError:
                    pass

            if should_delete:
                to_delete.append(version_dir)

        for version_dir in to_delete:
            try:
                size = self._get_dir_size(version_dir)
                if not dry_run:
                    self._remove_dir(version_dir)
                result.deleted_versions += 1
                result.freed_bytes += size
            except Exception as e:
                result.errors.append(f"Failed to delete {version_dir}: {e}")

        return result

    def cleanup_all(self, dry_run: bool = False) -> CleanupResult:
        """Apply retention policy to all stored maps.

        Args:
            dry_run: If True, only report what would be deleted.

        Returns:
            CleanupResult with aggregate details.
        """
        result = CleanupResult(dry_run=dry_run)
        base_dir = Path(self.config.base_dir)

        if not base_dir.exists():
            return result

        for map_dir in base_dir.iterdir():
            if not map_dir.is_dir() or not map_dir.name.startswith("map_"):
                continue

            map_result = self.cleanup(map_dir, dry_run=dry_run)
            result.deleted_versions += map_result.deleted_versions
            result.freed_bytes += map_result.freed_bytes
            result.errors.extend(map_result.errors)

            # Check if map directory is now empty
            if not dry_run:
                versions_remaining = [
                    c for c in map_dir.iterdir() if c.is_dir() and c.name.startswith("v")
                ]
                if not versions_remaining:
                    try:
                        # Remove empty map directory
                        for f in map_dir.iterdir():
                            if f.is_file():
                                f.unlink()
                        map_dir.rmdir()
                        result.deleted_maps += 1
                    except Exception as e:
                        result.errors.append(f"Failed to remove empty map dir {map_dir}: {e}")

        # Apply total size limit if configured
        if self.config.max_total_size_mb and not dry_run:
            size_result = self._enforce_size_limit()
            result.deleted_versions += size_result.deleted_versions
            result.freed_bytes += size_result.freed_bytes
            result.errors.extend(size_result.errors)

        logger.info(
            "map_cleanup_complete",
            deleted_versions=result.deleted_versions,
            deleted_maps=result.deleted_maps,
            freed_mb=result.freed_mb,
            errors=len(result.errors),
            dry_run=dry_run,
        )

        return result

    def _enforce_size_limit(self) -> CleanupResult:
        """Delete oldest versions until under size limit."""
        result = CleanupResult()
        if not self.config.max_total_size_mb:
            return result

        max_bytes = self.config.max_total_size_mb * 1024 * 1024
        base_dir = Path(self.config.base_dir)

        # Collect all versions with size and mtime
        versions: list[tuple[Path, int, float]] = []  # (path, size, mtime)
        for map_dir in base_dir.iterdir():
            if not map_dir.is_dir() or not map_dir.name.startswith("map_"):
                continue
            for child in map_dir.iterdir():
                if child.is_dir() and child.name.startswith("v"):
                    try:
                        size = self._get_dir_size(child)
                        mtime = child.stat().st_mtime
                        versions.append((child, size, mtime))
                    except OSError:
                        continue

        # Sort by mtime (oldest first)
        versions.sort(key=lambda x: x[2])

        total_size = sum(v[1] for v in versions)

        # Delete oldest versions until under limit, keeping at least 1 per map
        keep_last = max(1, self.config.keep_last)
        map_version_counts: dict[str, int] = {}

        # Count versions per map
        for v_path, _, _ in versions:
            map_name = v_path.parent.name
            map_version_counts[map_name] = map_version_counts.get(map_name, 0) + 1

        for v_path, v_size, _ in versions:
            if total_size <= max_bytes:
                break

            map_name = v_path.parent.name
            if map_version_counts.get(map_name, 0) <= keep_last:
                continue

            try:
                self._remove_dir(v_path)
                result.deleted_versions += 1
                result.freed_bytes += v_size
                total_size -= v_size
                map_version_counts[map_name] -= 1
            except Exception as e:
                result.errors.append(f"Failed to delete {v_path}: {e}")

        return result

    @staticmethod
    def _get_dir_size(path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except OSError:
            pass
        return total

    @staticmethod
    def _remove_dir(path: Path) -> None:
        """Remove a directory and all its contents."""
        shutil.rmtree(path, ignore_errors=True)
