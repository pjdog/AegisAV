"""3D Asset Manager for AegisAV Simulation.

Downloads, caches, and manages 3D models from various open-source repositories
for use in AirSim/Unreal Engine simulations.

Supported sources:
- Poly Haven (CC0, via API)
- Sketchfab (CC-BY/CC0, requires API token for downloads)
- Direct URLs (GLB/GLTF files)
- Local files

For Unreal Engine integration, assets need to be imported into the project.
This module provides the download/cache layer and metadata management.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger(__name__)


class AssetType(str, Enum):
    """Types of 3D assets for simulation."""

    HELIPAD = "helipad"
    LANDING_PAD = "landing_pad"
    SOLAR_PANEL = "solar_panel"
    WIND_TURBINE = "wind_turbine"
    DRONE = "drone"
    BUILDING = "building"
    VEHICLE = "vehicle"
    MARKER = "marker"
    CUSTOM = "custom"


class AssetFormat(str, Enum):
    """Supported 3D file formats."""

    GLB = "glb"
    GLTF = "gltf"
    FBX = "fbx"
    OBJ = "obj"
    UASSET = "uasset"  # Unreal Engine native


class AssetLicense(str, Enum):
    """Asset license types."""

    CC0 = "cc0"  # Public domain
    CC_BY = "cc-by"  # Attribution required
    CC_BY_SA = "cc-by-sa"  # Attribution + ShareAlike
    CC_BY_NC = "cc-by-nc"  # Attribution + NonCommercial
    MIT = "mit"
    UNKNOWN = "unknown"


@dataclass
class AssetMetadata:
    """Metadata for a 3D asset."""

    asset_id: str
    name: str
    asset_type: AssetType
    source: str  # "polyhaven", "sketchfab", "url", "local"
    source_url: str | None = None
    license: AssetLicense = AssetLicense.UNKNOWN
    author: str | None = None
    format: AssetFormat = AssetFormat.GLB
    file_path: Path | None = None
    unreal_asset_path: str | None = None  # e.g., "/Game/AegisAV/Assets/Helipad"
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    tags: list[str] = field(default_factory=list)
    downloaded: bool = False
    file_size_bytes: int = 0
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "asset_type": self.asset_type.value,
            "source": self.source,
            "source_url": self.source_url,
            "license": self.license.value,
            "author": self.author,
            "format": self.format.value,
            "file_path": str(self.file_path) if self.file_path else None,
            "unreal_asset_path": self.unreal_asset_path,
            "scale": list(self.scale),
            "tags": self.tags,
            "downloaded": self.downloaded,
            "file_size_bytes": self.file_size_bytes,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssetMetadata":
        """Create from dictionary."""
        return cls(
            asset_id=data["asset_id"],
            name=data["name"],
            asset_type=AssetType(data["asset_type"]),
            source=data["source"],
            source_url=data.get("source_url"),
            license=AssetLicense(data.get("license", "unknown")),
            author=data.get("author"),
            format=AssetFormat(data.get("format", "glb")),
            file_path=Path(data["file_path"]) if data.get("file_path") else None,
            unreal_asset_path=data.get("unreal_asset_path"),
            scale=tuple(data.get("scale", [1.0, 1.0, 1.0])),
            tags=data.get("tags", []),
            downloaded=data.get("downloaded", False),
            file_size_bytes=data.get("file_size_bytes", 0),
            checksum=data.get("checksum"),
        )


# Curated list of free assets for energy infrastructure simulation
# Direct download URLs where possible to avoid API authentication requirements
#
# IMPORTANT: For realistic simulation, you should download models from:
# - Sketchfab (https://sketchfab.com) - requires free account for GLB downloads
# - Poly Haven (https://polyhaven.com) - CC0, download GLTF+textures manually
# - TurboSquid (https://turbosquid.com) - some free models
#
# Then use: python -m simulation.setup_assets --register <id> <name> <type> <local_path>
#
CURATED_ASSETS: dict[str, dict[str, Any]] = {
    # === SAMPLE MODELS (CC0, direct download) ===
    # From Khronos glTF-Sample-Assets - useful for testing
    "box_basic": {
        "name": "Basic Box (Test)",
        "asset_type": AssetType.MARKER,
        "source": "url",
        "source_url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Box/glTF-Binary/Box.glb",
        "license": AssetLicense.CC0,
        "author": "Khronos Group",
        "tags": ["box", "cube", "test", "marker"],
        "scale": (1.0, 1.0, 1.0),
    },
    "box_textured": {
        "name": "Textured Box (Test)",
        "asset_type": AssetType.MARKER,
        "source": "url",
        "source_url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/BoxTextured/glTF-Binary/BoxTextured.glb",
        "license": AssetLicense.CC0,
        "author": "Khronos Group",
        "tags": ["box", "cube", "textured", "marker"],
        "scale": (1.0, 1.0, 1.0),
    },
    "lantern": {
        "name": "Lantern (Light Marker)",
        "asset_type": AssetType.MARKER,
        "source": "url",
        "source_url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Lantern/glTF-Binary/Lantern.glb",
        "license": AssetLicense.CC0,
        "author": "Microsoft",
        "tags": ["lantern", "light", "marker", "prop"],
        "scale": (0.1, 0.1, 0.1),  # Lantern is large, scale down
    },
    # === SKETCHFAB MODELS (require manual download with free account) ===
    # Instructions:
    # 1. Create free Sketchfab account
    # 2. Download GLB from the URL below
    # 3. Place in ~/.aegisav/assets/ or your Unreal project Content folder
    # 4. Run: python -m simulation.setup_assets --register helipad_basic "Helipad" helipad /path/to/file.glb
    "helipad_basic": {
        "name": "Basic Helipad (Manual Download)",
        "asset_type": AssetType.HELIPAD,
        "source": "sketchfab",
        "source_url": "https://sketchfab.com/3d-models/helipad-c76632cbfbbc4b4fa236e1a0155ed89a",
        "license": AssetLicense.CC_BY,
        "author": "srockzv",
        "tags": ["helipad", "landing", "platform"],
        "scale": (1.0, 1.0, 1.0),
    },
    "helipad_industrial": {
        "name": "Industrial Helipad (Manual Download)",
        "asset_type": AssetType.HELIPAD,
        "source": "sketchfab",
        "source_url": "https://sketchfab.com/3d-models/heliport-helipad-air-base-helicopter-5bc89e02a58b4ebca7404e5e35da2481",
        "license": AssetLicense.CC_BY,
        "author": "Mehdi Shahsavan",
        "tags": ["helipad", "heliport", "industrial"],
        "scale": (1.0, 1.0, 1.0),
    },
    "wind_turbine_large": {
        "name": "Wind Turbine (Manual Download)",
        "asset_type": AssetType.WIND_TURBINE,
        "source": "sketchfab",
        "source_url": "https://sketchfab.com/search?q=wind+turbine&type=models&licenses=322a749bcfa841b29dff1571c9b85ce8",
        "license": AssetLicense.CC_BY,
        "tags": ["wind", "turbine", "energy", "renewable"],
        "scale": (1.0, 1.0, 1.0),
    },
    # === POLY HAVEN MODELS (CC0, require manual GLTF+texture download) ===
    # Instructions:
    # 1. Visit the URL and download GLTF format
    # 2. Import into Blender and export as single GLB
    # 3. Place in ~/.aegisav/assets/
    "barrel_industrial": {
        "name": "Industrial Barrel (Manual Download)",
        "asset_type": AssetType.MARKER,
        "source": "polyhaven",
        "source_url": "https://polyhaven.com/a/Barrel_01",
        "license": AssetLicense.CC0,
        "author": "Poly Haven",
        "tags": ["barrel", "industrial", "marker"],
        "scale": (1.0, 1.0, 1.0),
    },
    "solar_panel_ground": {
        "name": "Solar Panel Array (Manual Download)",
        "asset_type": AssetType.SOLAR_PANEL,
        "source": "sketchfab",
        "source_url": "https://sketchfab.com/search?q=solar+panel&type=models&licenses=322a749bcfa841b29dff1571c9b85ce8",
        "license": AssetLicense.CC_BY,
        "tags": ["solar", "energy", "renewable", "panel"],
        "scale": (1.0, 1.0, 1.0),
    },
}


class AssetManager:
    """Manages 3D assets for AegisAV simulation.

    Handles downloading, caching, and tracking of 3D models from various sources.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        sketchfab_token: str | None = None,
    ):
        """Initialize asset manager.

        Args:
            cache_dir: Directory to cache downloaded assets
            sketchfab_token: Sketchfab API token for downloading models
        """
        self.cache_dir = cache_dir or Path.home() / ".aegisav" / "assets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / "assets.json"
        self.sketchfab_token = sketchfab_token

        # Load existing metadata
        self.assets: dict[str, AssetMetadata] = {}
        self._load_metadata()

        # Register curated assets
        self._register_curated_assets()

        logger.info(f"AssetManager initialized with {len(self.assets)} assets")

    def _load_metadata(self) -> None:
        """Load asset metadata from cache."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    data = json.load(f)
                for asset_id, asset_data in data.items():
                    self.assets[asset_id] = AssetMetadata.from_dict(asset_data)
            except Exception as e:
                logger.warning(f"Failed to load asset metadata: {e}")

    def _save_metadata(self) -> None:
        """Save asset metadata to cache."""
        try:
            data = {aid: asset.to_dict() for aid, asset in self.assets.items()}
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save asset metadata: {e}")

    def _register_curated_assets(self) -> None:
        """Register curated assets if not already in cache."""
        for asset_id, info in CURATED_ASSETS.items():
            if asset_id not in self.assets:
                self.assets[asset_id] = AssetMetadata(
                    asset_id=asset_id,
                    name=info["name"],
                    asset_type=info["asset_type"],
                    source=info["source"],
                    source_url=info.get("source_url"),
                    license=info.get("license", AssetLicense.UNKNOWN),
                    author=info.get("author"),
                    tags=info.get("tags", []),
                    scale=tuple(info.get("scale", (1.0, 1.0, 1.0))),
                )
        self._save_metadata()

    def get_asset(self, asset_id: str) -> AssetMetadata | None:
        """Get asset metadata by ID."""
        return self.assets.get(asset_id)

    def list_assets(
        self,
        asset_type: AssetType | None = None,
        downloaded_only: bool = False,
    ) -> list[AssetMetadata]:
        """List available assets.

        Args:
            asset_type: Filter by asset type
            downloaded_only: Only return downloaded assets

        Returns:
            List of matching assets
        """
        results = []
        for asset in self.assets.values():
            if asset_type and asset.asset_type != asset_type:
                continue
            if downloaded_only and not asset.downloaded:
                continue
            results.append(asset)
        return results

    def register_asset(
        self,
        asset_id: str,
        name: str,
        asset_type: AssetType,
        source: str,
        source_url: str | None = None,
        **kwargs: Any,
    ) -> AssetMetadata:
        """Register a new asset.

        Args:
            asset_id: Unique identifier for the asset
            name: Human-readable name
            asset_type: Type of asset
            source: Source of asset ("url", "local", "sketchfab", "polyhaven")
            source_url: URL to download from
            **kwargs: Additional metadata fields

        Returns:
            Created AssetMetadata
        """
        asset = AssetMetadata(
            asset_id=asset_id,
            name=name,
            asset_type=asset_type,
            source=source,
            source_url=source_url,
            **kwargs,
        )
        self.assets[asset_id] = asset
        self._save_metadata()
        return asset

    async def download_asset(
        self,
        asset_id: str,
        force: bool = False,
    ) -> AssetMetadata | None:
        """Download an asset to the cache.

        Args:
            asset_id: Asset to download
            force: Re-download even if already cached

        Returns:
            Updated AssetMetadata, or None if download failed
        """
        asset = self.assets.get(asset_id)
        if not asset:
            logger.error(f"Asset not found: {asset_id}")
            return None

        if asset.downloaded and not force:
            if asset.file_path and asset.file_path.exists():
                logger.info(f"Asset already downloaded: {asset_id}")
                return asset

        # Determine download method based on source
        try:
            if asset.source == "url" and asset.source_url:
                success = await self._download_from_url(asset)
            elif asset.source == "polyhaven":
                success = await self._download_from_polyhaven(asset)
            elif asset.source == "sketchfab":
                success = await self._download_from_sketchfab(asset)
            elif asset.source == "local":
                success = True  # Already local
            else:
                logger.error(f"Unknown source for asset {asset_id}: {asset.source}")
                return None

            if success:
                asset.downloaded = True
                self._save_metadata()
                logger.info(f"Downloaded asset: {asset_id} -> {asset.file_path}")
                return asset
            else:
                return None

        except Exception as e:
            logger.exception(f"Failed to download asset {asset_id}: {e}")
            return None

    async def _download_from_url(self, asset: AssetMetadata) -> bool:
        """Download asset from direct URL."""
        if not asset.source_url:
            return False

        # Determine filename from URL
        parsed = urlparse(asset.source_url)
        filename = Path(parsed.path).name
        if not filename:
            filename = f"{asset.asset_id}.glb"

        dest_path = self.cache_dir / asset.asset_id / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiohttp.ClientSession() as session:
            async with session.get(asset.source_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download {asset.source_url}: {response.status}")
                    return False

                content = await response.read()

                # Calculate checksum
                checksum = hashlib.sha256(content).hexdigest()

                # Save file
                with open(dest_path, "wb") as f:
                    f.write(content)

                asset.file_path = dest_path
                asset.file_size_bytes = len(content)
                asset.checksum = checksum

                # Detect format from extension
                ext = dest_path.suffix.lower().lstrip(".")
                if ext in ["glb", "gltf", "fbx", "obj"]:
                    asset.format = AssetFormat(ext)

                return True

    async def _download_from_polyhaven(self, asset: AssetMetadata) -> bool:
        """Download asset from Poly Haven API."""
        # Poly Haven API endpoints
        api_base = "https://api.polyhaven.com"

        async with aiohttp.ClientSession() as session:
            # First, get asset info
            if asset.source_url and "api.polyhaven.com" in asset.source_url:
                # Direct API URL provided
                info_url = asset.source_url
            else:
                # Search for asset by name/tags
                # Try to find a matching model
                async with session.get(f"{api_base}/assets?t=models") as response:
                    if response.status != 200:
                        logger.error(f"Failed to list Poly Haven models: {response.status}")
                        return False

                    models = await response.json()

                    # Find matching model
                    matching_id = None
                    for model_id, model_info in models.items():
                        model_name = model_info.get("name", "").lower()
                        model_tags = [t.lower() for t in model_info.get("tags", [])]

                        # Check if any of our tags match
                        for tag in asset.tags:
                            if tag.lower() in model_name or tag.lower() in model_tags:
                                matching_id = model_id
                                break
                        if matching_id:
                            break

                    if not matching_id:
                        logger.warning(f"No matching Poly Haven model for: {asset.name}")
                        return False

                    info_url = f"{api_base}/files/{matching_id}"

            # Get file info
            async with session.get(info_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to get Poly Haven file info: {response.status}")
                    return False

                file_info = await response.json()

                # Get GLB download URL
                glb_info = file_info.get("glb", {})
                if not glb_info:
                    gltf_info = file_info.get("gltf", {})
                    if gltf_info:
                        # Get the 1k or 2k version
                        for res in ["1k", "2k", "4k"]:
                            if res in gltf_info:
                                download_url = gltf_info[res].get("url")
                                break
                    else:
                        logger.error(f"No GLB/GLTF download for: {asset.name}")
                        return False
                else:
                    # Get highest available resolution GLB
                    for res in ["1k", "2k", "4k"]:
                        if res in glb_info:
                            download_url = glb_info[res].get("url")
                            break

                if not download_url:
                    logger.error(f"No download URL for: {asset.name}")
                    return False

                # Download the file
                asset.source_url = download_url
                return await self._download_from_url(asset)

    async def _download_from_sketchfab(self, asset: AssetMetadata) -> bool:
        """Download asset from Sketchfab.

        Note: Sketchfab requires authentication for downloads.
        """
        if not self.sketchfab_token:
            logger.warning(
                f"Sketchfab token required for download: {asset.name}. "
                "Set SKETCHFAB_TOKEN environment variable or pass to AssetManager."
            )
            return False

        if not asset.source_url:
            logger.error(f"No Sketchfab URL for: {asset.name}")
            return False

        # Extract model ID from URL
        # Format: https://sketchfab.com/3d-models/helipad-c76632cbfbbc4b4fa236e1a0155ed89a
        parts = asset.source_url.rstrip("/").split("-")
        model_id = parts[-1] if parts else None

        if not model_id or len(model_id) != 32:
            logger.error(f"Could not extract Sketchfab model ID from: {asset.source_url}")
            return False

        api_base = "https://api.sketchfab.com/v3"
        headers = {"Authorization": f"Token {self.sketchfab_token}"}

        async with aiohttp.ClientSession(headers=headers) as session:
            # Get download URL
            async with session.get(f"{api_base}/models/{model_id}/download") as response:
                if response.status == 401:
                    logger.error("Sketchfab authentication failed - check your token")
                    return False
                if response.status == 404:
                    logger.error(f"Model not downloadable or not found: {model_id}")
                    return False
                if response.status != 200:
                    logger.error(f"Sketchfab API error: {response.status}")
                    return False

                download_info = await response.json()

                # Prefer GLB format
                glb_url = download_info.get("glb", {}).get("url")
                if not glb_url:
                    gltf_url = download_info.get("gltf", {}).get("url")
                    if gltf_url:
                        asset.source_url = gltf_url
                        asset.format = AssetFormat.GLTF
                    else:
                        logger.error(f"No GLB/GLTF available for: {model_id}")
                        return False
                else:
                    asset.source_url = glb_url
                    asset.format = AssetFormat.GLB

                # Download the file
                return await self._download_from_url(asset)

    def get_unreal_asset_path(self, asset_id: str) -> str | None:
        """Get the Unreal Engine asset path for a downloaded asset.

        Args:
            asset_id: Asset identifier

        Returns:
            Unreal asset path (e.g., "/Game/AegisAV/Assets/Helipad") or None
        """
        asset = self.assets.get(asset_id)
        if not asset:
            return None

        # If we have a known Unreal path, use it
        if asset.unreal_asset_path:
            return asset.unreal_asset_path

        # Otherwise, return None - asset needs to be imported into UE
        return None

    def set_unreal_asset_path(self, asset_id: str, unreal_path: str) -> bool:
        """Set the Unreal Engine asset path after importing.

        Args:
            asset_id: Asset identifier
            unreal_path: Unreal asset path (e.g., "/Game/AegisAV/Assets/Helipad")

        Returns:
            True if updated successfully
        """
        asset = self.assets.get(asset_id)
        if not asset:
            return False

        asset.unreal_asset_path = unreal_path
        self._save_metadata()
        return True

    def get_asset_for_spawn(
        self,
        asset_type: AssetType,
        prefer_downloaded: bool = True,
    ) -> tuple[str | None, AssetMetadata | None]:
        """Get an asset suitable for spawning in AirSim.

        Args:
            asset_type: Type of asset needed
            prefer_downloaded: Prefer already-downloaded assets

        Returns:
            Tuple of (unreal_asset_name, metadata) or (None, None)
        """
        candidates = self.list_assets(asset_type=asset_type)

        if not candidates:
            return None, None

        # Sort: downloaded with Unreal path first, then downloaded, then others
        def sort_key(a: AssetMetadata) -> tuple[int, int]:
            has_ue_path = 1 if a.unreal_asset_path else 0
            is_downloaded = 1 if a.downloaded else 0
            return (-has_ue_path, -is_downloaded)

        candidates.sort(key=sort_key)

        best = candidates[0]

        # Return Unreal path if available
        if best.unreal_asset_path:
            return best.unreal_asset_path, best

        # Otherwise, if downloaded, user needs to import into UE first
        if best.downloaded and best.file_path:
            logger.info(
                f"Asset {best.asset_id} downloaded to {best.file_path} "
                "but needs to be imported into Unreal Engine. "
                "After importing, call set_unreal_asset_path() with the UE path."
            )

        return None, best

    async def ensure_asset_available(
        self,
        asset_type: AssetType,
    ) -> tuple[str | None, AssetMetadata | None]:
        """Ensure an asset of the given type is available.

        Downloads if necessary.

        Args:
            asset_type: Type of asset needed

        Returns:
            Tuple of (spawn_name, metadata) - spawn_name may be a primitive fallback
        """
        # Check if we already have a suitable asset with Unreal path
        asset_name, metadata = self.get_asset_for_spawn(asset_type)
        if asset_name:
            return asset_name, metadata

        # Try to download one
        candidates = self.list_assets(asset_type=asset_type, downloaded_only=False)
        for candidate in candidates:
            if not candidate.downloaded:
                result = await self.download_asset(candidate.asset_id)
                if result:
                    # Downloaded but still needs UE import
                    return None, result

        # No assets available - return fallback primitive name
        fallback_primitives = {
            AssetType.HELIPAD: "Cylinder",
            AssetType.LANDING_PAD: "Cylinder",
            AssetType.SOLAR_PANEL: "Cube",
            AssetType.WIND_TURBINE: "Cylinder",
            AssetType.DRONE: "Sphere",
            AssetType.BUILDING: "Cube",
            AssetType.VEHICLE: "Cube",
            AssetType.MARKER: "Sphere",
            AssetType.CUSTOM: "Cube",
        }
        return fallback_primitives.get(asset_type, "Cube"), None


# Convenience function to get global asset manager
_global_manager: AssetManager | None = None


def get_asset_manager(
    cache_dir: Path | None = None,
    sketchfab_token: str | None = None,
) -> AssetManager:
    """Get or create the global asset manager.

    Args:
        cache_dir: Optional cache directory (only used on first call)
        sketchfab_token: Optional Sketchfab API token

    Returns:
        AssetManager instance
    """
    global _global_manager
    if _global_manager is None:
        import os
        token = sketchfab_token or os.environ.get("SKETCHFAB_TOKEN")
        _global_manager = AssetManager(cache_dir=cache_dir, sketchfab_token=token)
    return _global_manager
