#!/usr/bin/env python3
"""Setup and download 3D assets for AegisAV simulation.

This script downloads open-source 3D models and prepares them for use
in the Unreal Engine AirSim environment.

Usage:
    python -m simulation.setup_assets --list
    python -m simulation.setup_assets --download helipad_basic
    python -m simulation.setup_assets --download-all
    python -m simulation.setup_assets --setup-unreal /path/to/UnrealProject
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.asset_manager import (
    AssetManager,
    AssetType,
    get_asset_manager,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def list_assets(manager: AssetManager) -> None:
    """List all registered assets."""
    print("\n=== Registered Assets ===\n")

    # Group by type
    by_type: dict[AssetType, list] = {}
    for asset in manager.assets.values():
        by_type.setdefault(asset.asset_type, []).append(asset)

    for asset_type, assets in sorted(by_type.items(), key=lambda x: x[0].value):
        print(f"\n{asset_type.value.upper()}:")
        print("-" * 40)
        for asset in assets:
            status = "✓ Downloaded" if asset.downloaded else "○ Not downloaded"
            ue_status = f" (UE: {asset.unreal_asset_path})" if asset.unreal_asset_path else ""
            print(f"  {asset.asset_id}")
            print(f"    Name: {asset.name}")
            print(f"    Source: {asset.source}")
            print(f"    License: {asset.license.value}")
            print(f"    Status: {status}{ue_status}")
            if asset.file_path:
                print(f"    File: {asset.file_path}")
            print()


async def download_asset(manager: AssetManager, asset_id: str) -> bool:
    """Download a single asset."""
    print(f"\nDownloading: {asset_id}")

    result = await manager.download_asset(asset_id)
    if result:
        print(f"✓ Downloaded to: {result.file_path}")
        print(f"  Format: {result.format.value}")
        print(f"  Size: {result.file_size_bytes / 1024:.1f} KB")
        return True
    else:
        print(f"✗ Failed to download: {asset_id}")
        return False


async def download_all_assets(manager: AssetManager) -> None:
    """Download all registered assets."""
    print("\n=== Downloading All Assets ===\n")

    success = 0
    failed = 0

    for asset_id in manager.assets:
        result = await download_asset(manager, asset_id)
        if result:
            success += 1
        else:
            failed += 1

    print("\n=== Download Summary ===")
    print(f"  Successful: {success}")
    print(f"  Failed: {failed}")


def setup_unreal_project(manager: AssetManager, project_path: Path) -> None:
    """Copy downloaded assets to Unreal project and generate import instructions.

    Args:
        manager: Asset manager with downloaded assets
        project_path: Path to Unreal Engine project root
    """
    content_dir = project_path / "Content"
    if not content_dir.exists():
        print(f"Error: Content directory not found at {content_dir}")
        print("Make sure you specified the correct Unreal project path.")
        return

    # Create AegisAV assets directory
    aegis_assets_dir = content_dir / "AegisAV" / "Assets"
    aegis_assets_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Setting Up Unreal Project ===")
    print(f"Project: {project_path}")
    print(f"Assets Dir: {aegis_assets_dir}\n")

    copied = 0
    skipped = 0

    for asset in manager.assets.values():
        if not asset.downloaded or not asset.file_path:
            continue

        if not asset.file_path.exists():
            logger.warning(f"Asset file not found: {asset.file_path}")
            continue

        # Create subdirectory based on asset type
        type_dir = aegis_assets_dir / asset.asset_type.value.title().replace("_", "")
        type_dir.mkdir(exist_ok=True)

        # Copy file
        dest_file = type_dir / asset.file_path.name
        if dest_file.exists():
            print(f"  Skipping (exists): {asset.asset_id}")
            skipped += 1
            continue

        shutil.copy2(asset.file_path, dest_file)
        copied += 1

        # Generate expected Unreal path
        rel_path = dest_file.relative_to(content_dir)
        ue_path = "/Game/" + str(rel_path.with_suffix("")).replace("\\", "/")

        print(f"  Copied: {asset.asset_id}")
        print(f"    -> {dest_file}")
        print(f"    Expected UE path: {ue_path}")

        # Update asset metadata with expected UE path
        manager.set_unreal_asset_path(asset.asset_id, ue_path)

    print("\n=== Summary ===")
    print(f"  Copied: {copied}")
    print(f"  Skipped: {skipped}")

    # Generate import instructions
    print("""
=== Next Steps ===

1. Open your Unreal Engine project
2. Navigate to Content/AegisAV/Assets in the Content Browser
3. Right-click and select "Import" or drag-drop the files
4. For GLB/GLTF files, use the Interchange import system:
   - Right-click on the GLB file
   - Select "Scripted Asset Actions" > "Interchange"
5. Configure import settings:
   - Enable "Import Mesh"
   - Set appropriate scale (usually 100 for cm to m conversion)
   - Enable "Import Materials" if desired
6. After import, the assets will be available as Static Meshes

For programmatic spawning, update the asset paths in your code:
   manager.set_unreal_asset_path("helipad_basic", "/Game/AegisAV/Assets/Helipad/helipad")
""")


def register_custom_asset(
    manager: AssetManager,
    asset_id: str,
    name: str,
    asset_type: str,
    url: str,
) -> None:
    """Register a custom asset from URL."""
    try:
        atype = AssetType(asset_type)
    except ValueError:
        print(f"Invalid asset type: {asset_type}")
        print(f"Valid types: {[t.value for t in AssetType]}")
        return

    asset = manager.register_asset(
        asset_id=asset_id,
        name=name,
        asset_type=atype,
        source="url",
        source_url=url,
    )
    print(f"Registered: {asset.asset_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setup and download 3D assets for AegisAV simulation"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered assets",
    )
    parser.add_argument(
        "--download",
        metavar="ASSET_ID",
        help="Download a specific asset",
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all registered assets",
    )
    parser.add_argument(
        "--setup-unreal",
        metavar="PROJECT_PATH",
        help="Copy assets to Unreal project",
    )
    parser.add_argument(
        "--register",
        nargs=4,
        metavar=("ID", "NAME", "TYPE", "URL"),
        help="Register a custom asset from URL",
    )
    parser.add_argument(
        "--cache-dir",
        metavar="DIR",
        help="Asset cache directory (default: ~/.aegisav/assets)",
    )
    parser.add_argument(
        "--sketchfab-token",
        metavar="TOKEN",
        help="Sketchfab API token (or set SKETCHFAB_TOKEN env var)",
    )

    args = parser.parse_args()

    # Get or create asset manager
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    token = args.sketchfab_token or os.environ.get("SKETCHFAB_TOKEN")
    manager = get_asset_manager(cache_dir=cache_dir, sketchfab_token=token)

    if args.list:
        list_assets(manager)
    elif args.download:
        asyncio.run(download_asset(manager, args.download))
    elif args.download_all:
        asyncio.run(download_all_assets(manager))
    elif args.setup_unreal:
        setup_unreal_project(manager, Path(args.setup_unreal))
    elif args.register:
        register_custom_asset(manager, *args.register)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
