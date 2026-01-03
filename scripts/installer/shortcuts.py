"""Cross-platform desktop shortcut creation for AegisAV.

Creates desktop shortcuts and application menu entries on Windows and Linux.
Uses native platform methods without external dependencies.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Platform detection
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"


@dataclass
class ShortcutConfig:
    """Configuration for a desktop shortcut.

    Attributes:
        name: Display name for the shortcut (e.g., "AegisAV Server")
        description: Tooltip/comment description
        target_script: Relative path to the script to run (e.g., "scripts/start_server.bat")
        icon_path: Path to the icon file (.ico for Windows, .png for Linux)
        working_dir: Working directory for the script (defaults to project root)
        arguments: Additional command-line arguments
        terminal: Whether to run in a terminal window
        categories: Desktop categories for Linux .desktop files
    """

    name: str
    description: str
    target_script: str
    icon_path: Path | None = None
    working_dir: Path | None = None
    arguments: str = ""
    terminal: bool = True
    categories: list[str] = field(default_factory=lambda: ["Development", "Simulation"])


class WindowsShortcutCreator:
    """Create Windows .lnk shortcuts using PowerShell."""

    def __init__(self, project_root: Path, log_callback: Callable[[str], None] | None = None):
        """Initialize the Windows shortcut creator.

        Args:
            project_root: Root directory of the AegisAV project
            log_callback: Optional callback for logging messages
        """
        self.project_root = project_root
        self.log = log_callback or (lambda msg: None)

    def get_desktop_path(self) -> Path:
        """Get the Windows Desktop folder path.

        Returns:
            Path to the user's Desktop folder
        """
        # Try USERPROFILE first
        desktop = Path(os.environ.get("USERPROFILE", "")) / "Desktop"
        if desktop.exists():
            return desktop

        # Fallback to OneDrive Desktop if it exists
        onedrive_desktop = Path(os.environ.get("USERPROFILE", "")) / "OneDrive" / "Desktop"
        if onedrive_desktop.exists():
            return onedrive_desktop

        return desktop

    def get_start_menu_path(self) -> Path:
        """Get the Windows Start Menu Programs folder path.

        Returns:
            Path to the user's Start Menu Programs folder
        """
        appdata = os.environ.get("APPDATA", "")
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "AegisAV"

    def create_lnk_shortcut(
        self,
        config: ShortcutConfig,
        destination: Path,
    ) -> bool:
        """Create a Windows .lnk shortcut file using PowerShell.

        Args:
            config: Shortcut configuration
            destination: Directory to create the shortcut in

        Returns:
            True if shortcut was created successfully
        """
        try:
            # Ensure destination directory exists
            destination.mkdir(parents=True, exist_ok=True)

            # Build the shortcut path
            shortcut_path = destination / f"{config.name}.lnk"

            # Get target path
            target_path = self.project_root / config.target_script
            if not target_path.exists():
                self.log(f"    Warning: Target script not found: {target_path}")
                # Create anyway, it might be generated later

            # Get working directory
            working_dir = config.working_dir or self.project_root

            # Build icon location
            icon_location = ""
            if config.icon_path and config.icon_path.exists():
                icon_location = str(config.icon_path)

            # PowerShell script to create shortcut
            # Using COM object WScript.Shell
            ps_script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{target_path}"
$Shortcut.WorkingDirectory = "{working_dir}"
$Shortcut.Description = "{config.description}"
"""
            if config.arguments:
                ps_script += f'$Shortcut.Arguments = "{config.arguments}"\n'

            if icon_location:
                ps_script += f'$Shortcut.IconLocation = "{icon_location}"\n'

            ps_script += "$Shortcut.Save()\n"

            # Execute PowerShell
            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    ps_script,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                self.log(f"    PowerShell error: {result.stderr}")
                return False

            return shortcut_path.exists()

        except subprocess.TimeoutExpired:
            self.log("    Timeout creating shortcut")
            return False
        except Exception as e:
            self.log(f"    Error creating shortcut: {e}")
            return False

    def create_all_shortcuts(
        self,
        shortcuts: list[ShortcutConfig],
        install_desktop: bool = True,
        install_start_menu: bool = True,
    ) -> dict[str, bool]:
        """Create all configured shortcuts.

        Args:
            shortcuts: List of shortcut configurations
            install_desktop: Whether to install to Desktop
            install_start_menu: Whether to install to Start Menu

        Returns:
            Dict mapping shortcut names to success status
        """
        results: dict[str, bool] = {}

        for config in shortcuts:
            success = True

            if install_desktop:
                desktop_result = self.create_lnk_shortcut(config, self.get_desktop_path())
                if not desktop_result:
                    self.log(f"    Failed to create desktop shortcut: {config.name}")
                    success = False

            if install_start_menu:
                start_menu_result = self.create_lnk_shortcut(config, self.get_start_menu_path())
                if not start_menu_result:
                    self.log(f"    Failed to create Start Menu shortcut: {config.name}")
                    success = False

            results[config.name] = success

        return results


class LinuxShortcutCreator:
    """Create Linux .desktop files following freedesktop.org specification."""

    def __init__(self, project_root: Path, log_callback: Callable[[str], None] | None = None):
        """Initialize the Linux shortcut creator.

        Args:
            project_root: Root directory of the AegisAV project
            log_callback: Optional callback for logging messages
        """
        self.project_root = project_root
        self.log = log_callback or (lambda msg: None)

    def get_applications_dir(self) -> Path:
        """Get the user's applications directory for .desktop files.

        Returns:
            Path to ~/.local/share/applications/
        """
        xdg_data_home = os.environ.get("XDG_DATA_HOME", "")
        if xdg_data_home:
            return Path(xdg_data_home) / "applications"
        return Path.home() / ".local" / "share" / "applications"

    def get_desktop_path(self) -> Path:
        """Get the user's Desktop folder path.

        Returns:
            Path to the Desktop folder
        """
        # Check XDG user dirs
        try:
            result = subprocess.run(
                ["xdg-user-dir", "DESKTOP"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return Path(result.stdout.strip())
        except Exception:
            pass

        # Fallback to ~/Desktop
        return Path.home() / "Desktop"

    def create_desktop_file(
        self,
        config: ShortcutConfig,
        destination: Path,
    ) -> bool:
        """Create a .desktop file following freedesktop.org specification.

        Args:
            config: Shortcut configuration
            destination: Directory to create the .desktop file in

        Returns:
            True if file was created successfully
        """
        try:
            # Ensure destination directory exists
            destination.mkdir(parents=True, exist_ok=True)

            # Sanitize name for filename
            safe_name = config.name.lower().replace(" ", "-")
            desktop_path = destination / f"aegisav-{safe_name}.desktop"

            # Get target script path
            target_path = self.project_root / config.target_script

            # Build Exec line
            if config.terminal:
                # Use the script directly, terminal=true handles the terminal
                exec_line = str(target_path)
            else:
                exec_line = str(target_path)

            if config.arguments:
                exec_line += f" {config.arguments}"

            # Build icon path
            icon_path = ""
            if config.icon_path and config.icon_path.exists():
                icon_path = str(config.icon_path)

            # Build categories
            categories = ";".join(config.categories) + ";"

            # Create .desktop file content
            content = f"""[Desktop Entry]
Version=1.1
Type=Application
Name={config.name}
GenericName=AegisAV
Comment={config.description}
Exec={exec_line}
Icon={icon_path}
Terminal={str(config.terminal).lower()}
Categories={categories}
StartupWMClass=AegisAV
StartupNotify=true
"""

            # Write the file
            desktop_path.write_text(content, encoding="utf-8")

            # Make executable
            desktop_path.chmod(0o755)

            # Also make the target script executable if it exists
            if target_path.exists():
                target_path.chmod(0o755)

            return True

        except Exception as e:
            self.log(f"    Error creating .desktop file: {e}")
            return False

    def install_to_desktop(self, config: ShortcutConfig) -> bool:
        """Install a .desktop file to the user's Desktop.

        Args:
            config: Shortcut configuration

        Returns:
            True if installed successfully
        """
        desktop_path = self.get_desktop_path()

        if not desktop_path.exists():
            self.log(f"    Desktop path does not exist: {desktop_path}")
            return False

        return self.create_desktop_file(config, desktop_path)

    def update_desktop_database(self) -> None:
        """Run update-desktop-database if available.

        This updates the desktop file cache so the shortcuts appear
        in application menus.
        """
        try:
            apps_dir = self.get_applications_dir()
            subprocess.run(
                ["update-desktop-database", str(apps_dir)],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            # Not critical if this fails
            pass

    def create_all_shortcuts(
        self,
        shortcuts: list[ShortcutConfig],
        install_desktop: bool = True,
        install_applications: bool = True,
    ) -> dict[str, bool]:
        """Create all configured shortcuts.

        Args:
            shortcuts: List of shortcut configurations
            install_desktop: Whether to install to Desktop
            install_applications: Whether to install to applications menu

        Returns:
            Dict mapping shortcut names to success status
        """
        results: dict[str, bool] = {}

        for config in shortcuts:
            success = True

            if install_desktop:
                desktop_result = self.install_to_desktop(config)
                if not desktop_result:
                    self.log(f"    Failed to create desktop shortcut: {config.name}")
                    success = False

            if install_applications:
                apps_result = self.create_desktop_file(config, self.get_applications_dir())
                if not apps_result:
                    self.log(f"    Failed to create applications entry: {config.name}")
                    success = False

            results[config.name] = success

        if install_applications:
            self.update_desktop_database()

        return results


def get_default_shortcuts(
    project_root: Path,
    icon_path: Path | None = None,
    airsim_enabled: bool = False,
) -> list[ShortcutConfig]:
    """Get the default list of shortcuts to create.

    Args:
        project_root: Root directory of the AegisAV project
        icon_path: Path to the icon file
        airsim_enabled: Whether to include AirSim launcher

    Returns:
        List of ShortcutConfig objects
    """
    # Determine script extension based on platform
    if IS_WINDOWS:
        ext = ".bat"
    else:
        ext = ".sh"

    shortcuts = [
        ShortcutConfig(
            name="AegisAV Server",
            description="Start the AegisAV agent server",
            target_script=f"scripts/start_server{ext}",
            icon_path=icon_path,
            working_dir=project_root,
            terminal=True,
            categories=["Development", "Science"],
        ),
        ShortcutConfig(
            name="AegisAV Dashboard",
            description="Open the AegisAV web dashboard",
            target_script=f"scripts/run_demo{ext}",
            icon_path=icon_path,
            working_dir=project_root,
            terminal=True,
            categories=["Development", "Science"],
        ),
        ShortcutConfig(
            name="AegisAV Demo",
            description="Run the AegisAV demo mode",
            target_script=f"scripts/run_demo{ext}",
            icon_path=icon_path,
            working_dir=project_root,
            terminal=True,
            categories=["Development", "Science"],
        ),
    ]

    if airsim_enabled and IS_WINDOWS:
        shortcuts.append(
            ShortcutConfig(
                name="Launch AirSim",
                description="Launch the AirSim drone simulator",
                target_script="start_airsim.bat",
                icon_path=icon_path,
                working_dir=project_root,
                terminal=False,
                categories=["Game", "Simulation"],
            )
        )

    return shortcuts


def create_shortcuts_for_platform(
    project_root: Path,
    shortcuts_to_create: list[str],
    icon_path: Path | None = None,
    install_desktop: bool = True,
    install_menu: bool = True,
    airsim_enabled: bool = False,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, bool]:
    """Create desktop shortcuts for the current platform.

    This is the main entry point for shortcut creation.

    Args:
        project_root: Root directory of the AegisAV project
        shortcuts_to_create: List of shortcut names to create
            Valid names: "server", "dashboard", "demo", "airsim"
        icon_path: Path to the icon file
        install_desktop: Whether to create desktop shortcuts
        install_menu: Whether to create Start Menu/applications menu entries
        airsim_enabled: Whether AirSim is enabled
        log_callback: Optional callback for logging messages

    Returns:
        Dict mapping shortcut names to success status
    """
    log = log_callback or (lambda msg: None)

    # Get all default shortcuts
    all_shortcuts = get_default_shortcuts(project_root, icon_path, airsim_enabled)

    # Filter to requested shortcuts
    name_map = {
        "server": "AegisAV Server",
        "dashboard": "AegisAV Dashboard",
        "demo": "AegisAV Demo",
        "airsim": "Launch AirSim",
    }

    shortcuts = [
        s
        for s in all_shortcuts
        if any(name_map.get(req, "") == s.name for req in shortcuts_to_create)
    ]

    if not shortcuts:
        log("  No shortcuts to create")
        return {}

    # Create shortcuts based on platform
    if IS_WINDOWS:
        creator = WindowsShortcutCreator(project_root, log)
        return creator.create_all_shortcuts(
            shortcuts,
            install_desktop=install_desktop,
            install_start_menu=install_menu,
        )
    elif IS_LINUX:
        creator = LinuxShortcutCreator(project_root, log)
        return creator.create_all_shortcuts(
            shortcuts,
            install_desktop=install_desktop,
            install_applications=install_menu,
        )
    elif IS_MACOS:
        # macOS not fully supported yet, but could add .app bundle creation
        log("  macOS shortcut creation not yet implemented")
        return {s.name: False for s in shortcuts}
    else:
        log(f"  Unknown platform: {platform.system()}")
        return {s.name: False for s in shortcuts}


def remove_shortcuts(
    shortcuts_to_remove: list[str] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, bool]:
    """Remove previously created shortcuts.

    Args:
        shortcuts_to_remove: List of shortcut names to remove, or None for all
        log_callback: Optional callback for logging messages

    Returns:
        Dict mapping shortcut names to removal success status
    """
    log = log_callback or (lambda msg: None)
    results: dict[str, bool] = {}

    name_map = {
        "server": "AegisAV Server",
        "dashboard": "AegisAV Dashboard",
        "demo": "AegisAV Demo",
        "airsim": "Launch AirSim",
    }

    if shortcuts_to_remove is None:
        shortcuts_to_remove = list(name_map.keys())

    for short_name in shortcuts_to_remove:
        full_name = name_map.get(short_name, short_name)

        try:
            if IS_WINDOWS:
                # Remove from Desktop
                desktop = Path(os.environ.get("USERPROFILE", "")) / "Desktop"
                shortcut = desktop / f"{full_name}.lnk"
                if shortcut.exists():
                    shortcut.unlink()

                # Remove from Start Menu
                appdata = os.environ.get("APPDATA", "")
                start_menu = (
                    Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "AegisAV"
                )
                shortcut = start_menu / f"{full_name}.lnk"
                if shortcut.exists():
                    shortcut.unlink()

                # Remove Start Menu folder if empty
                if start_menu.exists() and not any(start_menu.iterdir()):
                    start_menu.rmdir()

            elif IS_LINUX:
                safe_name = full_name.lower().replace(" ", "-")

                # Remove from Desktop
                desktop = Path.home() / "Desktop"
                desktop_file = desktop / f"aegisav-{safe_name}.desktop"
                if desktop_file.exists():
                    desktop_file.unlink()

                # Remove from applications
                apps_dir = Path.home() / ".local" / "share" / "applications"
                desktop_file = apps_dir / f"aegisav-{safe_name}.desktop"
                if desktop_file.exists():
                    desktop_file.unlink()

            results[short_name] = True
            log(f"  Removed shortcut: {full_name}")

        except Exception as e:
            log(f"  Failed to remove {full_name}: {e}")
            results[short_name] = False

    return results
