#!/usr/bin/env python3
"""
AegisAV Setup Wizard - Cross-Platform GUI Installer

A comprehensive setup wizard that guides users through installing and configuring
AegisAV with Unreal Engine and AirSim integration.

Usage:
    python setup_gui.py

Or on Windows:
    python setup_gui.py
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import socket
import subprocess  # noqa: S404
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from datetime import datetime
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Check for tkinter availability
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except ImportError:
    error_lines = [
        "Error: tkinter is required but not installed.",
        "",
        "To install tkinter:",
    ]
    if platform.system() == "Linux":
        error_lines.extend(
            [
                "  Ubuntu/Debian: sudo apt-get install python3-tk",
                "  Fedora: sudo dnf install python3-tkinter",
                "  Arch: sudo pacman -S tk",
            ]
        )
    elif platform.system() == "Darwin":
        error_lines.append("  macOS: brew install python-tk")
    else:
        error_lines.append("  Windows: Reinstall Python with tcl/tk option checked")
    sys.stderr.write("\n".join(error_lines) + "\n")
    sys.exit(1)


# =============================================================================
# Constants and Configuration
# =============================================================================

APP_NAME = "AegisAV Setup Wizard"
APP_VERSION = "1.0.0"

# Detect project root (assuming this script is in scripts/installer/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Platform detection
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MAC = platform.system() == "Darwin"

if IS_WINDOWS:
    local_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    LOG_DIR = local_root / "AegisAV" / "logs"
else:
    LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_PRIMARY = LOG_DIR / "setup_wizard.log"
LOG_FILE_FALLBACK = Path(tempfile.gettempdir()) / "aegis_setup.log"
LOG_FILE_ACTIVE = LOG_FILE_PRIMARY

# URLs
URLS = {
    "unreal_engine": "https://www.unrealengine.com/download",
    "epic_games": "https://store.epicgames.com/download",
    "airsim_releases": "https://github.com/microsoft/AirSim/releases",
    "airsim_docs": "https://microsoft.github.io/AirSim/",
    "airsim_ue5": "https://github.com/microsoft/AirSim/blob/main/docs/unreal_upgrade.md",
    "python": "https://www.python.org/downloads/",
    "nodejs": "https://nodejs.org/",
    "git": "https://git-scm.com/downloads",
    "redis_windows": "https://github.com/microsoftarchive/redis/releases",
    "cuda": "https://developer.nvidia.com/cuda-downloads",
    "vs_buildtools": "https://visualstudio.microsoft.com/visual-cpp-build-tools/",
}


class StepStatus(Enum):
    """Status of a setup step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class Dependency:
    """Represents a software dependency."""

    name: str
    check_cmd: list[str]
    version_pattern: str | None = None
    required: bool = True
    install_url: str | None = None
    install_cmd_linux: str | None = None
    install_cmd_windows: str | None = None
    min_version: str | None = None
    found: bool = False
    version: str = ""
    path: str = ""


def default_server_port() -> int:
    """Choose a default server port (prefer config file)."""
    default_port = 8080
    config_path = PROJECT_ROOT / "configs" / "aegis_config.yaml"
    if config_path.exists():
        try:
            import yaml

            with open(config_path, encoding="utf-8") as file_handle:
                data = yaml.safe_load(file_handle) or {}
            if isinstance(data, dict):
                server_block = data.get("server")
                if isinstance(server_block, dict):
                    port = server_block.get("port")
                    if isinstance(port, int) and 1 <= port <= 65535:
                        return port
                    if isinstance(port, str) and port.isdigit():
                        parsed = int(port)
                        if 1 <= parsed <= 65535:
                            return parsed
        except Exception:
            return default_port
    return default_port


def is_port_available(port: int) -> bool:
    """Check if a TCP port is available on localhost."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def pick_available_port(preferred: int) -> int:
    """Pick the first available port from a preferred list."""
    candidates = [preferred, 8000, 8080, 8090, 8100, 8181, 8200, 8500, 8765, 9000]
    seen: set[int] = set()
    for port in candidates:
        if port in seen:
            continue
        seen.add(port)
        if 1 <= port <= 65535 and is_port_available(port):
            return port
    return preferred


@dataclass
class SetupConfig:
    """Configuration gathered during setup."""

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    unreal_engine_path: Path | None = None
    airsim_path: Path | None = None
    ardupilot_path: Path | None = None
    python_path: Path | None = None
    uv_root: Path | None = None
    server_port: int = field(default_factory=default_server_port)
    use_redis: bool = False
    use_gpu: bool = False
    create_shortcuts: bool = True
    install_frontend: bool = True
    simulation_enabled: bool = True
    airsim_enabled: bool = True
    sitl_enabled: bool = True
    build_airsim_overlay: bool = False  # Build AirSim from source with AegisAV Overlay plugin

    def to_dict(self) -> dict:
        return {
            "project_root": str(self.project_root),
            "unreal_engine_path": str(self.unreal_engine_path) if self.unreal_engine_path else None,
            "airsim_path": str(self.airsim_path) if self.airsim_path else None,
            "ardupilot_path": str(self.ardupilot_path) if self.ardupilot_path else None,
            "python_path": str(self.python_path) if self.python_path else None,
            "uv_root": str(self.uv_root) if self.uv_root else None,
            "server_port": self.server_port,
            "use_redis": self.use_redis,
            "use_gpu": self.use_gpu,
            "create_shortcuts": self.create_shortcuts,
            "install_frontend": self.install_frontend,
            "simulation_enabled": self.simulation_enabled,
            "airsim_enabled": self.airsim_enabled,
            "sitl_enabled": self.sitl_enabled,
            "build_airsim_overlay": self.build_airsim_overlay,
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Utility Functions
# =============================================================================


def run_command(
    cmd: list[str], cwd: Path | None = None, timeout: int = 300
) -> tuple[bool, str, str]:
    """Run a command and return (success, stdout, stderr)."""
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except FileNotFoundError:
        return False, "", f"Command not found: {cmd[0]}"
    except Exception as exc:
        return False, "", str(exc)


def run_command_stream(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int = 300,
    on_output: Callable[[str], None] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[bool, str, int | None]:
    """Run a command and stream output line-by-line."""
    output_lines: list[str] = []
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        if process.stdout:
            for line in process.stdout:
                line = line.rstrip("\n")
                output_lines.append(line)
                if on_output:
                    on_output(line)
        process.wait(timeout=timeout)
        return process.returncode == 0, "\n".join(output_lines), process.returncode
    except subprocess.TimeoutExpired:
        if process is not None:
            process.kill()
        return False, "Command timed out", None
    except Exception as e:
        return False, str(e), None

def is_version_at_least(version: tuple[int, int, int], minimum: str) -> bool:
    """Compare a version tuple to a minimum version string."""
    parts = []
    for item in minimum.split("."):
        try:
            parts.append(int(item))
        except ValueError:
            parts.append(0)
    return version[:len(parts)] >= tuple(parts)


def find_uv_executable() -> str | None:
    """Locate uv executable, including common install locations."""
    exists, path = check_command_exists("uv")
    if exists:
        return path

    candidates = get_uv_candidate_paths()
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)

    if IS_WINDOWS:
        # Best-effort search under common roots
        local_appdata = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        search_roots = [
            local_appdata / "Programs",
            Path(os.environ.get("USERPROFILE", Path.home())),
        ]
        seen = 0
        for root in search_roots:
            if not root.exists():
                continue
            for candidate in root.rglob("uv.exe"):
                seen += 1
                if candidate.exists():
                    return str(candidate)
                if seen > 200:
                    break
    return None


def get_uv_candidate_paths() -> list[Path]:
    """Return common install paths for uv."""
    candidates: list[Path] = []
    if IS_WINDOWS:
        local_appdata = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        roaming_appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        user_profile = Path(os.environ.get("USERPROFILE", Path.home()))
        program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        candidates.extend([
            local_appdata / "Programs" / "uv" / "uv.exe",
            local_appdata / "Programs" / "uv" / "bin" / "uv.exe",
            roaming_appdata / "uv" / "uv.exe",
            roaming_appdata / "uv" / "bin" / "uv.exe",
            user_profile / ".cargo" / "bin" / "uv.exe",
            user_profile / ".local" / "bin" / "uv.exe",
            program_files / "uv" / "uv.exe",
            program_files_x86 / "uv" / "uv.exe",
        ])
    else:
        home = Path.home()
        candidates.extend([
            home / ".local" / "bin" / "uv",
            home / ".cargo" / "bin" / "uv",
            Path("/usr/local/bin/uv"),
            Path("/usr/bin/uv"),
        ])
    return candidates


def log_uv_candidates() -> None:
    """Log uv candidate paths and whether they exist."""
    append_log("Checking uv candidate paths:")
    for candidate in get_uv_candidate_paths():
        append_log(f"  {candidate} exists={candidate.exists()}")


def get_uv_python_path(uv_root: Path | None = None) -> Path | None:
    """Return the uv-managed Python interpreter if present."""
    if IS_WINDOWS:
        local_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        base_root = uv_root or (local_root / "AegisAV")
        candidate = base_root / "venvs" / "aegisav" / "Scripts" / "python.exe"
    else:
        home = Path.home()
        base_root = uv_root or (home / ".local" / "share" / "aegisav")
        candidate = base_root / "venvs" / "aegisav" / "bin" / "python"

    return candidate if candidate.exists() else None


def detect_unreal_engine_path() -> Path | None:
    """Detect a local Unreal Engine install path."""
    if not IS_WINDOWS:
        return None

    roots = [
        Path("C:/Program Files/Epic Games"),
        Path("D:/Program Files/Epic Games"),
        Path("C:/Program Files (x86)/Epic Games"),
        Path("D:/Program Files (x86)/Epic Games"),
        Path("D:/game_pass_games"),
        Path("C:/game_pass_games"),
    ]

    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("UE_"):
                candidates.append(child)

    if not candidates:
        return None

    def version_key(path: Path) -> tuple[int, int, int]:
        name = path.name
        if "_" not in name:
            return (0, 0, 0)
        version_str = name.split("_", 1)[1]
        parts = []
        for item in version_str.split("."):
            if item.isdigit():
                parts.append(int(item))
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])

    return sorted(candidates, key=version_key, reverse=True)[0]


def detect_airsimnh_path() -> Path | None:
    """Detect AirSimNH installation path."""
    if not IS_WINDOWS:
        return None

    user_profile = Path(os.environ.get("USERPROFILE", Path.home()))
    candidates = [
        user_profile / "AirSim" / "AirSimNH",
        user_profile / "Desktop" / "AirSimNH",
        Path("C:/AirSim/AirSimNH"),
        Path("D:/AirSim/AirSimNH"),
    ]

    for base in candidates:
        if (base / "AirSimNH.exe").exists():
            return base
    return None


def detect_ardupilot_path() -> Path | None:
    """Detect ArduPilot installation path."""
    candidates: list[Path] = []
    project_candidate = PROJECT_ROOT.parent / "ardupilot"
    candidates.append(project_candidate)

    if IS_WINDOWS:
        user_profile = Path(os.environ.get("USERPROFILE", Path.home()))
        candidates.append(user_profile / "ardupilot")
        # If running from WSL UNC, check for sibling in same distro
        project_str = str(PROJECT_ROOT)
        if project_str.startswith("\\\\wsl.localhost\\") or project_str.startswith("\\\\wsl$\\"):
            prefix = "\\\\wsl.localhost\\" if project_str.startswith("\\\\wsl.localhost\\") else "\\\\wsl$\\"
            remainder = project_str[len(prefix):]
            parts = remainder.split("\\", 1)
            if len(parts) == 2:
                distro = parts[0]
                candidates.append(Path(f"{prefix}{distro}\\home\\devcontainers\\ardupilot"))

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def wait_for_uv(timeout_seconds: int = 30) -> str | None:
    """Poll for uv to appear after installation."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        uv_path = find_uv_executable()
        if uv_path:
            return uv_path
        time.sleep(1)
    return None

def check_command_exists(cmd: str) -> tuple[bool, str]:
    """Check if a command exists and return its path."""
    where_cmd = ["where", cmd] if IS_WINDOWS else ["which", cmd]

    success, stdout, _ = run_command(where_cmd)
    if success and stdout.strip():
        return True, stdout.strip().split("\n")[0]
    return False, ""


def _yaml_format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f"\"{text}\""


def _update_yaml_section(text: str, section: str, updates: dict[str, object]) -> str:
    lines = text.splitlines()
    header = f"{section}:"
    section_start = None
    for idx, line in enumerate(lines):
        if line.strip() == header and line.startswith(header):
            section_start = idx
            break

    if section_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(header)
        for key, value in updates.items():
            lines.append(f"  {key}: {_yaml_format_value(value)}")
        return "\n".join(lines) + "\n"

    section_end = len(lines)
    for idx in range(section_start + 1, len(lines)):
        line = lines[idx]
        if line and not line.startswith(" ") and line.rstrip().endswith(":"):
            section_end = idx
            break

    existing_keys: set[str] = set()
    for idx in range(section_start + 1, section_end):
        raw = lines[idx]
        stripped = raw.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key = stripped.split(":", 1)[0].strip()
        if key in updates:
            lines[idx] = f"  {key}: {_yaml_format_value(updates[key])}"
            existing_keys.add(key)

    insert_at = section_end
    for key, value in updates.items():
        if key not in existing_keys:
            lines.insert(insert_at, f"  {key}: {_yaml_format_value(value)}")
            insert_at += 1

    return "\n".join(lines) + "\n"


def find_npm_executable() -> str | None:
    """Locate npm executable, including common install locations."""
    exists, path = check_command_exists("npm")
    if exists and path:
        return path

    if IS_WINDOWS:
        local_appdata = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        roaming_appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        user_profile = Path(os.environ.get("USERPROFILE", Path.home()))
        program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        candidates = [
            program_files / "nodejs" / "npm.cmd",
            program_files_x86 / "nodejs" / "npm.cmd",
            user_profile / "AppData" / "Roaming" / "npm" / "npm.cmd",
            roaming_appdata / "npm" / "npm.cmd",
            local_appdata / "Programs" / "nodejs" / "npm.cmd",
        ]
    else:
        candidates = [
            Path("/usr/local/bin/npm"),
            Path("/usr/bin/npm"),
            Path.home() / ".local" / "bin" / "npm",
        ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def get_command_version(cmd: list[str]) -> str:
    """Get version string from a command."""
    success, stdout, stderr = run_command(cmd)
    if success:
        return stdout.strip() or stderr.strip()
    return ""


def open_url(url: str) -> None:
    """Open a URL in the default browser."""
    webbrowser.open(url)


def open_path(path: Path) -> None:
    """Open a file or folder in the default OS handler."""
    if IS_WINDOWS:
        os.startfile(str(path))
        return
    if IS_MAC:
        subprocess.Popen(["open", str(path)])
        return
    subprocess.Popen(["xdg-open", str(path)])


def append_log(text: str) -> None:
    """Append a line to the installer log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = text.splitlines() or [text]
    targets = [LOG_FILE_ACTIVE]
    if LOG_FILE_FALLBACK not in targets:
        targets.append(LOG_FILE_FALLBACK)

    for path in targets:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as log_file:
                for line in lines:
                    log_file.write(f"[{timestamp}] {line}\n")
            globals()["LOG_FILE_ACTIVE"] = path
            return
        except OSError:
            continue


def init_log() -> None:
    """Create the log file early so it exists even before UI logs."""
    if not LOG_FILE_ACTIVE.exists():
        append_log("Setup wizard log started.")


# =============================================================================
# Dependency Definitions
# =============================================================================


def get_dependencies() -> list[Dependency]:
    """Get list of all dependencies to check."""
    return [
        Dependency(
            name="Python",
            check_cmd=[sys.executable, "--version"],
            required=True,
            min_version="3.10",
            install_url=URLS["python"],
        ),
        Dependency(
            name="pip",
            check_cmd=[sys.executable, "-m", "pip", "--version"],
            required=True,
            install_url=URLS["python"],
        ),
        Dependency(
            name="Git",
            check_cmd=["git", "--version"],
            required=True,
            install_url=URLS["git"],
            install_cmd_linux="sudo apt-get install -y git || sudo dnf install -y git",
        ),
        Dependency(
            name="Node.js",
            check_cmd=["node", "--version"],
            required=False,
            install_url=URLS["nodejs"],
            install_cmd_linux="curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt-get install -y nodejs",
        ),
        Dependency(
            name="npm",
            check_cmd=["npm", "--version"],
            required=False,
            install_url=URLS["nodejs"],
        ),
        Dependency(
            name="uv (Python package manager)",
            check_cmd=["uv", "--version"],
            required=False,
            install_cmd_linux="curl -LsSf https://astral.sh/uv/install.sh | sh",
            install_cmd_windows="powershell -NoProfile -ExecutionPolicy Bypass -Command \"irm https://astral.sh/uv/install.ps1 | iex\"",
        ),
    ]


# =============================================================================
# GUI Components
# =============================================================================


class StepFrame(ttk.Frame):
    """Base class for wizard step frames."""

    def __init__(self, parent: tk.Widget, wizard: SetupWizard) -> None:
        super().__init__(parent)
        self.wizard = wizard
        self.config = wizard.config

    def on_enter(self) -> None:
        """Called when this step becomes active."""
        pass

    def on_leave(self) -> bool:
        """Called when leaving this step. Return False to prevent navigation."""
        return True

    def can_proceed(self) -> bool:
        """Return True if user can proceed to next step."""
        return True

    def get_title(self) -> str:
        """Return the step title."""
        return "Step"


class WelcomeStep(StepFrame):
    """Welcome screen with overview."""

    def __init__(self, parent: tk.Widget, wizard: SetupWizard) -> None:
        super().__init__(parent, wizard)
        self._build_ui()

    def _build_ui(self) -> None:
        # Title
        title = ttk.Label(
            self,
            text="Welcome to AegisAV Setup",
            font=("Helvetica", 24, "bold"),
        )
        title.pack(pady=(40, 20))

        # Subtitle
        subtitle = ttk.Label(
            self,
            text="Autonomous Aerial Vehicle Decision System",
            font=("Helvetica", 12),
        )
        subtitle.pack(pady=(0, 30))

        # Description
        desc_frame = ttk.Frame(self)
        desc_frame.pack(fill="x", padx=50, pady=20)

        description = """This wizard will help you set up AegisAV with full Unreal Engine
and AirSim integration for realistic drone simulation.

What we'll set up:

  ✓ Check system requirements and dependencies
  ✓ Install Python packages and frontend
  ✓ Configure Unreal Engine 5 (if available)
  ✓ Set up AirSim for drone simulation
  ✓ Configure the AegisAV agent server
  ✓ Test the installation

The setup process takes approximately 15-30 minutes depending on
what components need to be installed."""

        desc_label = ttk.Label(
            desc_frame,
            text=description,
            justify="left",
            font=("Helvetica", 11),
        )
        desc_label.pack(anchor="w")

        # System info
        info_frame = ttk.LabelFrame(self, text="System Information", padding=10)
        info_frame.pack(fill="x", padx=50, pady=20)

        sys_info = f"""Operating System: {platform.system()} {platform.release()}
Architecture: {platform.machine()}
Python Version: {platform.python_version()}
Project Location: {PROJECT_ROOT}"""

        info_label = ttk.Label(info_frame, text=sys_info, font=("Courier", 10))
        info_label.pack(anchor="w")

    def get_title(self) -> str:
        return "Welcome"


class DependencyStep(StepFrame):
    """Check and install dependencies."""

    def __init__(self, parent: tk.Widget, wizard: SetupWizard) -> None:
        super().__init__(parent, wizard)
        self.dependencies = get_dependencies()
        self.check_complete = False
        self._build_ui()

    def _build_ui(self) -> None:
        # Title
        title = ttk.Label(
            self,
            text="System Requirements",
            font=("Helvetica", 18, "bold"),
        )
        title.pack(pady=(20, 10))

        subtitle = ttk.Label(
            self,
            text="Checking for required software...",
            font=("Helvetica", 11),
        )
        subtitle.pack(pady=(0, 20))

        # Dependencies list
        self.deps_frame = ttk.Frame(self)
        self.deps_frame.pack(fill="both", expand=True, padx=50, pady=10)

        # Create labels for each dependency
        self.dep_labels: dict[str, dict] = {}

        for dep in self.dependencies:
            row_frame = ttk.Frame(self.deps_frame)
            row_frame.pack(fill="x", pady=5)

            # Status indicator
            status_label = ttk.Label(row_frame, text="⏳", width=3, font=("Helvetica", 14))
            status_label.pack(side="left")

            # Name
            name_label = ttk.Label(
                row_frame,
                text=dep.name,
                font=("Helvetica", 11, "bold"),
                width=25,
                anchor="w",
            )
            name_label.pack(side="left", padx=(10, 0))

            # Required indicator
            req_text = "Required" if dep.required else "Optional"
            req_label = ttk.Label(
                row_frame,
                text=f"({req_text})",
                font=("Helvetica", 9),
                foreground="gray",
            )
            req_label.pack(side="left", padx=(5, 0))

            # Version/status
            version_label = ttk.Label(
                row_frame,
                text="Checking...",
                font=("Helvetica", 10),
                foreground="gray",
            )
            version_label.pack(side="right", padx=(0, 10))

            # Install button (hidden initially)
            install_btn = ttk.Button(
                row_frame,
                text="Install",
                width=10,
                command=lambda d=dep: self._install_dependency(d),
            )

            self.dep_labels[dep.name] = {
                "status": status_label,
                "version": version_label,
                "install_btn": install_btn,
                "dependency": dep,
            }

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(
            self,
            variable=self.progress_var,
            maximum=len(self.dependencies),
        )
        self.progress.pack(fill="x", padx=50, pady=20)

        # Action buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=50, pady=10)

        self.recheck_btn = ttk.Button(
            btn_frame,
            text="Re-check All",
            command=self._start_check,
        )
        self.recheck_btn.pack(side="left")

        self.install_all_btn = ttk.Button(
            btn_frame,
            text="Install Missing (Optional)",
            command=self._install_all_missing,
            state="disabled",
        )
        self.install_all_btn.pack(side="left", padx=10)

    def on_enter(self) -> None:
        if not self.check_complete:
            self._start_check()

    def _start_check(self) -> None:
        """Start checking dependencies in background thread."""
        self.check_complete = False
        self.progress_var.set(0)

        # Reset all labels
        for labels in self.dep_labels.values():
            labels["status"].config(text="⏳")
            labels["version"].config(text="Checking...", foreground="gray")
            labels["install_btn"].pack_forget()

        # Run check in background
        thread = threading.Thread(target=self._check_dependencies, daemon=True)
        thread.start()

    def _check_dependencies(self) -> None:
        """Check all dependencies (runs in background thread)."""
        for i, dep in enumerate(self.dependencies):
            self._check_single_dependency(dep)
            self.progress_var.set(i + 1)

        self.check_complete = True

        # Enable install all button if there are missing optional deps
        missing = [d for d in self.dependencies if not d.found and not d.required]
        if missing:
            self.install_all_btn.config(state="normal")

    def _check_single_dependency(self, dep: Dependency) -> None:
        """Check a single dependency."""
        labels = self.dep_labels[dep.name]

        if dep.name == "Python":
            current_version = platform.python_version()
            if dep.min_version and not is_version_at_least(sys.version_info[:3], dep.min_version):
                dep.found = False
                dep.version = f"{current_version} (need {dep.min_version}+)"
            else:
                dep.found = True
                dep.version = current_version
            self.after(0, lambda: self._update_dep_ui(dep, labels, dep.found))
            return

        if dep.name.startswith("uv"):
            log_uv_candidates()
            uv_path = find_uv_executable()
            if uv_path:
                uv_dir = str(Path(uv_path).parent)
                os.environ["PATH"] = f"{uv_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                success, stdout, stderr = run_command([uv_path, "--version"])
                if success:
                    dep.found = True
                    dep.version = (stdout.strip() or stderr.strip())[:50]
                else:
                    dep.found = False
                    dep.version = "Found but failed to run"
            else:
                dep.found = False
                dep.version = "Not found"
            self.after(0, lambda: self._update_dep_ui(dep, labels, dep.found))
            return

        if dep.name == "npm":
            npm_path = find_npm_executable()
            if npm_path:
                npm_parent = Path(npm_path).parent
                if npm_parent.is_absolute():
                    os.environ["PATH"] = f"{npm_parent}{os.pathsep}{os.environ.get('PATH', '')}"
                cmd = ["cmd", "/c", npm_path, "--version"] if IS_WINDOWS else [npm_path, "--version"]
                success, stdout, stderr = run_command(cmd)
                if success:
                    dep.found = True
                    dep.version = (stdout.strip() or stderr.strip())[:50]
                else:
                    dep.found = False
                    dep.version = "Found but failed to run"
            else:
                dep.found = False
                dep.version = "Not found"
            self.after(0, lambda: self._update_dep_ui(dep, labels, dep.found))
            return

        # Check if command exists
        success, stdout, stderr = run_command(dep.check_cmd)

        if success:
            dep.found = True
            dep.version = (stdout.strip() or stderr.strip())[:50]  # Truncate long output

            # Update UI (must be done in main thread)
            self.after(0, lambda: self._update_dep_ui(dep, labels, True))
        else:
            dep.found = False
            self.after(0, lambda: self._update_dep_ui(dep, labels, False))

    def _update_dep_ui(self, dep: Dependency, labels: dict, found: bool) -> None:
        """Update dependency UI after check."""
        if found:
            labels["status"].config(text="✅")
            labels["version"].config(text=dep.version or "Found", foreground="green")
        else:
            if dep.required:
                labels["status"].config(text="❌")
                labels["version"].config(
                    text=dep.version or "Missing (Required!)",
                    foreground="red",
                )
            else:
                labels["status"].config(text="⚠️")
                labels["version"].config(text=dep.version or "Not found", foreground="orange")

            # Show install button if we have install info
            if (
                dep.install_url
                or (IS_LINUX and dep.install_cmd_linux)
                or (IS_WINDOWS and dep.install_cmd_windows)
            ):
                labels["install_btn"].pack(side="right")

    def _install_dependency(self, dep: Dependency) -> None:
        """Handle install button click."""
        append_log(f"Dependency install requested: {dep.name}")
        if dep.install_url:
            if messagebox.askyesno(
                "Install Dependency", f"Open download page for {dep.name}?\n\n{dep.install_url}"
            ):
                append_log(f"Opening download page for {dep.name}: {dep.install_url}")
                open_url(dep.install_url)
        elif IS_LINUX and dep.install_cmd_linux:
            if messagebox.askyesno(
                "Install Dependency", f"Run the following command?\n\n{dep.install_cmd_linux}"
            ):
                append_log(f"Running install command for {dep.name} (linux)")
                # Run in terminal
                subprocess.Popen(  # noqa: S603
                    [  # noqa: S607
                        "x-terminal-emulator",
                        "-e",
                        f"bash -c '{dep.install_cmd_linux}; read -p \"Press Enter...\"'",
                    ],
                    start_new_session=True,
                )
        elif IS_WINDOWS and dep.install_cmd_windows:
            if messagebox.askyesno(
                "Install Dependency",
                f"Run the following command?\n\n{dep.install_cmd_windows}"
            ):
                append_log(f"Running install command for {dep.name} (windows)")
                thread = threading.Thread(
                    target=self._run_windows_install,
                    args=(dep,),
                    daemon=True,
                )
                thread.start()
                messagebox.showinfo(
                    "Install Started",
                    f"Installing {dep.name}...\n\nLogs: {LOG_FILE_ACTIVE}",
                )
        else:
            messagebox.showinfo(
                "Install Dependency",
                f"No installer available for {dep.name}.",
            )

    def _run_windows_install(self, dep: Dependency) -> None:
        """Run a Windows install command in the background."""
        append_log(f"Dependency install started: {dep.name}")
        if dep.name.startswith("uv"):
            cmd = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "$ErrorActionPreference='Stop'; try { irm https://astral.sh/uv/install.ps1 | iex } catch { $_ | Out-String; exit 1 }",
            ]
        else:
            cmd = ["cmd", "/c", dep.install_cmd_windows] if dep.install_cmd_windows else []
        append_log(f"Running: {' '.join(cmd)}")
        success, output, exit_code = run_command_stream(
            cmd,
            timeout=600,
            on_output=append_log,
        )
        if not output.strip():
            append_log("Installer produced no output.")
        append_log(f"Installer exit code: {exit_code}")
        if dep.name.startswith("uv"):
            uv_path = None
            if success:
                append_log("Waiting for uv to appear on disk...")
                uv_path = wait_for_uv()
            if not uv_path:
                winget_exists, _ = check_command_exists("winget")
                if winget_exists:
                    append_log("uv not found; attempting winget install.")
                    winget_cmd = [
                        "winget",
                        "install",
                        "--id",
                        "Astral.uv",
                        "--exact",
                        "--silent",
                        "--accept-package-agreements",
                        "--accept-source-agreements",
                    ]
                    append_log(f"Running: {' '.join(winget_cmd)}")
                    winget_success, winget_output, winget_exit = run_command_stream(
                        winget_cmd,
                        timeout=900,
                        on_output=append_log,
                    )
                    if not winget_output.strip():
                        append_log("winget produced no output.")
                    append_log(f"winget exit code: {winget_exit}")
                    if winget_success:
                        append_log("winget install completed; rechecking for uv.")
                        uv_path = wait_for_uv(timeout_seconds=60)
                else:
                    append_log("winget not available; cannot attempt winget install.")
            if uv_path:
                append_log(f"uv installed at: {uv_path}")
                uv_dir = str(Path(uv_path).parent)
                os.environ["PATH"] = f"{uv_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                append_log(f"Added uv to PATH: {uv_dir}")
                success = True
            else:
                append_log("uv install did not produce an executable in known paths.")
                success = False
                self.after(
                    0,
                    lambda: messagebox.showwarning(
                        "uv Not Found",
                        "uv install completed but no uv.exe was found.\n\n"
                        f"Log: {LOG_FILE_ACTIVE}\n\n"
                        "This may be blocked by policy. Try installing with winget or "
                        "run a new PowerShell and check `where uv`.",
                    ),
                )

        if success:
            append_log(f"Dependency install succeeded: {dep.name}")
            if dep.name.startswith("uv"):
                log_uv_candidates()
                uv_path = find_uv_executable()
                if uv_path:
                    uv_dir = str(Path(uv_path).parent)
                    os.environ["PATH"] = f"{uv_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                    append_log(f"Added uv to PATH: {uv_dir}")
        else:
            append_log(f"Dependency install failed: {dep.name}")
        self.after(0, lambda: self._check_single_dependency(dep))

    def _install_all_missing(self) -> None:
        """Install all missing optional dependencies."""
        missing = [d for d in self.dependencies if not d.found]
        if not missing:
            messagebox.showinfo("Info", "All dependencies are already installed!")
            return

        msg = "The following will be installed:\n\n"
        msg += "\n".join(f"  • {d.name}" for d in missing)

        if messagebox.askyesno("Install Dependencies", msg):
            for dep in missing:
                self._install_dependency(dep)

    def can_proceed(self) -> bool:
        if not self.check_complete:
            messagebox.showwarning("Please Wait", "Dependency check is still in progress.")
            return False

        # Check required dependencies
        missing_required = [d for d in self.dependencies if d.required and not d.found]
        if missing_required:
            msg = "The following required dependencies are missing:\n\n"
            msg += "\n".join(f"  • {d.name}" for d in missing_required)
            msg += "\n\nPlease install them before continuing."
            messagebox.showerror("Missing Dependencies", msg)
            return False

        return True

    def get_title(self) -> str:
        return "Dependencies"


class PythonSetupStep(StepFrame):
    """Install Python dependencies."""

    def __init__(self, parent: tk.Widget, wizard: SetupWizard) -> None:
        super().__init__(parent, wizard)
        self.install_complete = False
        self.install_failed = False
        self.uv_root_var = tk.StringVar()
        self.uv_path: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        # Title
        title = ttk.Label(
            self,
            text="Python Environment Setup",
            font=("Helvetica", 18, "bold"),
        )
        title.pack(pady=(20, 10))

        # Description
        desc = ttk.Label(
            self,
            text="Install Python packages required for AegisAV",
            font=("Helvetica", 11),
        )
        desc.pack(pady=(0, 20))

        # Options
        options_frame = ttk.LabelFrame(self, text="Installation Options", padding=15)
        options_frame.pack(fill="x", padx=50, pady=10)

        self.use_uv_var = tk.BooleanVar(value=True)
        uv_check = ttk.Checkbutton(
            options_frame,
            text="Use uv for fast installation (recommended)",
            variable=self.use_uv_var,
        )
        uv_check.pack(anchor="w")

        self.install_dev_var = tk.BooleanVar(value=True)
        dev_check = ttk.Checkbutton(
            options_frame,
            text="Install development dependencies (testing, linting)",
            variable=self.install_dev_var,
        )
        dev_check.pack(anchor="w", pady=(5, 0))

        if IS_WINDOWS:
            storage_frame = ttk.LabelFrame(self, text="Storage Location", padding=15)
            storage_frame.pack(fill="x", padx=50, pady=10)

            local_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            default_root = local_root / "AegisAV"
            self.uv_root_var.set(str(default_root))

            ttk.Label(
                storage_frame,
                text="UV cache and virtualenv location (choose a disk with free space):",
                font=("Helvetica", 10),
            ).pack(anchor="w")

            path_frame = ttk.Frame(storage_frame)
            path_frame.pack(fill="x", pady=(5, 0))

            path_entry = ttk.Entry(path_frame, textvariable=self.uv_root_var)
            path_entry.pack(side="left", fill="x", expand=True)

            browse_btn = ttk.Button(
                path_frame,
                text="Browse...",
                command=self._browse_uv_root,
            )
            browse_btn.pack(side="left", padx=(10, 0))

        # Console output
        console_frame = ttk.LabelFrame(self, text="Installation Output", padding=10)
        console_frame.pack(fill="both", expand=True, padx=50, pady=10)

        self.console = scrolledtext.ScrolledText(
            console_frame,
            height=15,
            font=("Courier", 9),
            state="disabled",
            bg="#f8fafc",
            fg="#0f172a",
            insertbackground="#0f172a",
            selectbackground="#dbeafe",
            selectforeground="#0f172a",
        )
        self.console.pack(fill="both", expand=True)

        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self, variable=self.progress_var, mode="indeterminate")
        self.progress.pack(fill="x", padx=50, pady=10)

        # Install button
        self.install_btn = ttk.Button(
            self,
            text="Install Python Packages",
            command=self._start_install,
        )
        self.install_btn.pack(pady=10)

        self.status_label = ttk.Label(self, text="Click 'Install' to begin", font=("Helvetica", 10))
        self.status_label.pack()

        self.log_btn = ttk.Button(
            self,
            text="Open Log File",
            command=self._open_log,
        )
        self.log_btn.pack(pady=(5, 0))

    def _log(self, text: str) -> None:
        """Append text to console."""
        append_log(text)
        self.console.config(state="normal")
        self.console.insert("end", text + "\n")
        self.console.see("end")
        self.console.config(state="disabled")

    def _open_log(self) -> None:
        """Open the installer log file."""
        append_log("Opening log file on user request.")
        open_path(LOG_FILE_ACTIVE)

    def _browse_uv_root(self) -> None:
        """Browse for UV storage location."""
        path = filedialog.askdirectory(title="Select UV Storage Location")
        if path:
            self.uv_root_var.set(path)

    def _start_install(self) -> None:
        """Start installation in background."""
        append_log("Python setup: install button clicked.")
        self.install_btn.config(state="disabled")
        self.progress.start(10)
        self.status_label.config(text="Installing...")
        self._log("Starting Python package installation...")

        thread = threading.Thread(target=self._run_install, daemon=True)
        thread.start()

    def _run_install(self) -> None:
        """Run the installation (background thread)."""
        try:
            def log_line(line: str) -> None:
                self.after(0, lambda: self._log(line))

            # Check for uv
            use_uv = self.use_uv_var.get()
            uv_available, _ = check_command_exists("uv")
            self.after(0, lambda: self._log(f"use_uv={use_uv}, uv_available={uv_available}"))

            if use_uv and not uv_available:
                self.after(0, lambda: self._log("Installing uv package manager..."))
                if IS_WINDOWS:
                    cmd = [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        "irm https://astral.sh/uv/install.ps1 | iex",
                    ]
                else:
                    cmd = ["bash", "-lc", "curl -LsSf https://astral.sh/uv/install.sh | sh"]
                self.after(0, lambda: self._log(f"Running: {' '.join(cmd)}"))
                success, output, exit_code = run_command_stream(cmd, timeout=300, on_output=log_line)
                if not output.strip():
                    self.after(0, lambda: self._log("uv installer produced no output."))
                self.after(0, lambda: self._log(f"uv installer exit code: {exit_code}"))
                if not success:
                    self.after(0, lambda: self._log("uv install failed. Falling back to pip."))

            uv_path = find_uv_executable()
            if uv_path:
                uv_dir = str(Path(uv_path).parent)
                os.environ["PATH"] = f"{uv_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                uv_available = True
                self.uv_path = uv_path
                self.after(0, lambda: self._log(f"Found uv at {uv_path}"))
            else:
                uv_available = False
                if use_uv:
                    self.after(0, lambda: self._log("uv still not found; using pip."))
                    use_uv = False

            # Install dependencies
            if use_uv and (uv_available or check_command_exists("uv")[0]):
                self.after(0, lambda: self._log("\n📦 Installing with uv..."))
                cmd = ["uv", "sync", "--preview-features", "extra-build-dependencies"]
                if self.install_dev_var.get():
                    cmd.append("--all-extras")
                env = None
                if IS_WINDOWS:
                    uv_root = self.uv_root_var.get().strip() or str(
                        Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "AegisAV"
                    )
                    uv_root_path = Path(uv_root)
                    self.config.uv_root = uv_root_path
                    uv_env_dir = uv_root_path / "venvs" / "aegisav"
                    uv_cache_dir = uv_root_path / "uv_cache"
                    uv_env_dir.mkdir(parents=True, exist_ok=True)
                    uv_cache_dir.mkdir(parents=True, exist_ok=True)
                    env = os.environ.copy()
                    env["UV_PROJECT_ENVIRONMENT"] = str(uv_env_dir)
                    env["UV_CACHE_DIR"] = str(uv_cache_dir)
                    self.after(0, lambda: self._log(f"Using UV_ROOT={uv_root_path}"))
                    self.after(0, lambda: self._log(f"Using UV_PROJECT_ENVIRONMENT={uv_env_dir}"))
                    self.after(0, lambda: self._log(f"Using UV_CACHE_DIR={uv_cache_dir}"))
            else:
                self.after(0, lambda: self._log("\n📦 Installing with pip..."))
                pip_cmd = [sys.executable, "-m", "pip"]
                cmd = [*pip_cmd, "install", "-e", "."]
                if self.install_dev_var.get():
                    cmd = [*pip_cmd, "install", "-e", ".[dev,test]"]
                env = None

            self.after(0, lambda: self._log(f"Running: {' '.join(cmd)}"))
            success, output, exit_code = run_command_stream(
                cmd,
                cwd=PROJECT_ROOT,
                timeout=600,
                on_output=log_line,
                env=env,
            )
            if not output.strip():
                self.after(0, lambda: self._log("Install command produced no output."))
            self.after(0, lambda: self._log(f"Install command exit code: {exit_code}"))

            if success:
                target_python = sys.executable
                uv_python = get_uv_python_path(self.config.uv_root)
                if use_uv and uv_python:
                    target_python = str(uv_python)
                    self.after(0, lambda: self._log(f"Using uv Python for AirSim: {target_python}"))
                airsim_ok = self._ensure_airsim_package(target_python, log_line)
                if airsim_ok:
                    self.after(0, self._install_success)
                else:
                    self.after(0, lambda: self._install_failed("AirSim package install failed"))
            else:
                if output and ("not enough space" in output.lower() or "os error 112" in output.lower()):
                    self.after(
                        0,
                        lambda: messagebox.showwarning(
                            "Out of Disk Space",
                            "UV ran out of disk space while extracting packages.\n\n"
                            "Choose a different storage location and retry.",
                        ),
                    )
                self.after(0, lambda: self._install_failed("Install command failed"))

        except Exception as exc:
            error = str(exc)
            self.after(0, lambda err=error: self._install_failed(err))

    def _ensure_airsim_package(
        self,
        python_exec: str,
        log_line: Callable[[str], None],
    ) -> bool:
        """Ensure AirSim Python package is installed in the target environment."""
        self.after(0, lambda: self._log("\n🔎 Checking AirSim Python package..."))
        check_cmd = [python_exec, "-c", "import airsim; print('OK')"]
        success, output, exit_code = run_command_stream(
            check_cmd,
            timeout=30,
            on_output=log_line,
        )
        if success and "OK" in output:
            self.after(0, lambda: self._log("✅ AirSim already installed."))
            return True

        self.after(0, lambda: self._log("Installing AirSim Python package..."))
        uv_path = self.uv_path or find_uv_executable()
        if uv_path:
            self.after(0, lambda: self._log(f"Using uv at {uv_path} for AirSim install."))
            install_cmd = [
                uv_path,
                "pip",
                "install",
                "--python",
                python_exec,
                "airsim",
                "msgpack-rpc-python",
                "backports.ssl_match_hostname",
            ]
            self.after(0, lambda: self._log(f"Running: {' '.join(install_cmd)}"))
            install_success, output, exit_code = run_command_stream(
                install_cmd,
                timeout=600,
                on_output=log_line,
            )
        else:
            ensure_cmd = [python_exec, "-m", "ensurepip", "--upgrade"]
            self.after(0, lambda: self._log(f"Running: {' '.join(ensure_cmd)}"))
            run_command_stream(ensure_cmd, timeout=300, on_output=log_line)
            install_cmd = [
                python_exec,
                "-m",
                "pip",
                "install",
                "airsim",
                "msgpack-rpc-python",
                "backports.ssl_match_hostname",
            ]
            self.after(0, lambda: self._log(f"Running: {' '.join(install_cmd)}"))
            install_success, output, exit_code = run_command_stream(
                install_cmd,
                timeout=600,
                on_output=log_line,
            )

        if not output.strip():
            self.after(0, lambda: self._log("AirSim install produced no output."))
        self.after(0, lambda: self._log(f"AirSim install exit code: {exit_code}"))
        if not install_success:
            return False

        success, output, exit_code = run_command_stream(
            check_cmd,
            timeout=30,
            on_output=log_line,
        )
        if success and "OK" in output:
            self.after(0, lambda: self._log("✅ AirSim installed successfully."))
            return True

        self.after(0, lambda: self._log(f"AirSim import failed: {output}"))
        return False

    def _install_success(self) -> None:
        """Handle successful installation."""
        self.progress.stop()
        self.install_complete = True
        self.status_label.config(text="✅ Installation complete!", foreground="green")
        self._log("\n✅ Python packages installed successfully!")
        self.install_btn.config(text="Reinstall", state="normal")

    def _install_failed(self, error: str) -> None:
        """Handle failed installation."""
        self.progress.stop()
        self.status_label.config(text="❌ Installation failed", foreground="red")
        self._log(f"\n❌ Installation failed: {error}")
        self.install_btn.config(state="normal")
        if messagebox.askyesno("Open Log File?", f"Open log file?\n\n{LOG_FILE_ACTIVE}"):
            append_log("Opening log file on user request.")
            open_path(LOG_FILE_ACTIVE)

    def can_proceed(self) -> bool:
        if not self.install_complete:
            if messagebox.askyesno(
                "Skip Installation?",
                "Python packages have not been installed.\n\n"
                "Skip this step? (You can install manually later)",
            ):
                return True
            return False
        return True

    def get_title(self) -> str:
        return "Python Setup"


class UnrealSetupStep(StepFrame):
    """Guide through Unreal Engine setup."""

    def __init__(self, parent: tk.Widget, wizard: SetupWizard) -> None:
        super().__init__(parent, wizard)
        self._build_ui()

    def _build_ui(self) -> None:
        # Title
        title = ttk.Label(
            self,
            text="Unreal Engine Setup",
            font=("Helvetica", 18, "bold"),
        )
        title.pack(pady=(20, 10))

        # Description
        desc = ttk.Label(
            self,
            text="Configure Unreal Engine 5 for drone simulation",
            font=("Helvetica", 11),
        )
        desc.pack(pady=(0, 20))

        # Notebook for different scenarios
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=30, pady=10)

        # Tab 1: Already Installed
        installed_frame = ttk.Frame(notebook, padding=15)
        notebook.add(installed_frame, text="Already Have UE5")

        ttk.Label(
            installed_frame,
            text="If you already have Unreal Engine 5 installed:",
            font=("Helvetica", 11, "bold"),
        ).pack(anchor="w")

        path_frame = ttk.Frame(installed_frame)
        path_frame.pack(fill="x", pady=15)

        ttk.Label(path_frame, text="UE5 Installation Path:").pack(anchor="w")

        path_entry_frame = ttk.Frame(path_frame)
        path_entry_frame.pack(fill="x", pady=5)

        self.ue_path_var = tk.StringVar()
        ue_entry = ttk.Entry(path_entry_frame, textvariable=self.ue_path_var, width=50)
        ue_entry.pack(side="left", fill="x", expand=True)

        browse_btn = ttk.Button(
            path_entry_frame,
            text="Browse...",
            command=self._browse_ue_path,
        )
        browse_btn.pack(side="left", padx=(10, 0))

        detect_btn = ttk.Button(
            path_entry_frame,
            text="Auto-Detect",
            command=self._auto_detect_ue,
        )
        detect_btn.pack(side="left", padx=(5, 0))

        detected_path = detect_unreal_engine_path()
        if detected_path:
            self.ue_path_var.set(str(detected_path))
            self.config.unreal_engine_path = detected_path

        # Default paths hint
        if IS_WINDOWS:
            default_path = str(detected_path) if detected_path else "C:\\Program Files\\Epic Games\\UE_5.x"
        else:
            default_path = "~/UnrealEngine or via Epic Games Launcher"

        ttk.Label(
            installed_frame,
            text=f"Default location: {default_path}",
            font=("Helvetica", 9),
            foreground="gray",
        ).pack(anchor="w")

        # Tab 2: Need to Install
        install_frame = ttk.Frame(notebook, padding=15)
        notebook.add(install_frame, text="Install UE5")

        install_text = """To install Unreal Engine 5:

1. Download the Epic Games Launcher
2. Sign in or create an Epic Games account
3. Go to "Unreal Engine" tab
4. Click "Install Engine" and select version 5.3+
5. Wait for download (requires ~50GB disk space)

After installation, return here and select the installation path."""

        ttk.Label(
            install_frame,
            text=install_text,
            justify="left",
            font=("Helvetica", 11),
        ).pack(anchor="w")

        btn_frame = ttk.Frame(install_frame)
        btn_frame.pack(pady=20)

        ttk.Button(
            btn_frame,
            text="Download Epic Games Launcher",
            command=lambda: open_url(URLS["epic_games"]),
        ).pack(side="left", padx=5)

        ttk.Button(
            btn_frame,
            text="Unreal Engine Website",
            command=lambda: open_url(URLS["unreal_engine"]),
        ).pack(side="left", padx=5)

        # Tab 3: Skip
        skip_frame = ttk.Frame(notebook, padding=15)
        notebook.add(skip_frame, text="Skip (OBS Only)")

        skip_text = """You can skip Unreal Engine setup if you only want to use the
OBS overlay visualization.

The OBS overlay works standalone and displays agent thought bubbles
over any video source - you can use screen capture, a webcam,
or pre-recorded footage.

You can always set up Unreal Engine later."""

        ttk.Label(
            skip_frame,
            text=skip_text,
            justify="left",
            font=("Helvetica", 11),
        ).pack(anchor="w", pady=10)

        self.skip_ue_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            skip_frame,
            text="Skip Unreal Engine setup",
            variable=self.skip_ue_var,
        ).pack(anchor="w", pady=10)

    def _browse_ue_path(self) -> None:
        """Browse for UE installation."""
        path = filedialog.askdirectory(title="Select Unreal Engine Installation")
        if path:
            self.ue_path_var.set(path)
            self.config.unreal_engine_path = Path(path)

    def _auto_detect_ue(self) -> None:
        """Try to auto-detect UE installation."""
        possible_paths = []

        if IS_WINDOWS:
            possible_paths = [
                Path("C:/Program Files/Epic Games/UE_5.4"),
                Path("C:/Program Files/Epic Games/UE_5.3"),
                Path("C:/Program Files/Epic Games/UE_5.2"),
                Path("C:/Program Files/Epic Games/UE_5.1"),
            ]
        else:
            home = Path.home()
            possible_paths = [
                home / "UnrealEngine",
                Path("/opt/UnrealEngine"),
                home / "Epic Games/UE_5.4",
            ]

        for path in possible_paths:
            if path.exists():
                self.ue_path_var.set(str(path))
                self.config.unreal_engine_path = path
                messagebox.showinfo("Found", f"Found Unreal Engine at:\n{path}")
                return

        messagebox.showinfo("Not Found", "Could not auto-detect Unreal Engine installation.")

    def on_leave(self) -> bool:
        if self.ue_path_var.get():
            self.config.unreal_engine_path = Path(self.ue_path_var.get())
        return True

    def get_title(self) -> str:
        return "Unreal Engine"


class AirSimSetupStep(StepFrame):
    """Guide through AirSim setup."""

    def __init__(self, parent: tk.Widget, wizard: SetupWizard) -> None:
        super().__init__(parent, wizard)
        self._build_ui()

    def _build_ui(self) -> None:
        # Title
        title = ttk.Label(
            self,
            text="AirSim Setup",
            font=("Helvetica", 18, "bold"),
        )
        title.pack(pady=(20, 10))

        desc = ttk.Label(
            self,
            text="Configure Microsoft AirSim for drone simulation",
            font=("Helvetica", 11),
        )
        desc.pack(pady=(0, 20))

        # Main content with scrolling
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        content_frame = ttk.Frame(canvas)

        content_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Step 1: Choose AirSim Setup Method
        step1 = ttk.LabelFrame(content_frame, text="Step 1: Choose AirSim Setup Method", padding=10)
        step1.pack(fill="x", padx=20, pady=10)

        step1_text = """AirSim is a simulator for drones built on Unreal Engine.

Choose how to set up AirSim:"""

        ttk.Label(step1, text=step1_text, justify="left").pack(anchor="w")

        # Radio buttons for setup method
        self.airsim_method_var = tk.StringVar(value="prebuilt")

        method_frame = ttk.Frame(step1)
        method_frame.pack(fill="x", pady=10)

        ttk.Radiobutton(
            method_frame,
            text="Use pre-built AirSim binary (faster, but no AegisAV overlay UI)",
            variable=self.airsim_method_var,
            value="prebuilt",
            command=self._on_method_change,
        ).pack(anchor="w", pady=2)

        ttk.Radiobutton(
            method_frame,
            text="Build from source with AegisAV Overlay plugin (recommended for full UI)",
            variable=self.airsim_method_var,
            value="build_overlay",
            command=self._on_method_change,
        ).pack(anchor="w", pady=2)

        # Info about the overlay
        self.overlay_info = ttk.Label(
            step1,
            text="The AegisAV Overlay adds native Unreal widgets showing agent thinking,\n"
                 "camera feeds, critic verdicts, telemetry, and battery status directly in AirSim.",
            foreground="gray",
            justify="left",
        )
        self.overlay_info.pack(anchor="w", pady=(5, 0))

        btn_frame1 = ttk.Frame(step1)
        btn_frame1.pack(pady=10)

        ttk.Button(
            btn_frame1,
            text="Download AirSim Releases",
            command=lambda: open_url(URLS["airsim_releases"]),
        ).pack(side="left", padx=5)

        ttk.Button(
            btn_frame1,
            text="AirSim Documentation",
            command=lambda: open_url(URLS["airsim_docs"]),
        ).pack(side="left", padx=5)

        # Step 2: Configure settings.json
        step2 = ttk.LabelFrame(content_frame, text="Step 2: Configure settings.json", padding=10)
        step2.pack(fill="x", padx=20, pady=10)

        settings_path = Path.home() / "Documents" / "AirSim" / "settings.json"
        if IS_LINUX:
            settings_path = Path.home() / "Documents" / "AirSim" / "settings.json"

        step2_text = f"""AirSim uses a settings.json file located at:
{settings_path}

We'll create an optimized configuration for AegisAV."""

        ttk.Label(step2, text=step2_text, justify="left").pack(anchor="w")

        ttk.Button(
            step2,
            text="Create AirSim Settings",
            command=self._create_airsim_settings,
        ).pack(pady=10)

        # Step 3: Test connection
        step3 = ttk.LabelFrame(content_frame, text="Step 3: Test Connection", padding=10)
        step3.pack(fill="x", padx=20, pady=10)

        step3_text = """After AirSim is running, test the connection:

1. Launch AirSim (or Blocks environment)
2. Click 'Test Connection' below"""

        ttk.Label(step3, text=step3_text, justify="left").pack(anchor="w")

        self.connection_status = ttk.Label(step3, text="Not tested", foreground="gray")
        self.connection_status.pack(pady=5)

        ttk.Button(
            step3,
            text="Test Connection",
            command=self._test_airsim_connection,
        ).pack(pady=10)

        # AirSim path
        path_frame = ttk.LabelFrame(
            content_frame, text="AirSim Installation Path (Required when AirSim enabled)", padding=10
        )
        path_frame.pack(fill="x", padx=20, pady=10)

        self.airsim_path_var = tk.StringVar()
        detected_airsim = detect_airsimnh_path()
        if detected_airsim:
            self.airsim_path_var.set(str(detected_airsim))
            self.config.airsim_path = detected_airsim

        path_entry = ttk.Entry(path_frame, textvariable=self.airsim_path_var, width=50)
        path_entry.pack(side="left", fill="x", expand=True)

        ttk.Button(
            path_frame,
            text="Browse...",
            command=self._browse_airsim_path,
        ).pack(side="left", padx=(10, 0))

        # Pack scrollable area
        canvas.pack(side="left", fill="both", expand=True, padx=20)
        scrollbar.pack(side="right", fill="y")

    def _create_airsim_settings(self) -> None:
        """Create AirSim settings.json for AegisAV."""
        settings_dir = Path.home() / "Documents" / "AirSim"
        settings_dir.mkdir(parents=True, exist_ok=True)

        settings_path = settings_dir / "settings.json"

        settings = {
            "SeeDocsAt": "https://microsoft.github.io/AirSim/settings/",
            "SettingsVersion": 1.2,
            "SimMode": "Multirotor",
            "ViewMode": "SpringArmChase",
            "ClockSpeed": 1.0,
            "Vehicles": {
                "Drone1": {
                    "VehicleType": "SimpleFlight",
                    "X": 0,
                    "Y": 0,
                    "Z": 0,
                    "EnableCollisionPassthroable": True,
                    "Cameras": {
                        "front_center": {
                            "CaptureSettings": [
                                {"ImageType": 0, "Width": 1280, "Height": 720, "FOV_Degrees": 90}
                            ],
                            "X": 0.25,
                            "Y": 0,
                            "Z": -0.1,
                            "Pitch": -10,
                            "Roll": 0,
                            "Yaw": 0,
                        }
                    },
                }
            },
            "SubWindows": [
                {"WindowID": 0, "CameraName": "front_center", "ImageType": 0, "VehicleName": "Drone1"}
            ],
            "Recording": {"RecordOnMove": False, "RecordInterval": 0.05},
        }

        # Backup existing
        if settings_path.exists():
            backup_path = settings_path.with_suffix(".json.backup")
            shutil.copy(settings_path, backup_path)

        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)

        messagebox.showinfo(
            "Settings Created",
            f"AirSim settings created at:\n{settings_path}\n\nRestart AirSim to apply changes.",
        )

    def _test_airsim_connection(self) -> None:
        """Test connection to AirSim."""
        self.connection_status.config(text="Testing...", foreground="orange")
        self.update()

        try:
            try:
                with socket.create_connection(("127.0.0.1", 41451), timeout=1.5):
                    append_log("AirSim RPC reachable on 127.0.0.1:41451")
            except OSError as e:
                append_log(f"AirSim RPC not reachable on 127.0.0.1:41451: {e}")

            try:
                airsim = importlib.import_module("airsim")
                client = airsim.MultirotorClient()
                client.confirmConnection()
                self.connection_status.config(
                    text="✅ Connected to AirSim successfully!", foreground="green"
                )
                return
            except Exception as exc:
                append_log(f"AirSim import/connection failed in current interpreter: {exc}")

            uv_python = get_uv_python_path(self.config.uv_root)
            if uv_python:
                append_log(f"Testing AirSim with uv Python: {uv_python}")
                cmd = [
                    str(uv_python),
                    "-c",
                    "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); print('OK')",
                ]
                success, output, _ = run_command_stream(
                    cmd,
                    timeout=30,
                    on_output=append_log,
                )
                if success and "OK" in output:
                    self.connection_status.config(
                        text="✅ Connected to AirSim successfully! (uv environment)",
                        foreground="green"
                    )
                    return
                append_log(f"uv AirSim test failed: {output}")
                self.connection_status.config(
                    text="❌ AirSim not available in uv environment.",
                    foreground="red"
                )
                return

            self.connection_status.config(
                text="❌ airsim package not installed for this Python.",
                foreground="red"
            )
            return

        except Exception as e:
            self.connection_status.config(
                text=f"❌ Connection failed: {str(e)[:50]}", foreground="red"
            )

    def _browse_airsim_path(self) -> None:
        """Browse for AirSim installation."""
        path = filedialog.askdirectory(title="Select AirSim Installation")
        if path:
            self.airsim_path_var.set(path)
            self.config.airsim_path = Path(path)

    def _on_method_change(self) -> None:
        """Handle AirSim setup method change."""
        method = self.airsim_method_var.get()
        self.config.build_airsim_overlay = (method == "build_overlay")

        if method == "build_overlay":
            self.overlay_info.config(
                text="Will clone AirSim, copy AegisAVOverlay plugin, and build.\n"
                     "Requires: UE5, Visual Studio 2022, Git. Build takes ~15-30 minutes.",
                foreground="blue",
            )
        else:
            self.overlay_info.config(
                text="The AegisAV Overlay adds native Unreal widgets showing agent thinking,\n"
                     "camera feeds, critic verdicts, telemetry, and battery status directly in AirSim.",
                foreground="gray",
            )

    def on_leave(self) -> bool:
        if self.airsim_path_var.get():
            self.config.airsim_path = Path(self.airsim_path_var.get())
        # Save overlay build preference
        self.config.build_airsim_overlay = (self.airsim_method_var.get() == "build_overlay")
        return True

    def get_title(self) -> str:
        return "AirSim"


class ConfigurationStep(StepFrame):
    """Configure AegisAV settings."""

    def __init__(self, parent: tk.Widget, wizard: SetupWizard) -> None:
        super().__init__(parent, wizard)
        self._build_ui()

    def _build_ui(self) -> None:
        # Title
        title = ttk.Label(
            self,
            text="AegisAV Configuration",
            font=("Helvetica", 18, "bold"),
        )
        title.pack(pady=(20, 10))

        desc = ttk.Label(
            self,
            text="Configure the AegisAV agent server",
            font=("Helvetica", 11),
        )
        desc.pack(pady=(0, 20))

        # Options frame
        options_frame = ttk.LabelFrame(self, text="Server Options", padding=15)
        options_frame.pack(fill="x", padx=50, pady=10)

        # Redis option
        self.use_redis_var = tk.BooleanVar(value=False)
        redis_check = ttk.Checkbutton(
            options_frame,
            text="Enable Redis for persistent storage (requires Redis server)",
            variable=self.use_redis_var,
        )
        redis_check.pack(anchor="w")

        # GPU option
        self.use_gpu_var = tk.BooleanVar(value=False)
        gpu_check = ttk.Checkbutton(
            options_frame,
            text="Enable GPU acceleration for vision (requires CUDA)",
            variable=self.use_gpu_var,
        )
        gpu_check.pack(anchor="w", pady=(5, 0))

        # Frontend option
        frontend_frame = ttk.LabelFrame(self, text="Dashboard Frontend", padding=15)
        frontend_frame.pack(fill="x", padx=50, pady=10)

        self.install_frontend_var = tk.BooleanVar(value=True)
        frontend_check = ttk.Checkbutton(
            frontend_frame,
            text="Build React dashboard (requires Node.js)",
            variable=self.install_frontend_var,
        )
        frontend_check.pack(anchor="w")

        ttk.Label(
            frontend_frame,
            text="The dashboard provides a web UI for monitoring agent decisions",
            font=("Helvetica", 9),
            foreground="gray",
        ).pack(anchor="w", pady=(5, 0))

        # Shortcuts option
        shortcuts_frame = ttk.LabelFrame(self, text="Convenience Options", padding=15)
        shortcuts_frame.pack(fill="x", padx=50, pady=10)

        self.create_shortcuts_var = tk.BooleanVar(value=True)
        shortcuts_check = ttk.Checkbutton(
            shortcuts_frame,
            text="Create launch scripts (start_server.sh/bat, run_demo.sh/bat)",
            variable=self.create_shortcuts_var,
        )
        shortcuts_check.pack(anchor="w")

        # Port configuration
        port_frame = ttk.Frame(shortcuts_frame)
        port_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(port_frame, text="Server Port:").pack(side="left")
        self.port_var = tk.StringVar(value=str(self.config.server_port))
        port_entry = ttk.Entry(port_frame, textvariable=self.port_var, width=10)
        port_entry.pack(side="left", padx=(10, 0))

        # Simulation options
        sim_frame = ttk.LabelFrame(self, text="Simulation (Optional)", padding=15)
        sim_frame.pack(fill="x", padx=50, pady=10)

        self.simulation_enabled_var = tk.BooleanVar(value=True)
        sim_enabled_check = ttk.Checkbutton(
            sim_frame,
            text="Enable simulation features (Unreal + AirSim + SITL)",
            variable=self.simulation_enabled_var,
        )
        sim_enabled_check.pack(anchor="w")

        self.airsim_enabled_var = tk.BooleanVar(value=True)
        airsim_enabled_check = ttk.Checkbutton(
            sim_frame,
            text="Auto-start AirSim bridge on server launch",
            variable=self.airsim_enabled_var,
        )
        airsim_enabled_check.pack(anchor="w", pady=(5, 0))

        self.sitl_enabled_var = tk.BooleanVar(value=True)
        sitl_enabled_check = ttk.Checkbutton(
            sim_frame,
            text="Enable ArduPilot SITL integration",
            variable=self.sitl_enabled_var,
        )
        sitl_enabled_check.pack(anchor="w", pady=(5, 0))

        ttk.Label(
            sim_frame,
            text="ArduPilot SITL Path (for drone simulation)",
            font=("Helvetica", 9),
            foreground="gray",
        ).pack(anchor="w")

        ardupilot_entry_frame = ttk.Frame(sim_frame)
        ardupilot_entry_frame.pack(fill="x", pady=(6, 0))

        self.ardupilot_path_var = tk.StringVar()
        detected_ardupilot = detect_ardupilot_path()
        if detected_ardupilot:
            self.ardupilot_path_var.set(str(detected_ardupilot))
            self.config.ardupilot_path = detected_ardupilot

        ardupilot_entry = ttk.Entry(
            ardupilot_entry_frame,
            textvariable=self.ardupilot_path_var,
            width=50,
        )
        ardupilot_entry.pack(side="left", fill="x", expand=True)

        ttk.Button(
            ardupilot_entry_frame,
            text="Browse...",
            command=self._browse_ardupilot_path,
        ).pack(side="left", padx=(10, 0))

        ttk.Label(
            sim_frame,
            text="Example (WSL): \\\\wsl.localhost\\Ubuntu\\home\\devcontainers\\ardupilot",
            font=("Helvetica", 9),
            foreground="gray",
        ).pack(anchor="w", pady=(6, 0))

    def on_leave(self) -> bool:
        self.config.use_redis = self.use_redis_var.get()
        self.config.use_gpu = self.use_gpu_var.get()
        self.config.install_frontend = self.install_frontend_var.get()
        self.config.create_shortcuts = self.create_shortcuts_var.get()
        self.config.simulation_enabled = self.simulation_enabled_var.get()
        self.config.airsim_enabled = (
            self.airsim_enabled_var.get() and self.config.simulation_enabled
        )
        self.config.sitl_enabled = self.sitl_enabled_var.get() and self.config.simulation_enabled
        if self.ardupilot_path_var.get():
            self.config.ardupilot_path = Path(self.ardupilot_path_var.get())
        try:
            port = int(self.port_var.get().strip())
            if 1 <= port <= 65535:
                self.config.server_port = port
        except ValueError:
            pass
        return True

    def _browse_ardupilot_path(self) -> None:
        """Browse for ArduPilot installation."""
        path = filedialog.askdirectory(title="Select ArduPilot Installation")
        if path:
            self.ardupilot_path_var.set(path)
            self.config.ardupilot_path = Path(path)

    def get_title(self) -> str:
        return "Configuration"


class InstallStep(StepFrame):
    """Run final installation and configuration."""

    def __init__(self, parent: tk.Widget, wizard: SetupWizard) -> None:
        super().__init__(parent, wizard)
        self.install_complete = False
        self._build_ui()

    def _build_ui(self) -> None:
        # Title
        title = ttk.Label(
            self,
            text="Final Installation",
            font=("Helvetica", 18, "bold"),
        )
        title.pack(pady=(20, 10))

        desc = ttk.Label(
            self,
            text="Applying configuration and finalizing setup",
            font=("Helvetica", 11),
        )
        desc.pack(pady=(0, 20))

        # Tasks list
        self.tasks_frame = ttk.LabelFrame(self, text="Installation Tasks", padding=10)
        self.tasks_frame.pack(fill="x", padx=50, pady=10)

        self.tasks = [
            ("Generate configuration files", self._generate_configs),
            ("Build AirSim with Overlay (if enabled)", self._build_airsim_overlay),
            ("Build frontend (if enabled)", self._build_frontend),
            ("Create launch scripts", self._create_scripts),
            ("Verify installation", self._verify_install),
        ]

        self.task_labels = []
        for task_name, _ in self.tasks:
            row = ttk.Frame(self.tasks_frame)
            row.pack(fill="x", pady=3)

            status = ttk.Label(row, text="⏳", width=3)
            status.pack(side="left")

            name = ttk.Label(row, text=task_name)
            name.pack(side="left", padx=(10, 0))

            self.task_labels.append(status)

        # Console
        console_frame = ttk.LabelFrame(self, text="Output", padding=10)
        console_frame.pack(fill="both", expand=True, padx=50, pady=10)

        self.console = scrolledtext.ScrolledText(
            console_frame,
            height=10,
            font=("Courier", 9),
            state="disabled",
            bg="#f8fafc",
            fg="#0f172a",
            insertbackground="#0f172a",
            selectbackground="#dbeafe",
            selectforeground="#0f172a",
        )
        self.console.pack(fill="both", expand=True)

        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self, variable=self.progress_var, maximum=len(self.tasks))
        self.progress.pack(fill="x", padx=50, pady=10)

        # Install button
        self.install_btn = ttk.Button(
            self,
            text="Run Installation",
            command=self._start_install,
        )
        self.install_btn.pack(pady=10)

        self.log_btn = ttk.Button(
            self,
            text="Open Log File",
            command=self._open_log,
        )
        self.log_btn.pack(pady=(0, 10))

    def _log(self, text: str) -> None:
        """Log to console."""
        append_log(text)
        self.console.config(state="normal")
        self.console.insert("end", text + "\n")
        self.console.see("end")
        self.console.config(state="disabled")

    def _open_log(self) -> None:
        """Open the installer log file."""
        append_log("Opening log file on user request.")
        open_path(LOG_FILE_ACTIVE)

    def _start_install(self) -> None:
        """Start installation."""
        append_log("Final install: run installation button clicked.")
        self.install_btn.config(state="disabled")
        thread = threading.Thread(target=self._run_install, daemon=True)
        thread.start()

    def _run_install(self) -> None:
        """Run all installation tasks."""
        append_log("Final install: installation thread started.")
        self.install_failed = False
        for i, (task_name, task_func) in enumerate(self.tasks):
            self.after(0, lambda i=i: self.task_labels[i].config(text="🔄"))
            self.after(0, lambda n=task_name: self._log(f"\n▶ {n}..."))

            try:
                success = task_func()
                if success:
                    self.after(0, lambda i=i: self.task_labels[i].config(text="✅"))
                else:
                    self.install_failed = True
                    self.after(0, lambda i=i: self.task_labels[i].config(text="⚠️"))
            except Exception as e:
                self.install_failed = True
                self.after(0, lambda i=i: self.task_labels[i].config(text="❌"))
                self.after(0, lambda e=e: self._log(f"Error: {e}"))

            self.progress_var.set(i + 1)

        self.install_complete = not self.install_failed
        if self.install_failed:
            self.after(
                0,
                lambda: self._log("\n⚠️ Installation completed with issues. Fix and re-run."),
            )
        else:
            self.after(0, lambda: self._log("\n✅ Installation complete!"))
        self.after(0, lambda: self.install_btn.config(text="Re-run", state="normal"))
        if self.install_failed:
            self.after(
                0,
                lambda: messagebox.askyesno(
                    "Open Log File?",
                    f"Some steps reported issues. Open log file?\n\n{LOG_FILE_ACTIVE}",
                )
                and open_path(LOG_FILE_ACTIVE),
            )

    def _generate_configs(self) -> bool:
        """Generate configuration files."""
        if self.config.airsim_enabled:
            airsim_root, error = self._resolve_airsim_install()
            if error:
                self._log(f"  {error}")
                return False
            self.config.airsim_path = airsim_root

        # Save setup config
        config_path = PROJECT_ROOT / ".aegis_setup.json"
        self.config.save(config_path)
        self._log(f"  Saved config to {config_path}")

        # Update server config if needed
        server_config = PROJECT_ROOT / "configs" / "aegis_config.yaml"
        if server_config.exists():
            try:
                try:
                    import yaml
                except Exception:
                    yaml = None

                ardupilot_path = self.config.ardupilot_path or detect_ardupilot_path()
                simulation_updates = {
                    "enabled": bool(self.config.simulation_enabled),
                    "airsim_enabled": bool(self.config.airsim_enabled),
                    "sitl_enabled": bool(self.config.sitl_enabled),
                }
                if ardupilot_path:
                    simulation_updates["ardupilot_path"] = str(ardupilot_path)

                if yaml:
                    with open(server_config, encoding="utf-8") as file_handle:
                        data = yaml.safe_load(file_handle) or {}
                    if not isinstance(data, dict):
                        data = {}
                    server_block = data.get("server")
                    if not isinstance(server_block, dict):
                        server_block = {}
                    server_block["port"] = int(self.config.server_port)
                    data["server"] = server_block
                    simulation_block = data.get("simulation")
                    if not isinstance(simulation_block, dict):
                        simulation_block = {}
                    simulation_block.update(simulation_updates)
                    data["simulation"] = simulation_block
                    with open(server_config, "w", encoding="utf-8") as file_handle:
                        yaml.dump(data, file_handle, sort_keys=False)
                    self._log(f"  Updated server port in {server_config}")
                else:
                    content = server_config.read_text(encoding="utf-8")
                    content = _update_yaml_section(
                        content, "server", {"port": int(self.config.server_port)}
                    )
                    content = _update_yaml_section(content, "simulation", simulation_updates)
                    server_config.write_text(content, encoding="utf-8")
                    self._log(f"  Updated server config in {server_config}")

                if ardupilot_path:
                    self._log(f"  Updated ArduPilot path in {server_config}")
            except Exception as exc:
                self._log(f"  Failed to update server port in {server_config}: {exc}")

        return True

    def _build_frontend(self) -> bool:
        """Build the React frontend."""
        if not self.config.install_frontend:
            self._log("  Skipped (disabled)")
            return True

        frontend_dir = PROJECT_ROOT / "frontend"
        if not frontend_dir.exists():
            self._log("  Frontend directory not found, skipping")
            return True

        # Check for npm
        npm_path = find_npm_executable()
        if not npm_path:
            self._log("  npm not found. Install Node.js or add npm to PATH, then retry.")
            return False
        if IS_WINDOWS and not npm_path.lower().endswith(".cmd"):
            cmd_candidate = f"{npm_path}.cmd"
            if Path(cmd_candidate).exists():
                npm_path = cmd_candidate
        npm_parent = Path(npm_path).parent
        if npm_parent.is_absolute():
            os.environ["PATH"] = f"{npm_parent}{os.pathsep}{os.environ.get('PATH', '')}"
        npm_cmd = ["cmd", "/c", npm_path] if IS_WINDOWS else [npm_path]
        self._log(f"  Using npm: {npm_path}")

        staging_dir: Path | None = None
        if IS_WINDOWS and str(frontend_dir).startswith("\\\\"):
            self._log("  Detected UNC path; staging frontend build locally.")
            local_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            staging_dir = local_root / "AegisAV" / "frontend_build"
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            shutil.copytree(
                frontend_dir,
                staging_dir,
                ignore=shutil.ignore_patterns("node_modules", "dist"),
            )
            self._log(f"  Staged frontend at: {staging_dir}")

        def run_npm(args: list[str]) -> tuple[bool, str, int | None]:
            target_dir = staging_dir or frontend_dir
            npm_logs = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "npm-cache" / "_logs"

            def log_line(line: str) -> None:
                if line:
                    self._log(f"  {line}")

            if IS_WINDOWS and str(target_dir).startswith("\\\\"):
                cmd_str = " ".join([f"\"{npm_path}\"", *args])
                self._log("  Using cmd pushd to handle UNC path for npm.")
                targets = [str(target_dir)]
                if str(target_dir).lower().startswith("\\\\wsl.localhost\\"):
                    mapped = "\\\\wsl$\\" + str(target_dir)[len("\\\\wsl.localhost\\"):]
                    targets.insert(0, mapped)
                last_output = ""
                last_code: int | None = None
                for target in targets:
                    self._log(f"  npm working directory: {target}")
                    success, output, exit_code = run_command_stream(
                        ["cmd", "/c", f"pushd \"{target}\" && {cmd_str}"],
                        timeout=300,
                        on_output=log_line,
                    )
                    if success:
                        return success, output, exit_code
                    last_output = output
                    last_code = exit_code
                if not last_output.strip():
                    self._log(f"  npm command failed with no output. Logs: {npm_logs}")
                return False, last_output, last_code
            success, output, exit_code = run_command_stream(
                [*npm_cmd, *args],
                cwd=target_dir,
                timeout=300,
                on_output=log_line,
            )
            if not success and not output.strip():
                self._log(f"  npm command failed with no output. Logs: {npm_logs}")
            return success, output, exit_code

        # Install dependencies
        self._log("  Installing npm dependencies...")
        success, _output, exit_code = run_npm(["install"])
        if exit_code is not None:
            self._log(f"  npm install exit code: {exit_code}")
        if not success:
            self._log("  npm install failed.")
            return False

        # Build
        self._log("  Building frontend...")
        success, _output, exit_code = run_npm(["run", "build"])
        if exit_code is not None:
            self._log(f"  npm build exit code: {exit_code}")
        if not success:
            self._log("  Build failed.")
            return False

        if staging_dir:
            staged_dist = staging_dir / "dist"
            if staged_dist.exists():
                dist_dir = frontend_dir / "dist"
                if dist_dir.exists():
                    shutil.rmtree(dist_dir, ignore_errors=True)
                shutil.copytree(staged_dist, dist_dir)
                self._log(f"  Synced build output to {dist_dir}")
            else:
                self._log("  Build succeeded but no dist folder found in staged output.")
                return False

        self._log("  Frontend built successfully")
        return True

    def _build_airsim_overlay(self) -> bool:
        """Build AirSim from source with AegisAVOverlay plugin."""
        if not self.config.build_airsim_overlay:
            self._log("  Skipped (not enabled)")
            return True

        if not IS_WINDOWS:
            self._log("  AirSim overlay build is only supported on Windows")
            return True

        # Check for UE5
        ue5_path = self.config.unreal_engine_path
        if not ue5_path:
            # Try to detect UE5
            ue5_candidates = [
                Path("C:/Program Files/Epic Games/UE_5.5"),
                Path("C:/Program Files/Epic Games/UE_5.4"),
                Path("C:/Program Files/Epic Games/UE_5.3"),
                Path("C:/Program Files/Epic Games/UE_5.2"),
                Path("C:/Program Files/Epic Games/UE_5.1"),
            ]
            for candidate in ue5_candidates:
                if candidate.exists():
                    ue5_path = candidate
                    break

        if not ue5_path or not ue5_path.exists():
            self._log("  ERROR: Unreal Engine 5 not found.")
            self._log("  Please install UE5 via Epic Games Launcher and set the path.")
            return False

        self._log(f"  Using UE5 at: {ue5_path}")

        # Check for Git
        git_path = shutil.which("git")
        if not git_path:
            self._log("  ERROR: Git not found. Please install Git and add to PATH.")
            return False

        airsim_dir = PROJECT_ROOT / "AirSim"
        overlay_src = PROJECT_ROOT / "unreal" / "AegisAVOverlay"

        # Check if overlay plugin source exists
        if not overlay_src.exists():
            self._log(f"  ERROR: AegisAVOverlay plugin not found at: {overlay_src}")
            return False

        # Step 1: Clone or update AirSim
        if (airsim_dir / ".git").exists():
            self._log("  AirSim already cloned, updating...")
            success, output, _ = run_command_stream(
                ["git", "pull", "origin", "main"],
                cwd=airsim_dir,
                timeout=120,
                on_output=lambda line: self.after(0, lambda msg=line: self._log(f"    {msg}")) if line else None,
            )
            if not success:
                self._log("  Warning: git pull failed, continuing with existing code")
        else:
            self._log("  Cloning AirSim repository (this may take a few minutes)...")
            success, output, _ = run_command_stream(
                ["git", "clone", "https://github.com/microsoft/AirSim.git", str(airsim_dir)],
                cwd=PROJECT_ROOT,
                timeout=300,
                on_output=lambda line: self.after(0, lambda msg=line: self._log(f"    {msg}")) if line else None,
            )
            if not success:
                self._log("  ERROR: Failed to clone AirSim")
                self._log(f"  Output: {output}")
                return False

        # Step 2: Run AirSim build script
        build_script = airsim_dir / "build.cmd"
        if build_script.exists():
            self._log("  Running AirSim build script...")
            success, output, _ = run_command_stream(
                ["cmd", "/c", str(build_script)],
                cwd=airsim_dir,
                timeout=600,
                on_output=lambda line: self.after(0, lambda msg=line: self._log(f"    {msg}")) if line else None,
            )
            if not success:
                self._log("  Warning: AirSim build script had errors, continuing...")
        else:
            self._log("  Warning: AirSim build.cmd not found, skipping native build")

        # Step 3: Copy AegisAVOverlay plugin to AirSim
        overlay_dst = airsim_dir / "Unreal" / "Plugins" / "AegisAVOverlay"
        self._log(f"  Copying AegisAVOverlay plugin to {overlay_dst}...")

        if overlay_dst.exists():
            shutil.rmtree(overlay_dst, ignore_errors=True)

        try:
            shutil.copytree(overlay_src, overlay_dst)
            self._log("  Plugin copied successfully")
        except Exception as e:
            self._log(f"  ERROR: Failed to copy plugin: {e}")
            return False

        # Step 4: Find and update project
        blocks_project = airsim_dir / "Unreal" / "Environments" / "Blocks" / "Blocks.uproject"
        if not blocks_project.exists():
            self._log("  Warning: Blocks.uproject not found")
            self._log("  You may need to generate project files manually")
        else:
            self._log("  Generating project files...")
            ubt_path = ue5_path / "Engine" / "Binaries" / "Win64" / "UnrealBuildTool.exe"
            if ubt_path.exists():
                success, output, _ = run_command_stream(
                    [str(ubt_path), "-projectfiles", f"-project={blocks_project}", "-game", "-engine"],
                    cwd=blocks_project.parent,
                    timeout=120,
                    on_output=lambda line: self.after(0, lambda msg=line: self._log(f"    {msg}")) if line else None,
                )
                if success:
                    self._log("  Project files generated")
                else:
                    self._log("  Warning: Project file generation may have issues")

        # Update config with new AirSim path
        self.config.airsim_path = airsim_dir / "Unreal" / "Environments" / "Blocks"

        self._log("")
        self._log("  ✅ AirSim with AegisAVOverlay setup complete!")
        self._log(f"  Project location: {blocks_project.parent if blocks_project.exists() else airsim_dir}")
        self._log("")
        self._log("  Next steps:")
        self._log("  1. Open Blocks.sln in Visual Studio 2022")
        self._log("  2. Build the project (Development Editor | Win64)")
        self._log("  3. Or open Blocks.uproject in Unreal Editor")
        return True

    def _resolve_airsim_install(self) -> tuple[Path | None, str | None]:
        if not IS_WINDOWS:
            return None, "AirSim launch is only supported on Windows."

        path = self.config.airsim_path or detect_airsimnh_path()
        if not path:
            return None, "AirSim installation path not set. Browse to AirSimNH folder."

        candidate = Path(path)
        if candidate.is_file():
            if candidate.name.lower() != "airsimnh.exe":
                return None, f"Selected file is not AirSimNH.exe: {candidate}"
            airsim_root = candidate.parent
        else:
            airsim_root = candidate

        exe_path = airsim_root / "AirSimNH.exe"
        if not exe_path.exists():
            return None, f"AirSimNH.exe not found at: {exe_path}"

        self.config.airsim_path = airsim_root
        return airsim_root, None

    def _create_scripts(self) -> bool:
        """Create launch scripts."""
        airsim_root: Path | None = None
        if self.config.airsim_enabled:
            airsim_root, error = self._resolve_airsim_install()
            if error:
                self._log(f"  {error}")
                return False

        if not self.config.create_shortcuts:
            self._log("  Skipped (disabled)")
            return True

        scripts_dir = PROJECT_ROOT / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        uv_python = get_uv_python_path(self.config.uv_root)
        python_exec = str(uv_python) if uv_python else sys.executable

        if IS_WINDOWS:
            # Windows batch scripts
            python_cmd = f"\"{python_exec}\""
            port = self.config.server_port
            project_root_str = str(PROJECT_ROOT)
            roots = [project_root_str]
            if project_root_str.lower().startswith("\\\\wsl.localhost\\"):
                roots.insert(0, "\\\\wsl$\\" + project_root_str[len("\\\\wsl.localhost\\"):])
            root_a = roots[0]
            root_b = roots[1] if len(roots) > 1 else roots[0]
            start_server = scripts_dir / "start_server.bat"
            start_server.write_text(f"""@echo off
setlocal
if "%AEGIS_PORT%"=="" call :select_port
if "%AEGIS_HOST%"=="" set "AEGIS_HOST=0.0.0.0"
if "%AEGIS_RELOAD%"=="" set "AEGIS_RELOAD=0"
pushd "{root_a}" 2>nul
if errorlevel 1 pushd "{root_b}" 2>nul
if errorlevel 1 (
  echo Failed to access project root.
  pause
  exit /b 1
)
{python_cmd} -m agent.server.main
popd
pause
exit /b 0

:select_port
for %%P in ({port} 8000 8080 8090 8100 8181 8200 8500 8765 9000) do (
  powershell -NoProfile -Command "try {{ $l=New-Object System.Net.Sockets.TcpListener([IPAddress]::Any, %%P); $l.Start(); $l.Stop(); exit 0 }} catch {{ exit 1 }}" >nul 2>&1
  if not errorlevel 1 (
    set "AEGIS_PORT=%%P"
    echo Using port %%P
    goto :eof
  )
)
set "AEGIS_PORT=8090"
echo Using port %AEGIS_PORT%
goto :eof
""")

            run_demo = scripts_dir / "run_demo.bat"
            run_demo.write_text(f"""@echo off
setlocal
if "%AEGIS_PORT%"=="" call :select_port
if "%AEGIS_HOST%"=="" set "AEGIS_HOST=0.0.0.0"
if "%AEGIS_RELOAD%"=="" set "AEGIS_RELOAD=0"
pushd "{root_a}" 2>nul
if errorlevel 1 pushd "{root_b}" 2>nul
if errorlevel 1 (
  echo Failed to access project root.
  pause
  exit /b 1
)
start "" {python_cmd} -m agent.server.main
timeout /t 3
start http://localhost:%AEGIS_PORT%/dashboard/
start http://localhost:%AEGIS_PORT%/overlay/
start http://localhost:%AEGIS_PORT%/api/docs
popd
pause
exit /b 0

:select_port
for %%P in ({port} 8000 8080 8090 8100 8181 8200 8500 8765 9000) do (
  powershell -NoProfile -Command "try {{ $l=New-Object System.Net.Sockets.TcpListener([IPAddress]::Any, %%P); $l.Start(); $l.Stop(); exit 0 }} catch {{ exit 1 }}" >nul 2>&1
  if not errorlevel 1 (
    set "AEGIS_PORT=%%P"
    echo Using port %%P
    goto :eof
  )
)
set "AEGIS_PORT=8090"
echo Using port %AEGIS_PORT%
goto :eof
""")
            self._log(f"  Created {start_server}")
            self._log(f"  Created {run_demo}")

            if self.config.airsim_enabled and airsim_root:
                airsim_script = PROJECT_ROOT / "start_airsim.bat"
                airsim_script.write_text(f"""@echo off
REM Start AirSim on Windows
REM Run this from Windows Command Prompt, not WSL

echo Starting AirSim...
cd /d "{airsim_root}"
start "" "AirSimNH.exe" -ResX=1920 -ResY=1080 -windowed
""")
                self._log(f"  Created {airsim_script}")
        else:
            # Linux/Mac shell scripts
            python_cmd = f"\"{python_exec}\""
            port = self.config.server_port
            start_server = scripts_dir / "start_server.sh"
            start_server.write_text(f"""#!/bin/bash
cd "{PROJECT_ROOT}"
export AEGIS_PORT={port}
export AEGIS_HOST=0.0.0.0
{python_cmd} -m agent.server.main
""")
            start_server.chmod(0o755)

            run_demo = scripts_dir / "run_demo.sh"
            run_demo.write_text(f"""#!/bin/bash
cd "{PROJECT_ROOT}"
export AEGIS_PORT={port}
export AEGIS_HOST=0.0.0.0
{python_cmd} -m agent.server.main &
SERVER_PID=$!
sleep 3
xdg-open http://localhost:{port}/dashboard/ 2>/dev/null || open http://localhost:{port}/dashboard/
xdg-open http://localhost:{port}/overlay/ 2>/dev/null || open http://localhost:{port}/overlay/
xdg-open http://localhost:{port}/api/docs 2>/dev/null || open http://localhost:{port}/api/docs
echo "Press Enter to stop server..."
read
kill $SERVER_PID
""")
            run_demo.chmod(0o755)

            self._log(f"  Created {start_server}")
            self._log(f"  Created {run_demo}")

        return True

    def _verify_install(self) -> bool:
        """Verify the installation."""
        # Check imports
        self._log("  Checking Python imports...")
        uv_python = get_uv_python_path(self.config.uv_root)
        python_exec = str(uv_python) if uv_python else sys.executable
        success, stdout, stderr = run_command(
            [python_exec, "-c", "from agent.server.main import app; print('OK')"],
            cwd=PROJECT_ROOT,
        )

        if success and "OK" in stdout:
            self._log("  ✓ Server imports working")
        else:
            self._log(f"  Import check failed: {stderr}")
            return False

        if self.config.install_frontend:
            dashboard_index = PROJECT_ROOT / "frontend" / "dist" / "index.html"
            if dashboard_index.exists():
                self._log("  ✓ Dashboard build ready")
            else:
                self._log("  Dashboard build missing (frontend/dist/index.html)")
                return False

        return True

    def can_proceed(self) -> bool:
        return self.install_complete

    def get_title(self) -> str:
        return "Install"


class CompleteStep(StepFrame):
    """Setup complete summary."""

    def __init__(self, parent: tk.Widget, wizard: SetupWizard) -> None:
        super().__init__(parent, wizard)
        self._build_ui()

    def _build_ui(self) -> None:
        # Title
        title = ttk.Label(
            self,
            text="🎉 Setup Complete!",
            font=("Helvetica", 24, "bold"),
        )
        title.pack(pady=(40, 20))

        # Summary
        summary_frame = ttk.LabelFrame(self, text="What's Ready", padding=15)
        summary_frame.pack(fill="x", padx=50, pady=20)

        items = [
            "✅ AegisAV agent server configured",
            "✅ Python dependencies installed",
            "✅ Launch scripts created",
            "✅ OBS overlay ready at /overlay/",
        ]
        if self.config.install_frontend:
            items.insert(3, "✅ Dashboard ready at /dashboard/")

        for item in items:
            ttk.Label(summary_frame, text=item, font=("Helvetica", 11)).pack(anchor="w", pady=2)

        # Next steps
        next_frame = ttk.LabelFrame(self, text="Next Steps", padding=15)
        next_frame.pack(fill="x", padx=50, pady=10)

        port = self.config.server_port
        if self.config.install_frontend:
            next_text = f"""1. Start the server:
   ./scripts/start_server.sh  (or .bat on Windows)

2. Open the dashboard:
   http://localhost:{port}/dashboard/

3. Open the overlay in OBS:
   Add Browser Source → http://localhost:{port}/overlay/

4. Run a simulation or connect AirSim

5. Watch agent decisions appear in the overlay!"""
        else:
            next_text = f"""1. Start the server:
   ./scripts/start_server.sh  (or .bat on Windows)

2. Open the overlay in OBS:
   Add Browser Source → http://localhost:{port}/overlay/

3. Run a simulation or connect AirSim

4. Watch agent decisions appear in the overlay!"""

        ttk.Label(next_frame, text=next_text, justify="left", font=("Courier", 10)).pack(anchor="w")

        # Quick launch buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=30)

        ttk.Button(
            btn_frame,
            text="🚀 Launch Server Now",
            command=self._launch_server,
        ).pack(side="left", padx=10)

        if self.config.install_frontend:
            ttk.Button(
                btn_frame,
                text="📊 Open Dashboard",
                command=lambda: open_url(f"http://localhost:{port}/dashboard/"),
            ).pack(side="left", padx=10)

        ttk.Button(
            btn_frame,
            text="📖 Open Documentation",
            command=lambda: open_url(f"file://{PROJECT_ROOT}/docs/"),
        ).pack(side="left", padx=10)

        ttk.Button(
            btn_frame,
            text="🎥 Open Overlay",
            command=lambda: open_url(f"http://localhost:{port}/overlay/"),
        ).pack(side="left", padx=10)

    def _launch_server(self) -> None:
        """Launch the server."""
        uv_python = get_uv_python_path(self.config.uv_root)
        python_exec = str(uv_python) if uv_python else sys.executable
        port = pick_available_port(self.config.server_port)
        self.config.server_port = port
        env = os.environ.copy()
        env["AEGIS_PORT"] = str(port)
        env["AEGIS_HOST"] = env.get("AEGIS_HOST", "0.0.0.0")
        env["AEGIS_RELOAD"] = env.get("AEGIS_RELOAD", "0")
        if IS_WINDOWS:
            script = PROJECT_ROOT / "scripts" / "start_server.bat"
            if script.exists():
                subprocess.Popen(
                    ["cmd", "/c", str(script)],
                    start_new_session=True,
                    env=env,
                )  # noqa: S603, S607
            else:
                project_root_str = str(PROJECT_ROOT)
                root_a = project_root_str
                root_b = project_root_str
                if project_root_str.lower().startswith("\\\\wsl.localhost\\"):
                    root_a = "\\\\wsl$\\" + project_root_str[len("\\\\wsl.localhost\\"):]
                    root_b = project_root_str
                subprocess.Popen(
                    [
                        "cmd",
                        "/c",
                        "set",
                        f"AEGIS_PORT={port}",
                        "&&",
                        "set",
                        "AEGIS_HOST=0.0.0.0",
                        "&&",
                        "set",
                        "AEGIS_RELOAD=0",
                        "&&",
                        "pushd",
                        f"\"{root_a}\"",
                        "2>nul",
                        "||",
                        "pushd",
                        f"\"{root_b}\"",
                        "2>nul",
                        "&&",
                        f"\"{python_exec}\" -m agent.server.main",
                    ],
                    start_new_session=True,
                    env=env,
                )
        else:
            script = PROJECT_ROOT / "scripts" / "start_server.sh"
            if script.exists():
                subprocess.Popen(
                    ["bash", str(script)],
                    start_new_session=True,
                    env=env,
                )  # noqa: S603, S607
            else:
                subprocess.Popen(
                    [
                        "bash",
                        "-c",
                        f"export AEGIS_PORT={port} "
                        f"AEGIS_HOST=0.0.0.0; cd {PROJECT_ROOT} && \"{python_exec}\" -m agent.server.main",
                    ],
                    start_new_session=True,
                    env=env,
                )

        port = self.config.server_port
        if self.config.install_frontend:
            self.after(2500, lambda: open_url(f"http://localhost:{port}/dashboard/"))
        self.after(3000, lambda: open_url(f"http://localhost:{port}/overlay/"))

        if self.config.install_frontend:
            notice = "Server is starting in a new terminal.\n\nDashboard and overlay will open shortly."
        else:
            notice = "Server is starting in a new terminal.\n\nOverlay will open shortly."
        messagebox.showinfo(
            "Server Starting",
            notice,
        )

    def get_title(self) -> str:
        return "Complete"


# =============================================================================
# Main Wizard
# =============================================================================


class SetupWizard:
    """Main setup wizard application."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)

        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - 800) // 2
        y = (self.root.winfo_screenheight() - 700) // 2
        self.root.geometry(f"+{x}+{y}")

        self.config = SetupConfig()
        self.current_step = 0

        init_log()
        append_log(f"Started {APP_NAME} v{APP_VERSION}")
        append_log(f"Project root: {PROJECT_ROOT}")
        append_log(f"Python: {platform.python_version()} ({sys.executable})")
        append_log(f"Log file: {LOG_FILE_ACTIVE}")
        if LOG_FILE_ACTIVE != LOG_FILE_FALLBACK:
            append_log(f"Fallback log: {LOG_FILE_FALLBACK}")

        self._setup_styles()
        self._build_ui()
        self._create_steps()
        self._show_step(0)

    def _setup_styles(self) -> None:
        """Configure ttk styles."""
        style = ttk.Style()

        # Try to use a modern theme
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")
        elif "alt" in available_themes:
            style.theme_use("alt")
        else:
            style.theme_use("default")

        # AegisAV Light Palette
        bg_base = "#f7f6f3"
        bg_surface = "#ffffff"
        bg_muted = "#eef1f4"
        fg_text = "#1f2933"
        fg_muted = "#52606d"
        accent = "#2563eb"
        accent_soft = "#dbeafe"
        border = "#d6d3d1"

        # Apply global settings
        style.configure(
            ".",
            background=bg_base,
            foreground=fg_text,
            troughcolor=bg_muted,
            selectbackground=accent_soft,
            selectforeground=fg_text,
        )

        self.root.configure(bg=bg_base)

        style.configure("TFrame", background=bg_base)
        style.configure("TLabel", background=bg_base, foreground=fg_text)
        style.configure(
            "TButton",
            background=bg_surface,
            foreground=fg_text,
            borderwidth=1,
            relief="flat",
            focuscolor=accent,
            focusthickness=1,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.map(
            "TButton",
            background=[("active", accent_soft), ("pressed", accent)],
            foreground=[("pressed", "#ffffff")],
        )

        style.configure("TLabelframe", background=bg_base, foreground=fg_text, bordercolor=border)
        style.configure("TLabelframe.Label", background=bg_base, foreground=fg_muted)

        style.configure(
            "TEntry",
            fieldbackground=bg_surface,
            foreground=fg_text,
            insertcolor=fg_text,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.map(
            "TEntry",
            fieldbackground=[("readonly", bg_muted), ("disabled", bg_muted)],
            foreground=[("disabled", "#9aa5b1")],
        )
        style.configure(
            "Horizontal.TProgressbar",
            background=accent,
            troughcolor=bg_muted,
            bordercolor=bg_base,
        )
        style.configure("TSeparator", background=border)

        strong_font = ("TkDefaultFont", 10, "bold")

        style.configure("Nav.TButton", font=strong_font, padding=(12, 6), background=bg_surface)
        style.map(
            "Nav.TButton",
            foreground=[("disabled", "#9aa5b1")],
            background=[("active", accent_soft), ("pressed", "#bfdbfe")],
        )

        style.configure(
            "PrimaryNav.TButton",
            font=strong_font,
            padding=(12, 6),
            foreground="#ffffff",
            background=accent,
            bordercolor=accent,
        )
        style.map(
            "PrimaryNav.TButton",
            foreground=[("disabled", "#e5e7eb")],
            background=[
                ("active", "#1d4ed8"),
                ("pressed", "#1e40af"),
                ("disabled", "#93c5fd"),
            ],
        )

    def _build_ui(self) -> None:
        """Build the main UI structure."""
        # Header with steps indicator
        self.header = ttk.Frame(self.root)
        self.header.pack(fill="x", padx=20, pady=10)

        # Steps indicator (will be populated after creating steps)
        self.steps_indicator = ttk.Frame(self.header)
        self.steps_indicator.pack()

        # Separator
        ttk.Separator(self.root).pack(fill="x", padx=20)

        # Content area
        self.content_container = ttk.Frame(self.root)
        self.content_container.pack(fill="both", expand=True, padx=20, pady=10)

        self.content_canvas = tk.Canvas(
            self.content_container,
            highlightthickness=0,
            borderwidth=0,
            background=self.root.cget("background"),
        )
        self.content_scrollbar = ttk.Scrollbar(
            self.content_container,
            orient="vertical",
            command=self.content_canvas.yview,
        )
        self.content_canvas.configure(yscrollcommand=self.content_scrollbar.set)

        self.content_canvas.pack(side="left", fill="both", expand=True)
        self.content_scrollbar.pack(side="right", fill="y")

        self.content = ttk.Frame(self.content_canvas)
        self._content_window = self.content_canvas.create_window(
            (0, 0),
            window=self.content,
            anchor="nw",
        )

        self.content.bind("<Configure>", self._on_content_configure)
        self.content_canvas.bind("<Configure>", self._on_canvas_configure)

        # Separator
        ttk.Separator(self.root).pack(fill="x", padx=20)

        # Navigation buttons
        self.nav_frame = ttk.Frame(self.root)
        self.nav_frame.pack(fill="x", padx=20, pady=15)

        self.back_btn = ttk.Button(
            self.nav_frame,
            text="← Back",
            command=self._go_back,
            style="Nav.TButton",
        )
        self.back_btn.pack(side="left")

        self.open_log_btn = ttk.Button(
            self.nav_frame,
            text="Open Log",
            command=self._open_log,
            style="Nav.TButton",
        )
        self.open_log_btn.pack(side="left", padx=(10, 0))

        self.next_btn = ttk.Button(
            self.nav_frame,
            text="Next →",
            command=self._go_next,
            style="PrimaryNav.TButton",
        )
        self.next_btn.pack(side="right")

        self.cancel_btn = ttk.Button(
            self.nav_frame,
            text="Cancel",
            command=self._cancel,
            style="Nav.TButton",
        )
        self.cancel_btn.pack(side="right", padx=(0, 10))

    def _open_log(self) -> None:
        """Open the installer log file."""
        append_log("Opening log file on user request.")
        open_path(LOG_FILE_ACTIVE)

    def _on_content_configure(self, _event: tk.Event) -> None:
        """Update scroll region when content size changes."""
        self.content_canvas.configure(scrollregion=self.content_canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        """Keep the content frame width in sync with the canvas."""
        self.content_canvas.itemconfigure(self._content_window, width=event.width)

    def _create_steps(self) -> None:
        """Create all wizard steps."""
        self.steps: list[StepFrame] = [
            WelcomeStep(self.content, self),
            DependencyStep(self.content, self),
            PythonSetupStep(self.content, self),
            UnrealSetupStep(self.content, self),
            AirSimSetupStep(self.content, self),
            ConfigurationStep(self.content, self),
            InstallStep(self.content, self),
            CompleteStep(self.content, self),
        ]

        # Create step indicators
        self.step_indicators = []
        for i, step in enumerate(self.steps):
            indicator = ttk.Label(
                self.steps_indicator,
                text=f" {i + 1}. {step.get_title()} ",
                font=("Helvetica", 9),
                foreground="gray",
            )
            indicator.pack(side="left", padx=2)
            self.step_indicators.append(indicator)

    def _show_step(self, index: int) -> None:
        """Show a specific step."""
        # Hide all steps
        for step in self.steps:
            step.pack_forget()

        # Show current step
        self.steps[index].pack(fill="both", expand=True)
        self.steps[index].on_enter()
        self.content_canvas.yview_moveto(0)

        # Update indicators
        for i, indicator in enumerate(self.step_indicators):
            if i < index:
                indicator.config(foreground="green")
            elif i == index:
                indicator.config(foreground="black", font=("Helvetica", 9, "bold"))
            else:
                indicator.config(foreground="gray", font=("Helvetica", 9))

        # Update navigation buttons
        self.back_btn.config(state="normal" if index > 0 else "disabled")

        if index == len(self.steps) - 1:
            self.next_btn.config(text="Finish", command=self._finish)
        else:
            self.next_btn.config(text="Next →", command=self._go_next)

        self.current_step = index

    def _go_next(self) -> None:
        """Go to next step."""
        current = self.steps[self.current_step]

        if not current.can_proceed():
            return

        if not current.on_leave():
            return

        if self.current_step < len(self.steps) - 1:
            self._show_step(self.current_step + 1)

    def _go_back(self) -> None:
        """Go to previous step."""
        if self.current_step > 0:
            self._show_step(self.current_step - 1)

    def _cancel(self) -> None:
        """Cancel setup."""
        if messagebox.askyesno("Cancel Setup", "Are you sure you want to cancel setup?"):
            self.root.destroy()

    def _finish(self) -> None:
        """Finish setup."""
        self.root.destroy()

    def run(self) -> None:
        """Run the wizard."""
        self.root.mainloop()


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main entry point."""
    wizard = SetupWizard()
    wizard.run()


if __name__ == "__main__":
    main()
