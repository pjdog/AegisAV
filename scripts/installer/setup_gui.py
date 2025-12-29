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
import platform
import shutil
import subprocess  # noqa: S404
import sys
import threading
import webbrowser
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


@dataclass
class SetupConfig:
    """Configuration gathered during setup."""

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    unreal_engine_path: Path | None = None
    airsim_path: Path | None = None
    python_path: Path | None = None
    use_redis: bool = False
    use_gpu: bool = False
    create_shortcuts: bool = True
    install_frontend: bool = True

    def to_dict(self) -> dict:
        return {
            "project_root": str(self.project_root),
            "unreal_engine_path": str(self.unreal_engine_path) if self.unreal_engine_path else None,
            "airsim_path": str(self.airsim_path) if self.airsim_path else None,
            "python_path": str(self.python_path) if self.python_path else None,
            "use_redis": self.use_redis,
            "use_gpu": self.use_gpu,
            "create_shortcuts": self.create_shortcuts,
            "install_frontend": self.install_frontend,
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
            shell=IS_WINDOWS,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except FileNotFoundError:
        return False, "", f"Command not found: {cmd[0]}"
    except Exception as exc:
        return False, "", str(exc)


def check_command_exists(cmd: str) -> tuple[bool, str]:
    """Check if a command exists and return its path."""
    where_cmd = ["where", cmd] if IS_WINDOWS else ["which", cmd]

    success, stdout, _ = run_command(where_cmd)
    if success and stdout.strip():
        return True, stdout.strip().split("\n")[0]
    return False, ""


def get_command_version(cmd: list[str]) -> str:
    """Get version string from a command."""
    success, stdout, stderr = run_command(cmd)
    if success:
        return stdout.strip() or stderr.strip()
    return ""


def open_url(url: str) -> None:
    """Open a URL in the default browser."""
    webbrowser.open(url)


# =============================================================================
# Dependency Definitions
# =============================================================================


def get_dependencies() -> list[Dependency]:
    """Get list of all dependencies to check."""
    return [
        Dependency(
            name="Python",
            check_cmd=["python3" if not IS_WINDOWS else "python", "--version"],
            required=True,
            min_version="3.10",
            install_url=URLS["python"],
        ),
        Dependency(
            name="pip",
            check_cmd=["pip3" if not IS_WINDOWS else "pip", "--version"],
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
            install_cmd_windows='powershell -c "irm https://astral.sh/uv/install.ps1 | iex"',
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

        # Check if command exists
        success, stdout, _ = run_command(dep.check_cmd)

        if success:
            dep.found = True
            dep.version = stdout.strip()[:50]  # Truncate long output

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
                labels["version"].config(text="Missing (Required!)", foreground="red")
            else:
                labels["status"].config(text="⚠️")
                labels["version"].config(text="Not found", foreground="orange")

            # Show install button if we have install info
            if (
                dep.install_url
                or (IS_LINUX and dep.install_cmd_linux)
                or (IS_WINDOWS and dep.install_cmd_windows)
            ):
                labels["install_btn"].pack(side="right")

    def _install_dependency(self, dep: Dependency) -> None:
        """Handle install button click."""
        if dep.install_url:
            if messagebox.askyesno(
                "Install Dependency", f"Open download page for {dep.name}?\n\n{dep.install_url}"
            ):
                open_url(dep.install_url)
        elif IS_LINUX and dep.install_cmd_linux:
            if messagebox.askyesno(
                "Install Dependency", f"Run the following command?\n\n{dep.install_cmd_linux}"
            ):
                # Run in terminal
                subprocess.Popen(  # noqa: S603
                    [  # noqa: S607
                        "x-terminal-emulator",
                        "-e",
                        f"bash -c '{dep.install_cmd_linux}; read -p \"Press Enter...\"'",
                    ],
                    start_new_session=True,
                )

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

        # Console output
        console_frame = ttk.LabelFrame(self, text="Installation Output", padding=10)
        console_frame.pack(fill="both", expand=True, padx=50, pady=10)

        self.console = scrolledtext.ScrolledText(
            console_frame,
            height=15,
            font=("Courier", 9),
            state="disabled",
            bg="#1e1e1e",
            fg="#d4d4d4",
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

    def _log(self, text: str) -> None:
        """Append text to console."""
        self.console.config(state="normal")
        self.console.insert("end", text + "\n")
        self.console.see("end")
        self.console.config(state="disabled")

    def _start_install(self) -> None:
        """Start installation in background."""
        self.install_btn.config(state="disabled")
        self.progress.start(10)
        self.status_label.config(text="Installing...")

        thread = threading.Thread(target=self._run_install, daemon=True)
        thread.start()

    def _run_install(self) -> None:
        """Run the installation (background thread)."""
        try:
            # Check for uv
            use_uv = self.use_uv_var.get()
            uv_available, _ = check_command_exists("uv")

            if use_uv and not uv_available:
                self.after(0, lambda: self._log("Installing uv package manager..."))
                if IS_WINDOWS:
                    cmd = ["powershell", "-c", "irm https://astral.sh/uv/install.ps1 | iex"]
                else:
                    cmd = ["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"]
                success, stdout, stderr = run_command(cmd, timeout=120)
                self.after(0, lambda: self._log(stdout or stderr))

            # Install dependencies
            if use_uv and (uv_available or check_command_exists("uv")[0]):
                self.after(0, lambda: self._log("\n📦 Installing with uv..."))
                cmd = ["uv", "sync"]
                if self.install_dev_var.get():
                    cmd.append("--all-extras")
            else:
                self.after(0, lambda: self._log("\n📦 Installing with pip..."))
                pip_cmd = "pip3" if not IS_WINDOWS else "pip"
                cmd = [pip_cmd, "install", "-e", "."]
                if self.install_dev_var.get():
                    cmd = [pip_cmd, "install", "-e", ".[dev,test]"]

            success, stdout, stderr = run_command(cmd, cwd=PROJECT_ROOT, timeout=600)
            self.after(0, lambda: self._log(stdout))
            if stderr:
                self.after(0, lambda: self._log(f"STDERR: {stderr}"))

            if success:
                self.after(0, self._install_success)
            else:
                self.after(0, lambda: self._install_failed(stderr))

        except Exception as exc:
            error = str(exc)
            self.after(0, lambda err=error: self._install_failed(err))

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

        # Default paths hint
        if IS_WINDOWS:
            default_path = "C:\\Program Files\\Epic Games\\UE_5.x"
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

        # Step 1: Download AirSim
        step1 = ttk.LabelFrame(content_frame, text="Step 1: Get AirSim", padding=10)
        step1.pack(fill="x", padx=20, pady=10)

        step1_text = """AirSim is a simulator for drones built on Unreal Engine.

Option A: Download pre-built binary (easier)
Option B: Build from source (more flexible)"""

        ttk.Label(step1, text=step1_text, justify="left").pack(anchor="w")

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
            content_frame, text="AirSim Installation Path (Optional)", padding=10
        )
        path_frame.pack(fill="x", padx=20, pady=10)

        self.airsim_path_var = tk.StringVar()
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
                "drone_001": {
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
                {
                    "WindowID": 0,
                    "CameraName": "front_center",
                    "ImageType": 0,
                    "VehicleName": "drone_001",
                }
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
            # Try to import airsim (optional dependency)
            spec = importlib.util.find_spec("airsim")

            if spec is None:
                self.connection_status.config(
                    text="❌ airsim package not installed. Run: pip install airsim",
                    foreground="red",
                )
                return

            airsim = importlib.import_module("airsim")

            client = airsim.MultirotorClient()
            client.confirmConnection()

            self.connection_status.config(
                text="✅ Connected to AirSim successfully!", foreground="green"
            )

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

    def on_leave(self) -> bool:
        if self.airsim_path_var.get():
            self.config.airsim_path = Path(self.airsim_path_var.get())
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
        self.port_var = tk.StringVar(value="8080")
        port_entry = ttk.Entry(port_frame, textvariable=self.port_var, width=10)
        port_entry.pack(side="left", padx=(10, 0))

    def on_leave(self) -> bool:
        self.config.use_redis = self.use_redis_var.get()
        self.config.use_gpu = self.use_gpu_var.get()
        self.config.install_frontend = self.install_frontend_var.get()
        self.config.create_shortcuts = self.create_shortcuts_var.get()
        return True

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
            bg="#1e1e1e",
            fg="#d4d4d4",
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

    def _log(self, text: str) -> None:
        """Log to console."""
        self.console.config(state="normal")
        self.console.insert("end", text + "\n")
        self.console.see("end")
        self.console.config(state="disabled")

    def _start_install(self) -> None:
        """Start installation."""
        self.install_btn.config(state="disabled")
        thread = threading.Thread(target=self._run_install, daemon=True)
        thread.start()

    def _run_install(self) -> None:
        """Run all installation tasks."""
        for i, (task_name, task_func) in enumerate(self.tasks):
            self.after(0, lambda i=i: self.task_labels[i].config(text="🔄"))
            self.after(0, lambda n=task_name: self._log(f"\n▶ {n}..."))

            try:
                success = task_func()
                if success:
                    self.after(0, lambda i=i: self.task_labels[i].config(text="✅"))
                else:
                    self.after(0, lambda i=i: self.task_labels[i].config(text="⚠️"))
            except Exception as e:
                self.after(0, lambda i=i: self.task_labels[i].config(text="❌"))
                self.after(0, lambda e=e: self._log(f"Error: {e}"))

            self.progress_var.set(i + 1)

        self.install_complete = True
        self.after(0, lambda: self._log("\n✅ Installation complete!"))
        self.after(0, lambda: self.install_btn.config(text="Re-run", state="normal"))

    def _generate_configs(self) -> bool:
        """Generate configuration files."""
        # Save setup config
        config_path = PROJECT_ROOT / ".aegis_setup.json"
        self.config.save(config_path)
        self._log(f"  Saved config to {config_path}")

        # Update server config if needed
        server_config = PROJECT_ROOT / "configs" / "server_config.yaml"
        if server_config.exists():
            self._log(f"  Server config exists at {server_config}")

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
        npm_exists, _ = check_command_exists("npm")
        if not npm_exists:
            self._log("  npm not found, skipping frontend build")
            return False

        # Install dependencies
        self._log("  Installing npm dependencies...")
        success, _stdout, stderr = run_command(["npm", "install"], cwd=frontend_dir, timeout=300)
        if not success:
            self._log(f"  npm install failed: {stderr}")
            return False

        # Build
        self._log("  Building frontend...")
        success, _stdout, stderr = run_command(
            ["npm", "run", "build"], cwd=frontend_dir, timeout=300
        )
        if not success:
            self._log(f"  Build failed: {stderr}")
            return False

        self._log("  Frontend built successfully")
        return True

    def _create_scripts(self) -> bool:
        """Create launch scripts."""
        if not self.config.create_shortcuts:
            self._log("  Skipped (disabled)")
            return True

        scripts_dir = PROJECT_ROOT / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        if IS_WINDOWS:
            # Windows batch scripts
            start_server = scripts_dir / "start_server.bat"
            start_server.write_text(f"""@echo off
cd /d "{PROJECT_ROOT}"
python -m agent.server.main
pause
""")

            run_demo = scripts_dir / "run_demo.bat"
            run_demo.write_text(f"""@echo off
cd /d "{PROJECT_ROOT}"
start python -m agent.server.main
timeout /t 3
start http://localhost:8080/overlay/
start http://localhost:8080/api/docs
pause
""")
            self._log(f"  Created {start_server}")
            self._log(f"  Created {run_demo}")
        else:
            # Linux/Mac shell scripts
            start_server = scripts_dir / "start_server.sh"
            start_server.write_text(f"""#!/bin/bash
cd "{PROJECT_ROOT}"
python -m agent.server.main
""")
            start_server.chmod(0o755)

            run_demo = scripts_dir / "run_demo.sh"
            run_demo.write_text(f"""#!/bin/bash
cd "{PROJECT_ROOT}"
python -m agent.server.main &
SERVER_PID=$!
sleep 3
xdg-open http://localhost:8080/overlay/ 2>/dev/null || open http://localhost:8080/overlay/
xdg-open http://localhost:8080/api/docs 2>/dev/null || open http://localhost:8080/api/docs
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
        success, stdout, stderr = run_command(
            [sys.executable, "-c", "from agent.server.main import app; print('OK')"],
            cwd=PROJECT_ROOT,
        )

        if success and "OK" in stdout:
            self._log("  ✓ Server imports working")
            return True
        else:
            self._log(f"  Import check failed: {stderr}")
            return False

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

        for item in items:
            ttk.Label(summary_frame, text=item, font=("Helvetica", 11)).pack(anchor="w", pady=2)

        # Next steps
        next_frame = ttk.LabelFrame(self, text="Next Steps", padding=15)
        next_frame.pack(fill="x", padx=50, pady=10)

        next_text = """1. Start the server:
   ./scripts/start_server.sh  (or .bat on Windows)

2. Open the overlay in OBS:
   Add Browser Source → http://localhost:8080/overlay/

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

        ttk.Button(
            btn_frame,
            text="📖 Open Documentation",
            command=lambda: open_url(f"file://{PROJECT_ROOT}/docs/"),
        ).pack(side="left", padx=10)

        ttk.Button(
            btn_frame,
            text="🎥 Open Overlay",
            command=lambda: open_url("http://localhost:8080/overlay/"),
        ).pack(side="left", padx=10)

    def _launch_server(self) -> None:
        """Launch the server."""
        if IS_WINDOWS:
            script = PROJECT_ROOT / "scripts" / "start_server.bat"
            if script.exists():
                subprocess.Popen(["cmd", "/c", str(script)], start_new_session=True)  # noqa: S603, S607
            else:
                subprocess.Popen(  # noqa: S603
                    [  # noqa: S607
                        "cmd",
                        "/c",
                        f"cd /d {PROJECT_ROOT} && python -m agent.server.main",
                    ],
                    start_new_session=True,
                )
        else:
            script = PROJECT_ROOT / "scripts" / "start_server.sh"
            if script.exists():
                subprocess.Popen(["bash", str(script)], start_new_session=True)  # noqa: S603, S607
            else:
                subprocess.Popen(  # noqa: S603
                    [  # noqa: S607
                        "bash",
                        "-c",
                        f"cd {PROJECT_ROOT} && python -m agent.server.main",
                    ],
                    start_new_session=True,
                )

        messagebox.showinfo(
            "Server Starting",
            "Server is starting in a new terminal.\n\nWait a few seconds, then open the overlay.",
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

        # AegisAV Dark Mode Palette
        bg_void = "#09090B"
        bg_deep = "#18181B"
        fg_text = "#FAFAFA"
        accent_cyan = "#06b6d4"

        # Apply global settings
        style.configure(
            ".",
            background=bg_void,
            foreground=fg_text,
            troughcolor=bg_deep,
            selectbackground=accent_cyan,
            selectforeground=bg_void,
        )

        style.configure("TFrame", background=bg_void)
        style.configure("TLabel", background=bg_void, foreground=fg_text)
        style.configure("TButton", background=bg_deep, foreground=fg_text, borderwidth=0, focuscolor=accent_cyan)
        style.map("TButton", background=[("active", accent_cyan), ("pressed", accent_cyan)], foreground=[("active", bg_void), ("pressed", bg_void)])

        style.configure("TLabelframe", background=bg_void, foreground=accent_cyan)
        style.configure("TLabelframe.Label", background=bg_void, foreground=accent_cyan)

        style.configure("TEntry", fieldbackground=bg_deep, foreground=fg_text, insertcolor=fg_text)
        style.configure("Horizontal.TProgressbar", background=accent_cyan, troughcolor=bg_deep, bordercolor=bg_void)

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
        self.content = ttk.Frame(self.root)
        self.content.pack(fill="both", expand=True, padx=20, pady=10)

        # Separator
        ttk.Separator(self.root).pack(fill="x", padx=20)

        # Navigation buttons
        self.nav_frame = ttk.Frame(self.root)
        self.nav_frame.pack(fill="x", padx=20, pady=15)

        self.back_btn = ttk.Button(
            self.nav_frame,
            text="← Back",
            command=self._go_back,
        )
        self.back_btn.pack(side="left")

        self.next_btn = ttk.Button(
            self.nav_frame,
            text="Next →",
            command=self._go_next,
        )
        self.next_btn.pack(side="right")

        self.cancel_btn = ttk.Button(
            self.nav_frame,
            text="Cancel",
            command=self._cancel,
        )
        self.cancel_btn.pack(side="right", padx=(0, 10))

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
