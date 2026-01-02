"""AegisAV Installer Branding and Theme Management.

Provides dark/light theme support and logo loading for the installer GUI.
Colors and design tokens match the main AegisAV dashboard design system.
"""

from __future__ import annotations

import platform
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

# Asset directory relative to this file
ASSETS_DIR = Path(__file__).parent / "assets"


@dataclass
class AegisTheme:
    """Design tokens for the AegisAV installer theme.

    Colors are derived from shared/styles/design-tokens.css to maintain
    visual consistency across the AegisAV ecosystem.
    """

    # Background colors
    bg_deep: str = "#09090B"        # Primary dark background
    bg_onyx: str = "#18181B"        # Elevated surface
    bg_elevated: str = "#27272A"    # Card/panel backgrounds
    bg_input: str = "#3F3F46"       # Input field background

    # Text colors
    text_primary: str = "#FAFAFA"   # Primary text (white)
    text_secondary: str = "#D4D4D8" # Secondary text
    text_muted: str = "#71717A"     # Muted/placeholder text
    text_dim: str = "#52525B"       # Very dim text

    # Accent colors
    accent_cyber: str = "#06B6D4"   # Primary cyan accent
    accent_security: str = "#10B981"  # Green (success)
    accent_alert: str = "#EF4444"     # Red (error/danger)
    accent_warning: str = "#F59E0B"   # Orange (warning)
    accent_purple: str = "#8B5CF6"    # Purple (info)

    # Borders
    border_subtle: str = "#27272A"
    border_default: str = "#3F3F46"

    # Status colors
    status_success: str = "#10B981"
    status_error: str = "#EF4444"
    status_warning: str = "#F59E0B"
    status_pending: str = "#71717A"


# Light theme - matches the original installer colors
LIGHT_THEME = AegisTheme(
    bg_deep="#F7F6F3",
    bg_onyx="#FFFFFF",
    bg_elevated="#FFFFFF",
    bg_input="#FFFFFF",
    text_primary="#1F2933",
    text_secondary="#52606D",
    text_muted="#9AA5B1",
    text_dim="#CBD2D9",
    accent_cyber="#2563EB",
    accent_security="#10B981",
    accent_alert="#EF4444",
    accent_warning="#F59E0B",
    accent_purple="#8B5CF6",
    border_subtle="#E4E7EB",
    border_default="#D6D3D1",
    status_success="#10B981",
    status_error="#EF4444",
    status_warning="#F59E0B",
    status_pending="#9AA5B1",
)

# Dark theme - matches dashboard design tokens
DARK_THEME = AegisTheme()  # Uses defaults which are dark theme values


class ThemeManager:
    """Manage dark/light/system theme switching for the installer.

    Applies consistent styling to all ttk widgets based on the selected theme.
    """

    def __init__(self, style: ttk.Style, root: tk.Tk) -> None:
        """Initialize the theme manager.

        Args:
            style: The ttk.Style instance to configure
            root: The root Tk window
        """
        self.style = style
        self.root = root
        self.current_theme: Literal["dark", "light", "system"] = "light"
        self._theme: AegisTheme = LIGHT_THEME

    @property
    def theme(self) -> AegisTheme:
        """Get the current theme colors."""
        return self._theme

    def apply_theme(self, theme: Literal["dark", "light", "system"]) -> None:
        """Apply the specified theme.

        Args:
            theme: The theme to apply - "dark", "light", or "system"
        """
        self.current_theme = theme

        if theme == "system":
            # Detect system theme preference
            is_dark = self._detect_system_dark_mode()
            self._theme = DARK_THEME if is_dark else LIGHT_THEME
        elif theme == "dark":
            self._theme = DARK_THEME
        else:
            self._theme = LIGHT_THEME

        self._apply_theme_styles()

    def _detect_system_dark_mode(self) -> bool:
        """Detect if the system is using dark mode.

        Returns:
            True if system appears to be in dark mode
        """
        system = platform.system()

        if system == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
                )
                value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                winreg.CloseKey(key)
                return value == 0  # 0 = dark mode
            except Exception:
                return False

        elif system == "Darwin":
            try:
                import subprocess
                result = subprocess.run(
                    ["defaults", "read", "-g", "AppleInterfaceStyle"],
                    capture_output=True,
                    text=True,
                )
                return "dark" in result.stdout.lower()
            except Exception:
                return False

        elif system == "Linux":
            # Check common desktop environment settings
            try:
                import subprocess
                # Try GNOME
                result = subprocess.run(
                    ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                    capture_output=True,
                    text=True,
                )
                if "dark" in result.stdout.lower():
                    return True

                # Try GTK theme name
                result = subprocess.run(
                    ["gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"],
                    capture_output=True,
                    text=True,
                )
                return "dark" in result.stdout.lower()
            except Exception:
                return False

        return False

    def _apply_theme_styles(self) -> None:
        """Apply theme colors to all ttk widget styles."""
        t = self._theme

        # Configure the root window background
        self.root.configure(bg=t.bg_deep)

        # Base style for all widgets
        self.style.configure(
            ".",
            background=t.bg_deep,
            foreground=t.text_primary,
            fieldbackground=t.bg_input,
            troughcolor=t.bg_elevated,
            bordercolor=t.border_default,
            darkcolor=t.bg_onyx,
            lightcolor=t.bg_elevated,
        )

        # Frame styles
        self.style.configure("TFrame", background=t.bg_deep)
        self.style.configure("Elevated.TFrame", background=t.bg_onyx)
        self.style.configure("Card.TFrame", background=t.bg_elevated)

        # Label styles
        self.style.configure(
            "TLabel",
            background=t.bg_deep,
            foreground=t.text_primary,
        )
        self.style.configure(
            "Title.TLabel",
            background=t.bg_deep,
            foreground=t.text_primary,
            font=("Helvetica", 24, "bold"),
        )
        self.style.configure(
            "Subtitle.TLabel",
            background=t.bg_deep,
            foreground=t.text_secondary,
            font=("Helvetica", 11),
        )
        self.style.configure(
            "Muted.TLabel",
            background=t.bg_deep,
            foreground=t.text_muted,
        )
        self.style.configure(
            "Success.TLabel",
            background=t.bg_deep,
            foreground=t.status_success,
        )
        self.style.configure(
            "Error.TLabel",
            background=t.bg_deep,
            foreground=t.status_error,
        )
        self.style.configure(
            "Warning.TLabel",
            background=t.bg_deep,
            foreground=t.status_warning,
        )
        self.style.configure(
            "Accent.TLabel",
            background=t.bg_deep,
            foreground=t.accent_cyber,
        )

        # Button styles
        self.style.configure(
            "TButton",
            background=t.bg_elevated,
            foreground=t.text_primary,
            padding=(12, 6),
            borderwidth=1,
        )
        self.style.map(
            "TButton",
            background=[
                ("active", t.accent_cyber),
                ("pressed", t.accent_cyber),
            ],
            foreground=[
                ("active", t.bg_deep),
                ("pressed", t.bg_deep),
            ],
        )

        self.style.configure(
            "Primary.TButton",
            background=t.accent_cyber,
            foreground=t.bg_deep,
        )
        self.style.map(
            "Primary.TButton",
            background=[
                ("active", t.text_primary),
                ("pressed", t.text_secondary),
            ],
        )

        self.style.configure(
            "Danger.TButton",
            background=t.accent_alert,
            foreground=t.text_primary,
        )

        # Entry/Input styles
        self.style.configure(
            "TEntry",
            fieldbackground=t.bg_input,
            foreground=t.text_primary,
            insertcolor=t.text_primary,
            bordercolor=t.border_default,
        )
        self.style.map(
            "TEntry",
            fieldbackground=[("focus", t.bg_onyx)],
            bordercolor=[("focus", t.accent_cyber)],
        )

        # Checkbutton and Radiobutton
        self.style.configure(
            "TCheckbutton",
            background=t.bg_deep,
            foreground=t.text_primary,
        )
        self.style.map(
            "TCheckbutton",
            background=[("active", t.bg_deep)],
            foreground=[("active", t.accent_cyber)],
        )

        self.style.configure(
            "TRadiobutton",
            background=t.bg_deep,
            foreground=t.text_primary,
        )
        self.style.map(
            "TRadiobutton",
            background=[("active", t.bg_deep)],
            foreground=[("active", t.accent_cyber)],
        )

        # LabelFrame
        self.style.configure(
            "TLabelframe",
            background=t.bg_deep,
            foreground=t.text_primary,
            bordercolor=t.border_default,
        )
        self.style.configure(
            "TLabelframe.Label",
            background=t.bg_deep,
            foreground=t.text_secondary,
            font=("Helvetica", 10, "bold"),
        )

        # Progressbar
        self.style.configure(
            "TProgressbar",
            background=t.accent_cyber,
            troughcolor=t.bg_elevated,
            bordercolor=t.border_subtle,
            lightcolor=t.accent_cyber,
            darkcolor=t.accent_cyber,
        )

        self.style.configure(
            "Success.Horizontal.TProgressbar",
            background=t.status_success,
        )
        self.style.configure(
            "Warning.Horizontal.TProgressbar",
            background=t.status_warning,
        )
        self.style.configure(
            "Error.Horizontal.TProgressbar",
            background=t.status_error,
        )

        # Combobox
        self.style.configure(
            "TCombobox",
            fieldbackground=t.bg_input,
            background=t.bg_elevated,
            foreground=t.text_primary,
            arrowcolor=t.text_secondary,
            bordercolor=t.border_default,
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", t.bg_elevated)],
            foreground=[("readonly", t.text_primary)],
            bordercolor=[("focus", t.accent_cyber)],
        )

        # Scrollbar
        self.style.configure(
            "TScrollbar",
            background=t.bg_elevated,
            troughcolor=t.bg_onyx,
            arrowcolor=t.text_muted,
        )

        # Notebook (tabs)
        self.style.configure(
            "TNotebook",
            background=t.bg_deep,
            bordercolor=t.border_subtle,
        )
        self.style.configure(
            "TNotebook.Tab",
            background=t.bg_elevated,
            foreground=t.text_secondary,
            padding=(12, 6),
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", t.bg_deep)],
            foreground=[("selected", t.accent_cyber)],
        )

        # Separator
        self.style.configure(
            "TSeparator",
            background=t.border_subtle,
        )

        # Treeview (if used)
        self.style.configure(
            "Treeview",
            background=t.bg_onyx,
            foreground=t.text_primary,
            fieldbackground=t.bg_onyx,
            bordercolor=t.border_subtle,
        )
        self.style.configure(
            "Treeview.Heading",
            background=t.bg_elevated,
            foreground=t.text_secondary,
        )
        self.style.map(
            "Treeview",
            background=[("selected", t.accent_cyber)],
            foreground=[("selected", t.bg_deep)],
        )

    def get_text_widget_config(self) -> dict:
        """Get configuration dict for tk.Text widgets (not ttk).

        Returns:
            Dict of configuration options for Text widgets
        """
        t = self._theme
        return {
            "bg": t.bg_onyx,
            "fg": t.text_primary,
            "insertbackground": t.text_primary,
            "selectbackground": t.accent_cyber,
            "selectforeground": t.bg_deep,
            "relief": "flat",
            "borderwidth": 1,
            "highlightthickness": 1,
            "highlightbackground": t.border_default,
            "highlightcolor": t.accent_cyber,
        }

    def get_canvas_config(self) -> dict:
        """Get configuration dict for tk.Canvas widgets.

        Returns:
            Dict of configuration options for Canvas widgets
        """
        t = self._theme
        return {
            "bg": t.bg_deep,
            "highlightthickness": 0,
        }


def load_logo_image(size: int = 48) -> tk.PhotoImage | None:
    """Load the AegisAV logo as a PhotoImage.

    Args:
        size: Desired logo size (will use closest available)

    Returns:
        PhotoImage if logo found, None otherwise
    """
    # Map requested size to available sizes
    available_sizes = [16, 32, 48, 64, 128, 256]
    closest_size = min(available_sizes, key=lambda x: abs(x - size))

    logo_path = ASSETS_DIR / f"aegis_logo_{closest_size}.png"

    if not logo_path.exists():
        return None

    try:
        return tk.PhotoImage(file=str(logo_path))
    except Exception:
        return None


def create_logo_canvas(
    parent: tk.Widget,
    size: int = 48,
    theme: AegisTheme | None = None,
) -> tk.Canvas:
    """Create a canvas widget with the AegisAV shield logo drawn on it.

    This is a fallback when PNG logos are not available. It renders
    a simplified version of the shield logo using canvas primitives.

    Args:
        parent: Parent widget
        size: Size of the canvas (square)
        theme: Theme to use for colors (defaults to dark theme)

    Returns:
        Canvas widget with the logo drawn
    """
    if theme is None:
        theme = DARK_THEME

    canvas = tk.Canvas(
        parent,
        width=size,
        height=size,
        bg=theme.bg_deep,
        highlightthickness=0,
    )

    # Scale factor
    s = size / 64

    # Center point
    cx, cy = size // 2, size // 2

    # Draw simplified shield outline
    shield_points = [
        cx, int(4 * s),           # Top center
        int(cx + 26 * s), int(10 * s),   # Top right
        int(cx + 28 * s), int(20 * s),   # Right upper
        int(cx + 24 * s), int(40 * s),   # Right lower
        cx, int(60 * s),          # Bottom point
        int(cx - 24 * s), int(40 * s),   # Left lower
        int(cx - 28 * s), int(20 * s),   # Left upper
        int(cx - 26 * s), int(10 * s),   # Top left
    ]

    canvas.create_polygon(
        shield_points,
        outline=theme.accent_cyber,
        fill="",
        width=max(1, int(2 * s)),
    )

    # Draw center lines
    line_len = int(12 * s)
    line_width = max(1, int(1.5 * s))

    # Upper right line
    canvas.create_line(
        cx, cy,
        cx + int(line_len * 0.707), cy - int(line_len * 0.707),
        fill=theme.accent_cyber,
        width=line_width,
    )

    # Upper left line
    canvas.create_line(
        cx, cy,
        cx - int(line_len * 0.707), cy - int(line_len * 0.707),
        fill=theme.accent_cyber,
        width=line_width,
    )

    # Down line
    canvas.create_line(
        cx, cy,
        cx, cy + int(16 * s),
        fill=theme.accent_cyber,
        width=line_width,
    )

    # Center circle
    r = int(6 * s)
    canvas.create_oval(
        cx - r, cy - r,
        cx + r, cy + r,
        fill=theme.accent_cyber,
        outline=theme.text_primary,
        width=max(1, int(0.5 * s)),
    )

    return canvas


def get_icon_path(for_windows: bool = True) -> Path | None:
    """Get the path to the appropriate icon file.

    Args:
        for_windows: If True, returns .ico path; otherwise .png

    Returns:
        Path to icon file if it exists, None otherwise
    """
    if for_windows:
        icon_path = ASSETS_DIR / "aegis_icon.ico"
    else:
        icon_path = ASSETS_DIR / "aegis_icon.png"

    return icon_path if icon_path.exists() else None
