#!/usr/bin/env python3
"""Generate PNG and ICO assets from the AegisAV logo design.

This script creates logo assets for the installer without requiring
external SVG conversion tools. It renders the shield logo using PIL.
"""

from pathlib import Path
import math
import struct
import io

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("PIL/Pillow is required. Install with: pip install Pillow")
    exit(1)

# Colors from design tokens
COLORS = {
    "bg_deep": "#09090B",
    "accent_cyber": "#06B6D4",
    "text_primary": "#FAFAFA",
}


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def draw_shield_logo(size: int = 512, padding: int = 32) -> Image.Image:
    """Draw the AegisAV shield logo.

    Args:
        size: Output image size (square)
        padding: Padding around the shield

    Returns:
        PIL Image with the rendered logo
    """
    img = Image.new("RGBA", (size, size), hex_to_rgb(COLORS["bg_deep"]) + (255,))
    draw = ImageDraw.Draw(img)

    # Scale factor for different sizes
    scale = size / 512

    # Shield outline points (from SVG path, centered at 256, scaled)
    # The shield shape: pointed bottom, curved sides, flat top
    cx, cy = size // 2, size // 2

    # Shield dimensions
    shield_width = int(384 * scale)
    shield_height = int(448 * scale)
    top_y = int(32 * scale) + padding // 2
    bottom_y = int(480 * scale) - padding // 2

    # Create shield path points
    shield_points = []

    # Top edge (flat)
    left_x = cx - shield_width // 2
    right_x = cx + shield_width // 2

    # Shield outline using bezier-like curve approximation
    # Top left to center bottom to top right
    num_points = 50

    for i in range(num_points + 1):
        t = i / num_points
        if t <= 0.5:
            # Left side: from top-left curving down to bottom point
            tt = t * 2  # 0 to 1 for left side
            x = left_x + (cx - left_x) * (tt ** 0.7)
            y = top_y + (bottom_y - top_y) * tt
        else:
            # Right side: from bottom point curving up to top-right
            tt = (t - 0.5) * 2  # 0 to 1 for right side
            x = cx + (right_x - cx) * (tt ** 0.7)
            y = bottom_y - (bottom_y - top_y) * tt
        shield_points.append((x, y))

    # Draw shield outline
    stroke_width = max(2, int(32 * scale / 4))
    cyan = hex_to_rgb(COLORS["accent_cyber"])

    # Draw as polygon outline
    draw.polygon(shield_points, outline=cyan, width=stroke_width)

    # Draw the three lines from center
    center_x, center_y = cx, cy
    line_width = max(2, int(24 * scale / 4))
    line_length = int(96 * scale)

    # Line to upper-right
    end_x = center_x + int(line_length * 0.707)
    end_y = center_y - int(line_length * 0.707)
    draw.line([(center_x, center_y), (end_x, end_y)], fill=cyan, width=line_width)

    # Line to upper-left
    end_x = center_x - int(line_length * 0.707)
    end_y = center_y - int(line_length * 0.707)
    draw.line([(center_x, center_y), (end_x, end_y)], fill=cyan, width=line_width)

    # Line downward
    end_y = center_y + int(128 * scale)
    draw.line([(center_x, center_y), (center_x, end_y)], fill=cyan, width=line_width)

    # Center circle
    circle_radius = int(48 * scale)
    circle_stroke = max(2, int(8 * scale / 2))

    # Filled cyan circle
    draw.ellipse(
        [
            center_x - circle_radius,
            center_y - circle_radius,
            center_x + circle_radius,
            center_y + circle_radius,
        ],
        fill=cyan,
        outline=hex_to_rgb(COLORS["text_primary"]),
        width=circle_stroke,
    )

    return img


def create_ico_file(images: list[Image.Image], output_path: Path) -> None:
    """Create an ICO file from multiple PNG images.

    Args:
        images: List of PIL Images at different sizes
        output_path: Path to write the ICO file
    """
    # ICO format: header + directory entries + image data
    # Use PNG compression within ICO (supported since Windows Vista)

    num_images = len(images)

    # Collect PNG data for each image
    png_data_list = []
    for img in images:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_data_list.append(buffer.getvalue())

    # ICO Header: 6 bytes
    # Reserved (2) + Type (2, 1=ICO) + Count (2)
    header = struct.pack("<HHH", 0, 1, num_images)

    # Directory entries: 16 bytes each
    # Width, Height, Colors, Reserved, Planes, BitCount, Size, Offset
    directory = b""
    offset = 6 + (16 * num_images)  # After header and all directory entries

    for i, (img, png_data) in enumerate(zip(images, png_data_list)):
        width = img.width if img.width < 256 else 0
        height = img.height if img.height < 256 else 0
        entry = struct.pack(
            "<BBBBHHII",
            width,      # Width (0 = 256)
            height,     # Height (0 = 256)
            0,          # Color palette (0 for true color)
            0,          # Reserved
            1,          # Color planes
            32,         # Bits per pixel
            len(png_data),  # Size of image data
            offset,     # Offset to image data
        )
        directory += entry
        offset += len(png_data)

    # Write the ICO file
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(directory)
        for png_data in png_data_list:
            f.write(png_data)


def main():
    """Generate all logo assets."""
    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(exist_ok=True)

    print("Generating AegisAV logo assets...")

    # Generate PNG files at different sizes
    sizes = [
        (16, "aegis_logo_16.png"),
        (32, "aegis_logo_32.png"),
        (48, "aegis_logo_48.png"),
        (64, "aegis_logo_64.png"),
        (128, "aegis_logo_128.png"),
        (256, "aegis_logo_256.png"),
    ]

    images_for_ico = []

    for size, filename in sizes:
        print(f"  Creating {filename}...")
        img = draw_shield_logo(size=size, padding=max(2, size // 16))
        img.save(assets_dir / filename, "PNG")
        images_for_ico.append(img)

    # Create Linux icon (128x128)
    print("  Creating aegis_icon.png (Linux)...")
    linux_icon = draw_shield_logo(size=128, padding=8)
    linux_icon.save(assets_dir / "aegis_icon.png", "PNG")

    # Create ICO file for Windows with multiple sizes
    print("  Creating aegis_icon.ico (Windows)...")
    ico_sizes = [16, 32, 48, 64, 128, 256]
    ico_images = [draw_shield_logo(size=s, padding=max(2, s // 16)) for s in ico_sizes]
    create_ico_file(ico_images, assets_dir / "aegis_icon.ico")

    print("\nAssets created successfully in:", assets_dir)
    print("\nFiles created:")
    for f in sorted(assets_dir.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
