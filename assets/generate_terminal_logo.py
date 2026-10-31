#!/usr/bin/env python3
"""Generate an ANSI half-cell SVG star for native CLI and server startup."""

import argparse
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def render(svg, output, rasterizer, image_tool):
    width, height = 24, 24
    with tempfile.TemporaryDirectory(prefix="betelgeuse-logo-") as temporary:
        icon = Path(temporary) / "icon.svg"
        tree = ET.parse(svg)
        tree.getroot().set("viewBox", "0 0 240 240")
        tree.write(icon)
        png = Path(temporary) / "logo.png"
        subprocess.run([rasterizer, "--width", str(width), "--height", str(height),
                        "--output", str(png), str(icon)], check=True)
        rgba = subprocess.run([image_tool, str(png), "-depth", "8", "RGBA:-"],
                              check=True, stdout=subprocess.PIPE).stdout
    if len(rgba) != width * height * 4:
        raise ValueError("unexpected raster dimensions")
    rows = []
    for y in range(0, height, 2):
        row = bytearray()
        state = None
        for x in range(width):
            top = rgba[(y * width + x) * 4:(y * width + x + 1) * 4]
            bottom = rgba[((y + 1) * width + x) * 4:((y + 1) * width + x + 1) * 4]
            top_visible, bottom_visible = top[3] >= 96, bottom[3] >= 96
            foreground = top[:3] if top_visible else bottom[:3] if bottom_visible else None
            background = bottom[:3] if top_visible and bottom_visible else None
            next_state = (foreground, background)
            if state != next_state:
                row += b"\x1b[0m"
                if foreground is not None:
                    row += f"\x1b[38;2;{foreground[0]};{foreground[1]};{foreground[2]}m".encode()
                if background is not None:
                    row += f"\x1b[48;2;{background[0]};{background[1]};{background[2]}m".encode()
                state = next_state
            row += b"\xe2\x96\x80" if top_visible else b"\xe2\x96\x84" if bottom_visible else b" "
        row += b"\x1b[0m"
        if y == 8:
            row += b"  Betelgeuse"
        elif y == 12:
            row += b"  Alpha-Orionis"
        row += b"\n"
        rows.append('    "' + ''.join(f"\\x{value:02x}" for value in row) + '"')
    content = "// Generated from assets/betelgeuse.svg.\n#pragma once\n\n"
    content += "static constexpr char betelgeuse_terminal_logo[] =\n" + "\n".join(rows) + ";\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rasterizer", required=True)
    parser.add_argument("--image-tool", required=True)
    args = parser.parse_args()
    render(args.svg, args.output, args.rasterizer, args.image_tool)
