"""Render the P1 screen-content sources from this repository's own files (docs/p1/clips.csv).

    uv run python scripts/make_screen_clips.py      # -> data/sources/screen_*.mkv (git-ignored)

Each source is a text file drawn once onto a tall page (matplotlib), then scrolled through a
1280x720 window at a constant speed, like an editor, a terminal or a document viewer. They are
rendered, not captured. The output is lossless; `vreg clips` cuts the masters from it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vreg import ff  # noqa: E402

OUT = Path("data/sources")
W, H, FPS, SECONDS = 1280, 720, 30, 4
LOSSLESS = ["-c:v", "libx264", "-preset", "veryfast", "-qp", "0", "-pix_fmt", "yuv444p"]

# name -> (file, font family, font size in px, line height in px, background, text colour, scroll px/s)
SCREENS = {
    "screen_code": ("src/vreg/precomp.py", "monospace", 18, 26, "#1e1e1e", "#d4d4d4", 60),
    "screen_terminal": ("docs/P0-plan.md", "monospace", 16, 22, "#000000", "#33ff66", 150),
    "screen_doc": ("idea.md", "sans-serif", 20, 30, "#ffffff", "#202020", 40),
    "screen_plan": ("docs/P1-plan.md", "monospace", 17, 24, "#f6f6f6", "#111111", 90),
}


def page(text: str, family: str, size: int, line_h: int, bg: str, fg: str, height: int) -> np.ndarray:
    """The text drawn onto a W x height RGB page, one line every line_h pixels."""
    dpi = 100
    fig = plt.figure(figsize=(W / dpi, height / dpi), dpi=dpi, facecolor=bg)
    for i, line in enumerate(text.expandtabs(4).splitlines()):
        fig.text(24 / W, 1 - (24 + i * line_h) / height, line, family=family, fontsize=size * 72 / dpi,
                 color=fg, va="top", ha="left", usetex=False, parse_math=False)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def render(name: str, spec: tuple) -> Path:
    path, family, size, line_h, bg, fg, speed = spec
    text = Path(path).read_text()
    n = FPS * SECONDS
    height = max(H + int(speed * SECONDS) + 2, 48 + line_h * (len(text.splitlines()) + 1))
    img = page(text, family, size, line_h, bg, fg, height)
    frames = np.stack([img[int(round(speed * i / FPS)):][:H] for i in range(n)])
    OUT.mkdir(parents=True, exist_ok=True)
    return ff.write_frames(frames, OUT / f"{name}.mkv", FPS, LOSSLESS)


if __name__ == "__main__":
    for name, spec in SCREENS.items():
        print(render(name, spec))
