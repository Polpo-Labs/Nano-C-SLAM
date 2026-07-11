"""
Saving simple animations as GIFs (R6).

A thin wrapper around matplotlib's FuncAnimation that writes a GIF with
PillowWriter -- Pillow ships with matplotlib, so this needs no external tools
(unlike MP4, which would require ffmpeg). The caller owns the figure and an
`update(frame_index)` function that redraws it; we just drive and save it.
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def save_gif(
    fig: plt.Figure,
    update: Callable[[int], object],
    frames: int,
    out_path: str,
    fps: int = 15,
) -> None:
    """Render `frames` frames by calling `update(i)` and save them as a GIF.

    fig    : the figure to animate (already set up by the caller).
    update : redraws the figure for frame i; called for i in range(frames).
    frames : number of frames.
    fps    : playback speed.
    """
    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    animation.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
