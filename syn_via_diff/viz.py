"""Visualization helpers for QA overlays."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image, ImageDraw


def overlay_mask(image: Image.Image, mask: Image.Image, color=(255, 0, 0), alpha: float = 0.35) -> Image.Image:
    """Overlay a mask boundary on top of an image."""
    base = image.convert("RGBA")
    mask_rgba = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_rgba)

    mask_arr = np.array(mask.convert("L"))
    edges = _mask_to_edges(mask_arr)
    edge_coords = list(zip(*np.where(edges)))
    for y, x in edge_coords:
        draw.point((x, y), fill=color + (int(alpha * 255),))
    return Image.alpha_composite(base, mask_rgba).convert("RGB")


def _mask_to_edges(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="edge")
    shifted = [
        padded[2:, 1:-1],
        padded[:-2, 1:-1],
        padded[1:-1, 2:],
        padded[1:-1, :-2],
    ]
    edges = np.zeros_like(mask, dtype=bool)
    for arr in shifted:
        edges |= arr != mask
    return edges
