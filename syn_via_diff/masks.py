"""Utilities for mask validation and alignment checks."""
from __future__ import annotations

import numpy as np
from PIL import Image


def validate_alignment(image: Image.Image, mask: Image.Image) -> None:
    """Raise if the image and mask dimensions differ."""
    if image.size != mask.size:
        raise ValueError(
            "Image and mask must share the same resolution, "
            f"but received image={image.size}, mask={mask.size}."
        )


def _to_numpy(mask: Image.Image) -> np.ndarray:
    arr = np.array(mask)
    if arr.ndim == 3:  # convert RGB mask to single channel if possible
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.view(dtype=np.uint32).reshape(arr.shape[:2])
    return arr


def iou(mask_a: Image.Image, mask_b: Image.Image) -> float:
    """Compute the Intersection over Union between two masks."""
    a = _to_numpy(mask_a)
    b = _to_numpy(mask_b)
    if a.shape != b.shape:
        raise ValueError("Masks must have identical spatial shape for IoU computation.")
    intersection = np.sum((a == b) & (a != 0))
    union = np.sum((a != 0) | (b != 0))
    if union == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(intersection) / float(union)


def assert_iou(mask_a: Image.Image, mask_b: Image.Image, threshold: float) -> float:
    """Assert that IoU exceeds the provided threshold."""
    score = iou(mask_a, mask_b)
    if score < threshold:
        raise ValueError(f"Mask alignment IoU {score:.6f} fell below threshold {threshold}.")
    return score
