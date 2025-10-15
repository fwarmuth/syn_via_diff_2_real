import numpy as np
from PIL import Image

from syn_via_diff.masks import assert_iou, iou, validate_alignment


def _make_mask(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype("uint8"))


def test_validate_alignment_rejects_mismatch() -> None:
    img = Image.new("RGB", (8, 8))
    mask = Image.new("L", (4, 4))
    try:
        validate_alignment(img, mask)
    except ValueError as exc:
        assert "same resolution" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for mismatched sizes")


def test_iou_exact_match() -> None:
    arr = np.zeros((4, 4), dtype=np.uint8)
    arr[1:3, 1:3] = 1
    mask_a = _make_mask(arr)
    mask_b = _make_mask(arr.copy())
    assert iou(mask_a, mask_b) == 1.0
    assert_iou(mask_a, mask_b, threshold=0.999)


def test_iou_handles_empty_masks() -> None:
    arr = np.zeros((4, 4), dtype=np.uint8)
    mask_a = _make_mask(arr)
    mask_b = _make_mask(arr)
    assert iou(mask_a, mask_b) == 1.0
