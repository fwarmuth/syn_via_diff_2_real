from pathlib import Path

from PIL import Image

from syn_via_diff.data_io import discover_pairs


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def _write_mask(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=value).save(path)


def test_discover_pairs_matches_by_stem(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"

    img_path = image_dir / "scene" / "sample01.jpg"
    mask_path = mask_dir / "scene" / "sample01.png"
    _write_image(img_path, (255, 0, 0))
    _write_mask(mask_path, 255)

    pairs = discover_pairs(image_dir, mask_dir, patterns=["*.jpg", "*.png"], recursive=True)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.image_relative == Path("scene/sample01.jpg")
    assert pair.mask_relative == Path("scene/sample01.png")
    assert pair.image_path.exists()
    assert pair.mask_path.exists()


def test_discover_pairs_missing_mask_raises(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    _write_image(image_dir / "a.png", (0, 0, 0))

    try:
        discover_pairs(image_dir, mask_dir, patterns=["*.png"], recursive=False)
    except FileNotFoundError as exc:
        assert "Mask missing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected FileNotFoundError when mask is missing")
