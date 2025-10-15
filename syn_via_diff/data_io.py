"""Data loading utilities for paired image and mask datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

from PIL import Image


@dataclass(slots=True)
class SamplePair:
    """Represents a paired synthetic image and segmentation mask."""

    image_path: Path
    mask_path: Path
    image_relative: Path
    mask_relative: Path

    def load_image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")

    def load_mask(self) -> Image.Image:
        return Image.open(self.mask_path)


def _resolve_files(base: Path, patterns: Sequence[str], recursive: bool) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        if recursive:
            files.extend(base.rglob(pattern))
        else:
            files.extend(base.glob(pattern))
    return sorted({path for path in files if path.is_file()})


def discover_pairs(
    image_dir: Path, mask_dir: Path, patterns: Sequence[str], recursive: bool = True
) -> List[SamplePair]:
    """Discover image/mask pairs by filename.

    Args:
        image_dir: Directory of source RGB images.
        mask_dir: Directory containing segmentation masks.
        patterns: Glob patterns for image files.
        recursive: Recursively search through subdirectories when True.

    Returns:
        List of :class:`SamplePair` objects mapping each image to its mask.
    """

    if not image_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_paths = _resolve_files(image_dir, patterns, recursive)
    mask_lookup = {path.relative_to(mask_dir): path for path in _resolve_files(mask_dir, patterns, recursive)}
    mask_lookup_stem = {
        (relative.parent, relative.stem): path for relative, path in mask_lookup.items()
    }

    pairs: List[SamplePair] = []
    for image_path in image_paths:
        relative = image_path.relative_to(image_dir)
        mask_path = mask_lookup.get(relative)
        if mask_path is None:
            key = (relative.parent, relative.stem)
            mask_path = mask_lookup_stem.get(key)
        if mask_path is None:
            raise FileNotFoundError(f"Mask missing for {relative} under {mask_dir}")
        mask_relative = mask_path.relative_to(mask_dir)
        pairs.append(
            SamplePair(
                image_path=image_path,
                mask_path=mask_path,
                image_relative=relative,
                mask_relative=mask_relative,
            )
        )
    return pairs


def ensure_output_structure(output_root: Path, sample: SamplePair) -> Path:
    """Create the output directory for the given sample and return the target path."""
    target_dir = output_root.joinpath(sample.image_relative).parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def write_metadata(target: Path, payload: dict) -> None:
    """Write JSON metadata next to the generated image."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def iter_batches(pairs: Sequence[SamplePair], batch_size: int) -> Iterator[Sequence[SamplePair]]:
    """Yield fixed-size batches from the discovered pairs."""
    batch: List[SamplePair] = []
    for pair in pairs:
        batch.append(pair)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
