"""Tile-based inference helper for high-resolution inputs."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from syn_via_diff.config import AppConfig
from syn_via_diff.controlnet_runner import ControlNetRunner
from syn_via_diff.masks import assert_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiled diffusion inference on a single image/mask pair.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config matching the main pipeline.")
    parser.add_argument("--image", type=Path, required=True, help="Input synthetic image.")
    parser.add_argument("--mask", type=Path, required=True, help="Aligned segmentation mask.")
    parser.add_argument("--output", type=Path, required=True, help="Destination for the generated image.")
    parser.add_argument("--output-mask", type=Path, required=True, help="Destination for the copied mask.")
    parser.add_argument("--tile-size", type=int, default=768, help="Tile size for inference.")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between tiles to reduce seams.")
    parser.add_argument("--seed", type=int, default=123, help="Base seed for deterministic tiles.")
    return parser.parse_args()


def _tile_positions(width: int, height: int, tile: int, overlap: int) -> Tuple[int, int, int, int]:
    step = max(tile - overlap, 1)
    for top in range(0, height, step):
        for left in range(0, width, step):
            bottom = min(top + tile, height)
            right = min(left + tile, width)
            yield left, top, right, bottom


def tile_and_infer(runner: ControlNetRunner, image: Image.Image, mask: Image.Image, tile: int, overlap: int, seed: int) -> Image.Image:
    width, height = image.size
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    for index, (left, top, right, bottom) in enumerate(_tile_positions(width, height, tile, overlap)):
        image_tile = image.crop((left, top, right, bottom))
        mask_tile = mask.crop((left, top, right, bottom))
        generated = runner.stylize(image_tile, mask_tile, seed=seed + index)
        tile_array = np.asarray(generated).astype(np.float32)
        canvas[top:bottom, left:right] += tile_array
        counts[top:bottom, left:right] += 1.0

    counts[counts == 0] = 1.0
    averaged = canvas / counts[..., None]
    averaged = np.clip(averaged, 0, 255).astype(np.uint8)
    return Image.fromarray(averaged)


def main() -> None:
    args = parse_args()
    config = AppConfig.from_yaml(args.config)

    runner = ControlNetRunner.from_config(config.model, config.infer, config.prompts)

    with Image.open(args.image) as image, Image.open(args.mask) as mask:
        generated = tile_and_infer(runner, image.convert("RGB"), mask.convert("RGB"), args.tile_size, args.overlap, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        generated.save(args.output)
        args.output_mask.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.mask, args.output_mask)

        with Image.open(args.output_mask) as mask_out:
            assert_iou(mask, mask_out, config.eval.iou_threshold)

    runner.close()


if __name__ == "__main__":
    main()
