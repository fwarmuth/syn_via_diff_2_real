"""Helpers for loading and blending LoRA adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


def apply_lora_weights(
    pipe,
    lora_paths: Sequence[Path],
    lora_weights: Sequence[float],
) -> None:
    """Apply one or multiple LoRA adapters to an existing pipeline."""
    if len(lora_paths) != len(lora_weights):
        raise ValueError("LoRA paths and weights must have the same length.")
    if not lora_paths:
        return
    for path, weight in zip(lora_paths, lora_weights):
        pipe.load_lora_weights(str(path), weight=weight)


def apply_single_lora(pipe, path: Path, weight: float) -> None:
    """Convenience helper for a single adapter."""
    apply_lora_weights(pipe, [path], [weight])
