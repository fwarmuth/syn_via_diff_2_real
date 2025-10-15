"""Evaluation utilities (FID, LPIPS, histogram checks)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional heavy deps
    from cleanfid import fid
except ImportError:  # pragma: no cover
    fid = None


def compute_fid_score(generated_dir: Path, reference_dir: Path, max_samples: Optional[int] = None) -> float:
    """Compute the FID score between two directories of images."""
    if fid is None:
        raise ImportError("cleanfid is required for FID computation. Install with `uv add clean-fid`.")
    LOGGER.info("Computing FID between %s and %s", generated_dir, reference_dir)
    return float(
        fid.compute_fid(
            str(generated_dir),
            str(reference_dir),
            num_workers=0,
            max_items=max_samples,
        )
    )
