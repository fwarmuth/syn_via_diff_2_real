"""Compute FID between generated and reference datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

from syn_via_diff.metrics import compute_fid_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID between two folders of images.")
    parser.add_argument("generated", type=Path, help="Directory with generated images.")
    parser.add_argument("reference", type=Path, help="Directory with reference real images.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional cap on the number of samples to use.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score = compute_fid_score(args.generated, args.reference, args.sample_limit)
    print(f"FID: {score:.3f}")


if __name__ == "__main__":
    main()
