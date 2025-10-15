# Synthetic-to-Real Diffusion Bridge

## Short Description / Motivation
Synthetic datasets are plentiful, but models trained on them often struggle when deployed on real-world imagery. This project explores how to use diffusion models to transform synthetic instance-segmentation data so that it looks like real photos while keeping the original masks unchanged. The goal is to boost downstream instance segmentation accuracy without the cost of manual data collection.

## Inputs
- Synthetic images (RGB) with corresponding instance segmentation masks.
- Optional configuration files describing diffusion parameters and dataset paths.

## Outputs
- Realistic-looking images aligned pixel-for-pixel with the original synthetic inputs.
- Original instance masks, preserved for downstream training.
- Metadata summarizing diffusion settings and processing logs.

## How It Works (high-level)
1. Load synthetic images and their masks into a preprocessing pipeline that normalizes sizes, color spaces, and annotations.
2. Feed the preprocessed data into a diffusion model fine-tuned to bridge the synthetic-to-real domain gap.
3. Decode the latent representations into images that mimic real-world appearance while maintaining spatial structure.
4. Package the generated images together with the untouched masks and optional metadata for training instance segmentation models.

## Installation
This project targets **Python 3.12** and uses the **[uv](https://github.com/astral-sh/uv)** package manager.

```bash
uv sync
```

The command resolves and installs all dependencies declared in `pyproject.toml` and creates the virtual environment if needed.

## Usage
Below is a minimal example that converts a folder of synthetic samples and writes the results to a new directory.

```bash
uv run python -m syn_via_diff.pipeline \
  --input-dir data/synthetic_dataset \
  --mask-dir data/synthetic_dataset/masks \
  --output-dir data/realistic_dataset \
  --config configs/default.yaml
```

The CLI assumes paired images and masks with matching filenames and will mirror the input folder structure under the output directory.

## Assumptions & Limitations
- Synthetic inputs and masks are perfectly aligned and share resolution; masks are not resized or warped.
- Diffusion checkpoints and configurations are provided by the user; the repo does not ship large model weights.
- The pipeline currently targets instance segmentation use cases and may require adaptation for other tasks (e.g., detection-only or semantic segmentation).
- Realism improvements depend on the quality of the fine-tuned diffusion model and may vary across domains.

## Potential Extensions / To Do
- Experiment with domain-specific conditioning (e.g., lighting, weather, camera parameters).
- Add automatic evaluation metrics comparing synthetic, transformed, and real datasets.
- Integrate active learning loops to refine diffusion checkpoints with human-in-the-loop feedback.
- Provide pre-trained checkpoints and dataset loaders for popular synthetic benchmarks.

## License & Acknowledgements
Copyright © 2024. See `LICENSE` for terms.  
Diffusion methodology inspired by recent work on synthetic-to-real domain adaptation; please cite upstream models and datasets where appropriate.
