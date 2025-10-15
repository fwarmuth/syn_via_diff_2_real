"""CLI entry point for the synthetic-to-real diffusion pipeline."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import typer
from rich.progress import Progress

from .config import AppConfig
from .controlnet_runner import ControlNetRunner
from .data_io import SamplePair, discover_pairs, ensure_output_structure, write_metadata
from .masks import assert_iou, validate_alignment
from .metrics import compute_fid_score
from .style_bank import StyleBank

LOGGER = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=True)


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="[%(levelname)s] %(name)s: %(message)s")


def _load_config(config_path: Path) -> AppConfig:
    return AppConfig.from_yaml(config_path)


def _prepare_style_bank(config: AppConfig) -> StyleBank:
    variations = config.prompts.bank or []
    return StyleBank(config.prompts.positive, variations=variations)


def _process_sample(
    runner: ControlNetRunner,
    sample: SamplePair,
    config: AppConfig,
    style_bank: StyleBank,
    index: int,
) -> dict:
    image = sample.load_image()
    mask = sample.load_mask()
    try:
        validate_alignment(image, mask)

        prompt = style_bank.next_prompt(index)
        generated = runner.stylize(
            image=image,
            mask=mask,
            seed=config.infer.seed + index,
            positive_prompt=prompt,
            negative_prompt=config.prompts.negative,
        )

        image_out_path = config.data.output_dir / sample.image_relative
        mask_out_path = config.data.output_mask_dir / sample.mask_relative

        ensure_output_structure(config.data.output_dir, sample)
        mask_out_path.parent.mkdir(parents=True, exist_ok=True)

        generated.save(image_out_path)
        shutil.copy2(sample.mask_path, mask_out_path)

        metadata = {
            "prompt": prompt,
            "negative_prompt": config.prompts.negative,
            "seed": config.infer.seed + index,
            "num_inference_steps": config.infer.num_inference_steps,
            "guidance_scale": config.infer.guidance_scale,
            "controlnet_conditioning_scale": config.infer.controlnet_conditioning_scale,
            "denoising_strength": config.infer.denoising_strength,
            "lora_weight": config.model.lora_weight,
            "source_image": str(sample.image_path),
            "source_mask": str(sample.mask_path),
            "output_image": str(image_out_path),
            "output_mask": str(mask_out_path),
        }

        if config.eval.check_alignment:
            from PIL import Image

            with Image.open(mask_out_path) as mask_out:
                score = assert_iou(mask, mask_out, config.eval.iou_threshold)
            metadata["mask_iou"] = score

        if config.logging.write_metadata:
            metadata_path = image_out_path.with_suffix(".json")
            write_metadata(metadata_path, metadata)

        return metadata
    finally:
        if hasattr(image, "close"):
            image.close()
        if hasattr(mask, "close"):
            mask.close()


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, readable=True),
) -> None:
    """Run the synthetic-to-real pipeline using a YAML configuration."""
    cfg = _load_config(config)
    _setup_logging(cfg.logging.verbose)

    cfg.data.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.data.output_mask_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(
        cfg.data.input_dir,
        cfg.data.mask_dir,
        cfg.data.patterns,
        recursive=cfg.data.recursive,
    )
    style_bank = _prepare_style_bank(cfg)

    runner = ControlNetRunner.from_config(cfg.model, cfg.infer, cfg.prompts)

    metadata_records = []
    with Progress() as progress:
        task = progress.add_task("Generating", total=len(pairs))
        for idx, sample in enumerate(pairs):
            record = _process_sample(runner, sample, cfg, style_bank, idx)
            metadata_records.append(record)
            progress.advance(task)

    runner.close()

    if cfg.eval.compute_fid and cfg.eval.real_ref_dir:
        if not cfg.eval.real_ref_dir.exists():
            LOGGER.warning("Reference directory %s does not exist; skipping FID.", cfg.eval.real_ref_dir)
        else:
            try:
                fid_score = compute_fid_score(cfg.data.output_dir, cfg.eval.real_ref_dir, cfg.eval.sample_limit)
                LOGGER.info("FID score: %.3f", fid_score)
            except Exception as exc:  # pragma: no cover - metrics optional
                LOGGER.warning("Failed to compute FID: %s", exc)

    summary_path = cfg.data.output_dir / "generation_summary.json"
    summary_path.write_text(json.dumps(metadata_records, indent=2), encoding="utf-8")
    LOGGER.info("Wrote generation summary to %s", summary_path)


if __name__ == "__main__":
    app()
