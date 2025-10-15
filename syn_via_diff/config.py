"""Configuration schema and helpers for the synthetic-to-real diffusion pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import yaml
from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    """Settings for locating the input dataset and writing outputs."""

    input_dir: Path = Field(..., description="Directory containing synthetic RGB images.")
    mask_dir: Path = Field(..., description="Directory containing segmentation masks.")
    output_dir: Path = Field(..., description="Directory where generated images are written.")
    output_mask_dir: Path = Field(..., description="Directory where copied masks are written.")
    recursive: bool = Field(True, description="Whether to search input directories recursively.")
    patterns: Sequence[str] = Field(
        default_factory=lambda: ["*.png", "*.jpg", "*.jpeg"],
        description="Glob patterns for discovering image files.",
    )

    @validator("output_dir", "output_mask_dir", pre=True)
    def _expand_output(cls, value: Path) -> Path:  # type: ignore[override]
        return Path(value).expanduser().resolve()

    @validator("input_dir", "mask_dir", pre=True)
    def _expand_input(cls, value: Path) -> Path:  # type: ignore[override]
        return Path(value).expanduser().resolve()


class ModelConfig(BaseModel):
    """Stable Diffusion, ControlNet, and LoRA configuration."""

    base: str = Field(..., description="Base Stable Diffusion checkpoint identifier.")
    controlnet: Optional[str] = Field(
        None, description="HuggingFace hub path or local directory for the ControlNet checkpoint."
    )
    lora_path: Optional[Path] = Field(
        None, description="Path to the LoRA weights (.safetensors)."
    )
    lora_weight: float = Field(0.7, ge=0.0, le=2.0, description="Scalar applied to the LoRA weights.")
    dtype: str = Field("fp16", description="Torch dtype for model weights (fp16/bf16/fp32).")
    device: str = Field("cuda", description="Device identifier passed to diffusers.")


class InferenceConfig(BaseModel):
    """Parameters controlling diffusion inference."""

    num_inference_steps: int = Field(34, ge=1, le=150)
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0)
    controlnet_conditioning_scale: float = Field(1.0, ge=0.0, le=5.0)
    denoising_strength: float = Field(0.4, ge=0.0, le=1.0)
    seed: int = Field(123, description="Global RNG seed for deterministic runs.")


class PromptConfig(BaseModel):
    """Prompt templates and optional prompt cycling settings."""

    positive: str = Field(..., description="Positive text prompt.")
    negative: str = Field("", description="Negative text prompt to steer the sampler away from artifacts.")
    bank: Optional[List[str]] = Field(
        None,
        description="Optional list of additional positive prompts used for cycling to add diversity.",
    )


class EvalConfig(BaseModel):
    """Evaluation settings for optional quality gates."""

    compute_fid: bool = Field(True, description="Whether to run FID after generation.")
    real_ref_dir: Optional[Path] = Field(
        None, description="Directory of reference real images for FID/LPIPS comparisons."
    )
    sample_limit: Optional[int] = Field(
        None, description="Optional cap on the number of samples used for evaluation metrics."
    )
    check_alignment: bool = Field(True, description="Whether to verify IoU(mask_in, mask_out).")
    iou_threshold: float = Field(0.999, ge=0.0, le=1.0, description="Minimum IoU required for acceptance.")


class LoggingConfig(BaseModel):
    """Logging and visualization settings."""

    write_metadata: bool = Field(True, description="Write JSON metadata next to each generated image.")
    save_grid_every: int = Field(50, ge=0, description="Frequency for QA grids (0 disables).")
    verbose: bool = Field(True, description="Enable progress bars and verbose logging.")


class AppConfig(BaseModel):
    """Top-level application configuration."""

    data: DataConfig
    model: ModelConfig
    infer: InferenceConfig = Field(default_factory=InferenceConfig)
    prompts: PromptConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AppConfig":
        """Load configuration from a YAML file."""
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.model_validate(payload)

    def to_dict(self) -> dict:
        """Return a serializable dictionary representation."""
        return self.model_dump()
