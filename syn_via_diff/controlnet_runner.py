"""Diffusion inference wrapper that binds Stable Diffusion, ControlNet, and LoRA."""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from .config import InferenceConfig, ModelConfig, PromptConfig
from .lora_utils import apply_single_lora

LOGGER = logging.getLogger(__name__)


try:  # pragma: no cover - heavy dependencies are optional in tests
    import torch
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )
except ImportError:  # pragma: no cover - fallback when diffusers isn't available
    torch = None  # type: ignore
    ControlNetModel = StableDiffusionControlNetPipeline = UniPCMultistepScheduler = None


@dataclass
class ControlNetRunner:
    """Thin wrapper around the diffusers pipeline for deterministic inference."""

    pipe: "StableDiffusionControlNetPipeline"
    model_config: ModelConfig
    infer_config: InferenceConfig
    prompt_config: PromptConfig

    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        infer_config: InferenceConfig,
        prompt_config: PromptConfig,
    ) -> "ControlNetRunner":
        if StableDiffusionControlNetPipeline is None:
            raise ImportError(
                "diffusers is required to instantiate ControlNetRunner. "
                "Install it via `uv add diffusers` and ensure torch is available."
            )

        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }.get(model_config.dtype, torch.float16)

        controlnet = None
        if model_config.controlnet:
            controlnet = ControlNetModel.from_pretrained(
                model_config.controlnet,
                torch_dtype=dtype,
            )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_config.base,
            controlnet=controlnet,
            torch_dtype=dtype,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        try:
            pipe.enable_model_cpu_offload()
        except AttributeError:
            LOGGER.debug("Model CPU offload not available; continuing without it.")

        if model_config.lora_path:
            apply_single_lora(pipe, model_config.lora_path, model_config.lora_weight)

        pipe.to(model_config.device)
        return cls(pipe=pipe, model_config=model_config, infer_config=infer_config, prompt_config=prompt_config)

    def stylize(
        self,
        image: Image.Image,
        mask: Image.Image,
        seed: Optional[int] = None,
        positive_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> Image.Image:
        """Generate a stylized image given an RGB input and segmentation mask."""
        if torch is None:
            raise RuntimeError("torch is required to run inference but is not installed.")

        generator = torch.Generator(device=self.model_config.device)
        generator.manual_seed(seed if seed is not None else self.infer_config.seed)

        control_image = mask.convert("RGB")

        with torch.inference_mode():
            output = self.pipe(
                image=image,
                control_image=control_image,
                prompt=positive_prompt or self.prompt_config.positive,
                negative_prompt=negative_prompt or self.prompt_config.negative,
                num_inference_steps=self.infer_config.num_inference_steps,
                guidance_scale=self.infer_config.guidance_scale,
                controlnet_conditioning_scale=self.infer_config.controlnet_conditioning_scale,
                generator=generator,
                strength=self.infer_config.denoising_strength,
            )

        return output.images[0]

    def close(self) -> None:
        """Free CUDA memory if possible."""
        if torch is None:
            return
        with contextlib.suppress(Exception):
            self.pipe.to("cpu")
        with contextlib.suppress(Exception):
            del self.pipe
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
