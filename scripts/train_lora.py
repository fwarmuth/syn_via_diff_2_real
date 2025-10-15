"""Minimal LoRA fine-tuning script using diffusers and torch."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor


@dataclass
class LoraTrainingConfig:
    base_model: str
    instance_data_dir: Path
    output_dir: Path
    prompt: str
    resolution: int = 512
    learning_rate: float = 1e-4
    max_train_steps: int = 500
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    rank: int = 8
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path) -> "LoraTrainingConfig":
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["instance_data_dir"] = Path(payload["instance_data_dir"])
        payload["output_dir"] = Path(payload["output_dir"])
        return cls(**payload)


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, resolution: int, prompt: str):
        self.paths = sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not self.paths:
            raise ValueError(f"No images found under {root}")
        self.resolution = resolution
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB").resize((self.resolution, self.resolution), Image.BICUBIC)
        array = np.asarray(image).astype(np.float32) / 255.0
        array = (array - 0.5) * 2.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _prepare_lora_layers(unet: nn.Module, rank: int) -> AttnProcsLayers:
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = unet.config.block_out_channels[0]
        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank)
    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)
    lora_layers.requires_grad_(True)
    return lora_layers


def train(config: LoraTrainingConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_everything(config.seed)

    pipeline = StableDiffusionPipeline.from_pretrained(config.base_model, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    pipeline.to(device)
    pipeline.enable_attention_slicing()
    pipeline.unet.train()

    for param in pipeline.vae.parameters():
        param.requires_grad = False
    for param in pipeline.text_encoder.parameters():
        param.requires_grad = False
    for param in pipeline.unet.parameters():
        param.requires_grad = False

    lora_layers = _prepare_lora_layers(pipeline.unet, config.rank)
    optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=config.learning_rate)
    noise_scheduler = DDPMScheduler.from_pretrained(config.base_model, subfolder="scheduler")

    dataset = ImageFolderDataset(config.instance_data_dir, config.resolution, config.prompt)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    text_inputs = pipeline.tokenizer(
        [config.prompt] * config.train_batch_size,
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        encoder_hidden_states = pipeline.text_encoder(input_ids)[0]
    encoder_hidden_states = encoder_hidden_states.to(device=device, dtype=pipeline.unet.dtype)

    optimizer.zero_grad(set_to_none=True)
    completed_steps = 0
    accumulation_step = 0

    while completed_steps < config.max_train_steps:
        for batch in dataloader:
            batch = batch.to(device=device, dtype=pipeline.unet.dtype)
            latents = pipeline.vae.encode(batch).latent_dist.sample() * pipeline.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.long)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noise_pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            (loss / config.gradient_accumulation_steps).backward()

            accumulation_step += 1
            if accumulation_step % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
                if completed_steps >= config.max_train_steps:
                    break
        if completed_steps >= config.max_train_steps:
            break

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save_lora_weights(config.output_dir)

    metadata = {
        "base_model": config.base_model,
        "prompt": config.prompt,
        "rank": config.rank,
        "max_train_steps": config.max_train_steps,
        "learning_rate": config.learning_rate,
        "train_batch_size": config.train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "seed": config.seed,
    }
    (config.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on a small real-image corpus.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config (see configs/lora.yaml).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LoraTrainingConfig.from_yaml(args.config)
    train(config)


if __name__ == "__main__":
    main()
