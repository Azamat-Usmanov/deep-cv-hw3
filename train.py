from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from utils import append_loss, load_config, plot_loss, save_image_grid, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small DDPM model on an image dataset.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint directory saved by save_pretrained.")
    return parser.parse_args()


def build_transform(cfg: dict) -> transforms.Compose:
    dataset_cfg = cfg["dataset"]
    ops: list[transforms.transforms.Transform] = [
        transforms.Resize(dataset_cfg["resolution"], interpolation=transforms.InterpolationMode.BILINEAR),
    ]
    if dataset_cfg.get("center_crop", True):
        ops.append(transforms.CenterCrop(dataset_cfg["resolution"]))
    else:
        ops.append(transforms.RandomCrop(dataset_cfg["resolution"]))
    if dataset_cfg.get("horizontal_flip", True):
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transforms.Compose(ops)


def build_model(cfg: dict) -> UNet2DModel:
    model_cfg = cfg["model"]
    return UNet2DModel(
        sample_size=model_cfg["sample_size"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        layers_per_block=model_cfg["layers_per_block"],
        block_out_channels=tuple(model_cfg["block_out_channels"]),
        down_block_types=tuple(model_cfg["down_block_types"]),
        up_block_types=tuple(model_cfg["up_block_types"]),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    diffusion_cfg = cfg["diffusion"]

    set_seed(train_cfg["seed"])
    output_dir = Path(train_cfg["output_dir"])
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        mixed_precision=train_cfg["mixed_precision"],
        log_with=None,
    )

    transform = build_transform(cfg)
    dataset = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["split"])
    if train_cfg.get("max_train_samples"):
        dataset = dataset.shuffle(seed=train_cfg["seed"]).select(range(train_cfg["max_train_samples"]))

    image_column = cfg["dataset"]["image_column"]

    def preprocess(batch: dict) -> dict:
        images = [image.convert("RGB") for image in batch[image_column]]
        batch["pixel_values"] = [transform(image) for image in images]
        return batch

    dataset = dataset.with_transform(preprocess)

    def collate_fn(examples: list[dict]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        return {"pixel_values": pixel_values}

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["train_batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
    )

    model = UNet2DModel.from_pretrained(args.resume) if args.resume else build_model(cfg)
    ema_model = None
    if train_cfg.get("use_ema", True):
        ema_model = EMAModel(
            model.parameters(),
            decay=train_cfg.get("ema_decay", 0.9999),
            use_ema_warmup=True,
            inv_gamma=1.0,
            power=0.75,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=diffusion_cfg["num_train_timesteps"],
        beta_schedule=diffusion_cfg["beta_schedule"],
        prediction_type=diffusion_cfg["prediction_type"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    steps_per_epoch = len(dataloader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=train_cfg["lr_warmup_steps"],
        num_training_steps=steps_per_epoch * train_cfg["num_epochs"],
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    if ema_model is not None:
        ema_model.to(accelerator.device)

    global_step = 0
    loss_csv = output_dir / "loss.csv"
    progress = tqdm(range(train_cfg["num_epochs"]), disable=not accelerator.is_local_main_process)

    for epoch in progress:
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            clean_images = batch["pixel_values"]
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            batch_size = clean_images.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=clean_images.device,
                dtype=torch.long,
            )
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients and ema_model is not None:
                    ema_model.step(model.parameters())

            global_step += 1
            loss_value = accelerator.gather(loss.detach()).mean().item()
            epoch_loss += loss_value
            if accelerator.is_local_main_process:
                append_loss(loss_csv, epoch + 1, global_step, loss_value)

        avg_loss = epoch_loss / max(1, steps_per_epoch)
        progress.set_postfix(epoch=epoch + 1, loss=f"{avg_loss:.4f}")

        if accelerator.is_local_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            if (epoch + 1) % train_cfg["validation_every_epochs"] == 0:
                if ema_model is not None:
                    ema_model.store(unwrapped_model.parameters())
                    ema_model.copy_to(unwrapped_model.parameters())
                images = sample_ddpm(
                    unwrapped_model,
                    noise_scheduler,
                    cfg["dataset"]["resolution"],
                    train_cfg["num_validation_images"],
                    accelerator.device,
                )
                if ema_model is not None:
                    ema_model.restore(unwrapped_model.parameters())
                save_image_grid(images, output_dir / f"validation_epoch_{epoch + 1:04d}.png")
                plot_loss(loss_csv, output_dir / "loss.png")

            if (epoch + 1) % train_cfg["save_every_epochs"] == 0 or (epoch + 1) == train_cfg["num_epochs"]:
                ckpt_path = checkpoint_dir / f"epoch_{epoch + 1:04d}"
                if ema_model is not None:
                    ema_model.store(unwrapped_model.parameters())
                    ema_model.copy_to(unwrapped_model.parameters())
                unwrapped_model.save_pretrained(ckpt_path)
                if ema_model is not None:
                    ema_model.restore(unwrapped_model.parameters())
                noise_scheduler.save_pretrained(ckpt_path)

    if accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        if ema_model is not None:
            ema_model.copy_to(unwrapped_model.parameters())
        unwrapped_model.save_pretrained(checkpoint_dir / "final")
        noise_scheduler.save_pretrained(checkpoint_dir / "final")
        plot_loss(loss_csv, output_dir / "loss.png")


@torch.no_grad()
def sample_ddpm(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    resolution: int,
    num_images: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    scheduler.set_timesteps(100)
    images = torch.randn((num_images, 3, resolution, resolution), device=device)
    for timestep in scheduler.timesteps:
        model_output = model(images, timestep).sample
        images = scheduler.step(model_output, timestep, images).prev_sample
    return images.detach().cpu()


if __name__ == "__main__":
    main()
