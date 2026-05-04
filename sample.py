from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from tqdm.auto import tqdm

from utils import load_config, save_image_grid, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with DDPM or DDIM from a trained UNet.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", default="checkpoints/final", help="Checkpoint directory.")
    parser.add_argument("--method", choices=["ddpm", "ddim"], default="ddpm", help="Sampling algorithm.")
    parser.add_argument("--steps", type=int, default=None, help="Number of denoising steps.")
    parser.add_argument("--num-images", type=int, default=None, help="Number of images to generate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Sampling batch size.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--output", default=None, help="Output image path.")
    return parser.parse_args()


def build_scheduler(method: str, checkpoint: str, cfg: dict):
    kwargs = {
        "num_train_timesteps": cfg["diffusion"]["num_train_timesteps"],
        "beta_schedule": cfg["diffusion"]["beta_schedule"],
        "prediction_type": cfg["diffusion"]["prediction_type"],
    }
    scheduler_cls = DDPMScheduler if method == "ddpm" else DDIMScheduler
    try:
        return scheduler_cls.from_pretrained(checkpoint)
    except OSError:
        return scheduler_cls(**kwargs)


@torch.no_grad()
def generate(
    model: UNet2DModel,
    scheduler,
    resolution: int,
    num_images: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    batches: list[torch.Tensor] = []
    for start in range(0, num_images, batch_size):
        current_batch = min(batch_size, num_images - start)
        images = torch.randn((current_batch, 3, resolution, resolution), device=device)
        for timestep in tqdm(scheduler.timesteps, desc=f"Sampling batch {start // batch_size + 1}"):
            model_output = model(images, timestep).sample
            images = scheduler.step(model_output, timestep, images).prev_sample
        batches.append(images.detach().cpu())
    return torch.cat(batches, dim=0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    sampling_cfg = cfg["sampling"]

    seed = args.seed if args.seed is not None else sampling_cfg["seed"]
    set_seed(seed)

    method_steps_key = "ddpm_steps" if args.method == "ddpm" else "ddim_steps"
    steps = args.steps if args.steps is not None else sampling_cfg[method_steps_key]
    num_images = args.num_images if args.num_images is not None else sampling_cfg["num_images"]
    batch_size = args.batch_size if args.batch_size is not None else sampling_cfg["batch_size"]
    output = Path(args.output or f"samples/{args.method}_{steps}_steps.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet2DModel.from_pretrained(args.checkpoint).to(device)
    scheduler = build_scheduler(args.method, args.checkpoint, cfg)
    scheduler.set_timesteps(steps)

    start = time.perf_counter()
    images = generate(
        model=model,
        scheduler=scheduler,
        resolution=cfg["dataset"]["resolution"],
        num_images=num_images,
        batch_size=batch_size,
        device=device,
    )
    elapsed = time.perf_counter() - start

    save_image_grid(images, output)
    print(f"method={args.method}")
    print(f"steps={steps}")
    print(f"num_images={num_images}")
    print(f"device={device}")
    print(f"elapsed_sec={elapsed:.3f}")
    print(f"sec_per_image={elapsed / num_images:.3f}")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
