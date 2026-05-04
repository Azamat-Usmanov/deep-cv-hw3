from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.utils import make_grid, save_image


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_grid(images: torch.Tensor, nrow: int = 4) -> torch.Tensor:
    images = (images.clamp(-1, 1) + 1) / 2
    return make_grid(images, nrow=nrow, padding=2)


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image_grid(images, nrow=nrow), path)


def save_pil_grid(images: list[Image.Image], path: str | Path, nrow: int = 4) -> None:
    if not images:
        raise ValueError("No images to save")
    width, height = images[0].size
    rows = (len(images) + nrow - 1) // nrow
    grid = Image.new("RGB", (nrow * width, rows * height), color=(255, 255, 255))
    for idx, image in enumerate(images):
        grid.paste(image.convert("RGB"), ((idx % nrow) * width, (idx // nrow) * height))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def append_loss(path: str | Path, epoch: int, step: int, loss: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["epoch", "global_step", "loss"])
        writer.writerow([epoch, step, f"{loss:.8f}"])


def plot_loss(csv_path: str | Path, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return

    steps: list[int] = []
    losses: list[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["global_step"]))
            losses.append(float(row["loss"]))

    if not steps:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses)
    plt.xlabel("Global step")
    plt.ylabel("MSE loss")
    plt.title("Training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
