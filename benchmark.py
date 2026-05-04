from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DDPM/DDIM generation benchmarks.")
    parser.add_argument("--checkpoint", default="checkpoints/final")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--ddpm-steps", type=int, default=1000)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--output-dir", default="samples")
    parser.add_argument("--metrics", default="outputs/sampling_benchmark.csv")
    return parser.parse_args()


def parse_sample_output(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def run_sample(method: str, steps: int, args: argparse.Namespace) -> dict[str, str]:
    output_path = Path(args.output_dir) / f"{method}_{steps}_steps.png"
    command = [
        sys.executable,
        "sample.py",
        "--checkpoint",
        args.checkpoint,
        "--method",
        method,
        "--steps",
        str(steps),
        "--num-images",
        str(args.num_images),
        "--output",
        str(output_path),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print(result.stdout)
    return parse_sample_output(result.stdout)


def main() -> None:
    args = parse_args()
    rows = [
        run_sample("ddpm", args.ddpm_steps, args),
        run_sample("ddim", args.ddim_steps, args),
    ]

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "steps", "num_images", "device", "elapsed_sec", "sec_per_image", "saved"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
