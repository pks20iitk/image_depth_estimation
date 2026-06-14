#!/usr/bin/env python3
"""
Batch depth estimation CLI.

Usage:
    python scripts/batch_infer.py \\
        --input  path/to/images/ \\
        --output path/to/results/ \\
        --model  relative            \\
        --colormap inferno           \\
        --device cuda

Processes all JPEG/PNG/WebP images in --input and writes:
  - <name>_depth.png  — colourised depth visualisation
  - <name>_depth.npy  — raw float32 depth array
  - results.json      — statistics for all images
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch depth estimation")
    p.add_argument("--input",     required=True, help="Input directory of images")
    p.add_argument("--output",    required=True, help="Output directory for results")
    p.add_argument("--model",     choices=["relative", "metric"], default="relative")
    p.add_argument("--colormap",  default="inferno", help="Matplotlib colormap name")
    p.add_argument("--device",    default="auto",   help="'cuda', 'cpu', or 'auto'")
    p.add_argument("--max-edge",  type=int, default=1024, help="Max image edge (px)")
    p.add_argument("--save-npy",  action="store_true", help="Also save raw depth as .npy")
    return p.parse_args()


def load_backend(model: str, device: str):
    """Instantiate and load the appropriate depth backend."""
    # Resolve device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    if model == "relative":
        from app.models.depth_model import DepthAnythingV2Backend
        backend = DepthAnythingV2Backend(
            checkpoint="depth-anything/Depth-Anything-V2-Small-hf"
        )
    else:
        from app.models.depth_model import ZoeDepthBackend
        backend = ZoeDepthBackend(checkpoint="Intel/zoedepth-nyu-kitti")

    backend.load(device)
    return backend


def process_image(
    backend,
    image_path: Path,
    output_dir: Path,
    colormap: str,
    max_edge: int,
    save_npy: bool,
) -> dict:
    from app.utils.image import (
        decode_upload,
        depth_statistics,
        encode_depth_to_png,
        resize_to_max_edge,
    )

    logger.info("Processing: %s", image_path.name)
    raw = image_path.read_bytes()
    image = decode_upload(raw)
    image = resize_to_max_edge(image, max_edge)

    result = backend.predict(image)
    stats = depth_statistics(result.depth_map)

    # Save colourised PNG
    png_path = output_dir / f"{image_path.stem}_depth.png"
    png_bytes = encode_depth_to_png(result.depth_map, colormap=colormap)
    png_path.write_bytes(png_bytes)

    # Optionally save raw array
    if save_npy:
        npy_path = output_dir / f"{image_path.stem}_depth.npy"
        np.save(npy_path, result.depth_map)

    return {
        "file": image_path.name,
        "depth_type": result.depth_type,
        "output_unit": result.output_unit,
        "original_size": list(result.original_size),
        "inference_time_ms": round(result.inference_time_ms, 2),
        **stats,
    }


def main() -> None:
    args = parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in sorted(input_dir.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not image_paths:
        logger.error("No supported images found in '%s'", input_dir)
        sys.exit(1)

    logger.info("Found %d images. Loading model …", len(image_paths))
    backend = load_backend(args.model, args.device)

    all_results = []
    t_total = time.perf_counter()

    for img_path in image_paths:
        try:
            result = process_image(
                backend, img_path, output_dir,
                args.colormap, args.max_edge, args.save_npy,
            )
            all_results.append(result)
        except Exception:
            logger.exception("Failed on %s", img_path.name)

    elapsed = time.perf_counter() - t_total

    # Save JSON summary
    summary = {
        "model": args.model,
        "device": args.device,
        "total_images": len(all_results),
        "total_time_seconds": round(elapsed, 2),
        "avg_time_ms": round(
            sum(r["inference_time_ms"] for r in all_results) / max(len(all_results), 1), 2
        ),
        "results": all_results,
    }
    summary_path = output_dir / "results.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Done. %d images in %.1fs (avg %.0f ms/img). Results → %s",
        len(all_results), elapsed,
        summary["avg_time_ms"],
        output_dir,
    )


if __name__ == "__main__":
    main()
