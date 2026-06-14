"""
standalone_demo.py
──────────────────
Zero-setup demo: runs depth estimation locally with Depth Anything V2
and ZoeDepth, saves colourised depth maps side-by-side.

Requirements:
    pip install transformers accelerate Pillow matplotlib torch torchvision

Usage:
    python notebooks/standalone_demo.py --image path/to/photo.jpg
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image


# ─────────────────────────────────────────────────────────────────
# Helper: colourised depth → PIL Image
# ─────────────────────────────────────────────────────────────────
def depth_to_pil(depth_np: np.ndarray, cmap: str = "inferno") -> Image.Image:
    d = depth_np.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    rgba = cm.get_cmap(cmap)(d)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


# ─────────────────────────────────────────────────────────────────
# Model 1: Depth Anything V2  (relative)
# ─────────────────────────────────────────────────────────────────
def run_depth_anything_v2(image: Image.Image, device: str) -> np.ndarray:
    print("  Loading Depth Anything V2 (Small) …")
    from transformers import pipeline

    pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if device == "cuda" else -1,
    )
    output = pipe(image)
    depth: torch.Tensor = output["predicted_depth"]
    depth_np = depth.squeeze().cpu().numpy().astype(np.float32)
    # Normalise to [0, 1]
    depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
    return depth_np


# ─────────────────────────────────────────────────────────────────
# Model 2: ZoeDepth  (metric — metres)
# ─────────────────────────────────────────────────────────────────
def run_zoedepth(image: Image.Image, device: str) -> np.ndarray:
    print("  Loading ZoeDepth (NYU+KITTI) …")
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
    model = AutoModelForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(device)
    model.eval()

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values)
        outputs_flipped = model(pixel_values=torch.flip(pixel_values, dims=[3]))

    post = processor.post_process_depth_estimation(
        outputs,
        source_sizes=[(image.height, image.width)],
        outputs_flipped=outputs_flipped,
    )
    depth_np = post[0]["predicted_depth"].squeeze().detach().cpu().numpy().astype(np.float32)
    return depth_np   # values are in metres


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="depth_comparison.png")
    args = parser.parse_args()

    # Resolve device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    image = Image.open(args.image).convert("RGB")
    print(f"Image: {image.width}×{image.height}")

    # Run both models
    print("\n[1/2] Depth Anything V2 …")
    depth_rel = run_depth_anything_v2(image, device)

    print("\n[2/2] ZoeDepth …")
    depth_met = run_zoedepth(image, device)

    # Print stats
    print(f"\n  ZoeDepth metric range: {depth_met.min():.2f} – {depth_met.max():.2f} m")

    # Plot side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Depth Estimation — Model Comparison", fontsize=14, fontweight="bold")

    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(depth_rel, cmap="inferno")
    axes[1].set_title("Depth Anything V2\n(relative, normalised)")
    axes[1].axis("off")
    plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, label="near → far")

    im = axes[2].imshow(depth_met, cmap="plasma")
    axes[2].set_title("ZoeDepth\n(metric, metres)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, label="metres")

    plt.tight_layout()
    out = Path(args.output)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison → {out.resolve()}")


if __name__ == "__main__":
    main()
