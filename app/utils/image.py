"""
Image utilities for the Depth Estimation API.

Covers:
  - Decoding uploaded files to PIL Images
  - Resizing to max edge (preserving aspect ratio)
  - Encoding depth maps to colourised PNG (streaming-safe)
  - Computing depth statistics
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

# Accepted MIME types for uploads
ACCEPTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
    "image/bmp",
}


def decode_upload(
    data: bytes,
    content_type: Optional[str] = None,
    max_size_bytes: int = 20 * 1024 * 1024,
) -> Image.Image:
    """
    Decode raw bytes to a PIL Image (RGB).

    Args:
        data:           Raw file bytes.
        content_type:   MIME type from the upload. Used for validation.
        max_size_bytes: Reject uploads larger than this.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If the file is too large, unsupported, or corrupt.
    """
    if len(data) > max_size_bytes:
        raise ValueError(
            f"Upload size {len(data) / 1e6:.1f} MB exceeds limit "
            f"of {max_size_bytes / 1e6:.0f} MB."
        )

    if content_type and content_type not in ACCEPTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported content type '{content_type}'. "
            f"Accepted: {sorted(ACCEPTED_MIME_TYPES)}"
        )

    try:
        image = Image.open(io.BytesIO(data))
        image.verify()              # catches truncated files
        image = Image.open(io.BytesIO(data))  # re-open after verify
        return image.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc


def resize_to_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    """
    Proportionally downscale image so its longest edge is ≤ max_edge.
    Returns the original if already within bounds.
    """
    h, w = image.height, image.width
    longest = max(h, w)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    logger.debug("Resizing %dx%d → %dx%d (max_edge=%d)", w, h, new_w, new_h, max_edge)
    return image.resize((new_w, new_h), Image.LANCZOS)


def encode_depth_to_png(
    depth_map: np.ndarray,
    colormap: str = "inferno",
    normalise: bool = True,
) -> bytes:
    """
    Convert a float32 depth array to a colourised PNG.

    Args:
        depth_map:  float32 array of shape (H, W).
        colormap:   Matplotlib colormap name.
        normalise:  If True, scale depth to [0, 1] before applying colormap.

    Returns:
        PNG bytes suitable for streaming to the client.
    """
    depth = depth_map.astype(np.float32)

    if normalise:
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-8:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

    cmap = cm.get_cmap(colormap)
    rgba = cmap(depth)                          # (H, W, 4) float64 in [0, 1]
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    buf.seek(0)
    return buf.read()


def depth_statistics(depth_map: np.ndarray) -> dict:
    """Return basic statistics for a depth array."""
    return {
        "min_depth":  float(np.nanmin(depth_map)),
        "max_depth":  float(np.nanmax(depth_map)),
        "mean_depth": float(np.nanmean(depth_map)),
        "std_depth":  float(np.nanstd(depth_map)),
    }
