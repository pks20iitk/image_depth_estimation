"""
Depth model registry.

Manages two model backends:
  - DepthAnythingV2Backend  (relative depth, HF pipeline)
  - ZoeDepthBackend         (metric depth, manual inference)

Both implement the DepthBackend protocol for a unified calling interface.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────
@dataclass
class DepthResult:
    depth_map: np.ndarray         # float32, shape (H, W)
    depth_type: str               # "relative" | "metric"
    output_unit: str              # "normalized [0-1]" | "metres"
    model_name: str
    checkpoint: str
    inference_time_ms: float
    original_size: tuple[int, int]   # (H, W)
    metadata: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# Abstract backend protocol
# ─────────────────────────────────────────────
class DepthBackend(ABC):
    """All depth backends must implement this interface."""

    name: str
    checkpoint: str
    depth_type: str
    output_unit: str

    @abstractmethod
    def load(self, device: str) -> None:
        """Load model weights onto device."""

    @abstractmethod
    def unload(self) -> None:
        """Release GPU memory."""

    @abstractmethod
    def predict(self, image: Image.Image) -> DepthResult:
        """Run inference synchronously. Returns DepthResult."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True when weights are in memory."""


# ─────────────────────────────────────────────
# Backend 1 — Depth Anything V2 (relative)
# ─────────────────────────────────────────────
class DepthAnythingV2Backend(DepthBackend):
    """
    Uses HuggingFace transformers pipeline for zero-shot relative depth.
    Checkpoint: depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf
    """

    name = "depth_anything_v2"
    depth_type = "relative"
    output_unit = "normalized [0-1]"

    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint
        self._pipe = None
        self._device: Optional[str] = None

    def load(self, device: str) -> None:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading Depth Anything V2 from '%s' on %s …", self.checkpoint, device)
        self._device = device
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=self.checkpoint,
            device=0 if device == "cuda" else -1,   # pipeline uses int device index
        )
        logger.info("Depth Anything V2 ready.")

    def unload(self) -> None:
        del self._pipe
        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Depth Anything V2 unloaded.")

    def predict(self, image: Image.Image) -> DepthResult:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        original_size = (image.height, image.width)
        t0 = time.perf_counter()
        output = self._pipe(image)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 'predicted_depth' is a torch tensor; 'depth' is a PIL visualisation
        predicted_depth: torch.Tensor = output["predicted_depth"]
        depth_np = predicted_depth.squeeze().cpu().numpy().astype(np.float32)

        # Normalise to [0, 1]
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 1e-6:
            depth_np = (depth_np - d_min) / (d_max - d_min)

        return DepthResult(
            depth_map=depth_np,
            depth_type=self.depth_type,
            output_unit=self.output_unit,
            model_name=self.name,
            checkpoint=self.checkpoint,
            inference_time_ms=elapsed_ms,
            original_size=original_size,
            metadata={"raw_min": float(d_min), "raw_max": float(d_max)},
        )

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None


# ─────────────────────────────────────────────
# Backend 2 — ZoeDepth (metric)
# ─────────────────────────────────────────────
class ZoeDepthBackend(DepthBackend):
    """
    Manual inference pipeline for ZoeDepth metric depth estimation.
    Supports flip-augmentation for improved accuracy.
    Checkpoint: Intel/zoedepth-nyu-kitti
    """

    name = "zoedepth"
    depth_type = "metric"
    output_unit = "metres"

    def __init__(self, checkpoint: str, flip_augment: bool = True):
        self.checkpoint = checkpoint
        self.flip_augment = flip_augment
        self._model = None
        self._processor = None
        self._device: Optional[str] = None

    def load(self, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        logger.info("Loading ZoeDepth from '%s' on %s …", self.checkpoint, device)
        self._device = device
        self._processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint)
        self._model = self._model.to(device)
        self._model.eval()
        logger.info("ZoeDepth ready.")

    def unload(self) -> None:
        del self._model, self._processor
        self._model = self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ZoeDepth unloaded.")

    def predict(self, image: Image.Image) -> DepthResult:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        original_size = (image.height, image.width)
        device = self._device

        t0 = time.perf_counter()

        # Preprocess
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values: torch.Tensor = inputs.pixel_values.to(device)

        with torch.no_grad():
            outputs = self._model(pixel_values)

            if self.flip_augment:
                # ZoeDepth recommendation: average forward + flipped pass
                pixel_values_flipped = torch.flip(pixel_values, dims=[3])
                outputs_flipped = self._model(pixel_values=pixel_values_flipped)
            else:
                outputs_flipped = None

        # Post-process — removes padding, resizes to original dims
        post = self._processor.post_process_depth_estimation(
            outputs,
            source_sizes=[(image.height, image.width)],
            outputs_flipped=outputs_flipped,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        predicted_depth: torch.Tensor = post[0]["predicted_depth"]
        depth_np = predicted_depth.squeeze().detach().cpu().numpy().astype(np.float32)

        return DepthResult(
            depth_map=depth_np,
            depth_type=self.depth_type,
            output_unit=self.output_unit,
            model_name=self.name,
            checkpoint=self.checkpoint,
            inference_time_ms=elapsed_ms,
            original_size=original_size,
            metadata={"flip_augment": self.flip_augment},
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────
class DepthModelRegistry:
    """
    Manages all depth backends.
    Loads models concurrently on startup using asyncio thread pools.
    """

    def __init__(self, settings):
        self._backends: Dict[str, DepthBackend] = {
            "relative": DepthAnythingV2Backend(settings.relative_model_checkpoint),
            "metric":   ZoeDepthBackend(settings.metric_model_checkpoint),
        }
        self._device = settings.device

    async def load_all(self) -> None:
        """Load all backends concurrently in thread-pool executors."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, backend.load, self._device)
            for backend in self._backends.values()
        ]
        await asyncio.gather(*tasks)

    def unload_all(self) -> None:
        for backend in self._backends.values():
            try:
                backend.unload()
            except Exception:
                logger.exception("Error unloading backend '%s'", backend.name)

    def get(self, depth_type: str) -> DepthBackend:
        """
        Retrieve a backend by depth type.
        Raises KeyError for unknown types.
        """
        if depth_type not in self._backends:
            raise KeyError(
                f"Unknown depth_type '{depth_type}'. "
                f"Available: {list(self._backends.keys())}"
            )
        return self._backends[depth_type]

    def list_models(self):
        from app.schemas import ModelInfo
        return [
            ModelInfo(
                name=b.name,
                checkpoint=b.checkpoint,
                depth_type=b.depth_type,
                output_unit=b.output_unit,
                loaded=b.is_loaded,
            )
            for b in self._backends.values()
        ]
