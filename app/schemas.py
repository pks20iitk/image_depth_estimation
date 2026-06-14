"""
Pydantic v2 schemas for all API request/response models.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: List[str] = Field(default_factory=list)
    device: str


# ─────────────────────────────────────────────
# Model info
# ─────────────────────────────────────────────
class ModelInfo(BaseModel):
    name: str
    checkpoint: str
    depth_type: str           # "relative" | "metric"
    output_unit: str          # "normalized [0-1]" | "metres"
    loaded: bool


class ModelListResponse(BaseModel):
    models: List[ModelInfo]


# ─────────────────────────────────────────────
# Depth results
# ─────────────────────────────────────────────
class DepthStats(BaseModel):
    min_depth: float = Field(description="Minimum depth value in output units.")
    max_depth: float = Field(description="Maximum depth value in output units.")
    mean_depth: float = Field(description="Mean depth value in output units.")
    std_depth: float = Field(description="Standard deviation of depth values.")


class DepthResponse(BaseModel):
    request_id: Optional[str] = None
    model: str
    depth_type: str
    output_unit: str
    original_size: List[int] = Field(description="[height, width] of input image.")
    depth_size: List[int]     = Field(description="[height, width] of depth map.")
    stats: DepthStats
    inference_time_ms: float
    # depth PNG is returned as a StreamingResponse; this schema is for JSON mode
    depth_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded PNG of the colourised depth map (JSON mode only).",
    )


# ─────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────
class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: Optional[str] = None
    details: Optional[Dict] = None
