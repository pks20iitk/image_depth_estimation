"""
Depth estimation endpoints.

  POST /api/v1/depth/relative   — relative depth (Depth Anything V2)
  POST /api/v1/depth/metric     — metric depth in metres (ZoeDepth)
  GET  /api/v1/depth/models     — list loaded models

Both POST endpoints accept:
  - form field: file  (image upload)
  - query param: response_format  ("png" | "json")

PNG response:   streams a colourised depth PNG directly.
JSON response:  returns DepthResponse with base64-encoded PNG + statistics.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.models.depth_model import DepthModelRegistry, DepthResult
from app.schemas import DepthResponse, DepthStats, ModelListResponse
from app.utils.image import (
    decode_upload,
    depth_statistics,
    encode_depth_to_png,
    resize_to_max_edge,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _get_registry(request: Request) -> DepthModelRegistry:
    return request.app.state.registry


async def _run_inference(registry: DepthModelRegistry, depth_type: str, image) -> DepthResult:
    """Run model inference in a thread pool (non-blocking)."""
    backend = registry.get(depth_type)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, backend.predict, image)


def _build_response(
    result: DepthResult,
    response_format: str,
    colormap: str,
    request_id: str | None,
):
    """
    Build either a StreamingResponse (PNG) or a DepthResponse (JSON).
    """
    stats = depth_statistics(result.depth_map)
    png_bytes = encode_depth_to_png(result.depth_map, colormap=colormap)

    if response_format == "png":
        return StreamingResponse(
            content=iter([png_bytes]),
            media_type="image/png",
            headers={
                "X-Depth-Type":         result.depth_type,
                "X-Output-Unit":        result.output_unit,
                "X-Model":              result.model_name,
                "X-Inference-Time-Ms":  f"{result.inference_time_ms:.1f}",
                "X-Min-Depth":          f"{stats['min_depth']:.4f}",
                "X-Max-Depth":          f"{stats['max_depth']:.4f}",
            },
        )

    # JSON response — include base64-encoded PNG
    return DepthResponse(
        request_id=request_id,
        model=result.model_name,
        depth_type=result.depth_type,
        output_unit=result.output_unit,
        original_size=list(result.original_size),
        depth_size=list(result.depth_map.shape),
        stats=DepthStats(**stats),
        inference_time_ms=result.inference_time_ms,
        depth_base64=base64.b64encode(png_bytes).decode("utf-8"),
    )


# ─────────────────────────────────────────────
# Endpoint factory
# ─────────────────────────────────────────────
def _make_depth_endpoint(depth_type: str):
    """
    Returns an async endpoint function for the given depth_type.
    Avoids code duplication between /relative and /metric.
    """
    async def endpoint(
        request: Request,
        file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, TIFF, BMP)."),
        response_format: Literal["png", "json"] = Query(
            default="png",
            description="'png' returns a colourised depth image; 'json' returns stats + base64 PNG.",
        ),
    ):
        settings = get_settings()
        request_id = getattr(request.state, "request_id", None)

        # 1. Read & validate upload
        try:
            raw = await file.read()
            image = decode_upload(
                raw,
                content_type=file.content_type,
                max_size_bytes=settings.max_upload_bytes,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        # 2. Resize to prevent OOM
        image = resize_to_max_edge(image, settings.max_image_size)

        # 3. Inference
        registry = _get_registry(request)
        try:
            result = await asyncio.wait_for(
                _run_inference(registry, depth_type, image),
                timeout=settings.inference_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Inference timed out after {settings.inference_timeout_seconds}s.",
            )
        except Exception as exc:
            logger.exception("Inference failed for %s", depth_type)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error: {exc}",
            ) from exc

        logger.info(
            "depth_type=%s | model=%s | size=%s | %.1f ms",
            depth_type, result.model_name, result.original_size, result.inference_time_ms,
        )

        # 4. Build & return response
        return _build_response(result, response_format, settings.colormap, request_id)

    return endpoint


# ─────────────────────────────────────────────
# Register endpoints
# ─────────────────────────────────────────────
router.add_api_route(
    "/relative",
    _make_depth_endpoint("relative"),
    methods=["POST"],
    summary="Relative depth estimation",
    description=(
        "Estimates depth order using **Depth Anything V2** (zero-shot, "
        "no real-world units). Best for scene understanding, AR masking, "
        "and segmentation aids."
    ),
)

router.add_api_route(
    "/metric",
    _make_depth_endpoint("metric"),
    methods=["POST"],
    summary="Metric (absolute) depth estimation",
    description=(
        "Estimates real-world depth in **metres** using **ZoeDepth** "
        "(NYU + KITTI). Best for robotics, autonomous driving, and 3D reconstruction."
    ),
)


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List loaded depth models",
)
async def list_models(request: Request):
    """Returns metadata for all loaded depth estimation backends."""
    registry = _get_registry(request)
    return ModelListResponse(models=registry.list_models())
