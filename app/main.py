"""
Depth Estimation API — Production-grade FastAPI application.

Supports:
  - Depth Anything V2  (relative depth, zero-shot)
  - ZoeDepth           (absolute/metric depth, NYU + KITTI)

Endpoints:
  POST /api/v1/depth/relative  — relative depth map
  POST /api/v1/depth/metric    — metric depth map (metres)
  GET  /api/v1/health          — liveness check
  GET  /api/v1/models          — list loaded models + metadata
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import Settings, get_settings
from app.models.depth_model import DepthModelRegistry
from app.schemas import DepthResponse, ErrorResponse, HealthResponse, ModelListResponse
from app.utils.image import decode_upload, encode_depth_to_png

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Application lifespan (startup / shutdown)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = get_settings()
    logger.info("Loading model registry …")
    registry = DepthModelRegistry(settings)
    await registry.load_all()
    app.state.registry = registry
    logger.info("Models ready — API is live")
    yield
    logger.info("Shutting down — releasing GPU memory")
    registry.unload_all()


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Depth Estimation API",
        description=(
            "Production depth estimation service using Depth Anything V2 "
            "(relative) and ZoeDepth (metric/absolute)."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — tighten allowed_origins in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Request ID middleware ──────────────────
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response

    # ── Global exception handler ───────────────
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s", request.url)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_server_error",
                message="An unexpected error occurred. Check server logs.",
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )

    # ── Routes ────────────────────────────────
    from app.routes import depth, health  # noqa: PLC0415
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(depth.router,  prefix="/api/v1/depth", tags=["Depth"])

    return app


app = create_app()
