"""
Centralised configuration via environment variables / .env file.
All settings have sensible defaults so the service starts with zero config.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Model checkpoints ──────────────────────────────────────────────
    relative_model_checkpoint: str = Field(
        default="depth-anything/Depth-Anything-V2-Small-hf",
        description=(
            "HuggingFace checkpoint for relative depth estimation. "
            "Use '-Small-hf' for development, '-Large-hf' for production."
        ),
    )
    metric_model_checkpoint: str = Field(
        default="Intel/zoedepth-nyu-kitti",
        description="HuggingFace checkpoint for metric depth estimation.",
    )

    # ── Inference ─────────────────────────────────────────────────────
    device: str = Field(
        default="auto",
        description="'cuda', 'cpu', or 'auto' (auto-detects GPU availability).",
    )
    max_image_size: int = Field(
        default=1920,
        description="Longest image edge is downscaled to this before inference.",
    )
    max_upload_mb: int = Field(
        default=20,
        description="Maximum upload size in megabytes.",
    )
    inference_timeout_seconds: float = Field(
        default=60.0,
        description="Maximum time allowed for a single inference call.",
    )

    # ── API ───────────────────────────────────────────────────────────
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins. Use specific domains in production.",
    )
    workers: int = Field(
        default=1,
        description="Number of uvicorn workers. Keep at 1 when sharing GPU.",
    )

    # ── Colour map for depth visualisation ────────────────────────────
    colormap: str = Field(
        default="inferno",
        description=(
            "Matplotlib colormap applied to depth visualisation. "
            "Options: 'inferno', 'plasma', 'magma', 'viridis', 'turbo'."
        ),
    )

    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        if v == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return v

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings — cached after first call."""
    return Settings()
