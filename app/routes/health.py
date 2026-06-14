"""Health and readiness endpoints."""

from fastapi import APIRouter, Request
from app.schemas import HealthResponse
from app.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health(request: Request):
    """Returns 200 when the service is running and models are loaded."""
    settings = get_settings()
    registry = request.app.state.registry
    loaded = [m.name for m in registry.list_models() if m.loaded]
    return HealthResponse(
        status="ok",
        models_loaded=loaded,
        device=settings.device,
    )
