"""Root and health endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from app.config import get_settings
from app.schemas.project import HealthResponse, WelcomeResponse
from app.services import ProjectService

router = APIRouter(tags=["General"])


@router.get("/", response_model=WelcomeResponse, summary="Welcome message")
def welcome() -> WelcomeResponse:
    """Simple welcome message that also points at the interactive docs."""
    settings = get_settings()
    return WelcomeResponse(
        message=f"Welcome to the {settings.app_name} API 🎧",
        app=settings.app_name,
        version=settings.version,
        docs_url="/docs",
        endpoints=[
            "GET /health",
            "POST /projects",
            "GET /projects/{project_id}",
            "POST /projects/{project_id}/upload-script",
            "POST /projects/{project_id}/analyze-narration",
            "GET /projects/{project_id}/narration-plan",
            "POST /projects/{project_id}/generate-audio",
            "GET /projects/{project_id}/audio",
        ],
    )


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health() -> HealthResponse:
    """Liveness probe: reports whether local storage is ready."""
    settings = get_settings()
    storage_ready = ProjectService(settings).ensure_storage().is_dir()

    return HealthResponse(
        status="healthy" if storage_ready else "degraded",
        app=settings.app_name,
        version=settings.version,
        storage_ready=storage_ready,
        timestamp=datetime.now(timezone.utc),
    )
