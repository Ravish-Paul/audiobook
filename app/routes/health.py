"""Root and health endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.config import get_settings
from app.schemas.project import HealthResponse, WelcomeResponse
from app.services import ProjectService

router = APIRouter(tags=["General"])

# Narration Preferences panel (served by FastAPI, no build step).
FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "static" / "index.html"


@router.get("/", response_model=WelcomeResponse, summary="Welcome message")
def welcome(request: Request) -> WelcomeResponse | FileResponse:
    """Welcome JSON for API clients, the frontend panel for browsers.

    Content negotiation keeps both working on the same URL:

    * a browser navigation sends ``Accept: text/html`` and receives
      ``index.html`` (the Narration Preferences panel);
    * ``fetch``/``curl``/tests send ``Accept: */*`` and receive the
      documented JSON welcome message.

    Static assets (``/static/app.js``, ``/static/style.css``) are served by
    the mount at ``/static`` registered in ``app.main``.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept and FRONTEND_INDEX.is_file():
        # `no-cache`: the browser revalidates every navigation, so a new
        # index.html (section order, casting markup) is never masked by a
        # stale cached copy — no hard refresh required.
        return FileResponse(
            FRONTEND_INDEX,
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

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
            "POST /projects/{project_id}/analyze-narration/preview",
            "GET /projects/{project_id}/narration-preferences",
            "PUT /projects/{project_id}/narration-preferences",
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
