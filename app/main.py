"""FastAPI application entry point.

Run it with:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routes import (
    audio,
    audio_analysis,
    casting,
    elevenlabs,
    health,
    media,
    narration,
    projects,
    usage,
)
from app.security import BetaSecurityMiddleware, auth_router
from app.services.audio_composer import resolve_ffmpeg
from app.services.errors import FFmpegNotFoundError, NarrationPlanError, ProjectError
from app.services.project_service import ProjectService

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Make sure the local storage tree exists before serving requests."""
    storage = ProjectService().ensure_storage()
    print(f"[startup] storage ready at: {storage}")

    # Never print API keys themselves — only whether they are configured.
    el_state = "configured" if settings.elevenlabs_configured else "NOT configured"
    print(
        f"[startup] elevenlabs: {el_state} | voice={settings.elevenlabs_voice_id or '-'} "
        f"| model={settings.elevenlabs_model_id}"
    )
    if not settings.elevenlabs_configured:
        print("[startup] hint: copy .env.example to .env and fill in your credentials")

    mimo_state = "configured" if settings.mimo_configured else "NOT configured"
    print(
        f"[startup] mimo: {mimo_state} | model={settings.mimo_model_id} "
        f"| base_url={settings.mimo_base_url}"
    )
    if not settings.mimo_configured:
        print(
            "[startup] hint: MIMO_API_KEY is required for /analyze-narration"
        )

    pyannote_state = (
        "configured" if settings.pyannote_configured else "NOT configured"
    )
    print(
        f"[startup] pyannote: {pyannote_state} | model={settings.pyannote_model_id} "
        f"| transcription={settings.pyannote_transcription_model}"
    )
    if not settings.pyannote_configured:
        print(
            "[startup] hint: PYANNOTE_API_KEY is required for "
            "POST /projects/{id}/audio-analysis (20s minimum billing per job)"
        )

    try:
        ffmpeg = resolve_ffmpeg(settings)
        print(f"[startup] ffmpeg: {ffmpeg}")
    except FFmpegNotFoundError:
        print(
            "[startup] ffmpeg: NOT FOUND — segment merging will fail until "
            "FFMPEG_PATH is set"
        )

    if settings.beta_enabled:
        testers = ", ".join(name for name, _ in settings.beta_access_tokens)
        print(
            f"[startup] private beta auth: ON | testers: {testers} | "
            f"docs: disabled | generation limit: "
            f"{settings.beta_generation_limit or 'unlimited'} per "
            f"{settings.beta_generation_window_hours}h per tester"
        )
    else:
        print(
            "[startup] private beta auth: OFF — set BETA_ACCESS_TOKEN(S) "
            "before exposing this server to testers"
        )
    yield
    # ASCII only: a cp1252 Windows console cannot encode emoji and would
    # crash the lifespan handler with UnicodeEncodeError.
    print("[shutdown] bye")


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.version,
    lifespan=lifespan,
    contact={"name": "Audiobook Web"},
)

app.include_router(auth_router)
app.include_router(health.router)
app.include_router(projects.router)
app.include_router(audio.router)
app.include_router(audio_analysis.router)
app.include_router(narration.router)
app.include_router(casting.router)
app.include_router(elevenlabs.router)
# Generation allowance of the authenticated tester (beta limits).
app.include_router(usage.router)

# Media downloads (generated audio + stored project files): same URLs as
# before, but served through an ownership-checked route instead of a bare
# StaticFiles mount. In beta mode the middleware also requires a tester
# token and project ownership for every /media/projects/{id}/… request.
app.include_router(media.router)

# Private-beta security (transparent when no BETA_ACCESS_TOKEN(S) configured):
# access control, project isolation, quotas, rate limiting, duplicate
# submission protection, /docs disabled for external testers.
app.add_middleware(BetaSecurityMiddleware)

# Frontend (one-page audiobook creator).
# Assets are served under /static (`GET /` in app.routes.health serves
# index.html to browsers via content negotiation). Mounted LAST so no API
# route above is shadowed — and deliberately NOT at "/": a root catch-all
# would swallow unknown API paths instead of returning the JSON 404.
# `Cache-Control: no-cache` makes browsers revalidate every asset
# (ETag/304), so a new app.js / style.css / index.html can never be masked
# by a stale cached copy — auth on these routes is unchanged.
class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles that always let the browser revalidate its cache."""

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


FRONTEND_DIR = Path(__file__).resolve().parent / "static"
if FRONTEND_DIR.is_dir():
    app.mount(
        "/static",
        _NoCacheStaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="web",
    )


@app.exception_handler(ProjectError)
async def project_error_handler(
    _: Request, exc: ProjectError
) -> JSONResponse:
    """Map service-layer domain errors onto their HTTP status code."""
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.exception_handler(NarrationPlanError)
async def narration_plan_error_handler(
    _: Request, exc: NarrationPlanError
) -> JSONResponse:
    """Structured, user-friendly error when plan generation fails.

    The `detail` is the message the frontend shows; `problems` and `attempts`
    are structured debug fields for API clients (never a stack trace).
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": str(exc),
            "problems": exc.problems,
            "attempts": exc.attempts,
        },
    )
