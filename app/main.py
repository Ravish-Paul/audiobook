"""FastAPI application entry point.

Run it with:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routes import audio, casting, elevenlabs, health, narration, projects
from app.services.audio_composer import resolve_ffmpeg
from app.services.errors import FFmpegNotFoundError, ProjectError
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

    try:
        ffmpeg = resolve_ffmpeg(settings)
        print(f"[startup] ffmpeg: {ffmpeg}")
    except FFmpegNotFoundError:
        print(
            "[startup] ffmpeg: NOT FOUND — segment merging will fail until "
            "FFMPEG_PATH is set"
        )
    yield
    print("[shutdown] bye 👋")


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.version,
    lifespan=lifespan,
    contact={"name": "Audiobook Web"},
)

app.include_router(health.router)
app.include_router(projects.router)
app.include_router(audio.router)
app.include_router(narration.router)
app.include_router(casting.router)
app.include_router(elevenlabs.router)

# Serve generated audio: /media/projects/{project_id}/audio/audiobook.mp3
# Only `storage/projects/` is exposed — the repository root (and therefore
# .env) is never mounted.
settings.media_root.mkdir(parents=True, exist_ok=True)
app.mount(
    settings.media_url_prefix,
    StaticFiles(directory=str(settings.media_root)),
    name="media",
)


@app.exception_handler(ProjectError)
async def project_error_handler(
    _: Request, exc: ProjectError
) -> JSONResponse:
    """Map service-layer domain errors onto their HTTP status code."""
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})
