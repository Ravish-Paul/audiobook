"""Audio endpoints: generate speech from the stored script and read it back."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.config import get_settings
from app.schemas.audio import AudioResponse
from app.services import AudioService

# Mounted under /projects so URLs stay consistent with the rest of the API.
router = APIRouter(prefix="/projects", tags=["Audio"])


def _audio_service() -> AudioService:
    """Fresh service per request (settings are re-read from the environment)."""
    return AudioService(get_settings())


@router.post(
    "/{project_id}/generate-audio",
    response_model=AudioResponse,
    summary="Generate audio from the uploaded script",
    response_description="Audio metadata including the playable audio_url",
)
async def generate_audio(project_id: str) -> AudioResponse:
    """Convert the already-uploaded `script.txt` into an MP3 via ElevenLabs.

    The script is read from local storage — users never upload it twice.

    Flow: project exists (404) -> script stored (404) -> script not empty (400)
    -> text within the ElevenLabs limit (400) -> ElevenLabs TTS call
    -> MP3 saved to `storage/projects/{id}/audio/audiobook.mp3`
    -> public URL returned.
    """
    return await _audio_service().generate_audio(project_id)


@router.get(
    "/{project_id}/audio",
    response_model=AudioResponse,
    summary="Get generated audio information",
    response_description="Audio metadata including the playable audio_url",
)
def get_audio(project_id: str) -> AudioResponse:
    """Return the generated MP3 details, or `404` when it does not exist yet."""
    return _audio_service().get_audio(project_id)
