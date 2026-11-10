"""Audio analysis endpoints: pyannoteAI diarization of the audiobook.

Stage 1 of the sound-design pipeline — analysis ONLY. These endpoints
never generate music/SFX and never mix audio.

Flow (the initial request NEVER waits for transcription):

    POST   /projects/{id}/audio-analysis          -> upload + submit job
    GET    /projects/{id}/audio-analysis/status    -> poll (on demand,
           exponential backoff) + fetch/persist results the moment the
           provider reports `succeeded`
    GET    /projects/{id}/audio-analysis           -> normalized timeline
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.schemas.audio_analysis import (
    AudioAnalysisNormalized,
    AudioAnalysisStartResponse,
    AudioAnalysisStatusResponse,
)
from app.services.audio_analysis_service import AudioAnalysisService

# Mounted under /projects so URLs stay consistent with the rest of the API.
router = APIRouter(prefix="/projects", tags=["Audio Analysis"])


def _service() -> AudioAnalysisService:
    """Fresh service per request (settings are re-read from the environment)."""
    return AudioAnalysisService(get_settings())


@router.post(
    "/{project_id}/audio-analysis",
    response_model=AudioAnalysisStartResponse,
    summary="Start pyannoteAI audio analysis of the generated audiobook",
    response_description="analysis_id + provider job status (job runs async)",
)
async def start_audio_analysis(
    project_id: str,
) -> AudioAnalysisStartResponse:
    """Upload the voice-only audiobook to pyannoteAI and start diarization.

    Flow: project exists (404) -> audiobook generated (404) ->
    PYANNOTE_API_KEY configured (500) -> SHA-256 + ffprobe ->
    duplicate check (identical analysis is REUSED, no credits spent) ->
    unique `media://` key -> pre-signed upload -> `POST /v1/diarize`.

    Returns immediately with `analysis_id` and the provider `job_status`
    (`created`/`pending`/`running`) — the job itself runs asynchronously
    on pyannoteAI's side. Poll `GET .../audio-analysis/status`.

    Cost warning: each SUCCESSFUL diarization job has a 20-second minimum
    charge at pyannoteAI; identical inputs are never submitted twice.
    """
    return await _service().start_analysis(project_id)


@router.get(
    "/{project_id}/audio-analysis/status",
    response_model=AudioAnalysisStatusResponse,
    summary="Get audio analysis status (local + provider)",
    response_description="Local and pyannoteAI job status",
)
async def get_audio_analysis_status(
    project_id: str,
) -> AudioAnalysisStatusResponse:
    """Return the current analysis status, polling the provider at most once.

    Polling is on demand with exponential backoff between provider calls
    (rate-limit friendly). When pyannoteAI reports `succeeded` this call
    IMMEDIATELY saves `audio_analysis_raw.json` and
    `audio_analysis_normalized.json` to project storage — the provider
    deletes job output after 24 hours, so we never rely on it being there.

    Errors: 404 unknown project / analysis never started; 504 provider
    timeout; 429 provider rate limit; 502 provider rejected our API key
    (never 401 — 401 means the caller's own session is invalid).
    """
    return await _service().get_status(project_id)


@router.get(
    "/{project_id}/audio-analysis",
    response_model=AudioAnalysisNormalized,
    summary="Get the normalized audio analysis",
    response_description="Word-level timeline with speaker attribution",
)
def get_audio_analysis(project_id: str) -> AudioAnalysisNormalized:
    """Return the completed normalized timeline.

    Contains `speakers`, `words` (raw seconds + ms + sample indexes),
    `speaker_turns` and optional turn-level transcriptions, all validated
    against the real duration/sample rate of the audiobook.

    Errors: 404 unknown project / analysis never started / files missing;
    409 analysis still running, failed or canceled (see `error`).
    """
    return _service().get_normalized(project_id)
