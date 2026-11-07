"""Casting & multi-voice endpoints (spec §11), following existing conventions.

Pipeline:

    POST /projects/{id}/analyze-characters      MiMo discovers the cast
    POST /projects/{id}/voice-casting           ElevenLabs discovery + top-3
    GET  /projects/{id}/voice-casting           review candidates / selections
    POST /projects/{id}/voice-casting/select    permanently lock one voice
    POST /projects/{id}/voice-casting/audition  optional short test line
    POST /projects/{id}/voice-casting/approve   lock the whole cast
    POST /projects/{id}/create-multi-voice-plan MiMo splits paragraphs into
                                                utterances (voices stamped
                                                from the locks)

Audio generation stays on the existing route
`POST /projects/{id}/generate-audio` — it prefers the multi-voice plan when
present and falls back to the paragraph plan / single pass otherwise.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from app import plan_identity
from app.config import get_settings
from app.schemas.casting import (
    AuditionRequest,
    CharacterAnalysisResponse,
    SelectVoiceRequest,
    VoiceCastingResponse,
)
from app.schemas.multi_voice import MultiVoicePlanResponse, MultiVoicePlanStatusResponse
from app.security import current_tester, mv_plan_joined
from app.services.character_analyzer import CharacterAnalyzer
from app.services.multi_voice_planner import MultiVoicePlanner
from app.services.voice_casting_service import VoiceCastingService

# Mounted under /projects so URLs stay consistent with the rest of the API.
router = APIRouter(prefix="/projects", tags=["Casting"])


def _analyzer() -> CharacterAnalyzer:
    """Fresh service per request (settings are re-read from the environment)."""
    return CharacterAnalyzer(get_settings())


def _casting_service() -> VoiceCastingService:
    return VoiceCastingService(get_settings())


def _planner() -> MultiVoicePlanner:
    return MultiVoicePlanner(get_settings())


# --------------------------------------------------------------------------- #
# 1. Story analysis + character discovery
# --------------------------------------------------------------------------- #
@router.post(
    "/{project_id}/analyze-characters",
    response_model=CharacterAnalysisResponse,
    summary="Analyse the script and discover the character cast",
    response_description="Story metadata + narrator/character voice profiles",
)
def analyze_characters(project_id: str) -> CharacterAnalysisResponse:
    """MiMo reads the whole story: tone, genre, language, narrator, characters.

    Attributes the script does not establish (gender/age/accent) stay null
    or are explicitly marked in `uncertain_attributes`. Only characters with
    spoken dialogue receive voice requirements. Saves `character_cast.json`.
    """
    return _analyzer().analyze(project_id)


# --------------------------------------------------------------------------- #
# 2/3. Voice discovery + casting review
# --------------------------------------------------------------------------- #
@router.post(
    "/{project_id}/voice-casting",
    response_model=VoiceCastingResponse,
    summary="Discover and rank ElevenLabs voices for every speaker",
    response_description="Up to 3 candidates per speaking role",
)
async def build_voice_casting(project_id: str) -> VoiceCastingResponse:
    """Search account voices first, then the shared library when needed.

    Candidates are ranked (language, accent, personality, demographics,
    audiobook fit, quality, restrictions) — weak matches are never proposed.
    Existing selections are preserved when re-casting.
    """
    return await _casting_service().build_casting(project_id)


@router.get(
    "/{project_id}/voice-casting",
    response_model=VoiceCastingResponse,
    summary="Get the current voice casting",
    response_description="Candidates, selections, approval state",
)
def get_voice_casting(project_id: str) -> VoiceCastingResponse:
    """Return the cast with candidates and locks, or `404` before casting."""
    return _casting_service().get_casting(project_id)


# --------------------------------------------------------------------------- #
# 4. Permanent selection / lock
# --------------------------------------------------------------------------- #
@router.post(
    "/{project_id}/voice-casting/select",
    response_model=VoiceCastingResponse,
    summary="Select (and permanently lock) a voice for one speaker",
    response_description="Updated casting with the new lock",
)
async def select_voice(
    project_id: str, payload: SelectVoiceRequest
) -> VoiceCastingResponse:
    """Pick a candidate (or any usable voice) for `narrator` / a character.

    Library voices need their `public_owner_id` so they can be added to the
    account first. Selections are verified against `GET /v1/voices/{id}` —
    restricted or unknown voices are rejected. Changing one speaker never
    affects the others and invalidates approval (re-approve to generate).
    """
    return await _casting_service().select(project_id, payload)


# --------------------------------------------------------------------------- #
# 5. Optional short audition (user-approved test line)
# --------------------------------------------------------------------------- #
@router.post(
    "/{project_id}/voice-casting/audition",
    summary="Render a short audition line in a candidate voice",
    response_description="MP3 audio of the test line (never the full story)",
)
async def audition(project_id: str, payload: AuditionRequest) -> Response:
    """Synthesise a ≤400-character test line — costs one small TTS credit.

    The full audiobook is never generated here and the cast is not modified.
    """
    audio = await _casting_service().audition(project_id, payload)
    return Response(content=audio, media_type="audio/mpeg")


# --------------------------------------------------------------------------- #
# 6. Approval
# --------------------------------------------------------------------------- #
@router.post(
    "/{project_id}/voice-casting/approve",
    response_model=VoiceCastingResponse,
    summary="Approve the cast (all speaking roles must be locked)",
    response_description="Approved casting",
)
def approve_voice_casting(project_id: str) -> VoiceCastingResponse:
    """Approve every lock so multi-voice audio generation may run."""
    return _casting_service().approve(project_id)


# --------------------------------------------------------------------------- #
# 7. Utterance plan
# --------------------------------------------------------------------------- #
@router.post(
    "/{project_id}/create-multi-voice-plan",
    response_model=MultiVoicePlanResponse,
    summary="Split paragraphs into speaker-attributed utterances",
    response_description="Validated utterance plan (saved to disk)",
)
def create_multi_voice_plan(project_id: str) -> MultiVoicePlanResponse:
    """MiMo divides each paragraph into ordered utterances per cast speaker.

    Narration and dialogue stay distinct; every utterance's words reproduce
    the original paragraph exactly; locked voice ids are stamped by the
    system (MiMo never assigns voices). Saves `multi_voice_plan.json`.

    Durable recovery (Case A/B): when the server's stored script, cast and
    narration preferences are unchanged and a plan workflow is already paid
    for, this request JOINS it (`mv_plan_joined`) and returns the saved
    plan — no second allowance, no second MiMo call, regardless of what the
    browser did with its localStorage.
    """
    return _planner().create(project_id, joined=mv_plan_joined())


@router.get(
    "/{project_id}/multi-voice-plan/status",
    response_model=MultiVoicePlanStatusResponse,
    summary="Status of the project's multi-voice plan (never charged)",
    response_description=(
        "Plan existence, lifecycle status, validity and readiness for generation"
    ),
)
def multi_voice_plan_status(
    project_id: str, tester: str = Depends(current_tester)
) -> MultiVoicePlanStatusResponse:
    """Server-authoritative plan status — the browser's recovery source.

    Everything is computed from the stored project data (script, character
    cast, narration preferences) and the persisted plan record; the request
    contributes no hashes or identities. Never reserves or consumes an
    allowance, and never exposes another tester's plan (the project
    ownership check runs before this handler and the record owner is
    re-verified here).
    """
    status = plan_identity.plan_status(
        get_settings(), project_id, owner=tester
    )
    return MultiVoicePlanStatusResponse(**status)
