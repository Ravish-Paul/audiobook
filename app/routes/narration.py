"""Narration endpoints: MiMo analyses the script, FastAPI stores the plan.

The Narration Preferences panel drives these:

    GET    /projects/{id}/narration-preferences          → hydrate the panel
    PUT    /projects/{id}/narration-preferences          → save without generating
    POST   /projects/{id}/analyze-narration/preview      → one paragraph
    POST   /projects/{id}/analyze-narration              → the whole plan
    GET    /projects/{id}/narration-plan                 → review the saved plan

Preferences are never frontend-only: whatever is sent here is persisted per
project, forwarded to MiMo, embedded in `narration_plan.json`, enforced by
`NarrationPolicyValidator` and part of the narration plan hash.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, status
from fastapi.responses import Response

from app.config import get_settings
from app.schemas.narration import NarrationPlanResponse, NarrationPreviewResponse
from app.schemas.narration_preferences import (
    AnalyzeNarrationRequest,
    NarrationPreferences,
    NarrationPreferencesResponse,
    PreviewNarrationAudioRequest,
    PreviewNarrationRequest,
)
from app.schemas.story_analysis import StoryAnalysisRequest, StoryAnalysisResponse
from app.services import NarrationPlanner, StoryAnalysisService

# Mounted under /projects so URLs stay consistent with the rest of the API.
router = APIRouter(prefix="/projects", tags=["Narration"])


def _planner() -> NarrationPlanner:
    """Fresh service per request (settings are re-read from the environment)."""
    return NarrationPlanner(get_settings())


def _story_service() -> StoryAnalysisService:
    return StoryAnalysisService(get_settings())


@router.post(
    "/{project_id}/analyze-story",
    response_model=StoryAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse the whole story (genre, mood progression, profile)",
    response_description="The directing analysis (also saved to disk)",
)
def analyze_story(
    project_id: str,
    request: Optional[StoryAnalysisRequest] = None,
) -> StoryAnalysisResponse:
    """Read the whole script and return a directing analysis.

    This is an auxiliary provider call: it never consumes a complete-audiobook
    allowance. A saved analysis that still describes the current script is
    reused unless `{"force": true}` is sent.
    """
    return _story_service().analyze(
        project_id, force=bool(request and request.force)
    )


@router.get(
    "/{project_id}/story-analysis",
    response_model=StoryAnalysisResponse,
    summary="Get the saved whole-story analysis",
    response_description="The saved analysis, or 404 when none exists",
)
def get_story_analysis(project_id: str) -> StoryAnalysisResponse:
    return _story_service().get(project_id)


@router.post(
    "/{project_id}/narration-preview/audio",
    summary="Render one representative narration line",
    response_description="MP3 audio for the previewed paragraph",
)
async def preview_narration_audio(
    project_id: str, request: PreviewNarrationAudioRequest
) -> Response:
    """Synthesise ONE paragraph as a short sample on explicit user action.

    Uses the existing narration-preview planner (aux MiMo) and the
    capability-filtered ElevenLabs client. Nothing is written to the project's
    audiobook files and no complete-audiobook allowance is consumed.
    """
    audio = await _planner().preview_audio(
        project_id,
        request.paragraph_index,
        preferences=request.preferences,
    )
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": 'inline; filename="narration-preview.mp3"',
        },
    )


@router.post(
    "/{project_id}/analyze-narration",
    response_model=NarrationPlanResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse the script with Xiaomi MiMo",
    response_description="The validated narration plan (also saved to disk)",
)
def analyze_narration(
    project_id: str,
    request: Optional[AnalyzeNarrationRequest] = None,
) -> NarrationPlanResponse:
    """Ask MiMo to direct every paragraph of the stored script.

    Optional body: the COMPLETE preferences object from the Narration
    Preferences panel (`{"preferences": {...}}`). It is persisted for the
    project, sent to MiMo and enforced by the policy validator; a plan that
    violates it is rejected and retried with targeted feedback.

    The original text is never modified; invalid model output is retried and
    finally rejected with a useful error (no audio is generated).
    """
    preferences = request.preferences if request else None
    return _planner().analyze(project_id, preferences=preferences)


@router.post(
    "/{project_id}/analyze-narration/preview",
    response_model=NarrationPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Preview ONE paragraph before generating the whole audiobook",
    response_description="How the selected paragraph would be narrated",
)
def preview_narration(
    project_id: str,
    request: PreviewNarrationRequest,
) -> NarrationPreviewResponse:
    """Direct a single paragraph (its neighbours are used as context).

    Nothing is written to `narration_plan.json`, so a preview never
    overwrites an existing plan. Preferences sent here are persisted.
    """
    return _planner().preview(
        project_id,
        request.paragraph_index,
        preferences=request.preferences,
    )


@router.get(
    "/{project_id}/narration-preferences",
    response_model=NarrationPreferencesResponse,
    summary="Get the project's narration preferences",
    response_description="Saved preferences, or the defaults when unset",
)
def get_narration_preferences(project_id: str) -> NarrationPreferencesResponse:
    """Return the stored preferences (`saved: false` → panel defaults)."""
    return _planner().get_preferences(project_id)


@router.put(
    "/{project_id}/narration-preferences",
    response_model=NarrationPreferencesResponse,
    summary="Save the narration preferences",
    response_description="The persisted preferences",
)
def put_narration_preferences(
    project_id: str, preferences: NarrationPreferences
) -> NarrationPreferencesResponse:
    """Persist the panel's preferences without generating a plan yet."""
    return _planner().save_preferences(project_id, preferences)


@router.get(
    "/{project_id}/narration-plan",
    response_model=NarrationPlanResponse,
    summary="Get the saved narration plan",
    response_description="The narration plan, ready for review before audio",
)
def get_narration_plan(project_id: str) -> NarrationPlanResponse:
    """Return the saved plan, or `404` when analysis has not run yet."""
    return _planner().get_plan(project_id)
