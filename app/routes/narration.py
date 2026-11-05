"""Narration endpoints: MiMo analyses the script, FastAPI stores the plan."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.config import get_settings
from app.schemas.narration import NarrationPlanResponse
from app.services import NarrationPlanner

# Mounted under /projects so URLs stay consistent with the rest of the API.
router = APIRouter(prefix="/projects", tags=["Narration"])


def _planner() -> NarrationPlanner:
    """Fresh service per request (settings are re-read from the environment)."""
    return NarrationPlanner(get_settings())


@router.post(
    "/{project_id}/analyze-narration",
    response_model=NarrationPlanResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse the script with Xiaomi MiMo",
    response_description="The validated narration plan (also saved to disk)",
)
def analyze_narration(project_id: str) -> NarrationPlanResponse:
    """Ask MiMo to direct every paragraph of the stored script.

    MiMo receives the complete story plus numbered paragraphs and returns
    per-paragraph emotion, delivery, voice settings and pauses. The result is
    validated with Pydantic and cross-checked against the original script
    before it is saved as `narration_plan.json`.

    The original text is never modified; invalid model output is retried and
    finally rejected with a useful error (no audio is generated).
    """
    return _planner().analyze(project_id)


@router.get(
    "/{project_id}/narration-plan",
    response_model=NarrationPlanResponse,
    summary="Get the saved narration plan",
    response_description="The narration plan, ready for review before audio",
)
def get_narration_plan(project_id: str) -> NarrationPlanResponse:
    """Return the saved plan, or `404` when analysis has not run yet."""
    return _planner().get_plan(project_id)
