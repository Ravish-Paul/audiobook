"""Whole-story analysis for the AI Narration Director.

    script.txt -> MiMo (one JSON analysis) -> Pydantic validation
              -> normalisation of the recommended profile -> story_analysis.json

The analysis answers the directing questions (genre, tone, scene progression,
recommended global profile, character tendencies, intimacy) and is used as
*context* by both narration planners. It never rewrites the script and it is
never a hard per-paragraph instruction.

The endpoint is an auxiliary provider call: it never consumes a
complete-audiobook allowance, and a saved analysis describing the current
script is reused unless the caller explicitly asks for a fresh one.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional, Sequence

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.schemas.narration_preferences import (
    NARRATION_PROFILES,
    PRESET_LABELS,
    normalize_profile_id,
)
from app.schemas.story_analysis import (
    StoryAnalysisResponse,
    StoryDirectorAnalysis,
)
from app.services.errors import (
    EmptyScriptError,
    StoryAnalysisError,
    StoryAnalysisNotFoundError,
)
from app.services.mimo_client import MiMoClient
from app.services.project_service import ProjectService

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
PROFILE_LINES = "\n".join(
    f"- {profile}: {PRESET_LABELS.get(profile, profile)}"
    for profile in NARRATION_PROFILES
)

SYSTEM_PROMPT = f"""You are a professional audiobook director and story analyst.

You receive a complete story script split into numbered paragraphs. Analyse the WHOLE story and return ONE JSON object describing how it should be directed. You never rewrite, translate or summarise the script.

Return exactly this shape (no markdown fences, no commentary):
{{
  "primary_genre": "<e.g. romantic drama, horror, comedy>",
  "secondary_genre": "<a second genre when genuinely present, else null>",
  "emotional_tone": "<short overall emotional tone>",
  "dominant_mood": "<the strongest mood of the story>",
  "language": "<language name or ISO code, e.g. Hindi / en>",
  "scene_progression": [
    {{
      "scene_index": 0,
      "start_paragraph_index": 0,
      "end_paragraph_index": 3,
      "mood": "<calm | playful | romantic | tense | fearful | sad | hopeful ...>",
      "intensity": 1,
      "summary": "<one short, script-grounded sentence>"
    }}
  ],
  "recommended_profile": "<exactly one profile id from the list below>",
  "recommended_direction": "<why this profile fits and how the mood should progress>",
  "character_tendencies": [
    {{"character_name": "<name exactly as the script writes it>", "tendency": "<stable emotional tendency>"}}
  ],
  "intimacy_level": "none | light | moderate | high"
}}

RULES
1. Base every value on the actual script. Never invent characters, events or relationships that the script does not support.
2. scene_progression must cover the story in order, with valid paragraph ranges (0..N-1) and non-overlapping ranges. A scene is a real beat, not every paragraph.
3. Do NOT describe one mood for the whole story: show how it changes. Preserve neutral and calm passages — a horror story has calm scenes and a sensual romance has ordinary conversation, conflict and neutral narration.
4. recommended_profile must be EXACTLY one of these ids (never `auto`, never a delivery-style value):
{PROFILE_LINES}
5. Choose the profile from the story's dominant need, not from a single scene. A story with one scary scene is not automatically horror; a love story with one intimate scene is not automatically sensual.
6. character_tendencies is for characters the script actually establishes. Keep each tendency stable and short; performance still changes per scene.
7. intimacy_level describes the script as written: none | light | moderate | high. Never add explicit content that is absent.
8. recommended_direction is guidance for the narrator's global approach, not a per-paragraph script."""


def build_user_prompt(
    paragraphs: Sequence[str], feedback: Optional[Sequence[str]] = None
) -> str:
    numbered = [
        {"paragraph_index": index, "text": text}
        for index, text in enumerate(paragraphs)
    ]
    parts = [
        "Analyse the complete audiobook script below and return the directing "
        "analysis as one JSON object.",
        "",
        f"Total paragraphs: {len(paragraphs)}",
        "",
        "Script paragraphs:",
        json.dumps(numbered, ensure_ascii=False, indent=2),
    ]
    if feedback:
        parts += [
            "",
            "Your previous attempt was REJECTED for these reasons:",
            *[f"- {problem}" for problem in feedback],
            "",
            "Fix every problem above and return the complete corrected JSON "
            "object only.",
        ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def script_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_analysis(
    analysis: StoryDirectorAnalysis, paragraph_count: int
) -> list[str]:
    """Cross-check the analysis against the script's paragraph count."""
    problems: list[str] = []

    profile = normalize_profile_id(analysis.recommended_profile)
    if profile is None:
        problems.append(
            "recommended_profile "
            f"{analysis.recommended_profile!r} is not one of the twelve "
            f"profiles ({', '.join(NARRATION_PROFILES)}) — "
            "choose exactly one profile id"
        )

    previous_end = -1
    for scene in analysis.scene_progression:
        label = f"scene {scene.scene_index}"
        if scene.start_paragraph_index > scene.end_paragraph_index:
            problems.append(f"{label}: start is after end")
        if scene.end_paragraph_index >= paragraph_count:
            problems.append(
                f"{label}: end_paragraph_index {scene.end_paragraph_index} is "
                f"outside the script (0..{paragraph_count - 1})"
            )
        if scene.start_paragraph_index <= previous_end:
            problems.append(
                f"{label}: scenes must be ordered and must not overlap"
            )
        previous_end = max(previous_end, scene.end_paragraph_index)

    return problems


def normalize_analysis(data: Any) -> tuple[Any, list[str]]:
    """Map a human profile label onto its id before validation."""
    notes: list[str] = []
    if not isinstance(data, dict):
        return data, notes
    raw = data.get("recommended_profile")
    if isinstance(raw, str):
        normalized = normalize_profile_id(raw)
        if normalized is not None and normalized != raw:
            notes.append(
                f"recommended_profile {raw!r} normalized to {normalized!r}"
            )
            data["recommended_profile"] = normalized
    # Never let the model invent an impossible intimacy value silently.
    return data, notes


def _format_validation_error(exc: ValidationError) -> list[str]:
    problems: list[str] = []
    for err in exc.errors()[:8]:
        loc = ".".join(str(part) for part in (err.get("loc") or ())) or "<root>"
        actual = repr(err.get("input", "<missing>"))
        if len(actual) > 120:
            actual = actual[:117] + "..."
        problems.append(f"{loc}: {err['msg']} (actual value: {actual})")
    return problems


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class StoryAnalysisService:
    """Build, reuse and persist the project's whole-story analysis."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        project_service: Optional[ProjectService] = None,
        mimo_client: Optional[MiMoClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.projects = project_service or ProjectService(self.settings)
        self.mimo = mimo_client or MiMoClient(self.settings)

    # ------------------------------------------------------------------ #
    def load(self, project_id: str) -> Optional[StoryDirectorAnalysis]:
        """The saved analysis, but only while it describes the current script."""
        self.projects.get_project(project_id)
        stored = self.projects.get_story_analysis(project_id)
        if stored is None:
            return None
        analysis, _saved_at, stored_hash = stored
        try:
            script = self.projects.read_script(project_id)
        except Exception:
            return None
        if stored_hash and stored_hash != script_hash(script):
            return None  # the script changed — old analysis is stale
        return analysis

    def get(self, project_id: str) -> StoryAnalysisResponse:
        self.projects.get_project(project_id)
        stored = self.projects.get_story_analysis(project_id)
        if stored is None:
            raise StoryAnalysisNotFoundError(project_id)
        analysis, saved_at, stored_hash = stored
        try:
            current_hash = script_hash(self.projects.read_script(project_id))
        except Exception:
            current_hash = ""
        if stored_hash and current_hash and stored_hash != current_hash:
            raise StoryAnalysisNotFoundError(project_id)
        return StoryAnalysisResponse(
            project_id=project_id,
            analysis=analysis,
            saved=True,
            saved_at=saved_at,
            reused=True,
        )

    def analyze(
        self, project_id: str, *, force: bool = False
    ) -> StoryAnalysisResponse:
        """Analyse the stored script (reusing a matching saved analysis)."""
        from app.services.narration_planner import split_paragraphs

        self.projects.get_project(project_id)
        script = self.projects.read_script(project_id)
        paragraphs = split_paragraphs(script)
        if not paragraphs:
            raise EmptyScriptError(project_id)
        current_hash = script_hash(script)

        if not force:
            stored = self.projects.get_story_analysis(project_id)
            if stored is not None:
                analysis, saved_at, stored_hash = stored
                if not stored_hash or stored_hash == current_hash:
                    return StoryAnalysisResponse(
                        project_id=project_id,
                        analysis=analysis,
                        saved=True,
                        saved_at=saved_at,
                        reused=True,
                    )

        attempts = self.settings.mimo_max_attempts
        problems: list[str] = ["no response from MiMo"]
        for attempt in range(1, attempts + 1):
            user = build_user_prompt(
                paragraphs, feedback=problems if attempt > 1 else None
            )
            raw = self.mimo.complete_json(system=SYSTEM_PROMPT, user=user)
            try:
                data: Any = json.loads(strip_code_fences(raw))
            except json.JSONDecodeError as exc:
                problems = [f"response was not valid JSON: {exc}"]
                continue
            if not isinstance(data, dict):
                problems = ["response must be a single JSON object"]
                continue

            data, _notes = normalize_analysis(data)
            try:
                analysis = StoryDirectorAnalysis.model_validate(data)
            except ValidationError as exc:
                problems = _format_validation_error(exc)
                continue

            problems = validate_analysis(analysis, len(paragraphs))
            if problems:
                continue

            # Store the canonical id (never a label) for deterministic use.
            profile = normalize_profile_id(analysis.recommended_profile)
            if profile and profile != analysis.recommended_profile:
                analysis = analysis.model_copy(
                    update={"recommended_profile": profile}
                )
            saved_at = self.projects.save_story_analysis(
                project_id, analysis, current_hash
            )
            return StoryAnalysisResponse(
                project_id=project_id,
                analysis=analysis,
                saved=True,
                saved_at=saved_at,
                reused=False,
            )

        raise StoryAnalysisError(problems, attempts)
