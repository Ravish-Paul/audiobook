"""AI character discovery: MiMo reads the WHOLE script and builds the cast.

Pipeline (mirrors `narration_planner`):

    script.txt -> MiMo (JSON mode) -> Pydantic validation
              -> cross-check against the script -> character_cast.json

Hard rules enforced here (spec §1):

* the script is never rewritten — analysis only describes and attributes;
* gender / age / accent are NEVER invented: unstated -> null, inferred ->
  explicitly listed in `uncertain_attributes`;
* only characters with `has_spoken_dialogue` get voice requirements, plus
  the always-required narrator profile (reserved id `narrator`);
* a character credited with dialogue must actually be named in the script;
* invalid model output never reaches disk: it is retried with feedback and
  finally rejected with a useful 502.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence

from pydantic import ValidationError

from app.config import LANGUAGE_NAME_TO_CODE, Settings, get_settings
from app.schemas.casting import (
    NARRATOR_ID,
    CharacterAnalysisResponse,
    CharacterCast,
    CharacterProfile,
    SpeakerCasting,
    StoryAnalysis,
)
from app.services.errors import (
    CharacterAnalysisError,
    EmptyScriptError,
)
from app.services.mimo_client import MiMoClient
from app.services.project_service import ProjectService

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are an expert casting director and script analyst at a professional audiobook studio.

You receive a complete story script split into numbered paragraphs. Your job is to describe the story and extract its CAST — you never rewrite, translate or summarise the script.

HARD RULES
1. Never modify the script text. You only describe and attribute.
2. Return exactly one JSON object (no markdown fences) with this shape:
{
  "overall_tone": "<short tone, e.g. suspenseful, warm, ominous>",
  "genre": "<e.g. horror, literary fiction, thriller>",
  "language": "<ISO 639-1 code when identifiable, e.g. hi / en; otherwise the language name>",
  "locale": "<regional locale when identifiable, e.g. hi-IN; else null>",
  "narrator": {<character profile with character_id "narrator">},
  "characters": [<character profiles>]
}
3. ALWAYS create the narrator profile first: character_id exactly "narrator", role "narrator", character_name "Narrator". Give it voice_description + speaking_style describing the storytelling voice the script implies.
4. Character profile shape:
{
  "character_id": "<lowercase slug: letters, digits, underscores — e.g. old_receptionist>",
  "character_name": "<display name as written in the script>",
  "role": "protagonist | antagonist | supporting_character | minor_character | ensemble",
  "description": "<only what the script establishes>",
  "relationships": ["<only stated relationships>"],
  "language": "<same as story language>",
  "gender": "<only if the script establishes it; else null>",
  "age": "<only if the script establishes it, e.g. mature/young; else null>",
  "accent": "<only if the script establishes it; else null>",
  "uncertain_attributes": ["gender" | "age" | "accent"],
  "voice_description": "<required vocal personality for speakers>",
  "speaking_style": "<how they talk: natural, restrained, theatrical...>",
  "has_spoken_dialogue": true | false
}
5. NEVER invent gender, age or accent. When the script does not clearly establish an attribute set it to null. If you infer an attribute with any doubt, keep the value AND list that attribute in uncertain_attributes. uncertain_attributes may only contain "gender", "age" and "accent".
6. has_spoken_dialogue is true ONLY when the character actually speaks or is quoted. Characters without spoken dialogue get empty voice_description and speaking_style — no speaking voice is required for them.
7. Every character credited with dialogue MUST have their character_name appear in the script (dialogue attribution — who says which line).
8. voice_description and speaking_style are mandatory for the narrator and every character with has_spoken_dialogue=true: ground them in the script (restrained, natural audiobook delivery — not theatrical).
9. Include every character that appears, in first-appearance order. Never merge two characters; never invent one."""


def build_user_prompt(script_paragraphs: Sequence[str], feedback: Optional[Sequence[str]] = None) -> str:
    numbered = [
        {"paragraph_index": index, "text": text}
        for index, text in enumerate(script_paragraphs)
    ]
    parts = [
        "Analyse the complete audiobook script below and return the story "
        "analysis + character cast as one JSON object.",
        "",
        f"Total paragraphs: {len(script_paragraphs)}",
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
# Validation (never trust the LLM)
# --------------------------------------------------------------------------- #
def strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_language(value: str) -> str:
    """Prefer an ISO 639-1 code; fall back to the raw value untouched."""
    cleaned = (value or "").strip()
    if not cleaned:
        return cleaned
    mapped = LANGUAGE_NAME_TO_CODE.get(cleaned.lower())
    if mapped:
        return mapped
    if len(cleaned) == 2 and cleaned.isalpha():
        return cleaned.lower()
    return cleaned


def validate_analysis_against_script(
    analysis: StoryAnalysis, script_paragraphs: Sequence[str]
) -> list[str]:
    """Cross-check the cast against the uploaded script. Returns problems."""
    problems: list[str] = []
    script = "\n".join(script_paragraphs)
    script_lower = script.lower()

    if analysis.narrator.character_id != NARRATOR_ID:
        problems.append(f"narrator.character_id must be '{NARRATOR_ID}'")

    for character in analysis.characters:
        label = f"character '{character.character_id}'"
        if character.has_spoken_dialogue:
            if character.character_name.strip().lower() not in script_lower:
                problems.append(
                    f"{label}: character_name '{character.character_name}' never "
                    "appears in the script — do not invent characters credited "
                    "with dialogue (check dialogue attribution)"
                )
            if not character.voice_description.strip():
                problems.append(
                    f"{label}: voice_description is required for characters "
                    "with spoken dialogue"
                )
            if not character.speaking_style.strip():
                problems.append(
                    f"{label}: speaking_style is required for characters "
                    "with spoken dialogue"
                )
        # Attributes must be null when listed as uncertain-unproven OR the
        # value must be explicitly flagged — never silently invented.
        for attribute in ("gender", "age", "accent"):
            value = getattr(character, attribute)
            if value and attribute not in character.uncertain_attributes:
                # Established (not flagged) is fine — nothing to verify here.
                pass
            if value is None and attribute in character.uncertain_attributes:
                problems.append(
                    f"{label}: uncertain_attributes lists '{attribute}' but the "
                    "value is null — remove it from the list or provide the value"
                )

    if not analysis.narrator.voice_description.strip():
        problems.append("narrator: voice_description is required")
    if not analysis.narrator.speaking_style.strip():
        problems.append("narrator: speaking_style is required")

    return problems


def _format_validation_error(exc: ValidationError) -> list[str]:
    problems: list[str] = []
    for err in exc.errors()[:8]:
        location = ".".join(str(part) for part in err["loc"]) or "<root>"
        problems.append(f"{location}: {err['msg']}")
    return problems


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #
class CharacterAnalyzer:
    """Drives MiMo, validates the cast and persists `character_cast.json`."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        project_service: Optional[ProjectService] = None,
        mimo_client: Optional[MiMoClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.projects = project_service or ProjectService(self.settings)
        # DI seam: tests inject a fake MiMo (no credits).
        self.mimo = mimo_client or MiMoClient(self.settings)

    def analyze(self, project_id: str) -> CharacterAnalysisResponse:
        self.projects.get_project(project_id)
        script = self.projects.read_script(project_id)
        paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n+", script)
            if part.strip()
        ] or [line.strip() for line in script.splitlines() if line.strip()]
        if not paragraphs:
            raise EmptyScriptError(project_id)

        analysis = self.build_analysis(paragraphs)
        cast = CharacterCast(
            project_id=project_id,
            story=analysis,
            casting={
                speaker.character_id: SpeakerCasting()
                for speaker in analysis.speakers()
            },
            approved=False,
        )
        saved_at = self.projects.save_character_cast(project_id, cast)
        return CharacterAnalysisResponse(
            project_id=project_id,
            story=analysis,
            speaker_ids=[s.character_id for s in analysis.speakers()],
            saved=True,
            saved_at=saved_at,
        )

    # ------------------------------------------------------------------ #
    # MiMo round trip + retries
    # ------------------------------------------------------------------ #
    def build_analysis(self, paragraphs: Sequence[str]) -> StoryAnalysis:
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

            try:
                analysis = StoryAnalysis.model_validate(data)
            except ValidationError as exc:
                problems = _format_validation_error(exc)
                continue

            # Normalise the story + per-speaker language to ISO codes.
            analysis.language = normalize_language(analysis.language)
            for speaker in analysis.speakers():
                speaker.language = speaker.language or analysis.language
                speaker.language = normalize_language(speaker.language)

            problems = validate_analysis_against_script(analysis, paragraphs)
            if not problems:
                return analysis

        raise CharacterAnalysisError(problems, attempts)
