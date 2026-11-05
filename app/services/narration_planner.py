"""AI narration analysis engine (MiMo = the professional director).

Pipeline:

    script.txt -> split paragraphs -> MiMo (JSON) -> Pydantic validation
               -> cross-check against the original script -> narration_plan.json

Hard rules enforced here:

* the original script is NEVER rewritten, shortened, translated or
  paraphrased — every `original_text` must match the uploaded paragraph;
* every paragraph index appears exactly once, in order;
* invalid model output never reaches audio generation: it is retried with
  feedback and finally rejected with a useful error.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.schemas.narration import (
    DELIVERY_STYLES,
    EMOTIONS,
    NarrationPlan,
    NarrationPlanResponse,
    ParagraphPlan,
)
from app.services.errors import (
    EmptyScriptError,
    NarrationPlanError,
    NarrationPlanNotFoundError,
)
from app.services.mimo_client import MiMoClient
from app.services.project_service import ProjectService

# Paragraphs are separated by one or more blank lines.
_BLANK_LINES = re.compile(r"\n\s*\n+")

MAX_DIRECTION_CHARS = 1000


def split_paragraphs(script: str) -> list[str]:
    """Split a script into paragraphs, preserving each paragraph's text.

    Blank lines separate paragraphs. When a script has no blank lines but
    several lines, each non-empty line becomes its own paragraph so long
    single-paragraph scripts stay within the ElevenLabs request limit.
    """
    blocks = [part.strip() for part in _BLANK_LINES.split(script) if part.strip()]
    if len(blocks) <= 1:
        lines = [line.strip() for line in script.splitlines() if line.strip()]
        if len(lines) > 1:
            blocks = lines
    return blocks


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = f"""You are an expert audiobook narration director at a professional audiobook studio.

You receive a complete story script that is already split into numbered paragraphs.
Your job is to decide HOW every single paragraph must be narrated — nothing else.

HARD RULES
1. Never rewrite, shorten, translate or paraphrase the script. `original_text` must be copied EXACTLY from the input text, character for character, including punctuation and line breaks.
2. Every paragraph index from 0 to N-1 must appear exactly once, in order. No duplicates, no omissions, no extra paragraphs.
3. Analyse each paragraph on its own WHILE considering the surrounding paragraphs, the overall story context and the emotional progression of the story. Settings must differ where the story shifts — never apply identical settings to every paragraph.
4. You are a director, not a summariser: give concrete, production-ready narration choices.

AUDIO TAGS (`tts_text` — Eleven v4 inline delivery tags)
5. Every paragraph MUST include `tts_text`: a copy of `original_text` with a FEW inline Eleven v4 audio tags inserted in square brackets. `tts_text` is never omitted — when no expression is needed it simply equals `original_text`.
6. EXACTNESS (hard rule): stripping every `[tag]` and its single separating space from `tts_text` must reproduce `original_text` byte-for-byte. Never change, remove, add, translate or reorder a single spoken word; never alter punctuation, line breaks or spacing of the original.
7. Natural tags to reach for: [whispers], [shouts], [sad], [curious], [laughs], [sighs], [low, ominous voice], [quiet, tense delivery]. Free-form cues in the same style are fine: [softly], [with quiet urgency], [dramatic pause], [voice dropping to a whisper].
8. Use tags ONLY where they materially improve the narration — typically 0-2 per paragraph (never more than 8). Prefer no tags over weak ones. Place a tag immediately before the words it changes: `[whispers] वह धीरे से बोला।`
9. SENTENCE-LEVEL TAGS: when the narration changes mid-paragraph, place a new tag at the sentence boundary. For a paragraph that moves from narrator narration to character dialogue, give EACH part its own delivery instruction — e.g. `[quiet, tense delivery] The corridor went silent. [whispers] "Don't turn around," he murmured.` The narrator stays in one voice quality, the quoted line switches.
10. MiMo decides: emotional delivery, narration intensity, when a whisper / shout / laugh / sigh / dramatic or ominous delivery is required, and sentence-level voice transitions (switch the voice quality mid-paragraph with a tag such as `[gruff, low voice]` before a character speaks).
11. NEVER put original words inside brackets, NEVER add or remove any spoken word, and NEVER write narration instructions outside the brackets — instructions outside brackets would be spoken aloud.
12. Use Eleven v4 audio tags only. NEVER emit SSML such as <break time="1s"/> — Eleven v4 does not support SSML; use tags, ellipses (...) and punctuation instead.
13. The script language (Hindi, English, …) never changes; tags stay short English delivery cues even for Hindi scripts.

ALLOWED VALUES
- emotion (exactly one): {", ".join(EMOTIONS)}
- delivery_style (exactly one): {", ".join(DELIVERY_STYLES)}
- intensity: integer 1..5
- pause_after_ms: integer 0..3000 (silence after the paragraph)
- narration_direction: 1-2 sentences telling the voice actor how the paragraph should sound (metadata, it is never spoken)
- voice_settings ranges:
    speed 0.7..1.2, stability 0.0..1.0, similarity_boost 0.0..1.0,
    style 0.0..1.0, use_speaker_boost true/false

OUTPUT
Return ONLY one valid JSON object (no markdown fences, no commentary) with exactly this shape:
{{
  "overall_story_tone": "<string>",
  "language": "<string, e.g. Hindi or English>",
  "paragraphs": [
    {{
      "paragraph_index": 0,
      "original_text": "<exact original text>",
      "tts_text": "<required: original text with only [audio tags] inserted — stripping the tags reproduces original_text byte-for-byte>",
      "emotion": "<one of the allowed emotions>",
      "delivery_style": "<one of the allowed delivery styles>",
      "intensity": 1,
      "narration_direction": "<short direction>",
      "voice_settings": {{
        "speed": 1.0, "stability": 0.5, "similarity_boost": 0.75,
        "style": 0.0, "use_speaker_boost": true
      }},
      "pause_after_ms": 500
    }}
  ]
}}"""


def build_user_prompt(
    paragraphs: Sequence[str], feedback: Optional[Sequence[str]] = None
) -> str:
    """Render the numbered script (JSON, so Hindi/line breaks survive)."""
    numbered = [
        {"paragraph_index": index, "text": text}
        for index, text in enumerate(paragraphs)
    ]
    parts = [
        "Analyze the complete audiobook script below and return the "
        "narration plan as one JSON object.",
        "",
        f"Total paragraphs: {len(paragraphs)}",
        "",
        "Script paragraphs (copy each `text` into `original_text` exactly):",
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
    """Tolerate ```json ... ``` wrappers even though JSON mode was requested."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def validate_against_script(
    plan: NarrationPlan, expected: Sequence[str]
) -> list[str]:
    """Cross-check the plan against the uploaded script. Returns problems."""
    from app.services.audio_tags import validate_tts_text

    problems: list[str] = []
    indices = [p.paragraph_index for p in plan.paragraphs]

    duplicates = sorted({i for i in indices if indices.count(i) > 1})
    if duplicates:
        problems.append(f"duplicate paragraph_index value(s): {duplicates}")

    expected_indices = list(range(len(expected)))
    if sorted(indices) != expected_indices:
        missing = sorted(set(expected_indices) - set(indices))
        extra = sorted(set(indices) - set(expected_indices))
        if missing:
            problems.append(f"missing paragraph_index value(s): {missing}")
        if extra:
            problems.append(f"unknown paragraph_index value(s): {extra}")

    by_index = {p.paragraph_index: p for p in plan.paragraphs}
    for index, text in enumerate(expected):
        paragraph = by_index.get(index)
        if paragraph is None:
            continue
        if paragraph.original_text.strip() != text:
            problems.append(
                f"paragraph {index}: original_text does not match the uploaded "
                "script exactly (the script must never be altered)"
            )
        if len(paragraph.narration_direction) > MAX_DIRECTION_CHARS:
            problems.append(
                f"paragraph {index}: narration_direction is too long "
                f"({len(paragraph.narration_direction)} chars)"
            )
        # tts_text may only insert [audio tags] — never change the words.
        for problem in validate_tts_text(paragraph.original_text, paragraph.tts_text):
            problems.append(f"paragraph {index}: {problem}")
    return problems


def require_tts_text(plan: NarrationPlan) -> list[str]:
    """Every paragraph must carry `tts_text` (tags optional, presence NOT).

    MiMo frequently omits newly-added optional fields, which previously let
    a plan save without `tts_text` at all. The field is the whole point of
    the v4 pipeline, so its absence is a validation failure that the retry
    loop feeds back to the model. (Old plans saved before the upgrade keep
    loading — this check only runs when a NEW plan is built.)
    """
    problems: list[str] = []
    for paragraph in plan.paragraphs:
        if paragraph.tts_text is None or not paragraph.tts_text.strip():
            problems.append(
                f"paragraph {paragraph.paragraph_index}: tts_text is required "
                "— copy original_text exactly and insert [audio tags] where "
                "the delivery changes; when no expression is needed, set "
                "tts_text equal to original_text (never omit the field)"
            )
    return problems


def _format_validation_error(exc: ValidationError) -> list[str]:
    problems: list[str] = []
    for err in exc.errors()[:8]:
        location = ".".join(str(part) for part in err["loc"]) or "<root>"
        problems.append(f"{location}: {err['msg']}")
    return problems


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
class NarrationPlanner:
    """Drives MiMo, validates its plan and persists it for the project."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        project_service: Optional[ProjectService] = None,
        mimo_client: Optional[MiMoClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.projects = project_service or ProjectService(self.settings)
        # DI seam: tests inject a fake MiMo transport (no real credits).
        self.mimo = mimo_client or MiMoClient(self.settings)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze(self, project_id: str) -> NarrationPlanResponse:
        """Read the stored script and build a validated narration plan."""
        self.projects.get_project(project_id)  # 404 for unknown project
        script = self.projects.read_script(project_id)  # 404 / 400
        paragraphs = split_paragraphs(script)
        if not paragraphs:
            raise EmptyScriptError(project_id)

        plan = self.build_plan(paragraphs)
        saved_at = self.projects.save_narration_plan(project_id, plan)
        return NarrationPlanResponse.from_plan(
            project_id, plan, saved=True, saved_at=saved_at
        )

    def get_plan(self, project_id: str) -> NarrationPlanResponse:
        """Return the saved narration plan (404 when analysis never ran)."""
        self.projects.get_project(project_id)
        stored = self.projects.get_narration_plan(project_id)
        if stored is None:
            raise NarrationPlanNotFoundError(project_id)
        plan, saved_at = stored
        return NarrationPlanResponse.from_plan(
            project_id, plan, saved=True, saved_at=saved_at
        )

    # ------------------------------------------------------------------ #
    # MiMo round trip + retries
    # ------------------------------------------------------------------ #
    def build_plan(self, paragraphs: Sequence[str]) -> NarrationPlan:
        """Call MiMo and validate/retry until the plan matches the script."""
        attempts = self.settings.mimo_max_attempts
        problems: list[str] = ["no response from MiMo"]

        for attempt in range(1, attempts + 1):
            user = build_user_prompt(paragraphs, feedback=problems if attempt > 1 else None)
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
                plan = NarrationPlan.model_validate(data)
            except ValidationError as exc:
                problems = _format_validation_error(exc)
                continue

            problems = validate_against_script(plan, paragraphs)
            # `tts_text` presence is mandatory for NEW plans (requirement 2):
            # MiMo must always populate it, with tags only where needed.
            problems += require_tts_text(plan)
            if not problems:
                return plan

        raise NarrationPlanError(problems, attempts)
