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
from app.security import current_generation_id
from app.schemas.narration import (
    DELIVERY_STYLES,
    EMOTIONS,
    EffectiveVoiceSettings,
    NarrationPlan,
    NarrationPlanResponse,
    NarrationPreviewResponse,
    ParagraphPlan,
    tts_projection,
)
from app.schemas.narration_preferences import (
    PRESET_LABELS,
    TAG_DENSITY_LABELS,
    NarrationPreferences,
    NarrationPreferencesResponse,
)
from app.services.eleven_models import get_capabilities
from app.services.errors import (
    EmptyScriptError,
    InvalidNarrationPreferencesError,
    NarrationPlanError,
    NarrationPlanNotFoundError,
    ParagraphPreviewError,
    ProjectError,
)
from app.services.mimo_client import MiMoClient
from app.services.narration_policy import (
    NarrationPolicyValidator,
    max_tags_per_paragraph,
    whisper_budget,
)
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
# The natural-delivery rules below are ALWAYS in force — they are what fix the
# overacting problem (excessive whispering, trembling, breathlessness).
# `build_system_prompt` appends the user's preferences and the capability
# block for the configured ElevenLabs model.
BASE_SYSTEM_PROMPT = f"""You are an expert audiobook narration director at a professional audiobook studio.

You receive a complete story script that is already split into numbered paragraphs.
Your job is to decide HOW every single paragraph must be narrated — nothing else.

HARD RULES
1. Never rewrite, shorten, translate or paraphrase the script. `original_text` must be copied EXACTLY from the input text, character for character, including punctuation and line breaks.
2. Every paragraph index from 0 to N-1 must appear exactly once, in order. No duplicates, no omissions, no extra paragraphs.
3. Analyse each paragraph on its own WHILE considering the surrounding paragraphs, the overall story context and the emotional progression of the story. Settings must differ where the story shifts — never apply identical settings to every paragraph.
4. You are a director, not a summariser: give concrete, production-ready narration choices.

NATURAL DELIVERY (the most important rule — do not overact)
5. Most of the narration must sound like a natural human being telling a story: controlled, consistent, effortless. The narrator performs the story, never "acts out" it.
6. An emotion label describes the scene; it does NOT demand a vocal cliché:
   - Fear does not automatically mean whispering.
   - Suspense does not automatically mean breathless delivery.
   - Sadness does not automatically mean crying.
   - Anger does not automatically mean shouting.
7. Emotional performance stays SUBTLE unless the NARRATION PREFERENCES below explicitly ask for more.
8. Do NOT insert an audio tag merely because a paragraph has an emotion label.
9. Avoid excessive whisper, hushed, trembling, breathless, shouting and dramatic instructions anywhere (tags, narration_direction, delivery_style). The narrator stays natural, controlled and consistent.
10. Create an audio tag ONLY when there is an intentional, audible delivery change — a real switch in how the line is spoken. When nothing changes audibly, `tts_text` equals `original_text`.

AUDIO TAGS (`tts_text` — Eleven v4 inline delivery tags)
11. Every paragraph MUST include `tts_text`: a copy of `original_text` with only the inline audio tags a genuine delivery change needs. `tts_text` is never omitted — when no expression is needed it simply equals `original_text`.
12. EXACTNESS (hard rule): stripping every `[tag]` and its single separating space from `tts_text` must reproduce `original_text` byte-for-byte. Never change, remove, add, translate or reorder a single spoken word; never alter punctuation, line breaks or spacing of the original.
13. Tags are sparse: most paragraphs carry NO tag. The NARRATION PREFERENCES below set a hard per-paragraph tag budget — never exceed it, and prefer no tag over a weak one. Place a tag immediately before the words it changes: `[quiet, tense delivery] "Don't turn around," he murmured.`
14. SENTENCE-LEVEL TAGS: only when the narration genuinely changes mid-paragraph (narrator narration -> quoted dialogue), place a new tag at the sentence boundary and give each part its own delivery instruction. The narrator stays in one voice quality; only the quoted line switches.
15. You decide emotional delivery and pacing INSIDE the given preferences, and may switch voice quality mid-paragraph for dialogue (e.g. a low, gruff voice before a character speaks).
16. NEVER put original words inside brackets, NEVER add or remove any spoken word, and NEVER write narration instructions outside the brackets — instructions outside brackets would be spoken aloud.
17. Use Eleven v4 audio tags only. NEVER emit SSML such as <break time="1s"/> — Eleven v4 does not support SSML; use tags, ellipses (...) and punctuation instead.
18. The script language (Hindi, English, …) never changes; tags stay short English delivery cues even for Hindi scripts.

DELIVERY STYLE vs NARRATION PRESET (never confuse the two)
19. The NARRATION PRESETS (natural, restrained_cinematic, expressive, dramatic) are GLOBAL directions for the whole book. A preset name is NEVER a valid `delivery_style` value.
20. `delivery_style` must be exactly one of: {", ".join(DELIVERY_STYLES)}. Never invent a new delivery_style — a value outside this list invalidates the whole plan.
21. Never return "cinematic", "natural" or "expressive" as `delivery_style`: those are preset names, not paragraph styles.
22. Choose `delivery_style` separately for each paragraph, based on the scene. Prefer calm or conversational for ordinary narration. Use mysterious or soft only when the scene justifies it.
23. The cinematic (restrained_cinematic) preset does NOT mean dramatic delivery: it is atmosphere through pacing, polish and restraint. Do not reach for dramatic or intense delivery just because the preset is cinematic.

ALLOWED VALUES
- emotion (exactly one): {", ".join(EMOTIONS)}
- delivery_style (exactly one, closed list — never any other value): {", ".join(DELIVERY_STYLES)}
- intensity: integer 1..5, never above the acting_intensity of the NARRATION PREFERENCES
- pause_after_ms: integer 0..3000 (silence after the paragraph)
- narration_direction: 1-2 sentences telling the voice actor how the paragraph should sound (metadata, it is never spoken; no overacting wording — see rule 9)
- voice_settings ranges:
__VOICE_RULES__

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
      "delivery_style": "<exactly one of: {", ".join(DELIVERY_STYLES)} — never a preset name such as cinematic/natural/expressive>",
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


# What each preset means, so the model never has to guess.
PRESET_RULES: dict[str, str] = {
    "natural": (
        "natural, controlled human storytelling; subtle emotion; no "
        "theatrical extremes, no shouting/trembling/breathless delivery"
    ),
    "restrained_cinematic": (
        "restrained cinematic tone: atmosphere over performance; measured, "
        "polished and still natural"
    ),
    "expressive": (
        "expressive and lively where the story earns it, but never overacted"
    ),
    "dramatic": (
        "heightened dramatic staging is welcome; strong emphasis is allowed "
        "where the scene supports it"
    ),
}


def _voice_settings_rules(capabilities: Any) -> str:
    """Capability-aware `voice_settings` rules — never advertise dead knobs."""
    lines: list[str] = []
    model = getattr(capabilities, "model_id", None) or "the selected model"
    if getattr(capabilities, "supports_speed", True):
        lines.append("    speed 0.7..1.2 — sent to the API")
    else:
        lines.append(
            f"    speed is NOT supported by {model} — keep speed 1.0. It is "
            "never sent, so never present it as an effective control; express "
            "pacing with pause_after_ms and narration_direction"
        )
    if getattr(capabilities, "supports_style", True):
        lines.append("    style 0.0..1.0 — sent to the API")
    else:
        lines.append(
            f"    style is NOT supported by {model} — keep style 0.0. It is "
            "never sent, so never present it as an effective control"
        )
    lines.append(
        "    stability 0.0..1.0, similarity_boost 0.0..1.0, "
        "use_speaker_boost true/false — these ARE sent; keep them consistent "
        "across paragraphs (one voice identity)"
    )
    if getattr(capabilities, "supports_clone_strength", False):
        lines.append(
            "    clone_strength 0.0..1.0 (only for an eligible v4 "
            "Professional Voice Clone)"
        )
    return "\n".join(lines)


def _whisper_rule(preferences: Any, paragraph_count: Optional[int]) -> str:
    """One prompt line per whisper policy (incl. the user's selected lines)."""
    policy = preferences.whisper_policy
    if policy == "never":
        return (
            "- whisper_policy: Never — no whisper or hushed delivery anywhere, "
            "not even for tense lines."
        )
    if policy == "user_selected":
        selected = sorted(set(preferences.whisper_paragraphs))
        shown = ", ".join(str(index) for index in selected) if selected else "none"
        return (
            "- whisper_policy: Only user-selected lines — whispering is allowed "
            f"ONLY in paragraph index(es) [{shown}]. Every other paragraph stays "
            "unwhispered; a scary moment that does not require it is played "
            "plainly."
        )
    budget = whisper_budget(paragraph_count) if paragraph_count else 1
    scope = (
        f"at most {budget} of the {paragraph_count} paragraphs"
        if paragraph_count
        else "very occasionally"
    )
    return (
        "- whisper_policy: AI may use rarely when essential — whisper in "
        f"{scope}, and only when plain narration would lose the moment."
    )


def _preferences_block(
    preferences: Optional[Any], paragraph_count: Optional[int] = None
) -> str:
    """Render the user's COMPLETE preferences as prompt instructions."""
    if preferences is None:
        return ""
    density = preferences.tag_density
    lines = [
        "NARRATION PREFERENCES (set by the user — they OVERRIDE your own "
        "judgement)",
        "- preset: "
        f"{PRESET_LABELS.get(preferences.preset, preferences.preset)} — "
        f"{PRESET_RULES.get(preferences.preset, '')}",
        f"- acting_intensity: {preferences.acting_intensity}/5 — NEVER set a "
        f"paragraph's intensity above {preferences.acting_intensity}; keep the "
        "performance controlled.",
        _whisper_rule(preferences, paragraph_count),
        f"- audio tag density: {TAG_DENSITY_LABELS.get(density, density)} — at "
        f"most {max_tags_per_paragraph(density)} audio tag(s) per paragraph, "
        "most paragraphs with none; only a genuine, intentional audible change "
        "earns a tag.",
    ]
    custom = (preferences.custom_direction or "").strip()
    if custom:
        lines.append(
            "- custom direction from the user (treat it as extra direction and "
            f'follow it): "{custom}"'
        )
    return "\n".join(lines)


def build_system_prompt(
    preferences: Optional[Any] = None,
    capabilities: Any = None,
    paragraph_count: Optional[int] = None,
) -> str:
    """Base natural-delivery rules + user preferences + model capabilities.

    `preferences=None` still yields the anti-overacting rules — they are not
    optional; only the preference-specific budgets are skipped.
    """
    from app.config import DEFAULT_ELEVENLABS_MODEL_ID
    from app.services.eleven_models import get_capabilities

    caps = capabilities or get_capabilities(DEFAULT_ELEVENLABS_MODEL_ID)
    parts = [
        BASE_SYSTEM_PROMPT.replace(
            "__VOICE_RULES__", _voice_settings_rules(caps)
        ).rstrip()
    ]
    preferences_block = _preferences_block(preferences, paragraph_count)
    if preferences_block:
        parts += ["", preferences_block]
    return "\n".join(parts)


# Legacy constant kept for anything importing a module-level prompt. The
# planner itself always calls `build_system_prompt(preferences, capabilities)`.
SYSTEM_PROMPT = build_system_prompt()


def build_user_prompt(
    paragraphs: Sequence[str],
    feedback: Optional[Sequence[str]] = None,
    preferences: Optional[Any] = None,
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
        "delivery_style is a per-paragraph choice from exactly this closed "
        "list (never invent a value, never use the global preset names "
        f'cinematic/natural/expressive): {", ".join(DELIVERY_STYLES)}.',
        "Prefer calm or conversational for ordinary narration; use mysterious "
        "or soft only when the scene justifies it. The selected preset is a "
        "GLOBAL direction only — it is not a delivery_style value.",
        "",
        "Script paragraphs (copy each `text` into `original_text` exactly):",
        json.dumps(numbered, ensure_ascii=False, indent=2),
    ]
    if preferences is not None:
        parts += [
            "",
            "Complete narration preferences object for this project (obey "
            "every field — this is the contract, not a suggestion):",
            json.dumps(
                preferences.model_dump(mode="json"), ensure_ascii=False, indent=2
            ),
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


# --------------------------------------------------------------------------- #
# Defensive normalization (before Pydantic validation)
# --------------------------------------------------------------------------- #
# Safe, unambiguous aliases — morphological variants of the exact enum members
# ONLY. Semantic guesses ("cinematic" -> "dramatic", "warm" -> "soft", …) are
# NEVER made: unknown values are left exactly as returned so Pydantic rejects
# them loudly and the retry feedback can name them. The global preset names
# (cinematic / natural / expressive) are deliberately absent — a preset must
# never be silently converted into a paragraph delivery style.
DELIVERY_STYLE_ALIASES: dict[str, str] = {
    "conversation": "conversational",
    "calmly": "calm",
    "softly": "soft",
    "slowly": "slow",
    "energetically": "energetic",
    "mysteriously": "mysterious",
    "intensely": "intense",
    "reflectively": "reflective",
    "dramatically": "dramatic",
}

# Same principle for emotions: adjectival / adverbial morphological variants
# of the exact enum members ONLY ("sad" -> "sadness"). Semantic guesses
# ("tense" -> "suspense", "joy" -> "happiness") are NEVER made — unknown
# values are left exactly as returned so validation rejects them loudly.
EMOTION_ALIASES: dict[str, str] = {
    "suspenseful": "suspense",
    "fearful": "fear",
    "sad": "sadness",
    "sadly": "sadness",
    "happy": "happiness",
    "happily": "happiness",
    "excited": "excitement",
    "excitedly": "excitement",
    "romantic": "romance",
    "curious": "curiosity",
    "curiously": "curiosity",
    "angry": "anger",
    "angrily": "anger",
    "confident": "confidence",
    "confidently": "confidence",
    "tender": "tenderness",
    "tenderly": "tenderness",
}


def normalize_emotion(value: str) -> tuple[str, bool]:
    """Best-effort cleanup of an `emotion` string.

    Returns (normalized_value, changed). Trims whitespace, folds casing and
    separators, then applies the safe alias table. Anything still outside the
    enum is returned UNCHANGED so validation rejects it — unknown emotions
    are never silently converted into arbitrary allowed values.
    """
    cleaned = " ".join(value.replace("_", " ").replace("-", " ").split()).lower()
    if cleaned in EMOTIONS:
        return cleaned, cleaned != value
    if cleaned in EMOTION_ALIASES:
        return EMOTION_ALIASES[cleaned], True
    return value, False


def normalize_delivery_style(value: str) -> tuple[str, bool]:
    """Best-effort cleanup of a `delivery_style` string.

    Returns (normalized_value, changed). Trims whitespace, folds casing and
    separators, then applies the safe alias table. Anything still outside the
    enum is returned UNCHANGED so validation rejects it — unknown styles are
    never silently converted into arbitrary allowed values.
    """
    cleaned = " ".join(value.replace("_", " ").replace("-", " ").split()).lower()
    if cleaned in DELIVERY_STYLES:
        return cleaned, cleaned != value
    if cleaned in DELIVERY_STYLE_ALIASES:
        return DELIVERY_STYLE_ALIASES[cleaned], True
    return value, False


def normalize_plan_payload(data: Any) -> tuple[Any, list[str]]:
    """Defensive pre-Pydantic cleanup of a decoded MiMo plan payload.

    Normalizes `paragraphs[].delivery_style` and `paragraphs[].emotion`
    (trim / casing / safe aliases) and records a note for every change,
    preserving the original raw value in the note for debugging. Everything
    else — including `preferences`, which the caller treats separately and
    never re-reads from the payload — is left untouched. Returns (data, notes).
    """
    notes: list[str] = []
    if not isinstance(data, dict):
        return data, notes
    paragraphs = data.get("paragraphs")
    if not isinstance(paragraphs, list):
        return data, notes

    for position, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, dict):
            continue
        index = paragraph.get("paragraph_index", position)
        for field, normalizer in (
            ("delivery_style", normalize_delivery_style),
            ("emotion", normalize_emotion),
        ):
            raw = paragraph.get(field)
            if not isinstance(raw, str):
                continue
            normalized, changed = normalizer(raw)
            if not changed:
                continue  # unknown values stay exactly as returned -> rejected
            notes.append(
                f"paragraph {index}: {field} {raw!r} normalized to "
                f"{normalized!r}"
            )
            paragraph[field] = normalized
    return data, notes


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
    """Turn Pydantic errors into targeted feedback for the next MiMo retry.

    Enum (`literal_error`) failures get the full treatment: exact paragraph
    index, exact field, the ACTUAL invalid value, the complete accepted list
    and an instruction to regenerate only that field. Everything else keeps
    the location + message, with the offending value attached (truncated).
    """
    problems: list[str] = []
    for err in exc.errors()[:8]:
        loc = tuple(err.get("loc") or ())
        location = ".".join(str(part) for part in loc) or "<root>"
        field = str(loc[-1]) if loc else "<root>"

        actual = err.get("input", "<missing>")
        actual_repr = repr(actual)
        if len(actual_repr) > 120:
            actual_repr = actual_repr[:117] + "..."

        if err.get("type") == "literal_error":
            expected = (err.get("ctx") or {}).get("expected") or "the documented enum"
            paragraph = loc[1] if len(loc) >= 3 and str(loc[0]) == "paragraphs" else None
            where = f" (paragraph {paragraph})" if paragraph is not None else ""
            problems.append(
                f"{location}: invalid value {actual_repr}{where} — "
                f"field '{field}'. Allowed values: {expected}. "
                f"Regenerate ONLY '{field}'{where} and keep every other plan "
                "value unchanged."
            )
        else:
            problems.append(f"{location}: {err['msg']} (actual value: {actual_repr})")
    return problems


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
class NarrationPlanner:
    """Drives MiMo, validates its plan and persists it for the project.

    Preferences flow: request body > stored `narration_preferences.json` >
    None (legacy plan, no policy enforcement). Whatever is resolved here is
    persisted, forwarded to MiMo, embedded in the saved plan and enforced by
    `NarrationPolicyValidator` — never a frontend-only setting.
    """

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
    def analyze(
        self,
        project_id: str,
        preferences: Optional[NarrationPreferences] = None,
    ) -> NarrationPlanResponse:
        """Read the stored script and build a validated narration plan."""
        self.projects.get_project(project_id)  # 404 for unknown project
        script = self.projects.read_script(project_id)  # 404 / 400
        paragraphs = split_paragraphs(script)
        if not paragraphs:
            raise EmptyScriptError(project_id)

        resolved = self._resolve_preferences(
            project_id, preferences, len(paragraphs)
        )
        plan = self.build_plan(
            paragraphs, preferences=resolved, project_id=project_id
        )
        plan.preferences = resolved  # authoritative — never MiMo's echo
        saved_at = self.projects.save_narration_plan(project_id, plan)
        return NarrationPlanResponse.from_plan(
            project_id, plan, saved=True, saved_at=saved_at, settings=self.settings
        )

    def get_plan(self, project_id: str) -> NarrationPlanResponse:
        """Return the saved narration plan (404 when analysis never ran)."""
        self.projects.get_project(project_id)
        stored = self.projects.get_narration_plan(project_id)
        if stored is None:
            raise NarrationPlanNotFoundError(project_id)
        plan, saved_at = stored
        return NarrationPlanResponse.from_plan(
            project_id, plan, saved=True, saved_at=saved_at, settings=self.settings
        )

    def preview(
        self,
        project_id: str,
        paragraph_index: int,
        preferences: Optional[NarrationPreferences] = None,
    ) -> NarrationPreviewResponse:
        """Direct ONE paragraph, using its neighbours as context.

        Nothing is written to `narration_plan.json` — a preview must never
        overwrite the plan the user already has. Preferences passed in ARE
        persisted (the panel saves them before previewing/generating).
        """
        self.projects.get_project(project_id)
        script = self.projects.read_script(project_id)
        paragraphs = split_paragraphs(script)
        if not paragraphs:
            raise EmptyScriptError(project_id)
        if not 0 <= paragraph_index < len(paragraphs):
            raise ParagraphPreviewError(paragraph_index, len(paragraphs))

        resolved = self._resolve_preferences(
            project_id, preferences, len(paragraphs)
        )

        # Context: the requested line plus at most one neighbour on each side.
        start = max(0, paragraph_index - 1)
        end = min(len(paragraphs), paragraph_index + 2)
        window = paragraphs[start:end]
        index_map = {position: start + position for position in range(len(window))}

        plan = self.build_plan(
            window, preferences=resolved, index_map=index_map, project_id=project_id
        )
        plan.preferences = resolved
        target_position = paragraph_index - start
        target = next(
            (
                item
                for item in plan.paragraphs
                if item.paragraph_index == target_position
            ),
            None,
        )
        if target is None:  # unreachable: validate_against_script guarantees it
            raise NarrationPlanError(
                [f"paragraph {paragraph_index} missing from the preview"], 1
            )

        voice_model, effective = tts_projection(plan, self.settings)
        target_settings = next(
            item.settings
            for item in effective
            if item.paragraph_index == target_position
        )
        return NarrationPreviewResponse(
            project_id=project_id,
            paragraph_index=paragraph_index,
            paragraph=target.model_copy(
                update={"paragraph_index": paragraph_index}
            ),
            preferences=resolved,
            voice_model=voice_model,
            tts_voice_settings=EffectiveVoiceSettings(
                paragraph_index=paragraph_index, settings=target_settings
            ),
            context_paragraph_indices=list(range(start, end)),
            saved=False,
        )

    # ------------------------------------------------------------------ #
    # Preferences (persisted per project)
    # ------------------------------------------------------------------ #
    def get_preferences(self, project_id: str) -> "NarrationPreferencesResponse":
        """Saved preferences, or the documented defaults (saved=False)."""
        self.projects.get_project(project_id)
        stored = self.projects.get_narration_preferences(project_id)
        if stored is None:
            return NarrationPreferencesResponse(
                project_id=project_id,
                preferences=NarrationPreferences(),
                saved=False,
                saved_at=None,
            )
        preferences, saved_at = stored
        return NarrationPreferencesResponse(
            project_id=project_id,
            preferences=preferences,
            saved=True,
            saved_at=saved_at,
        )

    def save_preferences(
        self, project_id: str, preferences: NarrationPreferences
    ) -> "NarrationPreferencesResponse":
        """Persist the panel's preferences without generating a plan."""
        self.projects.get_project(project_id)
        paragraph_count = self._paragraph_count(project_id)
        if paragraph_count:
            self._check_whisper_indexes(preferences, paragraph_count)
        saved_at = self.projects.save_narration_preferences(
            project_id, preferences
        )
        return NarrationPreferencesResponse(
            project_id=project_id,
            preferences=preferences,
            saved=True,
            saved_at=saved_at,
        )

    def _paragraph_count(self, project_id: str) -> Optional[int]:
        """Number of paragraphs in the stored script (None when no script)."""
        try:
            script = self.projects.read_script(project_id)
        except ProjectError:
            # No script yet (404) or an empty one (400) — nothing to validate
            # against; the indexes are checked again when a plan is built.
            return None
        count = len(split_paragraphs(script))
        return count or None

    def _check_whisper_indexes(
        self, preferences: NarrationPreferences, paragraph_count: int
    ) -> None:
        out_of_range = [
            index
            for index in preferences.whisper_paragraphs
            if index >= paragraph_count
        ]
        if out_of_range:
            raise InvalidNarrationPreferencesError(
                f"whisper_paragraphs {out_of_range} do not exist in this script "
                f"(valid paragraph indexes: 0..{paragraph_count - 1})"
            )

    def _resolve_preferences(
        self,
        project_id: str,
        provided: Optional[NarrationPreferences],
        paragraph_count: int,
    ) -> Optional[NarrationPreferences]:
        """Body > stored > None. Preferences sent by the client are persisted."""
        if provided is None:
            stored = self.projects.get_narration_preferences(project_id)
            return stored[0] if stored else None

        self._check_whisper_indexes(provided, paragraph_count)
        self.projects.save_narration_preferences(project_id, provided)
        return provided

    # ------------------------------------------------------------------ #
    # MiMo round trip + retries
    # ------------------------------------------------------------------ #
    def build_plan(
        self,
        paragraphs: Sequence[str],
        preferences: Optional[NarrationPreferences] = None,
        index_map: Optional[dict[int, int]] = None,
        project_id: Optional[str] = None,
    ) -> NarrationPlan:
        """Call MiMo and validate/retry until the plan matches the script
        AND the user's narration preferences (whenever any are set).

        Every raw MiMo response is preserved (attached to the raised error
        and written to `mimo_debug/` when a project is known) so an invalid
        value can always be traced back to the exact model output.
        """
        attempts = self.settings.mimo_max_attempts
        capabilities = get_capabilities(self.settings.elevenlabs_model_id)
        system = build_system_prompt(
            preferences, capabilities, paragraph_count=len(paragraphs)
        )
        problems: list[str] = ["no response from MiMo"]
        raw_attempts: list[str] = []
        normalization_notes: list[str] = []

        for attempt in range(1, attempts + 1):
            user = build_user_prompt(
                paragraphs,
                feedback=problems if attempt > 1 else None,
                preferences=preferences,
            )
            raw = self.mimo.complete_json(system=system, user=user)
            raw_attempts.append(raw)

            try:
                data: Any = json.loads(strip_code_fences(raw))
            except json.JSONDecodeError as exc:
                problems = [f"response was not valid JSON: {exc}"]
                continue

            if not isinstance(data, dict):
                problems = ["response must be a single JSON object"]
                continue

            # The model never dictates its own preferences — drop any echo.
            data.pop("preferences", None)
            # Defensive normalization (trim / casing / safe aliases) BEFORE
            # Pydantic — unknown values are never guessed, only rejected.
            data, notes = normalize_plan_payload(data)
            normalization_notes.extend(notes)

            try:
                plan = NarrationPlan.model_validate(data)
            except ValidationError as exc:
                problems = _format_validation_error(exc)
                continue

            plan.preferences = preferences
            problems = validate_against_script(plan, paragraphs)
            # `tts_text` presence is mandatory for NEW plans (requirement 2):
            # MiMo must always populate it, with tags only where needed.
            problems += require_tts_text(plan)

            # Policy validation runs LAST (after Pydantic + script checks) and
            # is never skipped when preferences are set: a violation is always
            # fed back to MiMo instead of silently accepted.
            if not problems and preferences is not None:
                problems = NarrationPolicyValidator(
                    preferences,
                    capabilities=capabilities,
                    index_map=index_map,
                ).validate(plan)

            if not problems:
                self._persist_mimo_debug(
                    project_id, raw_attempts, normalization_notes, problems=[]
                )
                return plan

        self._persist_mimo_debug(
            project_id, raw_attempts, normalization_notes, problems=problems
        )
        raise NarrationPlanError(
            problems,
            attempts,
            raw_responses=raw_attempts,
            normalization_notes=normalization_notes,
        )

    def _persist_mimo_debug(
        self,
        project_id: Optional[str],
        raw_attempts: Sequence[str],
        notes: Sequence[str],
        problems: Sequence[str],
    ) -> Optional[str]:
        """Preserve the raw MiMo responses of one run for debugging.

        Written to `storage/projects/{id}/mimo_debug/<timestamp>_attempts.json`.
        Best-effort: a failing disk must never break plan generation.
        """
        if not project_id:
            return None
        try:
            debug_dir = self.projects.storage_dir / project_id / "mimo_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            path = debug_dir / f"{stamp}_attempts.json"
            path.write_text(
                json.dumps(
                    {
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                        # Shared with the whole generation transaction:
                        # MiMo retries and plan validation run under it too.
                        "generation_id": current_generation_id(),
                        "problems": list(problems),
                        "normalization_notes": list(notes),
                        "raw_attempts": list(raw_attempts),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return str(path)
        except OSError:
            return None
