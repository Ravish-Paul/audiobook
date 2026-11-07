"""Pydantic models for the MiMo narration plan.

Every value MiMo returns is validated here — the LLM is never trusted
blindly. Ranges match the task specification:

    stability / similarity_boost / style : 0.0 .. 1.0
    speed                                 : 0.7 .. 1.2
    clone_strength                        : 0.0 .. 1.0 (optional, v4 PVC only)
    intensity                             : 1 .. 5
    pause_after_ms                        : 0 .. 3000

`original_text` is the uploaded paragraph, byte-for-byte. `tts_text` is the
same spoken words plus optional Eleven v4 inline audio tags (`[whispers]`,
`[low, ominous narrator voice]`, …) — validated so the spoken words can
never change and no instruction is ever spoken aloud.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.narration_preferences import NarrationPreferences

# NOTE: `app.services.audio_tags` is imported lazily inside the validator —
# importing it at module level would pull in the `app.services` package
# (which imports this schema) and create a circular import.

# --------------------------------------------------------------------------- #
# Controlled vocabularies (section 3 of the spec)
# --------------------------------------------------------------------------- #
Emotion = Literal[
    "neutral",
    "suspense",
    "fear",
    "sadness",
    "happiness",
    "excitement",
    "romance",
    "curiosity",
    "anger",
    "confidence",
    "tenderness",
]

DeliveryStyle = Literal[
    "calm",
    "conversational",
    "dramatic",
    "slow",
    "energetic",
    "soft",
    "mysterious",
    "intense",
    "reflective",
]

EMOTIONS: tuple[str, ...] = (
    "neutral",
    "suspense",
    "fear",
    "sadness",
    "happiness",
    "excitement",
    "romance",
    "curiosity",
    "anger",
    "confidence",
    "tenderness",
)

DELIVERY_STYLES: tuple[str, ...] = (
    "calm",
    "conversational",
    "dramatic",
    "slow",
    "energetic",
    "soft",
    "mysterious",
    "intense",
    "reflective",
)


# --------------------------------------------------------------------------- #
# Paragraph level decisions
# --------------------------------------------------------------------------- #
class VoiceSettings(BaseModel):
    """Numeric settings that are forwarded verbatim to ElevenLabs.

    The ElevenLabs client filters them against the selected model's
    capabilities: `speed`/`style` are dropped for eleven_v4 (not supported),
    `clone_strength` only reaches the API for an eligible v4 Professional
    Voice Clone. Unsupported keys are never sent.
    """

    model_config = ConfigDict(extra="forbid")

    speed: float = Field(
        default=1.0, ge=0.7, le=1.2, description="Narration speed multiplier."
    )
    stability: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Voice stability."
    )
    similarity_boost: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Similarity to the base voice."
    )
    style: float = Field(default=0.0, ge=0.0, le=1.0, description="Style exaggeration.")
    use_speaker_boost: bool = Field(
        default=True, description="Enhance speaker clarity."
    )
    clone_strength: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "How closely output follows a v4 Professional Voice Clone's "
            "training recordings. Only sent for eligible v4 PVC voices."
        ),
    )


class ParagraphPlan(BaseModel):
    """How a single paragraph must be narrated."""

    model_config = ConfigDict(extra="forbid")

    paragraph_index: int = Field(ge=0, description="0-based index in the script.")
    original_text: str = Field(
        min_length=1,
        description="Exact paragraph text — must match the uploaded script.",
    )
    tts_text: Optional[str] = Field(
        default=None,
        description=(
            "original_text plus optional Eleven v4 inline audio tags "
            "([whispers], [laughs], [low, ominous narrator voice], …). "
            "Spoken words must be identical to original_text; when absent "
            "the original text is spoken unchanged."
        ),
    )
    emotion: Emotion
    delivery_style: DeliveryStyle
    intensity: int = Field(ge=1, le=5, description="Emotional intensity 1..5.")
    narration_direction: str = Field(
        min_length=1,
        max_length=1000,
        description="Short director's note (metadata, never spoken).",
    )
    voice_settings: VoiceSettings
    pause_after_ms: int = Field(
        ge=0, le=3000, description="Silence inserted after this paragraph."
    )

    @model_validator(mode="after")
    def _validate_tts_text(self) -> "ParagraphPlan":
        """Reject any tts_text that would change the spoken words.

        Runs on every load path (MiMo output, saved plans, API requests) so
        an invalid TTS instruction can never reach ElevenLabs.
        """
        from app.services.audio_tags import validate_tts_text

        problems = validate_tts_text(self.original_text, self.tts_text)
        if problems:
            raise ValueError("; ".join(problems))
        return self


class NarrationPlan(BaseModel):
    """Complete director's plan for the whole audiobook.

    `preferences` is the snapshot of the narration preferences this plan was
    built with (None for plans created before preferences existed). It is
    part of the model — and therefore part of `plan_hash` — so changing the
    preferences changes the hash and forces a fresh plan.
    """

    model_config = ConfigDict(extra="ignore")

    overall_story_tone: str = Field(min_length=1, max_length=200)
    language: str = Field(min_length=1, max_length=60)
    paragraphs: list[ParagraphPlan] = Field(min_length=1)
    preferences: Optional[NarrationPreferences] = None


# --------------------------------------------------------------------------- #
# API responses
# --------------------------------------------------------------------------- #
class VoiceModelInfo(BaseModel):
    """What the configured ElevenLabs model ACTUALLY accepts.

    Keeps descriptive planning metadata separate from the real TTS request:
    for `eleven_v4` (no `speed`, no `style`) those parameters exist in
    `voice_settings` as planning notes only and are listed in
    `planning_only_parameters` — they are never sent.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(examples=["eleven_v4"])
    supports_speed: bool
    supports_style: bool
    supports_use_speaker_boost: bool
    supports_clone_strength: bool
    supports_audio_tags: bool
    planning_only_parameters: list[str] = Field(
        default_factory=list,
        description="voice_settings keys kept as metadata but NEVER sent.",
    )


class EffectiveVoiceSettings(BaseModel):
    """The exact voice_settings that reach ElevenLabs for one paragraph."""

    model_config = ConfigDict(extra="forbid")

    paragraph_index: int
    settings: dict[str, Any] = Field(
        description="Capability-filtered request settings (no unsupported keys)."
    )


def tts_projection(
    plan: NarrationPlan, settings: Any = None
) -> tuple[VoiceModelInfo, list[EffectiveVoiceSettings]]:
    """Split the plan into (model capabilities, effective per-paragraph TTS settings).

    Imported lazily: `app.services` imports this schema module.
    """
    from app.config import get_settings
    from app.services.elevenlabs_client import ElevenLabsClient

    client = ElevenLabsClient(settings or get_settings())
    caps = client.capabilities
    planning_only = [
        name
        for name, supported in (
            ("speed", caps.supports_speed),
            ("style", caps.supports_style),
            ("clone_strength", caps.supports_clone_strength),
        )
        if not supported
    ]
    voice_model = VoiceModelInfo(
        model_id=caps.model_id,
        supports_speed=caps.supports_speed,
        supports_style=caps.supports_style,
        supports_use_speaker_boost=caps.supports_use_speaker_boost,
        supports_clone_strength=caps.supports_clone_strength,
        supports_audio_tags=caps.supports_audio_tags,
        planning_only_parameters=planning_only,
    )
    effective = [
        EffectiveVoiceSettings(
            paragraph_index=paragraph.paragraph_index,
            settings=client.effective_voice_settings(
                paragraph.voice_settings.model_dump()
            ),
        )
        for paragraph in sorted(
            plan.paragraphs, key=lambda item: item.paragraph_index
        )
    ]
    return voice_model, effective


class NarrationPlanResponse(BaseModel):
    """The full narration plan plus bookkeeping fields."""

    project_id: str
    overall_story_tone: str
    language: str
    paragraph_count: int
    paragraphs: list[ParagraphPlan]
    preferences: Optional[NarrationPreferences] = Field(
        default=None,
        description="Preferences this plan was built with (None = not set).",
    )
    voice_model: VoiceModelInfo = Field(
        description="Capabilities of the configured TTS model (what is sent)."
    )
    tts_voice_settings: list[EffectiveVoiceSettings] = Field(
        description="Capability-filtered settings actually sent per paragraph."
    )
    saved: bool = Field(description="True when narration_plan.json is on disk.")
    saved_at: Optional[datetime] = None

    @classmethod
    def from_plan(
        cls,
        project_id: str,
        plan: NarrationPlan,
        *,
        saved: bool,
        saved_at: Optional[datetime] = None,
        settings: Any = None,
    ) -> "NarrationPlanResponse":
        voice_model, tts_voice_settings = tts_projection(plan, settings)
        return cls(
            project_id=project_id,
            overall_story_tone=plan.overall_story_tone,
            language=plan.language,
            paragraph_count=len(plan.paragraphs),
            paragraphs=plan.paragraphs,
            preferences=plan.preferences,
            voice_model=voice_model,
            tts_voice_settings=tts_voice_settings,
            saved=saved,
            saved_at=saved_at,
        )


class NarrationPreviewResponse(BaseModel):
    """`POST /projects/{project_id}/analyze-narration/preview` — one paragraph."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    paragraph_index: int = Field(description="Script index of the previewed line.")
    paragraph: ParagraphPlan = Field(description="How THIS paragraph would sound.")
    preferences: Optional[NarrationPreferences] = None
    voice_model: VoiceModelInfo
    tts_voice_settings: EffectiveVoiceSettings = Field(
        description="Capability-filtered settings that would be sent for it."
    )
    context_paragraph_indices: list[int] = Field(
        description=(
            "Neighbouring paragraphs sent as context (the preview itself "
            "never overwrites narration_plan.json)."
        ),
    )
    saved: bool = Field(
        default=False,
        description="Always false — a preview never overwrites the saved plan.",
    )
