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
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    """Complete director's plan for the whole audiobook."""

    model_config = ConfigDict(extra="ignore")

    overall_story_tone: str = Field(min_length=1, max_length=200)
    language: str = Field(min_length=1, max_length=60)
    paragraphs: list[ParagraphPlan] = Field(min_length=1)


# --------------------------------------------------------------------------- #
# API responses
# --------------------------------------------------------------------------- #
class NarrationPlanResponse(BaseModel):
    """The full narration plan plus bookkeeping fields."""

    project_id: str
    overall_story_tone: str
    language: str
    paragraph_count: int
    paragraphs: list[ParagraphPlan]
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
    ) -> "NarrationPlanResponse":
        return cls(
            project_id=project_id,
            overall_story_tone=plan.overall_story_tone,
            language=plan.language,
            paragraph_count=len(plan.paragraphs),
            paragraphs=plan.paragraphs,
            saved=saved,
            saved_at=saved_at,
        )
