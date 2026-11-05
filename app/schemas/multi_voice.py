"""Pydantic models for the utterance-level multi-voice narration plan (§7).

Each paragraph is split into ordered utterances; every utterance belongs to
exactly one speaker (the narrator or a character from `character_cast.json`).

Invariants enforced here / in the planner:

* `tts_text` may only insert Eleven v4 audio tags — stripping them must
  reproduce `original_text` byte-for-byte (same rule as the paragraph plan);
* per-paragraph, the utterances' words must reconstruct the original
  paragraph exactly (order preserved) — checked by the planner;
* `speaker_type` must agree with the reserved narrator id.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.casting import NARRATOR_ID
from app.schemas.narration import DeliveryStyle, Emotion

# NOTE: `app.services.audio_tags` is imported lazily inside the validator —
# importing it at module level would pull in the `app.services` package
# (which imports this schema) and create a circular import.

UtteranceSpeakerType = Literal["narrator", "character"]


class Utterance(BaseModel):
    """One continuous stretch of speech by a single speaker (spec §7)."""

    model_config = ConfigDict(extra="forbid")

    utterance_id: str = Field(
        min_length=1,
        description="Stable id, e.g. p0002-u0001 (paragraph 2, utterance 1).",
    )
    paragraph_index: int = Field(ge=0)
    sequence_index: int = Field(ge=0, description="0-based order inside the paragraph.")
    speaker_id: str = Field(
        min_length=1,
        description="'narrator' or a character_id from character_cast.json.",
    )
    speaker_type: UtteranceSpeakerType
    original_text: str = Field(min_length=1, description="Exact words from the script.")
    tts_text: Optional[str] = Field(
        default=None,
        description="original_text plus optional Eleven v4 audio tags.",
    )
    emotion: Emotion = Field()  # type: ignore[valid-type]
    acting_intensity: int = Field(
        ge=1, le=5, description="Emotional intensity 1..5 (restrained by default)."
    )
    delivery_style: DeliveryStyle = Field()  # type: ignore[valid-type]
    narration_direction: str = Field(
        min_length=1,
        max_length=1000,
        description="Director's note — metadata, never spoken.",
    )
    pause_after_ms: int = Field(ge=0, le=3000)
    voice_id: Optional[str] = Field(
        default=None,
        description="Filled from the locked cast (authoritative at generation).",
    )
    model_id: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "Utterance":
        from app.services.audio_tags import validate_tts_text

        if self.speaker_type == "narrator" and self.speaker_id != NARRATOR_ID:
            raise ValueError(
                f"speaker_type 'narrator' requires speaker_id '{NARRATOR_ID}' "
                f"(got '{self.speaker_id}')"
            )
        if self.speaker_type == "character" and self.speaker_id == NARRATOR_ID:
            raise ValueError(
                "speaker_type 'character' cannot use the reserved narrator id"
            )
        problems = validate_tts_text(self.original_text, self.tts_text)
        if problems:
            raise ValueError("; ".join(problems))
        return self


class MultiVoicePlan(BaseModel):
    """Complete ordered utterance plan for the whole audiobook."""

    model_config = ConfigDict(extra="ignore")

    language: str = Field(min_length=1, max_length=60)
    overall_tone: str = Field(default="", max_length=200)
    utterances: list[Utterance] = Field(min_length=1)


class MultiVoicePlanResponse(BaseModel):
    project_id: str
    language: str
    overall_tone: str
    utterance_count: int
    paragraph_count: int
    speakers: list[str]
    utterances: list[Utterance]
    saved: bool
    saved_at: Optional[datetime] = None

    @classmethod
    def from_plan(
        cls,
        project_id: str,
        plan: MultiVoicePlan,
        *,
        saved: bool,
        saved_at: Optional[datetime] = None,
    ) -> "MultiVoicePlanResponse":
        paragraphs = {u.paragraph_index for u in plan.utterances}
        speakers: list[str] = []
        for utterance in plan.utterances:
            if utterance.speaker_id not in speakers:
                speakers.append(utterance.speaker_id)
        return cls(
            project_id=project_id,
            language=plan.language,
            overall_tone=plan.overall_tone,
            utterance_count=len(plan.utterances),
            paragraph_count=len(paragraphs),
            speakers=speakers,
            utterances=plan.utterances,
            saved=saved,
            saved_at=saved_at,
        )
