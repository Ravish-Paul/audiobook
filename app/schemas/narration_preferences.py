"""User-controlled narration preferences (the anti-overacting contract).

These preferences are NOT a frontend-only setting: they are sent with
`POST /projects/{id}/analyze-narration`, persisted per project, embedded in
`narration_plan.json`, forwarded verbatim to MiMo, enforced afterwards by
`NarrationPolicyValidator` and included in the narration plan hash (so
changing them invalidates the previous plan's segments).

Defaults match the Narration Preferences panel:

    preset             natural
    acting_intensity   2        (1..5)
    whisper_policy     user_selected   ("only user-selected lines")
    tag_density        minimal
    custom_direction   ""  (free text, folded into the MiMo prompt)
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

NarrationPreset = Literal[
    "natural", "restrained_cinematic", "expressive", "dramatic"
]
WhisperPolicy = Literal["never", "user_selected", "rare"]
TagDensity = Literal["minimal", "standard", "expressive"]

NARRATION_PRESETS: tuple[str, ...] = (
    "natural",
    "restrained_cinematic",
    "expressive",
    "dramatic",
)
WHISPER_POLICIES: tuple[str, ...] = ("never", "user_selected", "rare")
TAG_DENSITIES: tuple[str, ...] = ("minimal", "standard", "expressive")

# Human labels used by the UI and by the MiMo prompt.
PRESET_LABELS: dict[str, str] = {
    "natural": "Natural",
    "restrained_cinematic": "Restrained Cinematic",
    "expressive": "Expressive",
    "dramatic": "Dramatic",
}
WHISPER_POLICY_LABELS: dict[str, str] = {
    "never": "Never",
    "user_selected": "Only user-selected lines",
    "rare": "AI may use rarely when essential",
}
TAG_DENSITY_LABELS: dict[str, str] = {
    "minimal": "Minimal",
    "standard": "Standard",
    "expressive": "Expressive",
}


class NarrationPreferences(BaseModel):
    """The complete preference object sent by the frontend."""

    model_config = ConfigDict(extra="forbid")

    preset: NarrationPreset = Field(
        default="natural",
        description=(
            "Overall direction style. `natural` and `restrained_cinematic` "
            "forbid shouting/trembling/breathless instructions almost "
            "entirely; `expressive`/`dramatic` allow more."
        ),
    )
    acting_intensity: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum intensity allowed for ANY paragraph (1..5).",
    )
    whisper_policy: WhisperPolicy = Field(
        default="user_selected",
        description=(
            "never = no whispering at all; user_selected = only the lines "
            "the user picked; rare = MiMo may whisper very occasionally."
        ),
    )
    whisper_paragraphs: list[int] = Field(
        default_factory=list,
        description=(
            "0-based paragraph indexes the user explicitly allows to be "
            "whispered (implements 'only user-selected lines')."
        ),
    )
    tag_density: TagDensity = Field(
        default="minimal",
        description=(
            "Maximum inline audio tags per paragraph: minimal=1, "
            "standard=3, expressive=8."
        ),
    )
    custom_direction: str = Field(
        default="",
        max_length=1000,
        description="Free-text direction added to the MiMo prompt verbatim.",
    )

    @model_validator(mode="after")
    def _clean_whisper_paragraphs(self) -> "NarrationPreferences":
        """Indexes are unique, non-negative and ordered (stable hashing)."""
        cleaned = sorted({int(index) for index in self.whisper_paragraphs})
        if any(index < 0 for index in cleaned):
            raise ValueError("whisper_paragraphs must not contain negative indexes")
        if cleaned != list(self.whisper_paragraphs):
            self.whisper_paragraphs = cleaned
        return self


class NarrationPreferencesResponse(BaseModel):
    """GET/PUT /projects/{project_id}/narration-preferences."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    preferences: NarrationPreferences
    saved: bool = Field(
        description=(
            "True when a preferences object was stored for this project; "
            "False when these are simply the defaults."
        )
    )
    saved_at: Optional[datetime] = None


class AnalyzeNarrationRequest(BaseModel):
    """Optional body of `POST /projects/{project_id}/analyze-narration`."""

    model_config = ConfigDict(extra="forbid")

    preferences: Optional[NarrationPreferences] = Field(
        default=None,
        description=(
            "Complete preferences object. When present it is persisted for "
            "the project and enforced on the generated plan."
        ),
    )


class PreviewNarrationRequest(BaseModel):
    """Body of `POST /projects/{project_id}/analyze-narration/preview`."""

    model_config = ConfigDict(extra="forbid")

    paragraph_index: int = Field(
        ge=0, description="0-based paragraph of the uploaded script to preview."
    )
    preferences: Optional[NarrationPreferences] = Field(
        default=None,
        description="Optional preferences to apply (and persist) for the preview.",
    )
