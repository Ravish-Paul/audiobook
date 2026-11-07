"""Pydantic models for AI character casting and voice profiles.

Design rules from the spec:

* attributes the script does NOT establish (gender / age / accent) are
  `None` and/or listed in `uncertain_attributes` — never invented;
* a character without spoken dialogue gets NO voice requirement
  (`has_spoken_dialogue: false`);
* the narrator is a separate profile with the reserved id `narrator`;
* a `VoiceLock` is the permanent, authoritative voice assignment
  (§6 of the spec) — generation reads ONLY from locks;
* `VoiceCandidate` carries everything the casting UI needs: name, id,
  metadata, preview audio URL, score breakdown and restriction notes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

NARRATOR_ID = "narrator"

Role = Literal[
    "narrator",
    "protagonist",
    "antagonist",
    "supporting_character",
    "minor_character",
    "ensemble",
]

SpeakerType = Literal["narrator", "character"]

# Attributes that MUST stay null / be marked uncertain when unstated.
UNCERTAIN_ATTRIBUTES: tuple[str, ...] = ("gender", "age", "accent")

_SLUG_RE = r"^[a-z0-9_]+$"


class CharacterProfile(BaseModel):
    """Permanent profile for one speaker (spec §2 example shape)."""

    model_config = ConfigDict(extra="forbid")

    character_id: str = Field(
        pattern=_SLUG_RE,
        description="Stable slug, e.g. old_receptionist. Narrator id: 'narrator'.",
    )
    character_name: str = Field(min_length=1, max_length=120)
    role: Role = "supporting_character"
    description: str = Field(
        default="",
        max_length=1000,
        description="What the script establishes about the character.",
    )
    relationships: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Relationships stated by the script (e.g. 'Samir's father').",
    )
    language: str = Field(
        default="",
        description="ISO 639-1 code (e.g. 'hi') when identifiable.",
    )
    gender: Optional[str] = Field(
        default=None,
        description="Only when the script establishes it; else null + uncertain.",
    )
    age: Optional[str] = Field(
        default=None,
        description="Only when the script establishes it (e.g. 'mature').",
    )
    accent: Optional[str] = Field(
        default=None,
        description="Only when the script establishes it.",
    )
    uncertain_attributes: list[str] = Field(
        default_factory=list,
        description="gender/age/accent values that are inferred, not established.",
    )
    voice_description: str = Field(
        default="",
        max_length=600,
        description="Required vocal personality for this speaker.",
    )
    speaking_style: str = Field(
        default="",
        max_length=400,
        description="How this speaker talks (natural, restrained, theatrical…).",
    )
    has_spoken_dialogue: bool = Field(
        default=True,
        description="False -> no speaking voice is required (unless user asks).",
    )
    # Set when a voice is permanently selected (kept in sync with VoiceLock).
    voice_id: Optional[str] = None
    voice_locked: bool = False

    @field_validator("uncertain_attributes")
    @classmethod
    def _known_uncertain_only(cls, value: list[str]) -> list[str]:
        for item in value:
            if item not in UNCERTAIN_ATTRIBUTES:
                raise ValueError(
                    f"unknown uncertain attribute '{item}' — allowed: "
                    + ", ".join(UNCERTAIN_ATTRIBUTES)
                )
        return value


class StoryAnalysis(BaseModel):
    """What MiMo extracts from the whole script (spec §1)."""

    model_config = ConfigDict(extra="forbid")

    overall_tone: str = Field(min_length=1, max_length=200)
    genre: str = Field(default="", max_length=120)
    language: str = Field(
        min_length=1,
        max_length=40,
        description="Language name or ISO 639-1 code, e.g. 'Hindi' or 'hi'.",
    )
    locale: Optional[str] = Field(
        default=None,
        description="Regional locale when identifiable (e.g. 'hi-IN').",
    )
    narrator: CharacterProfile
    characters: list[CharacterProfile] = Field(default_factory=list)

    @field_validator("narrator")
    @classmethod
    def _narrator_is_narrator(cls, value: CharacterProfile) -> CharacterProfile:
        if value.character_id != NARRATOR_ID:
            raise ValueError(f"narrator.character_id must be '{NARRATOR_ID}'")
        if value.role != "narrator":
            value.role = "narrator"
        value.has_spoken_dialogue = True
        return value

    @field_validator("characters")
    @classmethod
    def _unique_ids(cls, characters: list[CharacterProfile]) -> list[CharacterProfile]:
        seen: set[str] = set()
        for character in characters:
            if character.character_id == NARRATOR_ID:
                raise ValueError(
                    "'narrator' is reserved for the narrator profile"
                )
            if character.character_id in seen:
                raise ValueError(f"duplicate character_id '{character.character_id}'")
            seen.add(character.character_id)
        return characters

    def speakers(self) -> list[CharacterProfile]:
        """Profiles that actually speak: narrator + characters with dialogue."""
        return [self.narrator] + [
            c for c in self.characters if c.has_spoken_dialogue
        ]


# --------------------------------------------------------------------------- #
# Voice discovery / matching (spec §3, §4)
# --------------------------------------------------------------------------- #
class VoiceCandidate(BaseModel):
    """One ranked ElevenLabs voice option for a speaker."""

    model_config = ConfigDict(extra="forbid")

    voice_id: str
    name: str = ""
    origin: Literal["account", "shared"] = "account"
    public_owner_id: Optional[str] = None
    language: Optional[str] = None
    locale: Optional[str] = None
    accent: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    category: Optional[str] = None
    use_case: Optional[str] = None
    description: Optional[str] = None
    preview_url: Optional[str] = None
    labels: dict[str, str] = Field(default_factory=dict)
    restricted: bool = False
    custom_rate: bool = False
    notice_period_days: Optional[int] = None
    notes: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(
        default_factory=dict, description="Per-criterion score breakdown."
    )
    total_score: float = Field(ge=0.0, le=1.0, default=0.0)
    rank: int = Field(ge=1, default=1)


class VoiceLock(BaseModel):
    """Permanent voice assignment for one speaker (spec §6)."""

    model_config = ConfigDict(extra="forbid")

    character_id: str
    voice_id: str
    original_voice_id: Optional[str] = None
    voice_name: str = ""
    provider: str = "elevenlabs"
    language: Optional[str] = None
    accent: Optional[str] = None
    voice_metadata: dict = Field(default_factory=dict)
    model_id: str = ""
    supported_voice_settings: list[str] = Field(default_factory=list)
    preview_url: Optional[str] = None
    origin: Literal["account", "shared", "manual"] = "account"
    selected_at: datetime
    locked: bool = True


class SpeakerCasting(BaseModel):
    """Candidates + selection state for one speaker."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[VoiceCandidate] = Field(default_factory=list)
    selected: Optional[VoiceLock] = None
    status: Literal["pending", "selected", "approved"] = "pending"
    insufficient_candidates: bool = False


class CharacterCast(BaseModel):
    """Root document stored as `character_cast.json` (spec §10)."""

    model_config = ConfigDict(extra="ignore")  # tolerate older/newer files

    project_id: str = ""
    story: StoryAnalysis
    casting: dict[str, SpeakerCasting] = Field(default_factory=dict)
    approved: bool = False
    approved_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def required_speakers(self) -> list[CharacterProfile]:
        return self.story.speakers()

    def casting_for(self, speaker_id: str) -> SpeakerCasting:
        entry = self.casting.get(speaker_id)
        if entry is None:
            entry = SpeakerCasting()
            self.casting[speaker_id] = entry
        return entry


# --------------------------------------------------------------------------- #
# API responses (spec §5)
# --------------------------------------------------------------------------- #
class CharacterAnalysisResponse(BaseModel):
    project_id: str
    story: StoryAnalysis
    speaker_ids: list[str]
    saved: bool
    saved_at: Optional[datetime] = None


class VoiceCastingResponse(BaseModel):
    project_id: str
    approved: bool
    approved_at: Optional[datetime] = None
    narrator: SpeakerCasting
    characters: dict[str, SpeakerCasting]
    missing_speakers: list[str] = Field(default_factory=list)
    saved_at: Optional[datetime] = None
    story: Optional[StoryAnalysis] = Field(
        default=None,
        description=(
            "The stored story analysis (character identities/descriptions) — "
            "additive restore field so a reloaded browser can rebuild the "
            "casting UI without re-running character analysis."
        ),
    )


class SelectVoiceRequest(BaseModel):
    """Manual voice choice — from the candidate list or any usable voice."""

    model_config = ConfigDict(extra="forbid")

    speaker_id: str = Field(min_length=1, description="'narrator' or a character_id")
    voice_id: str = Field(min_length=1)
    public_owner_id: Optional[str] = Field(
        default=None,
        description="Required to add a shared library voice to the account.",
    )
    new_name: Optional[str] = Field(
        default=None,
        description="Display name when adding a shared voice to the account.",
    )


class AuditionRequest(BaseModel):
    """Short, user-approved test line — NEVER the full story (spec §5)."""

    model_config = ConfigDict(extra="forbid")

    speaker_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=400)
    voice_id: Optional[str] = Field(
        default=None,
        description="Defaults to the speaker's currently selected voice.",
    )


MAX_AUDITION_CHARS = 400
