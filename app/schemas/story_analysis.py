"""Whole-story analysis used by the AI Narration Director.

This is DIFFERENT from the casting `StoryAnalysis` (which extracts character
identities). This document answers the directing questions:

* what kind of story is this, and how does its mood progress scene by scene?
* which GLOBAL narration profile fits it, and why?
* which characters dominate the emotional tone, and how intimate is the story?

MiMo produces it from the complete script. It is persisted as
`story_analysis.json`, included in the multi-voice plan fingerprint and passed
into both narration prompts. It is guidance, never a hard per-paragraph
instruction: the source script and scene context always win.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

IntimacyLevel = Literal["none", "light", "moderate", "high"]

NarrationProfile = Literal[
    "natural",
    "cinematic",
    "horror",
    "romantic",
    "sensual",
    "emotional",
    "action",
    "mystery",
    "fantasy",
    "comedy",
    "professional",
    "children",
]


class SceneMood(BaseModel):
    """One scene/beat of the story and the mood it should carry."""

    model_config = ConfigDict(extra="forbid")

    scene_index: int = Field(ge=0, description="0-based scene order.")
    start_paragraph_index: int = Field(ge=0)
    end_paragraph_index: int = Field(ge=0)
    mood: str = Field(
        min_length=1,
        max_length=120,
        description="e.g. calm, playful, romantic, tense, reflective.",
    )
    intensity: int = Field(
        default=3, ge=1, le=5, description="Typical scene intensity 1..5."
    )
    summary: str = Field(
        default="",
        max_length=400,
        description="Short, script-grounded description of what changes here.",
    )

    @model_validator(mode="after")
    def _range(self) -> "SceneMood":
        if self.end_paragraph_index < self.start_paragraph_index:
            raise ValueError(
                "end_paragraph_index must be greater than or equal to "
                "start_paragraph_index"
            )
        return self


class CharacterTendency(BaseModel):
    """A character's stable emotional tendency (not a per-line instruction)."""

    model_config = ConfigDict(extra="forbid")

    character_name: str = Field(min_length=1, max_length=120)
    tendency: str = Field(
        min_length=1,
        max_length=300,
        description="e.g. nervous and curious; confident and caring.",
    )


class StoryDirectorAnalysis(BaseModel):
    """The complete directing analysis of the story."""

    model_config = ConfigDict(extra="forbid")

    primary_genre: str = Field(min_length=1, max_length=120)
    secondary_genre: Optional[str] = Field(default=None, max_length=120)
    emotional_tone: str = Field(min_length=1, max_length=200)
    dominant_mood: str = Field(min_length=1, max_length=120)
    language: str = Field(min_length=1, max_length=60)
    scene_progression: list[SceneMood] = Field(min_length=1)
    recommended_profile: str = Field(
        min_length=1,
        max_length=120,
        description="One of the twelve narration profile ids.",
    )
    recommended_direction: str = Field(
        min_length=1,
        max_length=1000,
        description="Why this profile fits and how the story should progress.",
    )
    character_tendencies: list[CharacterTendency] = Field(default_factory=list)
    intimacy_level: IntimacyLevel = Field(
        default="none",
        description="How intimate the story is: none | light | moderate | high.",
    )


class StoryAnalysisRequest(BaseModel):
    """Optional body of `POST /projects/{project_id}/analyze-story`."""

    model_config = ConfigDict(extra="forbid")

    force: bool = Field(
        default=False,
        description=(
            "Re-run the analysis even when a saved analysis describes the "
            "current script. Defaults to reusing the saved analysis."
        ),
    )


class StoryAnalysisResponse(BaseModel):
    """Response of the story-analysis endpoints."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    analysis: StoryDirectorAnalysis
    saved: bool = Field(description="True when story_analysis.json is on disk.")
    saved_at: Optional[datetime] = None
    reused: bool = Field(
        default=False,
        description="True when the saved analysis was returned without a new call.",
    )
