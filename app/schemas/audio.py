"""Pydantic models for the audio (text-to-speech) endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AudioResponse(BaseModel):
    """Shared shape for audio generation and audio information responses."""

    success: bool = Field(default=True, description="True when the MP3 is on disk.")
    message: str
    project_id: str
    audio_filename: str = Field(
        description="Name of the MP3 inside storage/projects/{project_id}/audio/."
    )
    audio_url: str = Field(
        description="Browser-playable URL of the generated MP3.",
        examples=["/media/projects/8f1c.../audio/audiobook.mp3"],
    )
    status: str = Field(
        description="'completed' once the MP3 exists on disk.",
        examples=["completed"],
    )
    size_bytes: Optional[int] = Field(
        default=None, description="Size of the stored MP3 file."
    )
    character_count: Optional[int] = Field(
        default=None, description="Characters of script that were synthesised."
    )
    voice_id: Optional[str] = Field(
        default=None, description="ElevenLabs voice used for this audio."
    )
    model_id: Optional[str] = Field(
        default=None, description="ElevenLabs model used for this audio."
    )
    output_format: Optional[str] = Field(
        default=None,
        description=(
            "Final audiobook container/quality requested from ElevenLabs/"
            "FFmpeg (e.g. mp3_44100_128)."
        ),
        examples=["mp3_44100_128"],
    )
    generated_at: Optional[datetime] = Field(
        default=None, description="When the audio was generated (UTC)."
    )
    generation_mode: Optional[str] = Field(
        default=None,
        description=(
            "'multi_voice' when the utterance plan with locked character "
            "voices was used, 'narration_plan' when MiMo's paragraph plan "
            "was used, 'single_pass' for the legacy whole-script generation."
        ),
        examples=["multi_voice"],
    )
    segment_count: Optional[int] = Field(
        default=None,
        description="Paragraph segments combined into the audiobook (plan mode).",
    )
