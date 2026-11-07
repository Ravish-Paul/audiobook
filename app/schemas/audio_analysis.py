"""Pydantic models for the pyannoteAI audio-analysis endpoints.

The normalized timeline is the contract used later by the Automatic Sound
Design Director: every word/turn keeps BOTH the raw provider timestamps
(fractional seconds, untouched) and the normalized values (milliseconds +
sample indexes computed with Decimal ROUND_HALF_UP at the file's REAL
sample rate).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Normalized timeline records
# --------------------------------------------------------------------------- #
class NormalizedWord(BaseModel):
    """One transcribed word on the internal audio timeline."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Word exactly as returned by pyannoteAI.")
    speaker_id: str = Field(description="Attributed speaker (e.g. SPEAKER_00).")
    raw_start_seconds: float = Field(
        description="Provider start time in fractional seconds (unmodified)."
    )
    raw_end_seconds: float = Field(
        description="Provider end time in fractional seconds (unmodified)."
    )
    start_ms: int = Field(description="start, Decimal ROUND_HALF_UP to milliseconds.")
    end_ms: int = Field(description="end, Decimal ROUND_HALF_UP to milliseconds.")
    start_sample: int = Field(description="start_ms * sample_rate / 1000 (HALF_UP).")
    end_sample: int = Field(description="end_ms * sample_rate / 1000 (HALF_UP).")


class NormalizedSpeakerTurn(BaseModel):
    """One speaker turn (diarization segment) on the internal timeline."""

    model_config = ConfigDict(extra="forbid")

    speaker_id: str
    raw_start_seconds: float
    raw_end_seconds: float
    start_ms: int
    end_ms: int
    start_sample: int
    end_sample: int


class NormalizedTurnTranscription(BaseModel):
    """Turn-level transcription segment (text + span), when the provider returned it."""

    model_config = ConfigDict(extra="forbid")

    text: str
    speaker_id: str
    raw_start_seconds: float
    raw_end_seconds: float
    start_ms: int
    end_ms: int
    start_sample: int
    end_sample: int


class NormalizedSpeaker(BaseModel):
    """Aggregate facts about one detected speaker (no invented confidence)."""

    model_config = ConfigDict(extra="forbid")

    speaker_id: str
    turn_count: int = 0
    word_count: int = 0
    first_start_ms: Optional[int] = None
    last_end_ms: Optional[int] = None


class ValidationIssue(BaseModel):
    """One provider record that failed validation and was excluded."""

    model_config = ConfigDict(extra="forbid")

    record_type: str = Field(description="'word', 'speaker_turn' or 'turn_transcription'.")
    index: Optional[int] = Field(
        default=None, description="Index of the record inside its provider array."
    )
    reason: str = Field(description="Machine readable rejection reason.")


class AudioAnalysisNormalized(BaseModel):
    """Normalized audio timeline — saved as audio_analysis_normalized.json."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    analysis_id: str
    status: str = Field(default="completed", examples=["completed"])
    model_id: str = Field(examples=["precision-3"])
    transcription_model: str = Field(
        examples=["faster-whisper-large-v3-turbo"]
    )
    time_unit: str = Field(default="milliseconds", description="Unit of *_ms fields.")
    sample_rate: int = Field(
        description="REAL sample rate of the source audiobook (ffprobe)."
    )
    audio_duration_ms: int = Field(description="REAL duration of the audiobook.")
    duration_tolerance_ms: int = Field(
        description="Allowed overshoot beyond audio_duration_ms during validation."
    )
    speakers: list[NormalizedSpeaker] = Field(default_factory=list)
    words: list[NormalizedWord] = Field(default_factory=list)
    speaker_turns: list[NormalizedSpeakerTurn] = Field(default_factory=list)
    turn_level_transcriptions: list[NormalizedTurnTranscription] = Field(
        default_factory=list,
        description="Present only when the provider returned turnLevelTranscription.",
    )
    validation_issues: list[ValidationIssue] = Field(
        default_factory=list,
        description="Provider records excluded by validation (raw file keeps them).",
    )
    generated_at: datetime


# --------------------------------------------------------------------------- #
# API responses
# --------------------------------------------------------------------------- #
class AudioAnalysisStartResponse(BaseModel):
    """POST /projects/{project_id}/audio-analysis."""

    success: bool = Field(default=True)
    message: str
    project_id: str
    analysis_id: str = Field(description="Local id of this analysis version.")
    status: str = Field(
        description="Local status: processing | completed | failed | canceled.",
        examples=["processing"],
    )
    job_status: str = Field(
        description="Provider job status: created | pending | running | succeeded "
        "| failed | canceled.",
        examples=["created"],
    )
    model_id: str = Field(examples=["precision-3"])
    transcription_model: str = Field(
        examples=["faster-whisper-large-v3-turbo"]
    )
    audio_hash: str = Field(description="SHA-256 of the analyzed audiobook.")
    reused: bool = Field(
        default=False,
        description=(
            "True when an identical analysis (same audio hash + models) "
            "already existed — no new pyannote credits were spent."
        ),
    )
    created_at: datetime


class AudioAnalysisStatusResponse(BaseModel):
    """GET /projects/{project_id}/audio-analysis/status."""

    success: bool = Field(default=True)
    message: str
    project_id: str
    analysis_id: str
    status: str = Field(
        examples=["processing"],
        description="Local status: processing | completed | failed | canceled.",
    )
    job_status: str = Field(examples=["running"])
    model_id: str
    transcription_model: str
    error: Optional[str] = Field(
        default=None, description="Provider or normalization error, when failed."
    )
    sample_rate: Optional[int] = None
    audio_duration_ms: Optional[int] = None
    word_count: Optional[int] = None
    speaker_count: Optional[int] = None
    poll_count: int = Field(
        default=0, description="How often the provider job has been polled."
    )
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
