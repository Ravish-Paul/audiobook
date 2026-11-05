"""Pydantic models for ElevenLabs integration endpoints."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AudioEffectsFilterPreset(BaseModel):
    """A filter preset that colours the voice (e.g. `phone`, `old_radio`)."""

    filter_preset_id: str
    name: str
    description: Optional[str] = None


class AudioEffectsEnvironment(BaseModel):
    """A reverb environment that places the voice in a space."""

    environment_id: str
    name: str
    description: Optional[str] = None


class AudioEffectsBackgroundNoise(BaseModel):
    """A background noise bed looped under the voice."""

    background_noise_id: str
    name: str
    description: Optional[str] = None


class AudioEffectsCatalog(BaseModel):
    """Valid IDs for the optional `audio_effects` request body.

    Source: `GET https://api.elevenlabs.io/v1/audio-effects/catalog`.
    Configure effects in `.env` (`ELEVENLABS_AUDIO_EFFECTS`) using only
    IDs listed here — unknown IDs are rejected upstream with a 404.
    """

    filter_presets: list[AudioEffectsFilterPreset] = Field(default_factory=list)
    environments: list[AudioEffectsEnvironment] = Field(default_factory=list)
    background_noises: list[AudioEffectsBackgroundNoise] = Field(default_factory=list)
