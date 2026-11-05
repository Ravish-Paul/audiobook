"""ElevenLabs integration endpoints (reference data for configuration).

The audio effects catalog lets you discover the valid IDs for the optional
`ELEVENLABS_AUDIO_EFFECTS` configuration instead of guessing them:

    GET /elevenlabs/audio-effects/catalog

Background music and sound effects for the audiobook itself are mixed in
the separate FFmpeg pipeline — the ElevenLabs `audio_effects` body only
colours the narrator's voice.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.schemas.elevenlabs import AudioEffectsCatalog
from app.services.elevenlabs_client import ElevenLabsClient

router = APIRouter(prefix="/elevenlabs", tags=["ElevenLabs"])


@router.get(
    "/audio-effects/catalog",
    response_model=AudioEffectsCatalog,
    summary="List valid ElevenLabs audio effects IDs",
    response_description="Filter presets, reverb environments and noise beds",
)
async def get_audio_effects_catalog() -> AudioEffectsCatalog:
    """Proxy for `GET /v1/audio-effects/catalog` (uses ELEVENLABS_API_KEY).

    Use the returned `*_id` values in `ELEVENLABS_AUDIO_EFFECTS`:

        ELEVENLABS_AUDIO_EFFECTS={"filter_preset_id":"…","environment_id":"…"}
    """
    client = ElevenLabsClient(get_settings())
    data = await client.fetch_audio_effects_catalog()
    return AudioEffectsCatalog.model_validate(data)
