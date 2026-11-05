"""ElevenLabs model capability registry.

Model-specific request parameters (voice `speed`/`style`, `clone_strength`,
inline audio tags, `language_code`, request stitching, character limits) may
only be sent when the selected model actually supports them. The registry
below encodes what the official docs state for each known model:

    https://elevenlabs.io/docs/models
    https://elevenlabs.io/docs/overview/capabilities/text-to-speech/eleven-v4

Rules applied here:

* `eleven_v4` / `eleven_v4_turbo` — no `style` and no `speed` voice settings,
  `clone_strength` only for Professional Voice Clones fine-tuned for v4,
  inline audio tags supported, SSML **not** supported, request stitching and
  `language_code` supported, 10,000 character limit;
* `eleven_v3` — audio tags yes, request stitching **no** (per docs), 5,000
  character limit;
* `eleven_multilingual_v2` — `speed`/`style` yes, audio tags **no**,
  `language_code` **not supported** per the API reference;
* unknown model ids get conservative legacy behaviour (speed/style allowed so
  older integrations keep working) but never receive v4-only parameters and
  never receive audio tags in the spoken text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapabilities:
    """What a given ElevenLabs TTS model can accept."""

    model_id: str
    max_chars: int = 10000
    supports_speed: bool = True
    supports_style: bool = True
    supports_use_speaker_boost: bool = True
    # v4 family only — and only for PVCs fine-tuned for v4 (extra config gate).
    supports_clone_strength: bool = False
    # Inline [audio tags] in the spoken text (v3/v4 families).
    supports_audio_tags: bool = False
    # previous/next text + request ids ("Request Stitching" feature).
    supports_request_stitching: bool = True
    # language_code body parameter (never for multilingual_v2).
    supports_language_code: bool = False


# Verified against the official docs (see module docstring).
_KNOWN_MODELS: dict[str, ModelCapabilities] = {
    "eleven_v4": ModelCapabilities(
        model_id="eleven_v4",
        max_chars=10000,
        supports_speed=False,          # "Style and Speed sliders are not available in Eleven v4"
        supports_style=False,
        supports_clone_strength=True,  # gated further by ELEVENLABS_VOICE_IS_PROFESSIONAL_CLONE
        supports_audio_tags=True,
        supports_request_stitching=True,
        supports_language_code=True,
    ),
    "eleven_v4_turbo": ModelCapabilities(
        model_id="eleven_v4_turbo",
        max_chars=10000,
        supports_speed=False,
        supports_style=False,
        supports_clone_strength=True,
        supports_audio_tags=True,
        supports_request_stitching=True,
        supports_language_code=True,
    ),
    "eleven_v3": ModelCapabilities(
        model_id="eleven_v3",
        max_chars=5000,
        supports_audio_tags=True,
        supports_request_stitching=False,  # "Request Stitching is not available for eleven_v3"
        supports_language_code=True,
    ),
    "eleven_v3_conversational": ModelCapabilities(
        model_id="eleven_v3_conversational",
        max_chars=5000,
        supports_audio_tags=True,
        supports_request_stitching=False,
        supports_language_code=True,
    ),
    "eleven_multilingual_v2": ModelCapabilities(
        model_id="eleven_multilingual_v2",
        max_chars=10000,
        supports_audio_tags=False,
        supports_request_stitching=True,
        supports_language_code=False,  # "This parameter is not supported for multilingual_v2"
    ),
    "eleven_flash_v2_5": ModelCapabilities(
        model_id="eleven_flash_v2_5",
        max_chars=40000,
        supports_audio_tags=False,
        supports_request_stitching=True,
        supports_language_code=True,
    ),
    "eleven_flash_v2": ModelCapabilities(
        model_id="eleven_flash_v2",
        max_chars=30000,
        supports_audio_tags=False,
        supports_request_stitching=True,
        supports_language_code=True,
    ),
}

# Model ids ElevenLabs has deprecated but that may still be configured.
_DEPRECATED_ALIASES: dict[str, str] = {
    "eleven_turbo_v2_5": "eleven_flash_v2_5",
    "eleven_turbo_v2": "eleven_flash_v2",
}

# Legacy behaviour for ids we do not know: keep the pre-upgrade contract
# (speed/style forwarded) but never send v4-only extras.
_LEGACY = ModelCapabilities(
    model_id="unknown",
    supports_speed=True,
    supports_style=True,
    supports_clone_strength=False,
    supports_audio_tags=False,
    supports_request_stitching=True,
    supports_language_code=False,
)


def get_capabilities(model_id: str) -> ModelCapabilities:
    """Return the capabilities for `model_id` (conservative for unknown ids)."""
    canonical = (model_id or "").strip()
    canonical = _DEPRECATED_ALIASES.get(canonical, canonical)
    caps = _KNOWN_MODELS.get(canonical)
    if caps is None:
        return ModelCapabilities(
            model_id=model_id,
            max_chars=_LEGACY.max_chars,
            supports_speed=_LEGACY.supports_speed,
            supports_style=_LEGACY.supports_style,
            supports_use_speaker_boost=_LEGACY.supports_use_speaker_boost,
            supports_clone_strength=_LEGACY.supports_clone_strength,
            supports_audio_tags=_LEGACY.supports_audio_tags,
            supports_request_stitching=_LEGACY.supports_request_stitching,
            supports_language_code=_LEGACY.supports_language_code,
        )
    return caps


def known_model_ids() -> tuple[str, ...]:
    """Model ids with an explicit capability entry (useful for docs/tests)."""
    return tuple(_KNOWN_MODELS)
