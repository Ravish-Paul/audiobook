"""Async client for the official ElevenLabs Text-to-Speech REST API.

Reference: https://elevenlabs.io/docs/api-reference/text-to-speech/convert

    POST {base_url}/text-to-speech/{voice_id}?output_format=mp3_44100_128
    headers:
        xi-api-key: <API key>          <- the only place the key is ever used
        accept: audio/mpeg
        (response header `request-id` -> stored for paragraph continuity)
    body:
        {"text": "...", "model_id": "...", "voice_settings": {...},
         "language_code": "...", "apply_text_normalization": "auto",
         "pronunciation_dictionary_locators": [...], "audio_effects": {...},
         "previous_text"/"next_text" | "previous_request_ids"/"next_request_ids"}
    response:
        200 -> raw audio bytes (audio/mpeg …)

Eleven v4 specifics honoured here (see `app.services.eleven_models`):

* `style` and `speed` are NOT supported by eleven_v4 — they are filtered out
  before the request is built; `clone_strength` is only sent for a voice
  explicitly flagged as an eligible v4 Professional Voice Clone;
* inline audio tags in `tts_text` are kept for tag-capable models and
  stripped for the rest, so a `[laughs]` cue is never spoken as words;
* continuity: `previous_text`/`next_text` for ordinary first-time
  generation, `previous_request_ids`/`next_request_ids` when regenerating —
  NEVER both for the same side (IDs win, text is dropped);
* `apply_text_normalization` defaults to `auto`;
  `apply_language_text_normalization` is only ever enabled for Japanese —
  never for Hindi;
* `pronunciation_dictionary_locators` are only built from explicitly
  configured IDs (never invented), max 3;
* `audio_effects` come from configuration only and are checked against the
  official audio effects catalog (`GET /v1/audio-effects/catalog`).

Security notes:
* the API key is sent as a header only, never logged, never returned to
  clients and excluded from `repr()` (see `Settings.elevenlabs_api_key`);
* upstream error bodies are truncated and never echo credentials.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.config import (
    ELEVENLABS_MAX_REQUEST_IDS,
    ELEVENLABS_OUTPUT_FORMATS,
    Settings,
    get_settings,
)
from app.services.audio_tags import sanitize_for_model
from app.services.eleven_models import ModelCapabilities, get_capabilities
from app.services.errors import (
    ElevenLabsAPIError,
    ElevenLabsAuthError,
    ElevenLabsConfigError,
    ElevenLabsRateLimitError,
    ElevenLabsTimeoutError,
    ElevenLabsVoiceError,
)

# Balanced defaults for narration. Kept server-side so clients cannot inject
# arbitrary voice settings. Keys the selected model does not support are
# removed again in `_merge_voice_settings`.
DEFAULT_VOICE_SETTINGS: dict[str, object] = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True,
}

# Only these keys are ever forwarded to ElevenLabs. Narration metadata such
# as `emotion`, `delivery_style` or `narration_direction` is deliberately
# NOT sent — they are direction for the director, not API parameters.
SUPPORTED_VOICE_SETTING_KEYS: tuple[str, ...] = (
    "speed",
    "stability",
    "similarity_boost",
    "style",
    "use_speaker_boost",
    "clone_strength",
)

# Continuity context is capped: ElevenLabs caps request-id lists at 3 and
# very long neighbour texts add no prosodic value.
CONTEXT_TEXT_MAX_CHARS = 1000

# Request IDs age out (docs: "no older than two hours") — leave a margin.
REQUEST_ID_MAX_AGE_SECONDS = 2 * 3600 - 300

_TEXT_NORMALIZATION_MODES = ("auto", "on", "off")

# Upstream error messages are useful but must stay short and safe to show.
_MAX_ERROR_DETAIL = 300

# The audio effects catalog rarely changes — cache it per client instance.
_CATALOG_TTL_SECONDS = 600.0


def _safe_detail(response: httpx.Response) -> str:
    """Extract a short, human readable message from an ElevenLabs error body."""
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            raw = payload.get("detail") or payload.get("error") or payload
            if isinstance(raw, dict):  # {"detail": {"message": "..."}}
                raw = raw.get("message") or raw.get("status_message") or raw
            detail = str(raw)
        else:
            detail = str(payload)
    except Exception:  # non-JSON body (HTML error page, empty body, ...)
        try:
            detail = response.text[:_MAX_ERROR_DETAIL]
        except Exception:
            detail = ""

    detail = " ".join(detail.split())
    if not detail:
        return ""
    return f" — {detail[:_MAX_ERROR_DETAIL]}"


def _clip_previous(text: str) -> str:
    """Keep the tail of the preceding text (closest to the current request)."""
    text = text or ""
    return text[-CONTEXT_TEXT_MAX_CHARS:] if len(text) > CONTEXT_TEXT_MAX_CHARS else text


def _clip_next(text: str) -> str:
    """Keep the head of the following text."""
    text = text or ""
    return text[:CONTEXT_TEXT_MAX_CHARS] if len(text) > CONTEXT_TEXT_MAX_CHARS else text


@dataclass(frozen=True)
class SynthesisContext:
    """Neighbouring context for paragraph continuity (Request Stitching).

    Per side, text and request IDs are MUTUALLY EXCLUSIVE: the client sends
    IDs when available and drops the text for that side, so ElevenLabs never
    receives conflicting context (IDs make the API ignore the text anyway).
    """

    previous_text: Optional[str] = None
    next_text: Optional[str] = None
    previous_request_ids: tuple[str, ...] = ()
    next_request_ids: tuple[str, ...] = ()

    def previous_ids(self) -> list[str]:
        """Up to 3 most recent IDs, oldest first / closest request last."""
        return list(self.previous_request_ids)[-ELEVENLABS_MAX_REQUEST_IDS:]

    def next_ids(self) -> list[str]:
        return list(self.next_request_ids)[-ELEVENLABS_MAX_REQUEST_IDS:]


@dataclass(frozen=True)
class SynthesisResult:
    """Audio bytes plus bookkeeping from one TTS request."""

    audio: bytes
    request_id: Optional[str] = None
    output_format: str = ""
    model_id: str = ""
    voice_id: str = ""


class ElevenLabsClient:
    """Small, dependency-free wrapper around `POST /v1/text-to-speech/{voice}`."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.settings = settings or get_settings()
        # Test seam: tests inject `httpx.MockTransport(...)` so no real request
        # is ever made and no API credits are consumed.
        self._transport = transport
        self.last_request_id: Optional[str] = None
        self._effects_catalog: Optional[dict[str, Any]] = None
        self._effects_catalog_at: float = 0.0

    # ------------------------------------------------------------------ #
    # Configuration helpers
    # ------------------------------------------------------------------ #
    @property
    def endpoint(self) -> str:
        """Full URL of the ElevenLabs convert endpoint for the configured voice."""
        voice_id = self.settings.elevenlabs_voice_id
        return f"{self.settings.elevenlabs_base_url}/text-to-speech/{voice_id}"

    @property
    def capabilities(self) -> ModelCapabilities:
        """Verified capabilities of the configured model (never guessed)."""
        return get_capabilities(self.settings.elevenlabs_model_id)

    @property
    def catalog_endpoint(self) -> str:
        return f"{self.settings.elevenlabs_base_url}/audio-effects/catalog"

    def _require_configuration(self) -> None:
        """Fail fast with a clear message when .env is not set up yet."""
        settings = self.settings
        if not settings.elevenlabs_api_key:
            raise ElevenLabsConfigError(
                "ELEVENLABS_API_KEY is not configured. Copy .env.example to .env "
                "and add your ElevenLabs API key."
            )
        if not settings.elevenlabs_voice_id:
            raise ElevenLabsConfigError(
                "ELEVENLABS_VOICE_ID is not configured. Copy .env.example to .env "
                "and add a voice ID from https://elevenlabs.io/app/voice-library"
            )
        # Validate formats locally — never send an unsupported value upstream.
        for label, fmt in (
            ("ELEVENLABS_OUTPUT_FORMAT", settings.elevenlabs_output_format),
            (
                "ELEVENLABS_INTERMEDIATE_OUTPUT_FORMAT",
                settings.elevenlabs_intermediate_output_format,
            ),
        ):
            if fmt and fmt not in ELEVENLABS_OUTPUT_FORMATS:
                allowed = ", ".join(ELEVENLABS_OUTPUT_FORMATS)
                raise ElevenLabsConfigError(
                    f"{label}='{fmt}' is not a valid ElevenLabs output format. "
                    f"Allowed values: {allowed}"
                )
        normalization = settings.elevenlabs_apply_text_normalization
        if normalization not in _TEXT_NORMALIZATION_MODES:
            raise ElevenLabsConfigError(
                "ELEVENLABS_APPLY_TEXT_NORMALIZATION must be one of: "
                + ", ".join(_TEXT_NORMALIZATION_MODES)
            )

    # ------------------------------------------------------------------ #
    # Voice settings (capability filtered — unsupported keys never sent)
    # ------------------------------------------------------------------ #
    def _merge_voice_settings(
        self,
        voice_settings: Optional[dict[str, object]],
    ) -> dict[str, object]:
        """Apply the plan's settings, keeping only supported API fields.

        * `speed` / `style`      — dropped when the model does not support
          them (eleven_v4 and eleven_v4_turbo);
        * `clone_strength`       — only for a v4 model with a voice explicitly
          flagged as an eligible Professional Voice Clone;
        * unknown keys (`emotion`, `narration_direction`, …) — always dropped.
        """
        caps = self.capabilities
        settings = self.settings

        merged: dict[str, object] = {}
        if caps.supports_style:
            merged["style"] = DEFAULT_VOICE_SETTINGS["style"]
        if caps.supports_speed:
            merged["speed"] = 1.0
        merged["stability"] = DEFAULT_VOICE_SETTINGS["stability"]
        merged["similarity_boost"] = DEFAULT_VOICE_SETTINGS["similarity_boost"]
        if caps.supports_use_speaker_boost:
            merged["use_speaker_boost"] = DEFAULT_VOICE_SETTINGS["use_speaker_boost"]

        if voice_settings:
            for key, value in voice_settings.items():
                if value is None:
                    continue
                if key == "speed" and not caps.supports_speed:
                    continue
                if key == "style" and not caps.supports_style:
                    continue
                if key == "clone_strength" and not self._clone_strength_allowed(
                    caps, settings
                ):
                    continue
                if key in SUPPORTED_VOICE_SETTING_KEYS:
                    merged[key] = value

        # Config-level default clone strength for eligible v4 PVC voices.
        if (
            caps.supports_clone_strength
            and settings.elevenlabs_voice_is_pvc
            and "clone_strength" not in merged
            and settings.elevenlabs_clone_strength is not None
        ):
            merged["clone_strength"] = settings.elevenlabs_clone_strength

        return merged

    @staticmethod
    def _clone_strength_allowed(
        caps: ModelCapabilities, settings: Settings
    ) -> bool:
        """clone_strength: v4 model AND voice confirmed as a v4 PVC."""
        return bool(
            caps.supports_clone_strength and settings.elevenlabs_voice_is_pvc
        )

    def effective_voice_settings(
        self, voice_settings: Optional[dict[str, object]] = None
    ) -> dict[str, object]:
        """Exactly the `voice_settings` that would be sent to ElevenLabs.

        Used for metadata/verification: the plan keeps `speed`/`style` as
        planning metadata, while this shows what the effective request
        contains for the configured model (v4: stability, similarity_boost,
        use_speaker_boost — plus clone_strength only for eligible PVCs).
        """
        return self._merge_voice_settings(voice_settings)

    # ------------------------------------------------------------------ #
    # Payload assembly
    # ------------------------------------------------------------------ #
    def _build_payload(
        self,
        text: str,
        voice_settings: Optional[dict[str, object]],
        *,
        tts_text: Optional[str] = None,
        context: Optional[SynthesisContext] = None,
        language_code: Optional[str] = None,
    ) -> dict[str, Any]:
        caps = self.capabilities
        settings = self.settings

        # Spoken text: tts_text carries the inline audio tags; strip them for
        # models that do not understand tags so they are never spoken aloud.
        spoken = tts_text if (tts_text and tts_text.strip()) else text
        spoken = sanitize_for_model(
            spoken, text, supports_tags=caps.supports_audio_tags
        )

        payload: dict[str, Any] = {
            "text": spoken,
            "model_id": settings.elevenlabs_model_id,
            "voice_settings": self._merge_voice_settings(voice_settings),
        }

        # Language hint (Hindi scripts etc.) — only when the model supports
        # `language_code` (multilingual_v2 explicitly does not).
        resolved_language = (settings.elevenlabs_language_code or "").strip()
        if not resolved_language and language_code:
            resolved_language = language_code.strip().lower()
        if resolved_language and caps.supports_language_code:
            payload["language_code"] = resolved_language

        # Text normalization: default `auto` (validated in _require_configuration).
        payload["apply_text_normalization"] = settings.elevenlabs_apply_text_normalization

        # Language-specific normalization is Japanese-only per the docs —
        # it must never be enabled for Hindi (it would only add latency).
        if (
            settings.elevenlabs_apply_language_text_normalization
            and resolved_language == "ja"
        ):
            payload["apply_language_text_normalization"] = True

        # Pronunciation dictionaries (explicitly configured IDs only).
        if settings.elevenlabs_pronunciation_dictionaries:
            payload["pronunciation_dictionary_locators"] = [
                dict(locator)
                for locator in settings.elevenlabs_pronunciation_dictionaries
            ]

        # Optional audio effects (configuration-driven, catalog-validated).
        effects = self._effective_audio_effects()
        if effects:
            payload["audio_effects"] = effects

        # Paragraph continuity — text XOR request IDs per side, always.
        if context is not None and caps.supports_request_stitching:
            self._apply_context(payload, context)

        return payload

    @staticmethod
    def _apply_context(payload: dict[str, Any], context: SynthesisContext) -> None:
        """Add neighbour context. Request IDs win; text is never sent with them."""
        previous_ids = context.previous_ids()
        if previous_ids:
            payload["previous_request_ids"] = previous_ids
        elif context.previous_text and context.previous_text.strip():
            payload["previous_text"] = _clip_previous(context.previous_text)

        next_ids = context.next_ids()
        if next_ids:
            payload["next_request_ids"] = next_ids
        elif context.next_text and context.next_text.strip():
            payload["next_text"] = _clip_next(context.next_text)

    def _effective_audio_effects(self) -> dict[str, Any]:
        """Configured effects, validated against the catalog when reachable.

        Effects are NEVER applied at random: they only exist when
        ELEVENLABS_AUDIO_EFFECTS is configured. Unknown IDs would return a
        404 from the API, so we proactively drop them when the catalog can be
        fetched; if the catalog is unreachable the API remains the judge.
        """
        configured = dict(self.settings.elevenlabs_audio_effects or {})
        if not configured:
            return {}

        # Reverb/pan produce stereo audio which pcm_*/ulaw_*/alaw_* outputs
        # reject with 400 — drop only those keys for those formats.
        fmt = self.settings.elevenlabs_segment_output_format
        if fmt.startswith(("pcm_", "ulaw_", "alaw_")):
            configured.pop("environment_id", None)
            configured.pop("pan", None)
            if "send_level" in configured and "environment_id" not in configured:
                configured.pop("send_level")
        if not configured:
            return {}

        catalog = self._effects_catalog
        if catalog is None:
            return configured  # no catalog cached -> API validates upstream

        allowed = {
            "filter_preset_id": {
                item.get("filter_preset_id")
                for item in catalog.get("filter_presets", [])
                if isinstance(item, dict)
            },
            "environment_id": {
                item.get("environment_id")
                for item in catalog.get("environments", [])
                if isinstance(item, dict)
            },
            "background_noise_id": {
                item.get("background_noise_id")
                for item in catalog.get("background_noises", [])
                if isinstance(item, dict)
            },
        }
        validated: dict[str, Any] = {}
        for key, value in configured.items():
            if key in allowed and value not in allowed[key]:
                continue  # unknown ID — never send it (404 upstream)
            validated[key] = value
        return validated

    # ------------------------------------------------------------------ #
    # Audio effects catalog
    # ------------------------------------------------------------------ #
    async def fetch_audio_effects_catalog(self) -> dict[str, Any]:
        """GET /v1/audio-effects/catalog (cached for `_CATALOG_TTL_SECONDS`)."""
        now = time.monotonic()
        if (
            self._effects_catalog is not None
            and now - self._effects_catalog_at < _CATALOG_TTL_SECONDS
        ):
            return self._effects_catalog

        settings = self.settings
        headers = {"accept": "application/json"}
        if settings.elevenlabs_api_key:
            headers["xi-api-key"] = settings.elevenlabs_api_key

        client_kwargs: dict[str, object] = {
            "timeout": settings.elevenlabs_timeout_seconds,
            "follow_redirects": False,
        }
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.get(self.catalog_endpoint, headers=headers)
        except httpx.TimeoutException as exc:
            raise ElevenLabsTimeoutError(
                "ElevenLabs audio effects catalog did not respond within "
                f"{settings.elevenlabs_timeout_seconds:.0f} seconds"
            ) from exc
        except httpx.HTTPError as exc:
            raise ElevenLabsAPIError(
                "Could not reach the ElevenLabs audio effects catalog "
                f"({exc.__class__.__name__})"
            ) from exc

        if response.status_code != httpx.codes.OK:
            detail = _safe_detail(response)
            if response.status_code in (
                httpx.codes.UNAUTHORIZED,
                httpx.codes.FORBIDDEN,
            ):
                raise ElevenLabsAuthError(
                    "ElevenLabs rejected the API key for the audio effects "
                    "catalog — check ELEVENLABS_API_KEY" + detail
                )
            raise ElevenLabsAPIError(
                "ElevenLabs audio effects catalog failed with HTTP "
                f"{response.status_code}" + detail
            )

        try:
            data = response.json()
        except Exception as exc:
            raise ElevenLabsAPIError(
                "ElevenLabs audio effects catalog returned invalid JSON"
            ) from exc
        if not isinstance(data, dict):
            raise ElevenLabsAPIError(
                "ElevenLabs audio effects catalog returned an unexpected payload"
            )

        self._effects_catalog = data
        self._effects_catalog_at = now
        return data

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #
    async def synthesize_result(
        self,
        text: str,
        voice_settings: Optional[dict[str, object]] = None,
        *,
        tts_text: Optional[str] = None,
        context: Optional[SynthesisContext] = None,
        language_code: Optional[str] = None,
        output_format: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> SynthesisResult:
        """Convert `text` to speech and return audio plus the request ID.

        The `request-id` response header is captured for paragraph
        continuity (stored per segment and reused when regenerating).
        `output_format` overrides the configured segment format (used by
        the legacy single-pass mode, which always wants the final format).
        `voice_id` selects a specific locked character voice for this
        request (multi-voice mode); the configured project voice is used
        when omitted.
        """
        # Warm the effects catalog once so IDs are validated before use.
        if self.settings.elevenlabs_audio_effects and self._effects_catalog is None:
            try:
                await self.fetch_audio_effects_catalog()
            except ElevenLabsAPIError:
                pass  # offline / mocked catalog: let the API be the judge

        self.last_request_id = None
        audio = await self.synthesize(
            text,
            voice_settings,
            tts_text=tts_text,
            context=context,
            language_code=language_code,
            output_format=output_format,
            voice_id=voice_id,
        )
        return SynthesisResult(
            audio=audio,
            request_id=self.last_request_id,
            output_format=output_format
            or self.settings.elevenlabs_segment_output_format,
            model_id=self.settings.elevenlabs_model_id,
            voice_id=(voice_id or "").strip()
            or self.settings.elevenlabs_voice_id,
        )

    async def synthesize(
        self,
        text: str,
        voice_settings: Optional[dict[str, object]] = None,
        *,
        tts_text: Optional[str] = None,
        context: Optional[SynthesisContext] = None,
        language_code: Optional[str] = None,
        output_format: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> bytes:
        """Convert `text` to speech and return raw audio bytes.

        `voice_settings` (from the MiMo narration plan) overrides the defaults
        key-by-key; unsupported keys and model-unsupported fields are dropped
        so only valid parameters reach ElevenLabs.

        Raises a `ProjectError` subclass for every failure mode so the route
        layer can map it straight onto an HTTP status code.
        """
        self._require_configuration()

        settings = self.settings
        # Multi-voice mode: per-request locked voice; else the project voice.
        voice = (voice_id or "").strip() or settings.elevenlabs_voice_id
        endpoint = f"{settings.elevenlabs_base_url}/text-to-speech/{voice}"
        self.last_request_id = None
        headers = {
            "xi-api-key": settings.elevenlabs_api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }
        payload = self._build_payload(
            text,
            voice_settings,
            tts_text=tts_text,
            context=context,
            language_code=language_code,
        )
        # Segments use the (optional) intermediate format; the final audiobook
        # is encoded ONCE by FFmpeg afterwards. Callers may override (the
        # legacy single-pass mode always requests the final format).
        params = {
            "output_format": (
                output_format or settings.elevenlabs_segment_output_format
            )
        }

        client_kwargs: dict[str, object] = {
            "timeout": settings.elevenlabs_timeout_seconds,
            "follow_redirects": False,
        }
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    params=params,
                )
        except httpx.TimeoutException as exc:
            raise ElevenLabsTimeoutError(
                f"ElevenLabs did not respond within "
                f"{settings.elevenlabs_timeout_seconds:.0f} seconds"
            ) from exc
        except httpx.HTTPError as exc:
            # DNS / connection refused / TLS problems -> upstream unreachable.
            raise ElevenLabsAPIError(
                f"Could not reach the ElevenLabs API ({exc.__class__.__name__})"
            ) from exc

        # Captured BEFORE the body is consumed: used for request stitching.
        self.last_request_id = response.headers.get("request-id") or None
        return self._to_audio(response)

    # ------------------------------------------------------------------ #
    # Response mapping
    # ------------------------------------------------------------------ #
    def _to_audio(self, response: httpx.Response) -> bytes:
        status = response.status_code

        if status == httpx.codes.OK:
            audio = response.content
            if not audio:
                raise ElevenLabsAPIError("ElevenLabs returned an empty audio payload")
            return audio

        detail = _safe_detail(response)

        if status in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            raise ElevenLabsAuthError(
                "ElevenLabs rejected the API key — check ELEVENLABS_API_KEY" + detail
            )
        if status == httpx.codes.NOT_FOUND:
            raise ElevenLabsVoiceError(
                "ElevenLabs could not find the configured voice — check "
                "ELEVENLABS_VOICE_ID" + detail
            )
        if status in (httpx.codes.BAD_REQUEST, httpx.codes.UNPROCESSABLE_ENTITY):
            raise ElevenLabsVoiceError(
                "ElevenLabs rejected the request — check ELEVENLABS_VOICE_ID / "
                "ELEVENLABS_MODEL_ID and the script text" + detail
            )
        if status == httpx.codes.TOO_MANY_REQUESTS:
            retry_after = response.headers.get("retry-after")
            hint = f"; retry after {retry_after}s" if retry_after else ""
            raise ElevenLabsRateLimitError(
                "ElevenLabs rate limit reached" + hint + detail
            )

        raise ElevenLabsAPIError(
            f"ElevenLabs API failed with HTTP {status}" + detail
        )
