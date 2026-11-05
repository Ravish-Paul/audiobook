"""Application configuration.

Settings are read from environment variables with sensible defaults, so the
project runs out-of-the-box without any extra setup:

    STORAGE_DIR=<repo>/storage/projects   (optional)
    MAX_SCRIPT_SIZE_MB=5                  (optional)

Secrets and ElevenLabs settings live in `<repo>/.env` (see `.env.example`).
They are loaded once at import time; real process env vars always win, so CI
and the test suite stay in control:

    ELEVENLABS_API_KEY=...
    ELEVENLABS_VOICE_ID=...
    ELEVENLABS_MODEL_ID=eleven_v4
    MIMO_API_KEY=...
    MIMO_MODEL_ID=mimo-v2.6-flash
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Repository root (parent of the `app/` package).
BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_STORAGE_DIR = BASE_DIR / "storage" / "projects"
DEFAULT_MAX_SCRIPT_SIZE_MB = 5

DEFAULT_ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
# Eleven v4 is the audiobook flagship model (audio tags, 90+ languages
# including Hindi, 10,000 chars/request). Override with ELEVENLABS_MODEL_ID.
DEFAULT_ELEVENLABS_MODEL_ID = "eleven_v4"
DEFAULT_ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"  # 128 kbps MP3 @ 44.1 kHz
DEFAULT_ELEVENLABS_TIMEOUT_SECONDS = 60.0
DEFAULT_ELEVENLABS_MAX_CHARS = 5000  # conservative; eleven_v4 allows 10000
DEFAULT_ELEVENLABS_TEXT_NORMALIZATION = "auto"  # auto | on | off

# Every output format accepted by POST /v1/text-to-speech/{voice_id}.
# (docs: allowed values of the `output_format` query parameter)
ELEVENLABS_OUTPUT_FORMATS: tuple[str, ...] = (
    "alaw_8000",
    "mp3_22050_32",
    "mp3_24000_48",
    "mp3_44100_32",
    "mp3_44100_64",
    "mp3_44100_96",
    "mp3_44100_128",
    "mp3_44100_192",  # requires Creator tier or above
    "opus_48000_32",
    "opus_48000_64",
    "opus_48000_96",
    "opus_48000_128",
    "opus_48000_192",
    "pcm_8000",
    "pcm_16000",
    "pcm_22050",
    "pcm_24000",
    "pcm_32000",
    "pcm_44100",  # 44.1 kHz PCM — requires Pro tier or above
    "pcm_48000",
    "ulaw_8000",
    "wav_8000",
    "wav_16000",
    "wav_22050",
    "wav_24000",
    "wav_32000",
    "wav_44100",  # 44.1 kHz WAV — requires Pro tier or above
    "wav_48000",
)

# Keys accepted inside ELEVENLABS_AUDIO_EFFECTS (EffectsSpec-Input).
ELEVENLABS_AUDIO_EFFECT_KEYS: tuple[str, ...] = (
    "filter_preset_id",
    "environment_id",
    "background_noise_id",
    "distance",
    "send_level",
    "pan",
)

# The ElevenLabs API accepts at most 3 pronunciation dictionary locators
# and at most 3 request ids per continuity side.
ELEVENLABS_MAX_DICTIONARY_LOCATORS = 3
ELEVENLABS_MAX_REQUEST_IDS = 3

# ISO 639-1 mapping for the languages MiMo commonly reports. Only used when
# ELEVENLABS_LANGUAGE_CODE is not set explicitly.
LANGUAGE_NAME_TO_CODE: dict[str, str] = {
    "hindi": "hi",
    "english": "en",
    "marathi": "mr",
    "bengali": "bn",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "gujarati": "gu",
    "punjabi": "pa",
    "urdu": "ur",
    "bhojpuri": "",  # no ISO 639-1 code — never guess
}

# Xiaomi MiMo — the AI narration director (OpenAI-compatible endpoint).
DEFAULT_MIMO_MODEL_ID = "mimo-v2.6-flash"
DEFAULT_MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_MIMO_TIMEOUT_SECONDS = 120.0
DEFAULT_MIMO_MAX_COMPLETION_TOKENS = 8192
DEFAULT_MIMO_MAX_ATTEMPTS = 3  # tries per analysis when the plan is invalid

# FFmpeg audio composition.
DEFAULT_FFMPEG_PATH = "ffmpeg"
DEFAULT_FFMPEG_TIMEOUT_SECONDS = 600

# Load <repo>/.env once. `override=False` => already-exported env vars are kept.
load_dotenv(BASE_DIR / ".env", override=False)


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _secret(name: str, default: str = "") -> str:
    """Read a secret-ish value, treating `.env.example` placeholders as unset.

    Returns an empty string when the variable is missing/blank or still holds
    an example placeholder such as `your_api_key_here`, so we can report
    "not configured" instead of sending garbage to the API.
    """
    value = _env(name, default).strip()
    if value.lower().startswith("your_"):
        return default
    return value


def _bool_env(name: str, default: bool = False) -> bool:
    value = _env(name, "true" if default else "false").strip().lower()
    return value in ("1", "true", "yes", "on")


def _json_env(name: str) -> Any:
    """Parse an optional JSON env var; malformed values are ignored (None)."""
    raw = _env(name, "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _optional_float_env(name: str) -> Optional[float]:
    raw = _env(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_pronunciation_dictionaries(raw: Any) -> tuple[dict[str, str], ...]:
    """Normalise `ELEVENLABS_PRONUNCIATION_DICTIONARIES` into API locators.

    Accepted JSON shapes (list of either):

        ["dictionary-id", ...]
        [{"id": "...", "version_id": "..."}, ...]
        [{"pronunciation_dictionary_id": "...", "version_id": "..."}, ...]

    Values that look like placeholders (`your_…`) or are blank are dropped —
    dictionary IDs are NEVER invented. At most 3 locators survive (API limit).
    """
    if not isinstance(raw, list):
        return ()
    locators: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            dict_id, version_id = item.strip(), ""
        elif isinstance(item, dict):
            dict_id = str(
                item.get("id")
                or item.get("pronunciation_dictionary_id")
                or ""
            ).strip()
            version_id = str(item.get("version_id") or "").strip()
        else:
            continue
        if not dict_id or dict_id.lower().startswith("your_"):
            continue  # placeholder / empty -> never sent, never invented
        locator: dict[str, str] = {"pronunciation_dictionary_id": dict_id}
        if version_id and not version_id.lower().startswith("your_"):
            locator["version_id"] = version_id
        locators.append(locator)
        if len(locators) >= ELEVENLABS_MAX_DICTIONARY_LOCATORS:
            break
    return tuple(locators)


def parse_audio_effects(raw: Any) -> Optional[dict[str, Any]]:
    """Normalise `ELEVENLABS_AUDIO_EFFECTS` into an `audio_effects` body.

    Unknown keys are dropped (only documented EffectsSpec fields survive) and
    numeric ranges are clamped to the documented bounds. Returns None when
    nothing usable is configured — effects are never applied at random.
    """
    if not isinstance(raw, dict):
        return None
    effects: dict[str, Any] = {}
    for key in ELEVENLABS_AUDIO_EFFECT_KEYS:
        if key not in raw or raw[key] is None:
            continue
        value = raw[key]
        if key in ("filter_preset_id", "environment_id", "background_noise_id"):
            text = str(value).strip()
            if text and not text.lower().startswith("your_"):
                effects[key] = text
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if key in ("distance", "send_level"):
            effects[key] = min(1.0, max(0.0, number))
        elif key == "pan":
            effects[key] = min(1.0, max(-1.0, number))
    return effects or None


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the API."""

    app_name: str = "AI Audiobook Platform"
    app_description: str = (
        "FastAPI backend for an AI audiobook platform. Create projects, "
        "upload .txt scripts and generate narrated MP3s with ElevenLabs TTS."
    )
    version: str = "0.2.0"

    # Where uploaded scripts live: storage/projects/{project_id}/script.txt
    storage_dir: Path = field(default=DEFAULT_STORAGE_DIR)

    # Hard limit for an uploaded script file.
    max_script_size_bytes: int = DEFAULT_MAX_SCRIPT_SIZE_MB * 1024 * 1024

    # Only plain-text scripts are accepted for now.
    allowed_script_extensions: tuple[str, ...] = (".txt",)

    # File the raw upload is stored as (per spec).
    stored_script_filename: str = "script.txt"

    # ---- Audio output ---------------------------------------------------- #
    # storage/projects/{project_id}/audio/audiobook.mp3
    audio_dir_name: str = "audio"
    audio_filename: str = "audiobook.mp3"
    # The projects folder is exposed at this URL prefix, so a generated file
    # lives at: /media/projects/{project_id}/audio/audiobook.mp3
    media_url_prefix: str = "/media/projects"

    # ---- ElevenLabs (loaded from .env — never hardcoded) ----------------- #
    elevenlabs_api_key: str = field(default="", repr=False)  # never print it
    elevenlabs_voice_id: str = ""
    elevenlabs_model_id: str = DEFAULT_ELEVENLABS_MODEL_ID
    elevenlabs_base_url: str = DEFAULT_ELEVENLABS_BASE_URL
    elevenlabs_output_format: str = DEFAULT_ELEVENLABS_OUTPUT_FORMAT
    # Optional PCM/WAV (or higher-quality MP3) intermediate: paragraphs are
    # fetched in this format and FFmpeg encodes the final audiobook ONCE.
    # Empty -> segments use `elevenlabs_output_format` directly (legacy).
    elevenlabs_intermediate_output_format: str = ""
    elevenlabs_timeout_seconds: float = DEFAULT_ELEVENLABS_TIMEOUT_SECONDS
    elevenlabs_max_chars: int = DEFAULT_ELEVENLABS_MAX_CHARS

    # ISO 639-1 language hint (e.g. "hi" for Hindi). Empty -> derived from
    # MiMo's plan language when the model supports language_code.
    elevenlabs_language_code: str = ""

    # Text normalization: auto | on | off (default auto per API docs).
    elevenlabs_apply_text_normalization: str = DEFAULT_ELEVENLABS_TEXT_NORMALIZATION
    # Language-specific normalization is only supported for Japanese — it is
    # NEVER enabled for Hindi (docs: "currently only supported for Japanese").
    elevenlabs_apply_language_text_normalization: bool = False

    # Pronunciation dictionaries (character names, Hindi names, hard words).
    # Only IDs explicitly configured here are ever sent — never invented.
    elevenlabs_pronunciation_dictionaries: tuple[dict[str, str], ...] = ()

    # Optional ElevenLabs audio effects (voice colouring only — background
    # music / SFX belong to the FFmpeg mixing pipeline). IDs must come from
    # GET /v1/audio-effects/catalog.
    elevenlabs_audio_effects: Optional[dict[str, Any]] = None

    # clone_strength is only valid for Professional Voice Clones fine-tuned
    # for eleven_v4 — set this flag once your configured voice is such a PVC.
    elevenlabs_voice_is_pvc: bool = False
    # Optional default clone strength (0..1) applied when the flag is on.
    elevenlabs_clone_strength: Optional[float] = None

    # ---- Xiaomi MiMo (AI narration director, from .env) ------------------ #
    mimo_api_key: str = field(default="", repr=False)  # never print it
    mimo_model_id: str = DEFAULT_MIMO_MODEL_ID
    mimo_base_url: str = DEFAULT_MIMO_BASE_URL
    mimo_timeout_seconds: float = DEFAULT_MIMO_TIMEOUT_SECONDS
    mimo_max_completion_tokens: int = DEFAULT_MIMO_MAX_COMPLETION_TOKENS
    mimo_max_attempts: int = DEFAULT_MIMO_MAX_ATTEMPTS

    # ---- Narration plan + audio composition ------------------------------ #
    narration_plan_filename: str = "narration_plan.json"
    character_cast_filename: str = "character_cast.json"
    multi_voice_plan_filename: str = "multi_voice_plan.json"
    segment_dir_name: str = "segments"
    audio_metadata_filename: str = "metadata.json"
    ffmpeg_path: str = DEFAULT_FFMPEG_PATH
    ffmpeg_timeout_seconds: int = DEFAULT_FFMPEG_TIMEOUT_SECONDS

    @property
    def max_script_size_mb(self) -> int:
        return self.max_script_size_bytes // (1024 * 1024)

    @property
    def elevenlabs_configured(self) -> bool:
        """True when both the API key and a voice ID are available."""
        return bool(self.elevenlabs_api_key and self.elevenlabs_voice_id)

    @property
    def elevenlabs_segment_output_format(self) -> str:
        """Format requested for each paragraph segment.

        The (optional) intermediate format wins so PCM/WAV masters can be
        edited locally; FFmpeg then encodes the final audiobook once.
        """
        intermediate = (self.elevenlabs_intermediate_output_format or "").strip()
        return intermediate or self.elevenlabs_output_format

    @property
    def elevenlabs_uses_intermediate(self) -> bool:
        """True when segments are fetched in a different format than final."""
        return bool(
            self.elevenlabs_intermediate_output_format
            and self.elevenlabs_intermediate_output_format
            != self.elevenlabs_output_format
        )

    @property
    def mimo_configured(self) -> bool:
        """True when a MiMo API key is available."""
        return bool(self.mimo_api_key)

    @property
    def media_root(self) -> Path:
        """Folder exposed at `media_url_prefix` (the projects directory)."""
        return self.storage_dir

    def media_url(self, path: Path) -> str:
        """Public, browser-playable URL for a file stored under `media_root`."""
        relative = path.resolve().relative_to(self.media_root.resolve())
        return f"{self.media_url_prefix}/{relative.as_posix()}"


def get_settings() -> Settings:
    """Build settings from the current environment (cheap, safe to call often)."""
    storage_dir = Path(_env("STORAGE_DIR", str(DEFAULT_STORAGE_DIR))).expanduser()
    max_mb = int(_env("MAX_SCRIPT_SIZE_MB", str(DEFAULT_MAX_SCRIPT_SIZE_MB)))

    return Settings(
        storage_dir=storage_dir,
        max_script_size_bytes=max_mb * 1024 * 1024,
        # Secrets: placeholder values from .env.example count as "not set".
        elevenlabs_api_key=_secret("ELEVENLABS_API_KEY"),
        elevenlabs_voice_id=_secret("ELEVENLABS_VOICE_ID"),
        elevenlabs_model_id=_env("ELEVENLABS_MODEL_ID", DEFAULT_ELEVENLABS_MODEL_ID),
        elevenlabs_base_url=_env(
            "ELEVENLABS_BASE_URL", DEFAULT_ELEVENLABS_BASE_URL
        ).rstrip("/"),
        elevenlabs_output_format=_env(
            "ELEVENLABS_OUTPUT_FORMAT", DEFAULT_ELEVENLABS_OUTPUT_FORMAT
        ),
        elevenlabs_intermediate_output_format=_env(
            "ELEVENLABS_INTERMEDIATE_OUTPUT_FORMAT", ""
        ),
        elevenlabs_timeout_seconds=float(
            _env("ELEVENLABS_TIMEOUT_SECONDS", str(DEFAULT_ELEVENLABS_TIMEOUT_SECONDS))
        ),
        elevenlabs_max_chars=int(
            _env("ELEVENLABS_MAX_CHARS", str(DEFAULT_ELEVENLABS_MAX_CHARS))
        ),
        # Language + normalization
        elevenlabs_language_code=_env("ELEVENLABS_LANGUAGE_CODE", "").strip().lower(),
        elevenlabs_apply_text_normalization=(
            _env(
                "ELEVENLABS_APPLY_TEXT_NORMALIZATION",
                DEFAULT_ELEVENLABS_TEXT_NORMALIZATION,
            )
            .strip()
            .lower()
        ),
        elevenlabs_apply_language_text_normalization=_bool_env(
            "ELEVENLABS_APPLY_LANGUAGE_TEXT_NORMALIZATION", False
        ),
        # Pronunciation dictionaries / audio effects / PVC clone strength
        elevenlabs_pronunciation_dictionaries=parse_pronunciation_dictionaries(
            _json_env("ELEVENLABS_PRONUNCIATION_DICTIONARIES")
        ),
        elevenlabs_audio_effects=parse_audio_effects(
            _json_env("ELEVENLABS_AUDIO_EFFECTS")
        ),
        elevenlabs_voice_is_pvc=_bool_env("ELEVENLABS_VOICE_IS_PROFESSIONAL_CLONE"),
        elevenlabs_clone_strength=_optional_float_env("ELEVENLABS_CLONE_STRENGTH"),
        # Xiaomi MiMo (narration director)
        mimo_api_key=_secret("MIMO_API_KEY"),
        mimo_model_id=_env("MIMO_MODEL_ID", DEFAULT_MIMO_MODEL_ID),
        mimo_base_url=_env("MIMO_BASE_URL", DEFAULT_MIMO_BASE_URL).rstrip("/"),
        mimo_timeout_seconds=float(
            _env("MIMO_TIMEOUT_SECONDS", str(DEFAULT_MIMO_TIMEOUT_SECONDS))
        ),
        mimo_max_completion_tokens=int(
            _env(
                "MIMO_MAX_COMPLETION_TOKENS",
                str(DEFAULT_MIMO_MAX_COMPLETION_TOKENS),
            )
        ),
        mimo_max_attempts=max(
            1, int(_env("MIMO_MAX_ATTEMPTS", str(DEFAULT_MIMO_MAX_ATTEMPTS)))
        ),
        # FFmpeg
        ffmpeg_path=_env("FFMPEG_PATH", DEFAULT_FFMPEG_PATH),
        ffmpeg_timeout_seconds=int(
            _env("FFMPEG_TIMEOUT_SECONDS", str(DEFAULT_FFMPEG_TIMEOUT_SECONDS))
        ),
    )
