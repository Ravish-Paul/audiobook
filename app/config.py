"""Application configuration.

Settings are read from environment variables with sensible defaults, so the
project runs out-of-the-box without any extra setup:

    STORAGE_DIR=<repo>/storage/projects   (optional)
    MAX_SCRIPT_SIZE_MB=5                  (optional)
    BETA_ACCESS_TOKENS=name:token,…       (private beta testers)
    BETA_TESTER_GENERATION_LIMITS=name:1,… (per-tester generation limits)

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

# The Eleven v4 supported language vocabulary (90+ languages) — the single
# source of truth for names MiMo may report, the upload detector shows and
# the UI override offers. Keys are lowercase display names; values are the
# provider's language codes (ISO 639-1 where one exists, otherwise the
# documented ISO 639-2/3 code). A language with no code maps to "" — never
# guess a code. Only used when ELEVENLABS_LANGUAGE_CODE is not set explicitly.
LANGUAGE_NAME_TO_CODE: dict[str, str] = {
    "afrikaans": "af",
    "amharic": "am",
    "arabic": "ar",
    "armenian": "hy",
    "assamese": "as",
    "asturian": "ast",
    "azerbaijani": "az",
    "belarusian": "be",
    "bengali": "bn",
    "bosnian": "bs",
    "bulgarian": "bg",
    "burmese": "my",
    "cantonese": "yue",
    "catalan": "ca",
    "cebuano": "ceb",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "filipino": "fil",
    "finnish": "fi",
    "french": "fr",
    "fula (pulaar)": "ff",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "hausa": "ha",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jv",
    "kamba": "kam",
    "kannada": "kn",
    "kazakh": "kk",
    "korean": "ko",
    "kyrgyz": "ky",
    "lao": "lo",
    "latvian": "lv",
    "lingala": "ln",
    "lithuanian": "lt",
    "luganda": "lg",
    "luxembourgish": "lb",
    "macedonian": "mk",
    "malay": "ms",
    "malayalam": "ml",
    "maltese": "mt",
    "mandarin chinese": "zh",
    "māori": "mi",
    "marathi": "mr",
    "mongolian": "mn",
    "nepali": "ne",
    "norwegian": "no",
    "occitan": "oc",
    "odia": "or",
    "pashto": "ps",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "shona": "sn",
    "sindhi": "sd",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "sorani kurdish": "ckb",
    "spanish": "es",
    "swahili": "sw",
    "swedish": "sv",
    "tajik": "tg",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "wolof": "wo",
    "zulu": "zu",
    # Unambiguous alternates MiMo or the detector may report.
    "chinese": "zh",
    "mandarin": "zh",
    "maori": "mi",
    "oriya": "or",
    "myanmar": "my",
    "fula": "ff",
    "pulaar": "ff",
    # Recognised languages outside the Eleven v4 vocabulary. The ISO code is
    # factually correct but NO provider voice may exist — casting then finds
    # nothing, which is the honest outcome.
    "sinhala": "si",
    "bhojpuri": "",  # no ISO 639-1 code — never guess
}


def language_to_iso(value: str) -> str:
    """Best-effort fold of a language name/code onto its provider code.

    Known names map through LANGUAGE_NAME_TO_CODE; code-shaped values
    (2-3 alpha, ISO 639-1/2/3) pass through lowercased; anything else is
    returned untouched (never guessed) so callers can reject or display the
    raw value loudly.
    """
    cleaned = (value or "").strip()
    if not cleaned:
        return ""
    mapped = LANGUAGE_NAME_TO_CODE.get(cleaned.lower())
    if mapped:
        return mapped
    if len(cleaned) in (2, 3) and cleaned.isalpha():
        return cleaned.lower()
    return cleaned

# Xiaomi MiMo — the AI narration director (OpenAI-compatible endpoint).
DEFAULT_MIMO_MODEL_ID = "mimo-v2.6-flash"
DEFAULT_MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_MIMO_TIMEOUT_SECONDS = 120.0
DEFAULT_MIMO_MAX_COMPLETION_TOKENS = 8192
DEFAULT_MIMO_MAX_ATTEMPTS = 3  # tries per analysis when the plan is invalid

# FFmpeg audio composition.
DEFAULT_FFMPEG_PATH = "ffmpeg"
DEFAULT_FFMPEG_TIMEOUT_SECONDS = 600
# ffprobe (ships with FFmpeg) reads the sample rate / duration of the
# finished audiobook — the analysis timeline is built from the REAL values.
DEFAULT_FFPROBE_PATH = "ffprobe"

# pyannoteAI (speaker diarization + word-level transcription).
# Docs: https://docs.pyannote.ai — billing has a 20-second minimum per
# successful /v1/diarize job, so live submissions must stay deliberate.
DEFAULT_PYANNOTE_BASE_URL = "https://api.pyannote.ai"
DEFAULT_PYANNOTE_MODEL_ID = "precision-3"
DEFAULT_PYANNOTE_TRANSCRIPTION_MODEL = "faster-whisper-large-v3-turbo"
DEFAULT_PYANNOTE_TIMEOUT_SECONDS = 30.0
# Controlled polling: exponential backoff between status checks.
DEFAULT_PYANNOTE_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_PYANNOTE_POLL_MAX_INTERVAL_SECONDS = 30.0
# Retries for IDEMPOTENT requests only (job lookup, media PUT upload).
# A /v1/diarize submission is NEVER retried automatically — a duplicate
# job would duplicate the charge.
DEFAULT_PYANNOTE_MAX_ATTEMPTS = 2
# Provider timestamps may overshoot the container duration slightly (MP3
# padding). Anything beyond duration + tolerance is rejected as invalid.
DEFAULT_PYANNOTE_DURATION_TOLERANCE_MS = 500

# ---- Private beta limits (all env-configurable, safe defaults) ----------- #
DEFAULT_BETA_SESSION_TTL_HOURS = 72
DEFAULT_BETA_GENERATION_LIMIT = 10  # credit-spending calls per tester (lifetime)
DEFAULT_BETA_GENERATION_WINDOW_HOURS = 24  # deprecated: no effect anymore
DEFAULT_BETA_RATE_LIMIT_PER_MINUTE = 120
DEFAULT_BETA_LOGIN_RATE_LIMIT_PER_MINUTE = 10
DEFAULT_BETA_DUPLICATE_WINDOW_SECONDS = 10
DEFAULT_MAX_SCRIPT_CHARACTERS = 200_000  # <= 0 disables the check
DEFAULT_MAX_AUDIO_DURATION_MINUTES = 120  # <= 0 disables the check
# Heuristic for the spoken duration of a plain-text script (~150 wpm).
DEFAULT_SPEECH_CHARS_PER_MINUTE = 900

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

    # Hard limits for the script text itself (private-beta cost guards).
    max_script_characters: int = DEFAULT_MAX_SCRIPT_CHARACTERS  # <= 0: unlimited
    max_audio_duration_minutes: int = DEFAULT_MAX_AUDIO_DURATION_MINUTES
    speech_chars_per_minute: int = DEFAULT_SPEECH_CHARS_PER_MINUTE

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

    # ---- pyannoteAI (audio analysis: diarization + transcription) ------- #
    pyannote_api_key: str = field(default="", repr=False)  # never print it
    pyannote_base_url: str = DEFAULT_PYANNOTE_BASE_URL
    pyannote_model_id: str = DEFAULT_PYANNOTE_MODEL_ID
    pyannote_transcription_model: str = DEFAULT_PYANNOTE_TRANSCRIPTION_MODEL
    pyannote_timeout_seconds: float = DEFAULT_PYANNOTE_TIMEOUT_SECONDS
    pyannote_poll_interval_seconds: float = DEFAULT_PYANNOTE_POLL_INTERVAL_SECONDS
    pyannote_poll_max_interval_seconds: float = (
        DEFAULT_PYANNOTE_POLL_MAX_INTERVAL_SECONDS
    )
    pyannote_max_attempts: int = DEFAULT_PYANNOTE_MAX_ATTEMPTS
    pyannote_duration_tolerance_ms: int = DEFAULT_PYANNOTE_DURATION_TOLERANCE_MS
    ffprobe_path: str = DEFAULT_FFPROBE_PATH

    # ---- Narration plan + audio composition ------------------------------ #
    narration_plan_filename: str = "narration_plan.json"
    narration_preferences_filename: str = "narration_preferences.json"
    character_cast_filename: str = "character_cast.json"
    multi_voice_plan_filename: str = "multi_voice_plan.json"
    segment_dir_name: str = "segments"
    audio_metadata_filename: str = "metadata.json"
    ffmpeg_path: str = DEFAULT_FFMPEG_PATH
    ffmpeg_timeout_seconds: int = DEFAULT_FFMPEG_TIMEOUT_SECONDS

    # ---- Private beta access control (from .env — never hardcoded) ------- #
    # `beta_access_tokens` maps tester name -> token. Populated from
    # BETA_ACCESS_TOKENS="alice:token-a,bob:token-b" and/or a single
    # BETA_ACCESS_TOKEN (tester name "tester"). Empty -> beta mode is OFF and
    # the app behaves exactly as before (development / tests).
    beta_access_tokens: tuple[tuple[str, str], ...] = ()
    beta_session_secret: str = field(default="", repr=False)
    beta_session_ttl_hours: int = DEFAULT_BETA_SESSION_TTL_HOURS
    beta_cookie_secure: bool = False  # set true when serving over HTTPS
    # Credit-spending calls (analyze/preview/generate/…) per tester.
    # `beta_tester_generation_limits` overrides `beta_generation_limit` per
    # tester identity (BETA_TESTER_GENERATION_LIMITS="name:limit,…"). Limits
    # are non-negative integers: 0 means the tester cannot generate at all.
    # Usage is counted persistently per tester (never reset on restart or
    # configuration change): remaining = max(0, effective_limit - used).
    beta_generation_limit: int = DEFAULT_BETA_GENERATION_LIMIT
    beta_tester_generation_limits: dict[str, int] = field(default_factory=dict)
    # Deprecated (no effect since per-tester lifetime usage replaced the
    # windowed counter) — still parsed so old .env files keep working.
    beta_generation_window_hours: int = DEFAULT_BETA_GENERATION_WINDOW_HOURS
    beta_rate_limit_per_minute: int = DEFAULT_BETA_RATE_LIMIT_PER_MINUTE
    beta_login_rate_limit_per_minute: int = DEFAULT_BETA_LOGIN_RATE_LIMIT_PER_MINUTE
    beta_duplicate_window_seconds: int = DEFAULT_BETA_DUPLICATE_WINDOW_SECONDS

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
    def pyannote_configured(self) -> bool:
        """True when a pyannoteAI API key is available."""
        return bool(self.pyannote_api_key)

    @property
    def media_root(self) -> Path:
        """Folder exposed at `media_url_prefix` (the projects directory)."""
        return self.storage_dir

    @property
    def beta_enabled(self) -> bool:
        """True when at least one beta tester token is configured."""
        return bool(self.beta_access_tokens)

    @property
    def beta_tokens_by_value(self) -> dict[str, str]:
        """Reverse lookup: token -> tester name (tokens are compared safely)."""
        return {token: name for name, token in self.beta_access_tokens}

    @property
    def beta_session_key(self) -> bytes:
        """Key for HMAC-signing session cookies.

        An explicit BETA_SESSION_SECRET wins; otherwise the key is derived
        from the configured tokens, so nothing is ever hardcoded and
        rotating tokens invalidates existing sessions.
        """
        import hashlib

        if self.beta_session_secret:
            return self.beta_session_secret.encode("utf-8")
        material = "|".join(f"{n}:{t}" for n, t in self.beta_access_tokens)
        return hashlib.sha256(f"beta-session:{material}".encode("utf-8")).digest()

    def media_url(self, path: Path) -> str:
        """Public, browser-playable URL for a file stored under `media_root`."""
        relative = path.resolve().relative_to(self.media_root.resolve())
        return f"{self.media_url_prefix}/{relative.as_posix()}"


def parse_beta_tokens(raw: str, single: str) -> tuple[tuple[str, str], ...]:
    """Parse `BETA_ACCESS_TOKENS="name:token,name2:token2"` (+ BETA_ACCESS_TOKEN).

    Returns an ordered tuple of (tester_name, token) pairs. A single
    BETA_ACCESS_TOKEN is registered under the name "tester". Malformed
    chunks are ignored — never guess.
    """
    pairs: list[tuple[str, str]] = []
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        name, token = chunk.split(":", 1)
        name, token = name.strip(), token.strip()
        valid_name = (
            1 <= len(name) <= 32
            and all(ch.isalnum() or ch in "-_" for ch in name)
        )
        if valid_name and token:
            pairs.append((name, token))
    if (single or "").strip():
        pairs.append(("tester", single.strip()))
    # Later definitions win for duplicated names.
    return tuple({name: token for name, token in pairs}.items())


def _is_non_negative_int(value: str) -> bool:
    """True for plain ASCII integer literals like "0", "7" (no sign/decimal)."""
    return bool(value) and value.isascii() and value.isdigit()


def parse_generation_limit(raw: str, variable: str) -> int:
    """Parse a generation limit variable: a non-negative integer (0 = none).

    Negative values and decimals are rejected loudly at startup instead of
    being silently reinterpreted.
    """
    value = (raw or "").strip()
    if _is_non_negative_int(value):
        return int(value)
    if value.startswith("-") and _is_non_negative_int(value[1:]):
        raise ValueError(
            f"{variable}: negative generation limit ({value}) — limits must be "
            "non-negative integers (use 0 to block generation entirely)"
        )
    raise ValueError(
        f"{variable}: invalid generation limit ({value!r}) — limits must be "
        "non-negative integers (no decimals)"
    )


def parse_tester_generation_limits(
    raw: str, known_testers: Any
) -> dict[str, int]:
    """Parse `BETA_TESTER_GENERATION_LIMITS="name:1,name2:3"`.

    Every entry must name an identity configured in `BETA_ACCESS_TOKENS` and
    carry a non-negative integer limit (0 = the tester cannot generate).
    Malformed entries, duplicate names, unknown identities, negative values
    and decimal values raise `ValueError` at startup — never guess, never
    silently ignore an entry. Error messages identify testers by name but
    never include authentication tokens.
    """
    limits: dict[str, int] = {}
    known = set(known_testers)
    raw = (raw or "").strip()
    if not raw:
        return {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk or chunk.count(":") != 1:
            raise ValueError(
                "BETA_TESTER_GENERATION_LIMITS: malformed entry "
                f"{chunk!r} (expected 'tester:limit')"
            )
        name, value = (part.strip() for part in chunk.split(":", 1))
        if not name or not value:
            raise ValueError(
                "BETA_TESTER_GENERATION_LIMITS: malformed entry "
                f"{chunk!r} (expected 'tester:limit')"
            )
        if name in limits:
            raise ValueError(
                "BETA_TESTER_GENERATION_LIMITS: duplicate tester name "
                f"'{name}'"
            )
        if name not in known:
            raise ValueError(
                "BETA_TESTER_GENERATION_LIMITS: unknown tester '{name}' — "
                "every entry must name an identity configured in "
                "BETA_ACCESS_TOKENS".format(name=name)
            )
        if value.startswith("-") and _is_non_negative_int(value[1:]):
            raise ValueError(
                "BETA_TESTER_GENERATION_LIMITS: negative limit "
                f"({value}) for tester '{name}' — limits must be "
                "non-negative integers (use 0 to block generation)"
            )
        if not _is_non_negative_int(value):
            raise ValueError(
                "BETA_TESTER_GENERATION_LIMITS: invalid limit "
                f"({value!r}) for tester '{name}' — limits must be "
                "non-negative integers (no decimals)"
            )
        limits[name] = int(value)
    return limits


def get_settings() -> Settings:
    """Build settings from the current environment (cheap, safe to call often).

    Beta configuration is validated here — so a malformed
    `BETA_TESTER_GENERATION_LIMITS` / `BETA_GENERATION_LIMIT` fails
    application startup with a clear error instead of being ignored.
    """
    storage_dir = Path(_env("STORAGE_DIR", str(DEFAULT_STORAGE_DIR))).expanduser()
    max_mb = int(_env("MAX_SCRIPT_SIZE_MB", str(DEFAULT_MAX_SCRIPT_SIZE_MB)))
    beta_access_tokens = parse_beta_tokens(
        _env("BETA_ACCESS_TOKENS", ""), _secret("BETA_ACCESS_TOKEN")
    )
    # Per-tester generation limits must name configured tester identities.
    beta_tester_generation_limits = parse_tester_generation_limits(
        _env("BETA_TESTER_GENERATION_LIMITS", ""),
        [name for name, _ in beta_access_tokens],
    )

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
        # pyannoteAI (audio analysis)
        pyannote_api_key=_secret("PYANNOTE_API_KEY"),
        pyannote_base_url=_env(
            "PYANNOTE_BASE_URL", DEFAULT_PYANNOTE_BASE_URL
        ).rstrip("/"),
        pyannote_model_id=_env("PYANNOTE_MODEL_ID", DEFAULT_PYANNOTE_MODEL_ID),
        pyannote_transcription_model=_env(
            "PYANNOTE_TRANSCRIPTION_MODEL", DEFAULT_PYANNOTE_TRANSCRIPTION_MODEL
        ),
        pyannote_timeout_seconds=float(
            _env("PYANNOTE_TIMEOUT_SECONDS", str(DEFAULT_PYANNOTE_TIMEOUT_SECONDS))
        ),
        pyannote_poll_interval_seconds=max(
            0.0,
            float(
                _env(
                    "PYANNOTE_POLL_INTERVAL_SECONDS",
                    str(DEFAULT_PYANNOTE_POLL_INTERVAL_SECONDS),
                )
            ),
        ),
        pyannote_poll_max_interval_seconds=max(
            0.0,
            float(
                _env(
                    "PYANNOTE_POLL_MAX_INTERVAL_SECONDS",
                    str(DEFAULT_PYANNOTE_POLL_MAX_INTERVAL_SECONDS),
                )
            ),
        ),
        pyannote_max_attempts=max(
            1, int(_env("PYANNOTE_MAX_ATTEMPTS", str(DEFAULT_PYANNOTE_MAX_ATTEMPTS)))
        ),
        pyannote_duration_tolerance_ms=max(
            0,
            int(
                _env(
                    "PYANNOTE_DURATION_TOLERANCE_MS",
                    str(DEFAULT_PYANNOTE_DURATION_TOLERANCE_MS),
                )
            ),
        ),
        # FFmpeg
        ffmpeg_path=_env("FFMPEG_PATH", DEFAULT_FFMPEG_PATH),
        ffmpeg_timeout_seconds=int(
            _env("FFMPEG_TIMEOUT_SECONDS", str(DEFAULT_FFMPEG_TIMEOUT_SECONDS))
        ),
        ffprobe_path=_env("FFPROBE_PATH", DEFAULT_FFPROBE_PATH),
        # Private-beta limits + access control
        max_script_characters=int(
            _env("MAX_SCRIPT_CHARACTERS", str(DEFAULT_MAX_SCRIPT_CHARACTERS))
        ),
        max_audio_duration_minutes=int(
            _env(
                "MAX_AUDIO_DURATION_MINUTES",
                str(DEFAULT_MAX_AUDIO_DURATION_MINUTES),
            )
        ),
        speech_chars_per_minute=max(
            1,
            int(
                _env("SPEECH_CHARS_PER_MINUTE", str(DEFAULT_SPEECH_CHARS_PER_MINUTE))
            ),
        ),
        beta_access_tokens=beta_access_tokens,
        beta_tester_generation_limits=beta_tester_generation_limits,
        beta_session_secret=_secret("BETA_SESSION_SECRET"),
        beta_session_ttl_hours=int(
            _env("BETA_SESSION_TTL_HOURS", str(DEFAULT_BETA_SESSION_TTL_HOURS))
        ),
        beta_cookie_secure=_bool_env("BETA_COOKIE_SECURE", False),
        beta_generation_limit=parse_generation_limit(
            _env("BETA_GENERATION_LIMIT", str(DEFAULT_BETA_GENERATION_LIMIT)),
            "BETA_GENERATION_LIMIT",
        ),
        beta_generation_window_hours=int(
            _env(
                "BETA_GENERATION_WINDOW_HOURS",
                str(DEFAULT_BETA_GENERATION_WINDOW_HOURS),
            )
        ),
        beta_rate_limit_per_minute=int(
            _env(
                "BETA_RATE_LIMIT_PER_MINUTE",
                str(DEFAULT_BETA_RATE_LIMIT_PER_MINUTE),
            )
        ),
        beta_login_rate_limit_per_minute=int(
            _env(
                "BETA_LOGIN_RATE_LIMIT_PER_MINUTE",
                str(DEFAULT_BETA_LOGIN_RATE_LIMIT_PER_MINUTE),
            )
        ),
        beta_duplicate_window_seconds=int(
            _env(
                "BETA_DUPLICATE_WINDOW_SECONDS",
                str(DEFAULT_BETA_DUPLICATE_WINDOW_SECONDS),
            )
        ),
    )
