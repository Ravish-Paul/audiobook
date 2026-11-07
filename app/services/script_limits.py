"""Private-beta script limits: length + estimated audiobook duration.

Cost guards applied wherever a script enters the system (upload/import) and
again before audio generation:

* `max_script_characters`      — hard cap on the decoded script length;
* `max_audio_duration_minutes` — cap on the ESTIMATED spoken duration of the
  script (chars / speech_chars_per_minute), so a runaway upload can never
  produce an unbounded TTS bill.

Both limits are env-configurable (`MAX_SCRIPT_CHARACTERS`,
`MAX_AUDIO_DURATION_MINUTES`, `SPEECH_CHARS_PER_MINUTE`) and `<= 0`
disables the corresponding check. The duration is a documented heuristic,
never a promise about the produced file.
"""

from __future__ import annotations

from app.config import Settings
from app.services.errors import ScriptValidationError


def estimate_duration_minutes(text: str, settings: Settings) -> float:
    """Estimated spoken duration of `text` in minutes (heuristic)."""
    chars_per_minute = max(1, settings.speech_chars_per_minute)
    return len(text or "") / chars_per_minute


def check_script_limits(settings: Settings, text: str) -> None:
    """Raise ScriptValidationError (400) when the script breaks a limit."""
    length = len(text or "")

    max_chars = settings.max_script_characters
    if max_chars > 0 and length > max_chars:
        raise ScriptValidationError(
            f"Script is too long: {length} characters "
            f"(limit: {max_chars} characters)"
        )

    max_minutes = settings.max_audio_duration_minutes
    if max_minutes > 0:
        minutes = estimate_duration_minutes(text, settings)
        if minutes > max_minutes:
            raise ScriptValidationError(
                f"Script would produce about {minutes:.0f} minutes of audio "
                f"(limit: {max_minutes} minutes) — please split it up"
            )
