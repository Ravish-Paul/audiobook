"""Audio probing: real sample rate + duration of a stored audio file.

The normalized audio analysis timeline converts provider timestamps into
milliseconds AND sample indexes. Both conversions need the ACTUAL sample
rate and duration of the source file — they are never hardcoded (the final
audiobook is MP3 @ 44.1 kHz today, but an intermediate/format change must
not silently break the timeline).

Strategy (mirrors `audio_composer.resolve_ffmpeg`):

* prefer `ffprobe` (ships with FFmpeg) and parse its JSON output;
* fall back to `ffmpeg -i` stderr parsing when ffprobe is unavailable;
* raise `AudioProbeError` (500) when neither can describe the file.

Only metadata is read — the audio file itself is never modified.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Optional

from app.config import Settings, get_settings
from app.services.errors import AudioProbeError

# Keep stderr excerpts short enough for an API error message.
_STDERR_TAIL_CHARS = 300

# ffmpeg -i fallback: "  Duration: 00:00:01.23, start: 0.000000, bitrate: ..."
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
# ffmpeg -i fallback: "Stream #0:0: Audio: mp3, 44100 Hz, stereo, 128 kb/s"
_SAMPLE_RATE_RE = re.compile(r"Audio:.*?(\d{4,6})\s*Hz")


@dataclass(frozen=True)
class AudioProbe:
    """Measured properties of one audio file."""

    sample_rate: int
    duration_ms: int

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0


def resolve_ffprobe(settings: Settings) -> str:
    """Locate the ffprobe binary (PATH, absolute path or FFPROBE_PATH)."""
    candidate = settings.ffprobe_path
    found = shutil.which(candidate)
    if found:
        return found
    path = Path(candidate)
    if path.is_file():
        return str(path)
    raise AudioProbeError(
        f"ffprobe was not found ('{candidate}'). Install FFmpeg (which "
        "includes ffprobe) and/or set FFPROBE_PATH in .env to the full path "
        "of the ffprobe binary."
    )


def probe_audio(path: Path, settings: Optional[Settings] = None) -> AudioProbe:
    """Read `sample_rate` and `duration_ms` from `path` (read-only)."""
    settings = settings or get_settings()
    if not path.is_file():
        raise AudioProbeError(f"the audio file '{path.name}' does not exist")

    try:
        probe = _probe_with_ffprobe(path, settings)
    except AudioProbeError:
        # ffprobe missing/broken -> single ffmpeg -i fallback.
        probe = _probe_with_ffmpeg(path, settings)
    if probe is None:
        raise AudioProbeError(
            f"could not detect an audio stream in '{path.name}'"
        )
    return probe


def _probe_with_ffprobe(path: Path, settings: Settings) -> Optional[AudioProbe]:
    ffprobe = resolve_ffprobe(settings)
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=settings.ffmpeg_timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioProbeError(f"ffprobe could not read '{path.name}': {exc}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip().replace("\n", " ")
        tail = stderr[-_STDERR_TAIL_CHARS:] if stderr else "no error output"
        raise AudioProbeError(f"ffprobe exited with an error: {tail}")

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AudioProbeError("ffprobe returned invalid JSON") from exc

    streams = payload.get("streams") or []
    stream = streams[0] if streams and isinstance(streams[0], dict) else {}
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}

    sample_rate = _positive_int(stream.get("sample_rate"))
    if sample_rate is None:
        raise AudioProbeError(
            f"ffprobe found no audio stream in '{path.name}'"
        )

    duration = _positive_float(stream.get("duration")) or _positive_float(
        fmt.get("duration")
    )
    if duration is None:
        raise AudioProbeError(
            f"ffprobe could not determine the duration of '{path.name}'"
        )
    return AudioProbe(sample_rate=sample_rate, duration_ms=_to_ms(duration))


def _probe_with_ffmpeg(path: Path, settings: Settings) -> Optional[AudioProbe]:
    """Fallback: parse `ffmpeg -i` banner output (no ffprobe required)."""
    from app.services.audio_composer import resolve_ffmpeg

    ffmpeg = resolve_ffmpeg(settings)
    command = [ffmpeg, "-hide_banner", "-i", str(path)]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=settings.ffmpeg_timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioProbeError(f"ffmpeg could not read '{path.name}': {exc}") from exc

    # `ffmpeg -i` without an output always exits non-zero; its stderr carries
    # the stream info we need, so the exit code is deliberately ignored.
    banner = f"{result.stdout or ''}{result.stderr or ''}"

    duration_match = _DURATION_RE.search(banner)
    rate_match = _SAMPLE_RATE_RE.search(banner)
    if duration_match is None or rate_match is None:
        tail = " ".join(banner.split())[-_STDERR_TAIL_CHARS:]
        raise AudioProbeError(
            f"ffmpeg could not describe the audio stream of '{path.name}'"
            + (f": {tail}" if tail else "")
        )

    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return AudioProbe(
        sample_rate=int(rate_match.group(1)),
        duration_ms=_to_ms(duration),
    )


def _positive_int(value: object) -> Optional[int]:
    try:
        number = int(str(value))  # ffprobe returns strings ("44100")
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_float(value: object) -> Optional[float]:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    # ffprobe prints "N/A" for some containers -> rejected by float() above.
    return number if number > 0 else None


def _to_ms(seconds: float) -> int:
    """Seconds -> milliseconds, rounded HALF_UP with Decimal (never binary float)."""
    from decimal import ROUND_HALF_UP, Decimal

    return int(
        (Decimal(str(seconds)) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
