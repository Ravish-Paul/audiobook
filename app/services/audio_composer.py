"""FFmpeg audio composition: paragraph segments + pauses -> audiobook.mp3

Strategy (single ffmpeg invocation, no intermediate silence files):

    for segment i (delay D_i = pause of the PREVIOUS paragraph):
        [i:a]aresample=44100, aformat=stereo, adelay=D_i:all=1 -> [a_i]
    [a0][a1]...concat=n=N:v=0:a=1 -> apad(last pause) -> libmp3lame

Two details matter:

* the concat inputs must be labelled explicitly — upstream outputs are
  labelled too, so ffmpeg cannot auto-link them;
* `adelay` pads the *beginning* of each clip and `concat` then sums the
  clips, so each clip gets only the pause that precedes it (using
  cumulative delays would double-count every pause).

`adelay` only *prepends* silence, so the beginning and the ending of every
speech clip are preserved (nothing is ever trimmed) and the original
paragraph order is kept.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from app.config import Settings, get_settings
from app.services.errors import FFmpegCompositionError, FFmpegNotFoundError

# Keep stderr excerpts short enough for an API error message.
_STDERR_TAIL_CHARS = 500

# Normalise every input so the concat filter sees identical streams.
_INPUT_CHAIN = "aresample=44100,aformat=channel_layouts=stereo"

# Headerless intermediate formats fetched from ElevenLabs (pcm_*, ulaw_*,
# alaw_*): ffmpeg cannot probe them, so the decoder needs explicit hints.
_RAW_FORMAT_BY_SUFFIX = {
    ".pcm": "s16le",
    ".ulaw": "mulaw",
    ".alaw": "alaw",
}


def _raw_input_args(segment_path: Path, settings: Settings) -> list[str]:
    """`-f/-ar/-ac` flags ffmpeg needs before `-i` for raw intermediates."""
    suffix = segment_path.suffix.lower()
    raw_format = _RAW_FORMAT_BY_SUFFIX.get(suffix)
    if raw_format is None:
        return []
    # Sample rate comes from the configured format (e.g. pcm_44100 -> 44100).
    configured = settings.elevenlabs_segment_output_format
    rate = configured.rsplit("_", 1)[-1]
    if not rate.isdigit():
        rate = "44100"
    return ["-f", raw_format, "-ar", rate, "-ac", "2"]


@dataclass(frozen=True)
class Segment:
    """One generated paragraph file plus the silence that follows it."""

    path: Path
    pause_after_ms: int = 0


def resolve_ffmpeg(settings: Settings) -> str:
    """Locate the ffmpeg binary (PATH, absolute path or FFMPEG_PATH)."""
    candidate = settings.ffmpeg_path
    found = shutil.which(candidate)
    if found:
        return found
    path = Path(candidate)
    if path.is_file():
        return str(path)
    raise FFmpegNotFoundError(candidate)


class FFmpegAudioComposer:
    """Assembles ordered MP3 segments into a single audiobook file."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ #
    def compose(self, segments: Sequence[Segment], output_path: Path) -> None:
        """Write `output_path` from `segments` (in order, with pauses)."""
        if not segments:
            raise FFmpegCompositionError("there are no audio segments to combine")

        for segment in segments:
            if not segment.path.is_file():
                raise FFmpegCompositionError(
                    f"missing segment '{segment.path.name}'"
                )

        ffmpeg = resolve_ffmpeg(self.settings)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        inputs: list[str] = []
        filters: list[str] = []
        concat_inputs: list[str] = []
        # Delay of the NEXT clip = pause that follows the current one.
        delay_ms = 0
        for index, segment in enumerate(segments):
            # Raw PCM/μ-law/A-law intermediates need explicit decoder flags.
            inputs += _raw_input_args(segment.path, self.settings)
            inputs += ["-i", str(segment.path)]
            filters.append(
                f"[{index}:a]{_INPUT_CHAIN},adelay={delay_ms}:all=1[a{index}]"
            )
            concat_inputs.append(f"[a{index}]")
            delay_ms = max(0, int(segment.pause_after_ms))

        last_pause_ms = max(0, int(segments[-1].pause_after_ms))
        concat_label = "[cat]" if last_pause_ms else "[out]"
        filter_graph = (
            ";".join(filters)
            + ";"
            + "".join(concat_inputs)
            + f"concat=n={len(segments)}:v=0:a=1{concat_label}"
        )
        if last_pause_ms:
            # Honour the final paragraph's pause as trailing silence.
            filter_graph += (
                f";[cat]apad=pad_dur={last_pause_ms / 1000:.3f}s[out]"
            )

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            *inputs,
            "-filter_complex", filter_graph,
            "-map", "[out]",
            "-vn",
            "-c:a", "libmp3lame",
            "-q:a", "2",
            str(output_path),
        ]

        self._run(command, output_path)

    # ------------------------------------------------------------------ #
    def _run(self, command: list[str], output_path: Path) -> None:
        timeout = self.settings.ffmpeg_timeout_seconds
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise FFmpegNotFoundError(command[0]) from exc
        except subprocess.TimeoutExpired as exc:
            raise FFmpegCompositionError(
                f"it timed out after {timeout} seconds"
            ) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip().replace("\n", " ")
            tail = stderr[-_STDERR_TAIL_CHARS:] if stderr else "no error output"
            raise FFmpegCompositionError(f"ffmpeg exited with an error: {tail}")

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise FFmpegCompositionError(
                f"'{output_path.name}' was not created"
            )
