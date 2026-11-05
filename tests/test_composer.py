"""FFmpeg composition tests.

The composition tests run the **real** ffmpeg binary (found on PATH or via
FFMPEG_PATH) against locally generated sine-wave MP3s — no API credits are
involved. They are skipped automatically when ffmpeg is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.config import get_settings
from app.services.audio_composer import (
    FFmpegAudioComposer,
    Segment,
    resolve_ffmpeg,
)
from app.services.errors import FFmpegCompositionError, FFmpegNotFoundError

settings = get_settings()


def _ffmpeg_available() -> bool:
    try:
        resolve_ffmpeg(settings)
        return True
    except FFmpegNotFoundError:
        return False


requires_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(), reason="ffmpeg is not installed / not on PATH"
)


def _make_tone(path: Path, frequency: int = 440, duration: float = 0.3) -> None:
    """Generate a short MP3 locally with ffmpeg (no external assets)."""
    ffmpeg = resolve_ffmpeg(settings)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency={frequency}:duration={duration}",
            "-c:a", "libmp3lame",
            "-q:a", "5",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )


def _duration_seconds(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        pytest.skip("ffprobe is not available")
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return float(result.stdout.strip())


# --------------------------------------------------------------------------- #
# Error handling (no ffmpeg run needed)
# --------------------------------------------------------------------------- #
def test_no_segments_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FFmpegCompositionError) as exc:
        FFmpegAudioComposer(settings).compose([], tmp_path / "out.mp3")
    assert "no audio segments" in str(exc.value)


def test_missing_segment_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FFmpegCompositionError) as exc:
        FFmpegAudioComposer(settings).compose(
            [Segment(path=tmp_path / "does_not_exist.mp3", pause_after_ms=100)],
            tmp_path / "out.mp3",
        )
    assert "missing segment" in str(exc.value)


def test_missing_ffmpeg_binary_is_reported(tmp_path: Path) -> None:
    import dataclasses

    # compose() validates the segment files before locating ffmpeg.
    segment = tmp_path / "a.mp3"
    segment.write_bytes(b"stub")

    broken = dataclasses.replace(
        settings, ffmpeg_path="ffmpeg-not-installed-xyz"
    )
    with pytest.raises(FFmpegNotFoundError) as exc:
        FFmpegAudioComposer(broken).compose(
            [Segment(path=segment)], tmp_path / "out.mp3"
        )
    assert "FFMPEG_PATH" in str(exc.value)


def test_resolve_ffmpeg_finds_a_binary() -> None:
    found = resolve_ffmpeg(settings)
    assert Path(found).is_file()


# --------------------------------------------------------------------------- #
# Real composition
# --------------------------------------------------------------------------- #
@requires_ffmpeg
def test_compose_merges_segments_in_order_with_pauses(tmp_path: Path) -> None:
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    # Three 0.3s tones -> 0.9s of speech in total.
    paths = []
    for index in range(3):
        tone = segments_dir / f"paragraph_{index + 1:04d}.mp3"
        _make_tone(tone, frequency=330 + (index * 110), duration=0.3)
        paths.append(tone)

    pauses = [500, 300, 0]  # 0.8s of inserted silence
    segments = [
        Segment(path=path, pause_after_ms=pause)
        for path, pause in zip(paths, pauses)
    ]

    output = tmp_path / "audiobook.mp3"
    FFmpegAudioComposer(settings).compose(segments, output)

    assert output.is_file()
    assert output.stat().st_size > 0

    total = _duration_seconds(output)
    source_total = sum(_duration_seconds(path) for path in paths)
    expected = source_total + (sum(pauses) / 1000.0)

    # Nothing may be trimmed: the result must contain every speech clip.
    assert total >= source_total - 0.05, f"{total:.3f}s < speech {source_total:.3f}s"
    # ...plus the pauses MiMo asked for (mp3 padding gives some slack).
    assert abs(total - expected) < 0.45, f"{total:.3f}s vs expected {expected:.3f}s"


@requires_ffmpeg
def test_compose_does_not_trim_speech(tmp_path: Path) -> None:
    """A single segment must survive byte-for-byte in duration."""
    tone = tmp_path / "only.mp3"
    _make_tone(tone, duration=0.5)

    output = tmp_path / "out.mp3"
    FFmpegAudioComposer(settings).compose(
        [Segment(path=tone, pause_after_ms=0)], output
    )

    assert _duration_seconds(output) >= 0.5


@requires_ffmpeg
def test_compose_creates_parent_directory(tmp_path: Path) -> None:
    tone = tmp_path / "only.mp3"
    _make_tone(tone, duration=0.2)

    output = tmp_path / "deep" / "nested" / "audiobook.mp3"
    FFmpegAudioComposer(settings).compose([Segment(path=tone)], output)

    assert output.is_file()
