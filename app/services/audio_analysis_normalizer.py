"""Timestamp normalization: pyannoteAI seconds -> internal audio timeline.

Why Decimal instead of float?
    Pyannote returns fractional seconds such as 0.748932. The obvious
    `round(0.748932 * 1000)` works on binary floats only by luck
    (2.675 * 1000 -> 2674.9999999999995) and Python's banker's rounding
    can go the wrong way at .5 boundaries (1.0005 s -> 1000 ms instead of
    1001 ms). Every conversion here therefore goes through
    `decimal.Decimal` with `ROUND_HALF_UP`:

        0.748932 s -> Decimal("0.748932") * 1000 -> 748.932 -> 749 ms
        1.0005   s -> Decimal("1.0005")   * 1000 -> 1000.5  -> 1001 ms

Sample indexes use the REAL sample rate of the source file (detected with
ffprobe), never a hardcoded 44100 Hz:

        start_sample = ROUND_HALF_UP(start_seconds * sample_rate)

Honesty note: converting to sample indexes does NOT improve pyannote's
underlying timing accuracy — it only re-expresses the same timestamps on
the sample grid so later mixing stages can align audio exactly.

Validation is strict (see `normalize_analysis_output`) but NEVER reorders,
rewrites or drops *valid* records: overlapping speaker intervals are
legitimate (two people can speak at once) and are always kept. Invalid
records are excluded from the normalized output and reported in
`validation_issues`; the raw provider file is preserved untouched either
way, so nothing is lost for debugging.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Optional

from app.schemas.audio_analysis import (
    AudioAnalysisNormalized,
    NormalizedSpeaker,
    NormalizedSpeakerTurn,
    NormalizedTurnTranscription,
    NormalizedWord,
    ValidationIssue,
)
from app.services.errors import AudioAnalysisNormalizationError

# Reasons recorded in `validation_issues` (stable, machine readable).
REASON_MISSING_TIMESTAMP = "missing_timestamp"
REASON_NON_NUMERIC_TIMESTAMP = "non_numeric_timestamp"
REASON_NEGATIVE_START = "negative_start_timestamp"
REASON_END_NOT_AFTER_START = "end_not_greater_than_start"
REASON_EMPTY_TEXT = "empty_text"
REASON_MISSING_SPEAKER = "missing_speaker_id"
REASON_EXCEEDS_DURATION = "exceeds_audio_duration"
REASON_INVALID_SAMPLE_INDEX = "invalid_sample_index"
REASON_MALFORMED_RECORD = "malformed_record"

# Structural problems -> normalization fails (raw output still preserved).
PROBLEM_NOT_AN_OBJECT = "provider output is not a JSON object"
PROBLEM_MISSING_DIARIZATION = "provider output has no 'diarization' array"
PROBLEM_MISSING_WORDS = (
    "provider output has no 'wordLevelTranscription' array "
    "(transcription was requested but produced nothing usable)"
)
PROBLEM_NO_USABLE_OUTPUT = (
    "the job completed but produced no usable word or speaker records"
)


def seconds_to_ms(value: Any) -> int:
    """Fractional seconds -> integer milliseconds (Decimal ROUND_HALF_UP).

    Raises ValueError/TypeError for non-numeric input so callers can
    record a validation issue instead of inventing a timestamp.
    """
    seconds = _to_decimal(value)
    if seconds is None:
        raise ValueError(f"not a numeric timestamp: {value!r}")
    return int((seconds * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def seconds_to_samples(value: Any, sample_rate: int) -> int:
    """Fractional seconds -> sample index at `sample_rate` (HALF_UP)."""
    seconds = _to_decimal(value)
    if seconds is None:
        raise ValueError(f"not a numeric timestamp: {value!r}")
    if sample_rate <= 0:
        raise ValueError(f"invalid sample rate: {sample_rate!r}")
    samples = seconds * Decimal(int(sample_rate))
    return int(samples.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _to_decimal(value: Any) -> Optional[Decimal]:
    """Parse a provider timestamp into Decimal (None when impossible).

    Floats go through `str()` first: `repr(0.748932)` is the shortest
    round-trip form, so the decimal literal the provider sent is restored
    instead of inheriting binary float noise. Booleans are rejected.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            parsed = Decimal(value.strip())
        except (InvalidOperation, ValueError):
            return None
        return parsed if parsed.is_finite() else None
    return None


def _ms_to_samples(ms: int, sample_rate: int) -> int:
    """Milliseconds -> sample index (Decimal HALF_UP, consistent with seconds)."""
    samples = (Decimal(int(ms)) * Decimal(int(sample_rate))) / Decimal(1000)
    return int(samples.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


# --------------------------------------------------------------------------- #
# Record validation + conversion
# --------------------------------------------------------------------------- #
class _SpanResult:
    """Outcome of validating one provider record."""

    __slots__ = ("ok", "reason", "raw_start", "raw_end", "start_ms", "end_ms",
                 "start_sample", "end_sample")

    def __init__(self) -> None:
        self.ok = False
        self.reason = ""
        self.raw_start: float = 0.0
        self.raw_end: float = 0.0
        self.start_ms = 0
        self.end_ms = 0
        self.start_sample = 0
        self.end_sample = 0


def _validate_span(
    record: dict[str, Any],
    *,
    sample_rate: int,
    audio_duration_ms: int,
    duration_tolerance_ms: int,
    total_samples: int,
    sample_tolerance: int,
) -> _SpanResult:
    """Validate start/end/speaker of one record and convert the span.

    Returns a result with `ok=False` and a stable `reason` on the first
    violation. Overlap with other records is explicitly NOT checked —
    simultaneous speakers are valid.
    """
    result = _SpanResult()

    raw_start = record.get("start")
    raw_end = record.get("end")
    if raw_start is None or raw_end is None:
        result.reason = REASON_MISSING_TIMESTAMP
        return result

    start_decimal = _to_decimal(raw_start)
    end_decimal = _to_decimal(raw_end)
    if start_decimal is None or end_decimal is None:
        result.reason = REASON_NON_NUMERIC_TIMESTAMP
        return result

    if start_decimal < 0:
        result.reason = REASON_NEGATIVE_START
        return result
    if end_decimal <= start_decimal:
        result.reason = REASON_END_NOT_AFTER_START
        return result

    start_ms = seconds_to_ms(start_decimal)
    end_ms = seconds_to_ms(end_decimal)

    # Duration check (documented tolerance absorbs MP3 padding/encoder lag).
    limit_ms = audio_duration_ms + max(0, duration_tolerance_ms)
    if start_ms > limit_ms or end_ms > limit_ms:
        result.reason = REASON_EXCEEDS_DURATION
        return result

    start_sample = seconds_to_samples(start_decimal, sample_rate)
    end_sample = seconds_to_samples(end_decimal, sample_rate)
    limit_sample = total_samples + max(0, sample_tolerance)
    if start_sample < 0 or end_sample <= start_sample or end_sample > limit_sample:
        result.reason = REASON_INVALID_SAMPLE_INDEX
        return result

    result.ok = True
    result.raw_start = float(start_decimal)
    result.raw_end = float(end_decimal)
    result.start_ms = start_ms
    result.end_ms = end_ms
    result.start_sample = start_sample
    result.end_sample = end_sample
    return result


def _speaker_of(record: dict[str, Any]) -> str:
    speaker = record.get("speaker")
    return speaker.strip() if isinstance(speaker, str) else ""


def _text_of(record: dict[str, Any]) -> str:
    text = record.get("text")
    return text if isinstance(text, str) else ""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def normalize_analysis_output(
    output: Any,
    *,
    project_id: str,
    analysis_id: str,
    model_id: str,
    transcription_model: str,
    sample_rate: int,
    audio_duration_ms: int,
    duration_tolerance_ms: int = 500,
    generated_at: Optional[datetime] = None,
) -> AudioAnalysisNormalized:
    """Validate a successful pyannoteAI `output` and build the timeline.

    Raises AudioAnalysisNormalizationError (502) when the response has an
    unexpected structure, misses an expected transcription array, or holds
    no usable records — the caller must already have saved the raw file.
    """
    problems: list[str] = []
    issues: list[ValidationIssue] = []

    if not isinstance(output, dict):
        raise AudioAnalysisNormalizationError([PROBLEM_NOT_AN_OBJECT])

    diarization = output.get("diarization")
    if not isinstance(diarization, list):
        raise AudioAnalysisNormalizationError([PROBLEM_MISSING_DIARIZATION])

    words_raw = output.get("wordLevelTranscription")
    if not isinstance(words_raw, list) or not words_raw:
        raise AudioAnalysisNormalizationError([PROBLEM_MISSING_WORDS])

    if sample_rate <= 0 or audio_duration_ms <= 0:
        raise AudioAnalysisNormalizationError(
            [f"invalid source audio metadata (sample_rate={sample_rate}, "
             f"audio_duration_ms={audio_duration_ms})"]
        )

    total_samples = _ms_to_samples(audio_duration_ms, sample_rate)
    sample_tolerance = _ms_to_samples(duration_tolerance_ms, sample_rate)

    # --- Speaker turns (diarization) — overlaps are preserved ----------- #
    turns: list[NormalizedSpeakerTurn] = []
    for index, item in enumerate(diarization):
        if not isinstance(item, dict):
            issues.append(ValidationIssue(
                record_type="speaker_turn", index=index,
                reason=REASON_MALFORMED_RECORD,
            ))
            continue
        speaker = _speaker_of(item)
        if not speaker:
            issues.append(ValidationIssue(
                record_type="speaker_turn", index=index,
                reason=REASON_MISSING_SPEAKER,
            ))
            continue
        span = _validate_span(
            item,
            sample_rate=sample_rate,
            audio_duration_ms=audio_duration_ms,
            duration_tolerance_ms=duration_tolerance_ms,
            total_samples=total_samples,
            sample_tolerance=sample_tolerance,
        )
        if not span.ok:
            issues.append(ValidationIssue(
                record_type="speaker_turn", index=index, reason=span.reason,
            ))
            continue
        turns.append(NormalizedSpeakerTurn(
            speaker_id=speaker,
            raw_start_seconds=span.raw_start,
            raw_end_seconds=span.raw_end,
            start_ms=span.start_ms,
            end_ms=span.end_ms,
            start_sample=span.start_sample,
            end_sample=span.end_sample,
        ))

    # --- Word-level transcription --------------------------------------- #
    words: list[NormalizedWord] = []
    for index, item in enumerate(words_raw):
        if not isinstance(item, dict):
            issues.append(ValidationIssue(
                record_type="word", index=index,
                reason=REASON_MALFORMED_RECORD,
            ))
            continue
        text = _text_of(item)
        if not text.strip():
            issues.append(ValidationIssue(
                record_type="word", index=index, reason=REASON_EMPTY_TEXT,
            ))
            continue
        speaker = _speaker_of(item)
        if not speaker:
            issues.append(ValidationIssue(
                record_type="word", index=index,
                reason=REASON_MISSING_SPEAKER,
            ))
            continue
        span = _validate_span(
            item,
            sample_rate=sample_rate,
            audio_duration_ms=audio_duration_ms,
            duration_tolerance_ms=duration_tolerance_ms,
            total_samples=total_samples,
            sample_tolerance=sample_tolerance,
        )
        if not span.ok:
            issues.append(ValidationIssue(
                record_type="word", index=index, reason=span.reason,
            ))
            continue
        words.append(NormalizedWord(
            text=text,
            speaker_id=speaker,
            raw_start_seconds=span.raw_start,
            raw_end_seconds=span.raw_end,
            start_ms=span.start_ms,
            end_ms=span.end_ms,
            start_sample=span.start_sample,
            end_sample=span.end_sample,
        ))

    # --- Turn-level transcription (OPTIONAL — treated safely) ----------- #
    turn_texts: list[NormalizedTurnTranscription] = []
    turn_texts_raw = output.get("turnLevelTranscription")
    if isinstance(turn_texts_raw, list):
        for index, item in enumerate(turn_texts_raw):
            if not isinstance(item, dict):
                issues.append(ValidationIssue(
                    record_type="turn_transcription", index=index,
                    reason=REASON_MALFORMED_RECORD,
                ))
                continue
            text = _text_of(item)
            speaker = _speaker_of(item)
            if not text.strip():
                issues.append(ValidationIssue(
                    record_type="turn_transcription", index=index,
                    reason=REASON_EMPTY_TEXT,
                ))
                continue
            if not speaker:
                issues.append(ValidationIssue(
                    record_type="turn_transcription", index=index,
                    reason=REASON_MISSING_SPEAKER,
                ))
                continue
            span = _validate_span(
                item,
                sample_rate=sample_rate,
                audio_duration_ms=audio_duration_ms,
                duration_tolerance_ms=duration_tolerance_ms,
                total_samples=total_samples,
                sample_tolerance=sample_tolerance,
            )
            if not span.ok:
                issues.append(ValidationIssue(
                    record_type="turn_transcription", index=index,
                    reason=span.reason,
                ))
                continue
            turn_texts.append(NormalizedTurnTranscription(
                text=text,
                speaker_id=speaker,
                raw_start_seconds=span.raw_start,
                raw_end_seconds=span.raw_end,
                start_ms=span.start_ms,
                end_ms=span.end_ms,
                start_sample=span.start_sample,
                end_sample=span.end_sample,
            ))

    # --- Usable output? -------------------------------------------------- #
    if not words and not turns:
        reasons = sorted({issue.reason for issue in issues}) or ["no_records"]
        problems.append(
            PROBLEM_NO_USABLE_OUTPUT + f" (rejections: {', '.join(reasons)})"
        )
    if problems:
        raise AudioAnalysisNormalizationError(problems)

    # --- Speakers (facts only — no invented confidence) ------------------ #
    speaker_ids = sorted(
        {turn.speaker_id for turn in turns} | {word.speaker_id for word in words}
    )
    speakers: list[NormalizedSpeaker] = []
    for speaker_id in speaker_ids:
        speaker_turns = [t for t in turns if t.speaker_id == speaker_id]
        speaker_words = [w for w in words if w.speaker_id == speaker_id]
        boundaries = [t.start_ms for t in speaker_turns] + [
            w.start_ms for w in speaker_words
        ]
        ends = [t.end_ms for t in speaker_turns] + [w.end_ms for w in speaker_words]
        speakers.append(NormalizedSpeaker(
            speaker_id=speaker_id,
            turn_count=len(speaker_turns),
            word_count=len(speaker_words),
            first_start_ms=min(boundaries) if boundaries else None,
            last_end_ms=max(ends) if ends else None,
        ))

    return AudioAnalysisNormalized(
        project_id=project_id,
        analysis_id=analysis_id,
        status="completed",
        model_id=model_id,
        transcription_model=transcription_model,
        time_unit="milliseconds",
        sample_rate=sample_rate,
        audio_duration_ms=audio_duration_ms,
        duration_tolerance_ms=max(0, duration_tolerance_ms),
        speakers=speakers,
        words=words,
        speaker_turns=turns,
        turn_level_transcriptions=turn_texts,
        validation_issues=issues,
        generated_at=generated_at or datetime.now(timezone.utc),
    )
