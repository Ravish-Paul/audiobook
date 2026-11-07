"""Domain errors raised by the service layer.

The API layer maps them onto HTTP status codes in `app/main.py`.
"""

from __future__ import annotations


class ProjectError(Exception):
    """Base class for project related errors."""

    status_code = 500


class ProjectNotFoundError(ProjectError):
    status_code = 404

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(f"Project '{project_id}' not found")


class ScriptValidationError(ProjectError):
    """Wrong file type / not decodable as UTF-8."""

    status_code = 400


class ScriptTooLargeError(ProjectError):
    status_code = 413

    def __init__(self, size: int, limit: int) -> None:
        self.size = size
        self.limit = limit
        super().__init__(
            f"Script is {size} bytes which exceeds the {limit} byte limit"
        )


class ScriptNotFoundError(ProjectError):
    """The project exists but no script.txt has been uploaded yet."""

    status_code = 404

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(
            f"No script has been uploaded for project '{project_id}' yet"
        )


class EmptyScriptError(ProjectError):
    """The stored script contains no speakable text."""

    status_code = 400

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(
            f"The script for project '{project_id}' is empty — upload some text first"
        )


class ScriptTooLongError(ProjectError):
    """The script exceeds the ElevenLabs per-request character limit."""

    status_code = 400

    def __init__(self, length: int, limit: int) -> None:
        self.length = length
        self.limit = limit
        super().__init__(
            f"Script has {length} characters which exceeds the ElevenLabs "
            f"per-request limit of {limit} characters"
        )


class AudioNotFoundError(ProjectError):
    status_code = 404

    def __init__(self, project_id: str, reason: str = "not generated yet") -> None:
        self.project_id = project_id
        super().__init__(
            f"No audio available for project '{project_id}' ({reason})"
        )


class AudioSaveError(ProjectError):
    """The MP3 could not be written to local storage."""

    status_code = 500


# --------------------------------------------------------------------------- #
# ElevenLabs / TTS
# --------------------------------------------------------------------------- #
class ElevenLabsError(ProjectError):
    """Base class for ElevenLabs integration errors."""


class ElevenLabsConfigError(ElevenLabsError):
    """Missing API key or voice ID (server misconfiguration)."""

    status_code = 500


class ElevenLabsAuthError(ElevenLabsError):
    """ElevenLabs rejected the API key (401/403)."""

    status_code = 401


class ElevenLabsVoiceError(ElevenLabsError):
    """Invalid voice ID / model ID / rejected request (400/404/422)."""

    status_code = 400


class ElevenLabsRateLimitError(ElevenLabsError):
    """ElevenLabs rate limit reached (429)."""

    status_code = 429


class ElevenLabsTimeoutError(ElevenLabsError):
    """No response from ElevenLabs within the configured timeout."""

    status_code = 504


class ElevenLabsAPIError(ElevenLabsError):
    """Any other ElevenLabs failure / network problem -> bad gateway."""

    status_code = 502


# --------------------------------------------------------------------------- #
# Xiaomi MiMo (narration director)
# --------------------------------------------------------------------------- #
class MiMoError(ProjectError):
    """Base class for Xiaomi MiMo integration errors."""


class MiMoConfigError(MiMoError):
    """Missing MIMO_API_KEY (server misconfiguration)."""

    status_code = 500


class MiMoAuthError(MiMoError):
    """MiMo rejected the API key (401/403)."""

    status_code = 401


class MiMoRateLimitError(MiMoError):
    status_code = 429


class MiMoTimeoutError(MiMoError):
    status_code = 504


class MiMoAPIError(MiMoError):
    """Any other MiMo failure / network problem."""

    status_code = 502


class NarrationPlanNotFoundError(ProjectError):
    status_code = 404

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(
            f"No narration plan for project '{project_id}' yet — "
            "run POST /projects/{project_id}/analyze-narration first"
        )


class NarrationPlanError(ProjectError):
    """MiMo returned a plan that failed validation after all retries.

    The response never generates audio; `problems` explains exactly what was
    wrong with the model output and `raw_responses` preserves the original
    raw MiMo outputs for debugging (also written to `mimo_debug/`).
    """

    status_code = 502

    def __init__(
        self,
        problems: list[str],
        attempts: int,
        raw_responses: list[str] | None = None,
        normalization_notes: list[str] | None = None,
    ) -> None:
        self.problems = problems
        self.attempts = attempts
        self.raw_responses = list(raw_responses or [])
        self.normalization_notes = list(normalization_notes or [])
        preview = " | ".join(problems[:6])
        more = "" if len(problems) <= 6 else f" (+{len(problems) - 6} more)"
        super().__init__(
            "Unable to prepare the narration plan. Please try again — no audio "
            f"was generated. (MiMo returned an invalid narration plan after "
            f"{attempts} attempt(s). Problems: {preview}{more})"
        )


# --------------------------------------------------------------------------- #
# FFmpeg composition / segment generation
# --------------------------------------------------------------------------- #
class FFmpegError(ProjectError):
    """Base class for audio composition errors."""

    status_code = 500


class FFmpegNotFoundError(FFmpegError):
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(
            f"FFmpeg was not found ('{path}'). Install FFmpeg and/or set "
            "FFMPEG_PATH in .env to the full path of the ffmpeg binary."
        )


class FFmpegCompositionError(FFmpegError):
    def __init__(self, message: str) -> None:
        super().__init__(f"FFmpeg could not assemble the audiobook: {message}")


class AudioProbeError(FFmpegError):
    """ffprobe/ffmpeg could not read the sample rate or duration."""

    def __init__(self, message: str) -> None:
        super().__init__(f"audio probing failed: {message}")


class StaleNarrationPlanError(ProjectError):
    """The script changed after the plan was analysed (or vice versa)."""

    status_code = 409

    def __init__(self, project_id: str, problems: list[str]) -> None:
        self.project_id = project_id
        self.problems = problems
        preview = " | ".join(problems[:4])
        super().__init__(
            f"The narration plan for project '{project_id}' no longer matches "
            f"the uploaded script ({preview}). Re-run "
            f"POST /projects/{project_id}/analyze-narration"
        )


class StaleNarrationPreferencesError(ProjectError):
    """The narration preferences changed after this plan was generated."""

    status_code = 409

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(
            f"The narration preferences of project '{project_id}' changed after "
            "this narration plan was generated. Re-run "
            f"POST /projects/{project_id}/analyze-narration so the plan is "
            "rebuilt under the new preferences (the plan hash changed, so old "
            "segments must not be reused)."
        )


class InvalidNarrationPreferencesError(ProjectError):
    """Preferences that cannot apply to this project's script."""

    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(f"Invalid narration preferences: {message}")


class ParagraphPreviewError(ProjectError):
    """The preview requested a paragraph that the script does not contain."""

    status_code = 400

    def __init__(self, paragraph_index: int, paragraph_count: int) -> None:
        self.paragraph_index = paragraph_index
        super().__init__(
            f"paragraph_index {paragraph_index} is outside this script "
            f"(0..{paragraph_count - 1})"
        )


class SegmentGenerationError(ProjectError):
    """One paragraph failed to synthesise; finished segments are preserved."""

    status_code = 502

    def __init__(
        self, failed_index: int, completed: int, total: int, reason: str
    ) -> None:
        self.failed_index = failed_index
        self.completed = completed
        self.total = total
        super().__init__(
            f"Paragraph {failed_index} failed after {completed}/{total} segment(s) "
            f"were generated; completed segments were preserved. Reason: {reason}"
        )


# --------------------------------------------------------------------------- #
# Character casting / multi-voice narration
# --------------------------------------------------------------------------- #
class CharacterCastNotFoundError(ProjectError):
    """No character analysis exists yet for this project."""

    status_code = 404

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(
            f"No character cast for project '{project_id}' yet — run "
            "POST /projects/{project_id}/analyze-characters first"
        )


class CharacterAnalysisError(ProjectError):
    """MiMo returned a character analysis that failed validation after retries."""

    status_code = 502

    def __init__(self, problems: list[str], attempts: int) -> None:
        self.problems = problems
        self.attempts = attempts
        preview = " | ".join(problems[:6])
        more = "" if len(problems) <= 6 else f" (+{len(problems) - 6} more)"
        super().__init__(
            f"MiMo returned an invalid character analysis after {attempts} "
            f"attempt(s); nothing was saved. Problems: {preview}{more}"
        )


class CastingNotApprovedError(ProjectError):
    """The voice cast is incomplete or was not approved yet."""

    status_code = 409

    def __init__(self, missing: list[str], *, generating: bool = False) -> None:
        self.missing = missing
        if not missing:
            super().__init__(
                "The voice cast has not been approved yet — run "
                "POST /projects/{project_id}/voice-casting/approve before "
                "generating audio"
            )
            return
        preview = ", ".join(missing[:8])
        action = (
            "Select and approve voices before generating audio"
            if generating
            else "Select a voice for every speaking role"
        )
        super().__init__(f"{action}. Still missing a voice for: {preview}")


class VoiceSelectionError(ProjectError):
    """Invalid manual voice selection (unknown speaker / unavailable voice)."""

    status_code = 400


class VoiceNotAssignedError(ProjectError):
    """Generation found a speaker without a locked voice."""

    status_code = 409

    def __init__(self, speaker_id: str) -> None:
        self.speaker_id = speaker_id
        super().__init__(
            f"Speaker '{speaker_id}' has no locked voice — select a voice via "
            "POST /projects/{project_id}/voice-casting/select"
        )


class MultiVoicePlanNotFoundError(ProjectError):
    status_code = 404

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(
            f"No multi-voice plan for project '{project_id}' yet — run "
            "POST /projects/{project_id}/create-multi-voice-plan first"
        )


class MultiVoicePlanError(ProjectError):
    """MiMo returned an utterance plan that failed validation after retries."""

    status_code = 502

    def __init__(self, problems: list[str], attempts: int) -> None:
        self.problems = problems
        self.attempts = attempts
        preview = " | ".join(problems[:6])
        more = "" if len(problems) <= 6 else f" (+{len(problems) - 6} more)"
        super().__init__(
            f"MiMo returned an invalid multi-voice plan after {attempts} "
            f"attempt(s); no audio was generated. Problems: {preview}{more}"
        )


class StaleMultiVoicePlanError(ProjectError):
    """The script changed after the multi-voice plan was created."""

    status_code = 409

    def __init__(self, project_id: str, problems: list[str]) -> None:
        self.project_id = project_id
        self.problems = problems
        preview = " | ".join(problems[:4])
        super().__init__(
            f"The multi-voice plan for project '{project_id}' no longer matches "
            f"the uploaded script ({preview}). Re-run "
            "POST /projects/{project_id}/create-multi-voice-plan"
        )


class UtteranceGenerationError(ProjectError):
    """One utterance failed to synthesise; finished utterances are preserved."""

    status_code = 502

    def __init__(
        self, failed_id: str, completed: int, total: int, reason: str
    ) -> None:
        self.failed_id = failed_id
        self.completed = completed
        self.total = total
        super().__init__(
            f"Utterance {failed_id} failed after {completed}/{total} utterance(s) "
            f"were generated; completed utterances were preserved. "
            f"Reason: {reason}"
        )


# --------------------------------------------------------------------------- #
# pyannoteAI (audio analysis: diarization + word-level transcription)
# --------------------------------------------------------------------------- #
class PyannoteError(ProjectError):
    """Base class for pyannoteAI integration errors."""


class PyannoteConfigError(PyannoteError):
    """Missing PYANNOTE_API_KEY (server misconfiguration)."""

    status_code = 500


class PyannoteAuthError(PyannoteError):
    """pyannoteAI rejected the API key (401/403)."""

    status_code = 401


class PyannoteRequestError(PyannoteError):
    """pyannoteAI rejected our request payload (400/422)."""

    status_code = 400


class PyannoteRateLimitError(PyannoteError):
    """pyannoteAI rate limit reached (429)."""

    status_code = 429


class PyannoteTimeoutError(PyannoteError):
    """No response from pyannoteAI within the configured timeout."""

    status_code = 504


class PyannoteAPIError(PyannoteError):
    """Any other pyannoteAI failure / network problem -> bad gateway."""

    status_code = 502


class AudioAnalysisNotFoundError(ProjectError):
    """No audio analysis exists yet for this project."""

    status_code = 404

    def __init__(self, project_id: str, reason: str = "not started yet") -> None:
        self.project_id = project_id
        super().__init__(
            f"No audio analysis available for project '{project_id}' ({reason})"
        )


class AudioAnalysisIncompleteError(ProjectError):
    """The analysis exists but has not completed successfully yet."""

    status_code = 409

    def __init__(self, project_id: str, status: str, error: str = "") -> None:
        self.project_id = project_id
        self.status = status
        detail = f" (last error: {error})" if error else ""
        super().__init__(
            f"The audio analysis for project '{project_id}' is '{status}' — "
            f"the normalized result is not available yet{detail}. Check "
            "GET /projects/{project_id}/audio-analysis/status first"
        )


class AudioAnalysisNormalizationError(ProjectError):
    """pyannoteAI succeeded but its output could not be normalized.

    The raw provider response is always preserved on disk first, so the
    failure can be debugged without re-running (and re-paying for) the job.
    """

    status_code = 502

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        preview = " | ".join(problems[:6])
        more = "" if len(problems) <= 6 else f" (+{len(problems) - 6} more)"
        super().__init__(
            "The pyannoteAI output could not be normalized into the internal "
            f"audio timeline. Problems: {preview}{more}"
        )
