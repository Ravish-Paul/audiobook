"""Audio generation service: stored script -> speech -> local audiobook.

Two generation modes:

* **narration_plan** (new) — when `narration_plan.json` exists, every
  paragraph is generated separately using the voice settings MiMo decided
  (and its inline `tts_text` audio tags), then FFmpeg combines the segments
  with MiMo's `pause_after_ms` gaps into `audio/audiobook.mp3`.
* **single_pass** (legacy, backwards compatible) — the whole script is sent
  to ElevenLabs in one request, exactly as before.

Eleven v4 continuity rules implemented here:

* neighbouring TEXT (`previous_text` / `next_text`) provides context for
  ordinary first-time paragraph generation;
* stored ElevenLabs **request IDs** are captured from every response and
  persisted per segment; when a segment is regenerated while its neighbours
  already exist (same plan), those IDs provide continuity instead of text —
  never both for the same side (the client enforces the XOR);
* request IDs are only reused when they are fresh (< ~2 hours), from the
  same model and from the same plan.

Route handlers stay thin: they only call into this service.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app import plan_identity
from app.config import Settings, get_settings, language_to_iso
from app.security import current_generation_id
from app.services.script_limits import check_script_limits
from app.schemas.audio import AudioResponse
from app.schemas.multi_voice import MultiVoicePlan, Utterance
from app.schemas.narration import NarrationPlan, ParagraphPlan
from app.services.audio_composer import FFmpegAudioComposer, Segment
from app.services.elevenlabs_client import ElevenLabsClient, SynthesisContext
from app.services.errors import (
    AudioNotFoundError,
    AudioSaveError,
    CastingNotApprovedError,
    CharacterCastNotFoundError,
    FFmpegError,
    ProjectError,
    ScriptTooLongError,
    SegmentGenerationError,
    StaleMultiVoicePlanError,
    StaleNarrationPlanError,
    StaleNarrationPreferencesError,
    UtteranceGenerationError,
    VoiceNotAssignedError,
)
from app.services.multi_voice_planner import validate_plan_against_script
from app.services.narration_planner import split_paragraphs, validate_against_script
from app.services.project_service import ProjectService

MODE_NARRATION_PLAN = "narration_plan"
MODE_MULTI_VOICE = "multi_voice"
MODE_SINGLE_PASS = "single_pass"

# Request IDs older than this cannot be used for stitching (API limit 2h).
_REQUEST_ID_MAX_AGE_SECONDS = 2 * 3600 - 300


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def plan_hash(plan: Any) -> str:
    """Stable fingerprint of a plan model — lets us reuse finished segments.

    Narration preferences are PART of the hash: changing them changes the
    fingerprint, so segments produced under the old preferences are never
    reused. Plans created before preferences existed carry `preferences: None`
    and keep their previous fingerprint (otherwise every existing project
    would silently re-bill its finished segments after this upgrade).
    """
    data = plan.model_dump(mode="json")
    if isinstance(data, dict) and data.get("preferences") is None:
        data.pop("preferences", None)
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _segment_extension(output_format: str) -> str:
    """File extension for a segment fetched in `output_format`."""
    if output_format.startswith("mp3_"):
        return ".mp3"
    if output_format.startswith("wav_"):
        return ".wav"
    if output_format.startswith("pcm_"):
        return ".pcm"
    if output_format.startswith("opus_"):
        return ".opus"
    if output_format.startswith("ulaw_"):
        return ".ulaw"
    if output_format.startswith("alaw_"):
        return ".alaw"
    return ".bin"


def _iso_age_seconds(iso_timestamp: Optional[str]) -> Optional[float]:
    if not iso_timestamp:
        return None
    try:
        moment = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (_utcnow() - moment).total_seconds()


class AudioService:
    """Orchestrates speech generation and audio storage."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        project_service: Optional[ProjectService] = None,
        tts_client: Optional[ElevenLabsClient] = None,
        composer: Optional[FFmpegAudioComposer] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.projects = project_service or ProjectService(self.settings)
        # DI seams: tests inject a mocked ElevenLabs client / fake composer.
        self.tts = tts_client or ElevenLabsClient(self.settings)
        self.composer = composer or FFmpegAudioComposer(self.settings)

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    async def generate_audio(self, project_id: str) -> AudioResponse:
        """Generate the audiobook: multi-voice plan > paragraph plan > single pass."""
        # 1) project must exist -> 404
        self.projects.get_project(project_id)

        # 2) stored script: 404 when missing, 400 when empty
        text = self.projects.read_script(project_id)

        # 2b) private-beta length / estimated-duration limits (cost guard)
        check_script_limits(self.settings, text)

        # 3) multi-voice mode when an utterance plan exists (spec §8) …
        multi_voice = self.projects.get_multi_voice_plan(project_id)
        if multi_voice is not None:
            mv_plan, _ = multi_voice
            # Plan workflow lifecycle (durable recovery): started -> completed
            # / failed, keyed by the middleware's generation_id (joined when
            # the workflow was already paid for, fresh otherwise). The
            # allowance is never refunded on failure (existing policy).
            plan_identity.mark_started(
                self.settings, project_id, current_generation_id()
            )
            try:
                response = await self._generate_multi_voice(
                    project_id, text, mv_plan
                )
            except Exception:
                plan_identity.mark_failed(
                    self.settings, project_id, current_generation_id()
                )
                raise
            plan_identity.mark_completed(
                self.settings, project_id, current_generation_id()
            )
            return response

        # 4) … else paragraph-plan mode when analysis ran, else legacy single pass
        stored = self.projects.get_narration_plan(project_id)
        if stored is None:
            return await self._generate_single_pass(project_id, text)

        plan, _ = stored
        return await self._generate_from_plan(project_id, text, plan)

    # ------------------------------------------------------------------ #
    # Mode A: whole script in one request (unchanged legacy behaviour)
    # ------------------------------------------------------------------ #
    async def _generate_single_pass(
        self, project_id: str, text: str
    ) -> AudioResponse:
        max_chars = self.settings.elevenlabs_max_chars
        if len(text) > max_chars:
            raise ScriptTooLongError(len(text), max_chars)

        # Legacy mode writes the file directly, so it always requests the
        # FINAL format (intermediates only apply to plan-driven segments).
        result = await self.tts.synthesize_result(
            text, output_format=self.settings.elevenlabs_output_format
        )
        audio_bytes = result.audio

        audio_path = self.projects.audio_path(project_id)
        try:
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(audio_bytes)
        except OSError as exc:
            raise AudioSaveError(
                f"Could not save the generated audio to {audio_path.name}: {exc}"
            ) from exc

        info = {
            "filename": audio_path.name,
            "size_bytes": len(audio_bytes),
            "character_count": len(text),
            "voice_id": self.settings.elevenlabs_voice_id,
            "model_id": self.settings.elevenlabs_model_id,
            "output_format": self.settings.elevenlabs_output_format,
            "request_id": result.request_id,
            "generated_at": _utcnow().isoformat(),
            "mode": MODE_SINGLE_PASS,
            "segment_count": None,
            "generation_id": current_generation_id(),
        }
        self.projects.save_audio_metadata(project_id, info)

        return self._to_response(
            project_id, info, message="Audio generated successfully"
        )

    # ------------------------------------------------------------------ #
    # Mode B: MiMo paragraph plan + ElevenLabs segments + FFmpeg merge
    # ------------------------------------------------------------------ #
    async def _generate_from_plan(
        self, project_id: str, text: str, plan: NarrationPlan
    ) -> AudioResponse:
        paragraphs = split_paragraphs(text)

        # The plan must still describe the current script exactly.
        problems = validate_against_script(plan, paragraphs)
        if problems:
            raise StaleNarrationPlanError(project_id, problems)

        # …and it must have been built with the preferences currently stored
        # for the project: changing preferences triggers a NEW plan (the plan
        # hash changes with them, so old segments must not be reused).
        stored_preferences = self.projects.get_narration_preferences(project_id)
        current_preferences = (
            stored_preferences[0] if stored_preferences else None
        )
        if plan.preferences != current_preferences:
            raise StaleNarrationPreferencesError(project_id)

        ordered = sorted(plan.paragraphs, key=lambda p: p.paragraph_index)
        max_chars = self.settings.elevenlabs_max_chars
        for paragraph in ordered:
            # Character limit applies to what is actually sent (tts_text may
            # be a little longer than original_text because of the tags).
            spoken = paragraph.tts_text or paragraph.original_text
            if len(spoken) > max_chars:
                raise ScriptTooLongError(len(spoken), max_chars)

        current_hash = plan_hash(plan)
        existing = self.projects.read_audio_metadata(project_id) or {}
        prior_records = self._prior_records(existing, current_hash)
        reusable = self._reusable_segments(project_id, ordered, prior_records)

        # Fresh metadata for this run (segments found above stay reusable).
        self._update_metadata(
            project_id,
            mode=MODE_NARRATION_PLAN,
            status="generating",
            plan_hash=current_hash,
            overall_story_tone=plan.overall_story_tone,
            language=plan.language,
            output_format=self.settings.elevenlabs_output_format,
            segment_output_format=self.settings.elevenlabs_segment_output_format,
            model_id=self.settings.elevenlabs_model_id,
            voice_id=self.settings.elevenlabs_voice_id,
            segment_count=len(ordered),
            segments=[],
            created_at=_utcnow().isoformat(),
        )

        language_code = self._language_code(plan)
        records: list[dict[str, Any]] = []
        for position, paragraph in enumerate(ordered):
            context = self._build_context(
                position=position,
                ordered=ordered,
                prior_records=prior_records,
                using_prior_ids=bool(prior_records),
            )
            record = await self._ensure_segment(
                project_id,
                paragraph,
                context=context,
                language_code=language_code,
                reusable=reusable,
                prior=prior_records.get(paragraph.paragraph_index),
                completed=records,
                total=len(ordered),
            )
            records.append(record)
            self._update_metadata(project_id, status="generating", segments=records)

        # --- FFmpeg composition --------------------------------------- #
        audio_path = self.projects.audio_path(project_id)
        ext = _segment_extension(self.settings.elevenlabs_segment_output_format)
        segments = [
            Segment(
                path=self.projects.segment_path(
                    project_id, p.paragraph_index, extension=ext
                ),
                pause_after_ms=p.pause_after_ms,
            )
            for p in ordered
        ]
        try:
            self.composer.compose(segments, audio_path)
        except FFmpegError as exc:
            self._update_metadata(
                project_id,
                status="composition_failed",
                segments=records,
                error=str(exc),
            )
            raise

        info = {
            "filename": audio_path.name,
            "size_bytes": audio_path.stat().st_size,
            "character_count": len(text),
            "voice_id": self.settings.elevenlabs_voice_id,
            "model_id": self.settings.elevenlabs_model_id,
            "output_format": self.settings.elevenlabs_output_format,
            "segment_output_format": self.settings.elevenlabs_segment_output_format,
            "generated_at": _utcnow().isoformat(),
            "mode": MODE_NARRATION_PLAN,
            "segment_count": len(records),
            "generation_id": current_generation_id(),
        }
        self.projects.save_audio_metadata(project_id, info)
        self._update_metadata(
            project_id,
            status="completed",
            segments=records,
            audiobook={
                "filename": audio_path.name,
                "size_bytes": info["size_bytes"],
                "generated_at": info["generated_at"],
            },
            total_pause_ms=sum(p.pause_after_ms for p in ordered),
        )

        return self._to_response(
            project_id,
            info,
            message=(
                f"Audio generated successfully from {len(records)} "
                "narration segments"
            ),
        )

    # ------------------------------------------------------------------ #
    # Paragraph continuity (Eleven v4 Request Stitching)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prior_records(
        existing: dict[str, Any], current_hash: str
    ) -> dict[int, dict[str, Any]]:
        """Segment records from a previous run of the SAME plan.

        Only these carry usable request IDs for regeneration continuity; a
        different plan means fresh, first-time generation (neighbour TEXT).
        """
        if existing.get("mode") != MODE_NARRATION_PLAN:
            return {}
        if existing.get("plan_hash") != current_hash:
            return {}
        prior: dict[int, dict[str, Any]] = {}
        for record in existing.get("segments", []):
            if isinstance(record, dict) and isinstance(record.get("index"), int):
                prior[record["index"]] = record
        return prior

    @staticmethod
    def _usable_request_id(record: Optional[dict[str, Any]], model_id: str) -> Optional[str]:
        """A stored request ID that is fresh, from the same model and real."""
        if not record or record.get("status") != "generated":
            return None
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            return None
        if record.get("model_id") not in (None, model_id):
            return None  # stitching works best within one model
        age = _iso_age_seconds(record.get("generated_at"))
        if age is None or age > _REQUEST_ID_MAX_AGE_SECONDS:
            return None  # IDs expire after ~2 hours
        return request_id

    def _build_context(
        self,
        *,
        position: int,
        ordered: list[ParagraphPlan],
        prior_records: dict[int, dict[str, Any]],
        using_prior_ids: bool,
    ) -> Optional[SynthesisContext]:
        """Neighbour context for `ordered[position]`.

        * first-time generation (no prior run of this plan): neighbouring
          TEXT — `previous_text` / `next_text`;
        * regeneration with stored IDs available: `previous_request_ids` /
          `next_request_ids` — the text for that side is NOT sent (the
          client also enforces this XOR).
        """
        model_id = self.settings.elevenlabs_model_id
        index = ordered[position].paragraph_index

        previous_text: Optional[str] = None
        next_text: Optional[str] = None
        previous_ids: list[str] = []
        next_ids: list[str] = []

        if position > 0:
            previous_text = self._spoken_text(ordered[position - 1])
        if position < len(ordered) - 1:
            next_text = self._spoken_text(ordered[position + 1])

        if using_prior_ids:
            # Up to 3 neighbours per side; closest to this request last for
            # `previous_*` (API requirement) and first for `next_*`.
            for offset in range(1, 4):
                neighbour = position - offset
                if neighbour < 0:
                    break
                request_id = self._usable_request_id(
                    prior_records.get(ordered[neighbour].paragraph_index), model_id
                )
                if request_id:
                    previous_ids.append(request_id)
            previous_ids.reverse()  # oldest first, closest request last

            for offset in range(1, 4):
                neighbour = position + offset
                if neighbour >= len(ordered):
                    break
                request_id = self._usable_request_id(
                    prior_records.get(ordered[neighbour].paragraph_index), model_id
                )
                if request_id:
                    next_ids.append(request_id)  # closest first

        context = SynthesisContext(
            previous_text=None if previous_ids else previous_text,
            next_text=None if next_ids else next_text,
            previous_request_ids=tuple(previous_ids),
            next_request_ids=tuple(next_ids),
        )
        if (
            not context.previous_text
            and not context.next_text
            and not context.previous_request_ids
            and not context.next_request_ids
        ):
            return None
        return context

    @staticmethod
    def _spoken_text(paragraph: ParagraphPlan) -> str:
        """What ElevenLabs will actually say for this paragraph."""
        return paragraph.tts_text or paragraph.original_text

    def _language_code(self, plan: NarrationPlan) -> Optional[str]:
        """ISO 639-1 code for `language` (config wins; never guessed)."""
        configured = (self.settings.elevenlabs_language_code or "").strip()
        if configured:
            return configured.lower()
        code = language_to_iso(plan.language)
        # Only a real ISO-looking code is ever sent — an unrecognised raw
        # name stays a display value and never becomes an API parameter.
        if len(code) in (2, 3) and code.isalpha():
            return code
        return None

    @staticmethod
    def _cast_language_code(cast) -> Optional[str]:
        """Story language when it already looks like an ISO 639-1 code."""
        language = (cast.story.language or "").strip().lower()
        if len(language) in (2, 3) and language.isalpha():
            return language
        return None

    # ------------------------------------------------------------------ #
    # Mode C: multi-voice utterance plan (spec §8/§9)
    # ------------------------------------------------------------------ #
    async def _generate_multi_voice(
        self, project_id: str, text: str, plan: MultiVoicePlan
    ) -> AudioResponse:
        paragraphs = split_paragraphs(text)

        cast = self.projects.get_character_cast(project_id)
        if cast is None:
            raise CharacterCastNotFoundError(project_id)

        # Approval gate — no full-story TTS before the cast is approved.
        missing = [
            speaker.character_id
            for speaker in cast.required_speakers()
            if not cast.casting_for(speaker.character_id).selected
        ]
        if missing:
            raise CastingNotApprovedError(missing, generating=True)
        if not cast.approved:
            raise CastingNotApprovedError([], generating=True)

        # The plan must still describe the current script + cast exactly.
        problems = validate_plan_against_script(plan, paragraphs, cast)
        if problems:
            raise StaleMultiVoicePlanError(project_id, problems)

        ordered = sorted(
            plan.utterances, key=lambda u: (u.paragraph_index, u.sequence_index)
        )
        max_chars = self.settings.elevenlabs_max_chars
        for utterance in ordered:
            spoken = utterance.tts_text or utterance.original_text
            if len(spoken) > max_chars:
                raise ScriptTooLongError(len(spoken), max_chars)

        # Voices come ONLY from the locked cast — never re-rolled, never from
        # a possibly stale plan snapshot.
        locked_voices: dict[str, str] = {}
        for speaker in cast.required_speakers():
            lock = cast.casting_for(speaker.character_id).selected
            if lock is not None:
                locked_voices[speaker.character_id] = lock.voice_id

        language_code = self._cast_language_code(cast)
        current_hash = plan_hash(plan)
        existing = self.projects.read_audio_metadata(project_id) or {}
        prior = self._prior_utterance_records(existing, current_hash)
        # Reuse only when the utterance was made with the SAME locked voice —
        # replacing a character's voice must regenerate its audio (spec §6).
        reusable = self._reusable_utterances(project_id, ordered, prior, locked_voices)

        self._update_metadata(
            project_id,
            mode=MODE_MULTI_VOICE,
            status="generating",
            plan_hash=current_hash,
            overall_story_tone=cast.story.overall_tone,
            language=cast.story.language,
            voice_id=None,  # mixed voices — see per-utterance records
            output_format=self.settings.elevenlabs_output_format,
            segment_output_format=self.settings.elevenlabs_segment_output_format,
            model_id=self.settings.elevenlabs_model_id,
            segment_count=len(ordered),
            segments=[],
            created_at=_utcnow().isoformat(),
        )

        records: list[dict[str, Any]] = []
        for position, utterance in enumerate(ordered):
            voice_id = locked_voices.get(utterance.speaker_id)
            if not voice_id:
                raise VoiceNotAssignedError(utterance.speaker_id)
            context = self._build_utterance_context(
                position=position,
                ordered=ordered,
                prior_records=prior,
                using_prior_ids=bool(prior),
            )
            record = await self._ensure_utterance(
                project_id,
                utterance,
                position=position,
                voice_id=voice_id,
                language_code=language_code,
                context=context,
                reusable=reusable,
                prior=prior.get(utterance.utterance_id),
                completed=records,
                total=len(ordered),
            )
            records.append(record)
            self._update_metadata(
                project_id, status="generating", segments=records
            )

        # --- FFmpeg assembly in exact story order (spec §9) -------------- #
        audio_path = self.projects.audio_path(project_id)
        ext = _segment_extension(self.settings.elevenlabs_segment_output_format)
        segments = [
            Segment(
                path=self.projects.utterance_path(
                    project_id, u.paragraph_index, u.sequence_index, extension=ext
                ),
                pause_after_ms=u.pause_after_ms,
            )
            for u in ordered
        ]
        try:
            self.composer.compose(segments, audio_path)
        except FFmpegError as exc:
            self._update_metadata(
                project_id,
                status="composition_failed",
                segments=records,
                error=str(exc),
            )
            raise

        info = {
            "filename": audio_path.name,
            "size_bytes": audio_path.stat().st_size,
            "character_count": len(text),
            "voice_id": None,  # multiple locked voices — see segment records
            "model_id": self.settings.elevenlabs_model_id,
            "output_format": self.settings.elevenlabs_output_format,
            "segment_output_format": self.settings.elevenlabs_segment_output_format,
            "generated_at": _utcnow().isoformat(),
            "mode": MODE_MULTI_VOICE,
            "segment_count": len(records),
            "generation_id": current_generation_id(),
        }
        self.projects.save_audio_metadata(project_id, info)
        self._update_metadata(
            project_id,
            status="completed",
            segments=records,
            audiobook={
                "filename": audio_path.name,
                "size_bytes": info["size_bytes"],
                "generated_at": info["generated_at"],
            },
            total_pause_ms=sum(u.pause_after_ms for u in ordered),
        )

        return self._to_response(
            project_id,
            info,
            message=(
                f"Audio generated successfully from {len(records)} "
                "multi-voice utterances"
            ),
        )

    @staticmethod
    def _prior_utterance_records(
        existing: dict[str, Any], current_hash: str
    ) -> dict[str, dict[str, Any]]:
        """Utterance records from a previous run of the SAME multi-voice plan."""
        if existing.get("mode") != MODE_MULTI_VOICE:
            return {}
        if existing.get("plan_hash") != current_hash:
            return {}
        prior: dict[str, dict[str, Any]] = {}
        for record in existing.get("segments", []):
            if isinstance(record, dict) and isinstance(
                record.get("utterance_id"), str
            ):
                prior[record["utterance_id"]] = record
        return prior

    def _reusable_utterances(
        self,
        project_id: str,
        ordered: list[Utterance],
        prior_records: dict[str, dict[str, Any]],
        voices: dict[str, str],
    ) -> set[str]:
        """Utterance ids whose audio exists for the same plan AND voice.

        A segment generated with a DIFFERENT locked voice is never reused —
        when the user replaces a character's voice, only that speaker's
        utterances regenerate (spec §6: permanent locks must be honoured).
        """
        if not prior_records:
            return set()
        reusable: set[str] = set()
        for utterance in ordered:
            record = prior_records.get(utterance.utterance_id)
            if not record or record.get("status") != "generated":
                continue
            if record.get("voice_id") != voices.get(utterance.speaker_id):
                continue  # the lock changed -> synthesise again in the new voice
            filename = record.get("filename")
            if not isinstance(filename, str):
                continue
            if (self.projects.segments_dir(project_id) / filename).is_file():
                reusable.add(utterance.utterance_id)
        return reusable

    def _build_utterance_context(
        self,
        *,
        position: int,
        ordered: list[Utterance],
        prior_records: dict[str, dict[str, Any]],
        using_prior_ids: bool,
    ) -> Optional[SynthesisContext]:
        """Neighbouring-utterance context — text first-time, IDs on regeneration."""
        model_id = self.settings.elevenlabs_model_id

        previous_text: Optional[str] = None
        next_text: Optional[str] = None
        previous_ids: list[str] = []
        next_ids: list[str] = []

        if position > 0:
            previous_text = self._spoken(ordered[position - 1])
        if position < len(ordered) - 1:
            next_text = self._spoken(ordered[position + 1])

        if using_prior_ids:
            for offset in range(1, 4):
                neighbour = position - offset
                if neighbour < 0:
                    break
                record = prior_records.get(ordered[neighbour].utterance_id)
                request_id = self._usable_request_id(record, model_id)
                if request_id:
                    previous_ids.append(request_id)
            previous_ids.reverse()

            for offset in range(1, 4):
                neighbour = position + offset
                if neighbour >= len(ordered):
                    break
                record = prior_records.get(ordered[neighbour].utterance_id)
                request_id = self._usable_request_id(record, model_id)
                if request_id:
                    next_ids.append(request_id)

        context = SynthesisContext(
            previous_text=None if previous_ids else previous_text,
            next_text=None if next_ids else next_text,
            previous_request_ids=tuple(previous_ids),
            next_request_ids=tuple(next_ids),
        )
        if (
            not context.previous_text
            and not context.next_text
            and not context.previous_request_ids
            and not context.next_request_ids
        ):
            return None
        return context

    @staticmethod
    def _spoken(utterance: Utterance) -> str:
        return utterance.tts_text or utterance.original_text

    # ------------------------------------------------------------------ #
    # Segment generation
    # ------------------------------------------------------------------ #
    async def _ensure_utterance(
        self,
        project_id: str,
        utterance: Utterance,
        *,
        position: int,
        voice_id: str,
        language_code: Optional[str],
        context: Optional[SynthesisContext],
        reusable: set[str],
        prior: Optional[dict[str, Any]],
        completed: list[dict[str, Any]],
        total: int,
    ) -> dict[str, Any]:
        """Generate (or reuse) one utterance in its LOCKED voice."""
        ext = _segment_extension(self.settings.elevenlabs_segment_output_format)
        path = self.projects.utterance_path(
            project_id, utterance.paragraph_index, utterance.sequence_index,
            extension=ext,
        )
        spoken = self._spoken(utterance)
        base: dict[str, Any] = {
            "utterance_id": utterance.utterance_id,
            "filename": path.name,
            "paragraph_index": utterance.paragraph_index,
            "sequence_index": utterance.sequence_index,
            "speaker_id": utterance.speaker_id,
            "speaker_type": utterance.speaker_type,
            "voice_id": voice_id,
            "model_id": self.settings.elevenlabs_model_id,
            "original_text": utterance.original_text,
            "tts_text": utterance.tts_text,
            "has_audio_tags": utterance.tts_text is not None
            and utterance.tts_text != utterance.original_text,
            "emotion": utterance.emotion,
            "acting_intensity": utterance.acting_intensity,
            "delivery_style": utterance.delivery_style,
            "narration_direction": utterance.narration_direction,
            "pause_after_ms": utterance.pause_after_ms,
            "character_count": len(utterance.original_text),
            "spoken_character_count": len(spoken),
            # Exactly the settings this request will send (v4: no speed/style).
            "effective_voice_settings": self.tts.effective_voice_settings(None),
        }

        # Resume: never re-bill an utterance finished for the SAME plan.
        if utterance.utterance_id in reusable and path.is_file():
            return {
                **base,
                "status": "generated",
                "reused": True,
                "request_id": (prior or {}).get("request_id"),
                "size_bytes": path.stat().st_size,
                "generated_at": (prior or {}).get("generated_at")
                or _utcnow().isoformat(),
            }

        try:
            result = await self.tts.synthesize_result(
                utterance.original_text,
                voice_settings=None,  # restrained defaults; capability-filtered
                tts_text=utterance.tts_text,
                context=context,
                language_code=language_code,
                voice_id=voice_id,
            )
            audio_bytes = result.audio
        except ProjectError as exc:
            self._update_metadata(
                project_id,
                status="partial",
                segments=completed,
                failed_index=position,
                failed_utterance_id=utterance.utterance_id,
                error=str(exc),
            )
            raise UtteranceGenerationError(
                utterance.utterance_id, len(completed), total, str(exc)
            ) from exc

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio_bytes)
        except OSError as exc:
            raise AudioSaveError(
                f"Could not save utterance {path.name}: {exc}"
            ) from exc

        return {
            **base,
            "status": "generated",
            "reused": False,
            "request_id": result.request_id,
            "output_format": result.output_format,
            "size_bytes": len(audio_bytes),
            "generated_at": _utcnow().isoformat(),
        }

    async def _ensure_segment(
        self,
        project_id: str,
        paragraph: ParagraphPlan,
        *,
        context: Optional[SynthesisContext],
        language_code: Optional[str],
        reusable: set[int],
        prior: Optional[dict[str, Any]],
        completed: list[dict[str, Any]],
        total: int,
    ) -> dict[str, Any]:
        """Generate (or reuse) one paragraph's segment and describe it."""
        index = paragraph.paragraph_index
        ext = _segment_extension(self.settings.elevenlabs_segment_output_format)
        path = self.projects.segment_path(project_id, index, extension=ext)
        spoken = self._spoken_text(paragraph)
        plan_voice_settings = paragraph.voice_settings.model_dump()
        base: dict[str, Any] = {
            "index": index,
            "filename": path.name,
            "emotion": paragraph.emotion,
            "delivery_style": paragraph.delivery_style,
            "intensity": paragraph.intensity,
            "narration_direction": paragraph.narration_direction,
            "pause_after_ms": paragraph.pause_after_ms,
            # Planning metadata: keeps speed/style for review/editing …
            "voice_settings": plan_voice_settings,
            # … while this is what eleven_v4 actually receives (no
            # speed/style; clone_strength only for eligible v4 PVCs).
            "effective_voice_settings": self.tts.effective_voice_settings(
                plan_voice_settings
            ),
            "character_count": len(paragraph.original_text),
            "spoken_character_count": len(spoken),
            "has_audio_tags": paragraph.tts_text is not None
            and paragraph.tts_text != paragraph.original_text,
            "tts_text": paragraph.tts_text,
            "model_id": self.settings.elevenlabs_model_id,
        }

        # Resume: a finished segment for the same plan is never re-billed.
        if index in reusable and path.is_file():
            return {
                **base,
                "status": "generated",
                "reused": True,
                "request_id": (prior or {}).get("request_id"),
                "size_bytes": path.stat().st_size,
                "generated_at": (prior or {}).get("generated_at")
                or _utcnow().isoformat(),
            }

        try:
            # Only the spoken text (tts_text with MiMo's inline audio tags),
            # the numeric voice settings, continuity context and the language
            # hint are sent — emotion/direction are never API parameters.
            result = await self.tts.synthesize_result(
                paragraph.original_text,
                voice_settings=paragraph.voice_settings.model_dump(),
                tts_text=paragraph.tts_text,
                context=context,
                language_code=language_code,
            )
            audio_bytes = result.audio
        except ProjectError as exc:
            # Preserve everything generated so far and report partial failure.
            self._update_metadata(
                project_id,
                status="partial",
                segments=completed,
                failed_index=index,
                error=str(exc),
            )
            raise SegmentGenerationError(
                index, len(completed), total, str(exc)
            ) from exc

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio_bytes)
        except OSError as exc:
            raise AudioSaveError(
                f"Could not save segment {path.name}: {exc}"
            ) from exc

        return {
            **base,
            "status": "generated",
            "reused": False,
            # Stored for continuity when this segment is regenerated later.
            "request_id": result.request_id,
            "output_format": result.output_format,
            "size_bytes": len(audio_bytes),
            "generated_at": _utcnow().isoformat(),
        }

    def _reusable_segments(
        self,
        project_id: str,
        ordered: list[ParagraphPlan],
        prior_records: dict[int, dict[str, Any]],
    ) -> set[int]:
        """Indices whose segment exists from a previous run of the SAME plan.

        `prior_records` already guarantees the same plan hash (see
        `_prior_records`), so only the file itself must still be on disk.
        """
        if not prior_records:
            return set()

        reusable: set[int] = set()
        for record in prior_records.values():
            index = record.get("index")
            filename = record.get("filename")
            if not isinstance(index, int) or not isinstance(filename, str):
                continue
            if record.get("status") != "generated":
                continue
            if index not in {p.paragraph_index for p in ordered}:
                continue
            if (self.projects.segments_dir(project_id) / filename).is_file():
                reusable.add(index)
        return reusable

    def _update_metadata(self, project_id: str, **updates: Any) -> dict[str, Any]:
        """Merge updates into audio/metadata.json (progress is never lost)."""
        data = self.projects.read_audio_metadata(project_id) or {}
        data.update(updates)
        now = _utcnow().isoformat()
        data.setdefault("created_at", now)
        data["updated_at"] = now
        self.projects.write_audio_metadata(project_id, data)
        return data

    # ------------------------------------------------------------------ #
    # Read access
    # ------------------------------------------------------------------ #
    def get_audio(self, project_id: str) -> AudioResponse:
        """Return information about the generated audio (404 when absent)."""
        self.projects.get_project(project_id)  # 404 for unknown project

        info = self.projects.get_audio_metadata(project_id)
        if not info:
            raise AudioNotFoundError(project_id)

        # Metadata says audio exists — make sure the file is really on disk.
        if not self.projects.audio_path(project_id).is_file():
            raise AudioNotFoundError(
                project_id, reason="the audio file is missing on disk"
            )

        return self._to_response(
            project_id, info, message="Audio is available for this project"
        )

    def audio_exists(self, project_id: str) -> bool:
        return self.projects.audio_path(project_id).is_file()

    def generation_metadata(self, project_id: str) -> Optional[dict[str, Any]]:
        """Contents of audio/metadata.json (segment level detail)."""
        return self.projects.read_audio_metadata(project_id)

    # ------------------------------------------------------------------ #
    # Mapping
    # ------------------------------------------------------------------ #
    def _to_response(
        self, project_id: str, info: dict, message: str
    ) -> AudioResponse:
        audio_path: Path = self.projects.audio_path(project_id)
        generated_at = info.get("generated_at")
        return AudioResponse(
            success=True,
            message=message,
            project_id=project_id,
            audio_filename=info["filename"],
            audio_url=self.settings.media_url(audio_path),
            status="completed",
            size_bytes=info.get("size_bytes"),
            character_count=info.get("character_count"),
            voice_id=info.get("voice_id"),
            model_id=info.get("model_id"),
            output_format=info.get("output_format"),
            generated_at=datetime.fromisoformat(generated_at)
            if generated_at
            else None,
            generation_mode=info.get("mode"),
            segment_count=info.get("segment_count"),
        )
