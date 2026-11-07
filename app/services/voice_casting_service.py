"""Voice casting service (spec §3–§6): discovery, ranking, selection, approval.

Flow:

    analyze-characters (CharacterAnalyzer)   -> character_cast.json
    POST voice-casting                        -> discover + rank candidates
    POST voice-casting/select                 -> permanent VoiceLock (§6)
    POST voice-casting/approve                -> all speaking roles locked
    POST voice-casting/audition               -> short test line (optional)

Rules enforced here:

* account voices are searched FIRST; the shared library only fills the gap
  when fewer than 3 acceptable account candidates exist;
* characters without spoken dialogue need no voice — only `narrator` plus
  characters with `has_spoken_dialogue` are "required speakers";
* selecting a voice locks it permanently for that speaker; generation reads
  ONLY from locks (never re-rolls). Re-selecting one speaker never touches
  the others;
* any voice change invalidates approval (must re-approve before audio);
* restricted voices are rejected; manual selection requires a real,
  fetchable voice (no invented IDs).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.config import Settings, get_settings, language_to_iso
from app.schemas.casting import (
    AuditionRequest,
    CharacterCast,
    CharacterProfile,
    SelectVoiceRequest,
    SpeakerCasting,
    VoiceCastingResponse,
    VoiceLock,
)
from app.services.eleven_models import get_capabilities
from app.services.elevenlabs_client import ElevenLabsClient
from app.services.errors import (
    CastingNotApprovedError,
    CharacterCastNotFoundError,
    ElevenLabsAPIError,
    VoiceSelectionError,
)
from app.services.project_service import ProjectService
from app.services.voice_discovery import (
    ElevenLabsVoiceDiscovery,
    VoiceDetailsNotFoundError,
    VoiceSearchFilters,
)
from app.services.voice_matcher import (
    RawVoice,
    normalize_account_voice,
    normalize_age,
    normalize_gender,
    normalize_shared_voice,
    rank_candidates,
)

# How many acceptable candidates we try to present per speaker.
CANDIDATES_PER_SPEAKER = 3


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _dedupe(pool: list[RawVoice]) -> list[RawVoice]:
    """Unique voices by id — the first occurrence wins (account before shared)."""
    seen: set[str] = set()
    unique: list[RawVoice] = []
    for voice in pool:
        if not voice.voice_id or voice.voice_id in seen:
            continue
        seen.add(voice.voice_id)
        unique.append(voice)
    return unique


def _supported_voice_settings(model_id: str) -> list[str]:
    """Voice-settings keys the selected model actually accepts."""
    caps = get_capabilities(model_id)
    keys = ["stability", "similarity_boost"]
    if caps.supports_use_speaker_boost:
        keys.append("use_speaker_boost")
    if caps.supports_speed:
        keys.append("speed")
    if caps.supports_style:
        keys.append("style")
    if caps.supports_clone_strength:
        keys.append("clone_strength")
    return keys


class VoiceCastingService:
    """Orchestrates ElevenLabs voice discovery and the permanent cast."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        project_service: Optional[ProjectService] = None,
        discovery: Optional[ElevenLabsVoiceDiscovery] = None,
        tts_client: Optional[ElevenLabsClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.projects = project_service or ProjectService(self.settings)
        # DI seams: tests inject MockTransport-backed clients (no credits).
        self.discovery = discovery or ElevenLabsVoiceDiscovery(self.settings)
        self.tts = tts_client or ElevenLabsClient(self.settings)

    # ------------------------------------------------------------------ #
    # Load / persist helpers
    # ------------------------------------------------------------------ #
    def _load(self, project_id: str) -> CharacterCast:
        self.projects.get_project(project_id)
        cast = self.projects.get_character_cast(project_id)
        if cast is None:
            raise CharacterCastNotFoundError(project_id)
        return cast

    def _save(self, cast: CharacterCast) -> datetime:
        return self.projects.save_character_cast(cast.project_id, cast)

    @staticmethod
    def _response(cast: CharacterCast, saved_at: Optional[datetime] = None) -> VoiceCastingResponse:
        characters = {
            character_id: entry
            for character_id, entry in cast.casting.items()
            if character_id != "narrator"
        }
        missing = [
            speaker.character_id
            for speaker in cast.required_speakers()
            if not cast.casting_for(speaker.character_id).selected
        ]
        return VoiceCastingResponse(
            project_id=cast.project_id,
            approved=cast.approved,
            approved_at=cast.approved_at,
            narrator=cast.casting_for("narrator"),
            characters=characters,
            missing_speakers=missing,
            saved_at=saved_at,
            story=cast.story,
        )

    # ------------------------------------------------------------------ #
    # Discovery + ranking (POST /voice-casting)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _filters_for(profile: CharacterProfile, story_language: str, locale: Optional[str]) -> VoiceSearchFilters:
        """Server-side filters — ONLY attributes the script established.

        Age/gender are canonicalised to the provider's enum vocabulary
        ("mature" -> "middle_aged", "elderly" -> "old", …); an unrecognised
        value is dropped instead of being sent and emptying the results.
        """
        established_gender = profile.gender and "gender" not in profile.uncertain_attributes
        established_age = profile.age and "age" not in profile.uncertain_attributes
        established_accent = profile.accent and "accent" not in profile.uncertain_attributes
        return VoiceSearchFilters(
            language=language_to_iso(profile.language or story_language or "") or None,
            locale=locale,
            gender=normalize_gender(profile.gender) if established_gender else None,
            age=normalize_age(profile.age) if established_age else None,
            accent=profile.accent.lower() if established_accent else None,
        )

    async def _pool(
        self, filters: VoiceSearchFilters, *, shared: bool
    ) -> list[RawVoice]:
        if shared:
            raw = await self.discovery.list_shared_voices(filters)
            return [v for v in (normalize_shared_voice(i) for i in raw) if v]
        raw = await self.discovery.list_account_voices(filters)
        return [v for v in (normalize_account_voice(i) for i in raw) if v]

    async def build_casting(self, project_id: str) -> VoiceCastingResponse:
        cast = self._load(project_id)
        story = cast.story

        for speaker in cast.required_speakers():
            entry = cast.casting_for(speaker.character_id)
            filters = self._filters_for(
                speaker, story.language, story.locale
            )

            # 1) account voices first …
            account_pool = await self._pool(filters, shared=False)
            candidates = rank_candidates(
                speaker,
                account_pool,
                required_language=story.language,
                required_locale=story.locale,
                limit=CANDIDATES_PER_SPEAKER,
            )

            # 2) … library voices only when the account pool is too thin.
            merged: list[RawVoice] = account_pool
            if len(candidates) < CANDIDATES_PER_SPEAKER:
                merged = account_pool + await self._pool(filters, shared=True)
                candidates = rank_candidates(
                    speaker,
                    merged,
                    required_language=story.language,
                    required_locale=story.locale,
                    limit=CANDIDATES_PER_SPEAKER,
                )

            # 3) EVERY speaking role ends up with candidates — "No candidate
            #    available" blocks the whole cast. Relax the demographics
            #    filters and the match-score threshold before ever showing
            #    none; language compatibility stays a hard rule (an English
            #    voice never narrates a Hindi story) and restricted voices
            #    stay excluded at every level.
            if not candidates:
                relaxed = VoiceSearchFilters(
                    language=filters.language, locale=filters.locale
                )
                merged = _dedupe(
                    merged
                    + await self._pool(relaxed, shared=False)
                    + await self._pool(relaxed, shared=True)
                )
                candidates = rank_candidates(
                    speaker,
                    merged,
                    required_language=story.language,
                    required_locale=story.locale,
                    min_accept_score=0.0,
                    limit=CANDIDATES_PER_SPEAKER,
                )

            # Existing locks survive a re-cast; only candidates refresh.
            entry.candidates = candidates
            entry.insufficient_candidates = not candidates

        cast.updated_at = _utcnow()
        saved_at = self._save(cast)
        return self._response(cast, saved_at)

    # ------------------------------------------------------------------ #
    # Read (GET /voice-casting)
    # ------------------------------------------------------------------ #
    def get_casting(self, project_id: str) -> VoiceCastingResponse:
        return self._response(self._load(project_id))

    # ------------------------------------------------------------------ #
    # Permanent selection / lock (POST /voice-casting/select)
    # ------------------------------------------------------------------ #
    async def select(self, project_id: str, request: SelectVoiceRequest) -> VoiceCastingResponse:
        cast = self._load(project_id)

        speaker = next(
            (
                s for s in cast.required_speakers()
                if s.character_id == request.speaker_id
            ),
            None,
        )
        if speaker is None:
            raise VoiceSelectionError(
                f"Unknown speaker '{request.speaker_id}' — select a voice for "
                "'narrator' or a character from the cast (characters without "
                "spoken dialogue do not need voices)."
            )

        voice_id = request.voice_id.strip()
        if not voice_id:
            raise VoiceSelectionError("voice_id must not be empty")

        # The candidate we already hold for this voice — its name is the
        # voice's REAL name and must survive the import/lock round trip
        # (never the character's name).
        entry = cast.casting_for(speaker.character_id)
        known = next(
            (c for c in entry.candidates if c.voice_id == voice_id), None
        )
        original_name = (known.name if known else "") or ""

        # Import a library voice into the account when its owner is given —
        # never for voices we cannot verify. The import keeps the voice's own
        # name so the account and the UI always agree.
        origin: str = "account"
        original_voice_id: Optional[str] = None
        if request.public_owner_id:
            origin = "shared"
            original_voice_id = voice_id
            voice_id = await self.discovery.add_shared_voice(
                request.public_owner_id.strip(),
                voice_id,
                (request.new_name or "").strip()
                or original_name
                or speaker.character_name,
            )

        # Verify the voice really exists (GET /v1/voices/{voice_id}) —
        # invalid or unavailable IDs are rejected before anything is locked.
        try:
            details = await self.discovery.get_voice(voice_id)
        except VoiceDetailsNotFoundError as exc:
            raise VoiceSelectionError(
                f"Voice '{voice_id}' was not found in this account. For a "
                "library voice, pass its public_owner_id so it can be added "
                "first."
            ) from exc
        except ElevenLabsAPIError:
            if origin == "shared":
                # Added to the account but details lag — fall back to the
                # candidate metadata we already hold.
                details = {}
            else:
                raise

        normalized = normalize_account_voice(details) if details else None
        if normalized and normalized.restricted_reason:
            raise VoiceSelectionError(
                f"Voice '{voice_id}' is restricted and cannot be used "
                f"({normalized.restricted_reason})"
            )

        lock = VoiceLock(
            character_id=speaker.character_id,
            voice_id=voice_id,
            original_voice_id=original_voice_id,
            voice_name=original_name
            or (normalized.name if normalized else "")
            or voice_id,
            provider="elevenlabs",
            language=language_to_iso(speaker.language or cast.story.language or "") or None,
            accent=(normalized.accent if normalized else None) or speaker.accent,
            voice_metadata={
                k: v
                for k, v in {
                    "origin": origin,
                    "category": normalized.category if normalized else None,
                    "description": normalized.description if normalized else None,
                    "labels": normalized.labels if normalized else {},
                    "custom_rate": normalized.custom_rate if normalized else False,
                    "notice_period_days": normalized.notice_period_days if normalized else None,
                    "verified_languages": (details or {}).get("verified_languages") or [],
                }.items()
                if v not in (None, {}, [])
            },
            model_id=self.settings.elevenlabs_model_id,
            supported_voice_settings=_supported_voice_settings(
                self.settings.elevenlabs_model_id
            ),
            preview_url=(normalized.preview_url if normalized else None),
            origin=origin if origin == "shared" else "account",
            selected_at=_utcnow(),
            locked=True,
        )

        speaker.voice_id = lock.voice_id
        speaker.voice_locked = True
        entry = cast.casting_for(speaker.character_id)
        entry.selected = lock
        entry.status = "selected"
        # Any voice change requires a fresh approval before generation.
        cast.approved = False
        cast.approved_at = None
        cast.updated_at = _utcnow()
        saved_at = self._save(cast)
        return self._response(cast, saved_at)

    # ------------------------------------------------------------------ #
    # Approval (POST /voice-casting/approve)
    # ------------------------------------------------------------------ #
    def approve(self, project_id: str) -> VoiceCastingResponse:
        cast = self._load(project_id)
        missing = [
            speaker.character_id
            for speaker in cast.required_speakers()
            if not cast.casting_for(speaker.character_id).selected
        ]
        if missing:
            raise CastingNotApprovedError(missing)

        for speaker in cast.required_speakers():
            cast.casting_for(speaker.character_id).status = "approved"
        cast.approved = True
        cast.approved_at = _utcnow()
        cast.updated_at = _utcnow()
        saved_at = self._save(cast)
        return self._response(cast, saved_at)

    # ------------------------------------------------------------------ #
    # Short audition (optional, user-approved test line — spec §5)
    # ------------------------------------------------------------------ #
    async def audition(self, project_id: str, request: AuditionRequest) -> bytes:
        """Synthesise a SHORT test line in a candidate/selected voice.

        Uses the existing TTS client (capability-filtered). The full story
        is never generated here and the cast is never modified.
        """
        cast = self._load(project_id)
        speaker = next(
            (s for s in cast.required_speakers()
             if s.character_id == request.speaker_id),
            None,
        )
        if speaker is None:
            raise VoiceSelectionError(f"Unknown speaker '{request.speaker_id}'")

        voice_id = (request.voice_id or "").strip()
        if not voice_id:
            lock = cast.casting_for(speaker.character_id).selected
            if lock is None:
                raise VoiceSelectionError(
                    f"Speaker '{request.speaker_id}' has no selected voice yet "
                    "and no voice_id was given for the audition"
                )
            voice_id = lock.voice_id

        result = await self.tts.synthesize_result(
            request.text,
            voice_id=voice_id,
            output_format=self.settings.elevenlabs_output_format,
        )
        return result.audio
