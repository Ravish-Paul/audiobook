"""Multi-voice utterance planner (spec §7).

    script.txt + character_cast.json -> MiMo (JSON) -> Pydantic validation
        -> word-level cross-check against the script -> multi_voice_plan.json

Hard rules enforced here:

* every script paragraph becomes 1..k ordered utterances; the utterances'
  words (ignoring punctuation) must reconstruct the paragraph EXACTLY —
  no word may be changed, added, removed, translated or reordered;
* narration and dialogue stay distinct: each utterance belongs to exactly
  one cast speaker (`narrator` or a character with spoken dialogue);
* per-utterance `tts_text` follows the same byte-exact audio-tag rule as
  the paragraph plan (restrained, natural tags only);
* canonical `utterance_id`s and locked voice ids are stamped by the system
  after validation — MiMo never chooses voices;
* invalid output is retried with feedback, then rejected (502).
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Optional, Sequence

from pydantic import ValidationError

from app import plan_identity
from app.config import Settings, get_settings
from app.schemas.casting import CharacterCast
from app.schemas.multi_voice import (
    MultiVoicePlan,
    MultiVoicePlanResponse,
    Utterance,
)
from app.schemas.narration import DELIVERY_STYLES, EMOTIONS
from app.security import current_generation_id
from app.services.errors import (
    CharacterCastNotFoundError,
    EmptyScriptError,
    MultiVoicePlanError,
    MultiVoicePlanNotFoundError,
)
from app.services.mimo_client import MiMoClient
from app.services.narration_planner import (
    normalize_delivery_style,
    normalize_emotion,
    split_paragraphs,
)
from app.services.project_service import ProjectService

# Mid-paragraph utterances must not inject long silences (spec §9).
MAX_MID_PARAGRAPH_PAUSE_MS = 1000


def canonical_utterance_id(paragraph_index: int, sequence_index: int) -> str:
    return f"p{paragraph_index:04d}-u{sequence_index:04d}"


def _spoken_skeleton(text: str) -> str:
    """Punctuation-free view of the spoken words (order preserved).

    Keeps letters, digits AND combining marks (category M) — Devanagari
    matras are combining marks and must survive the comparison.
    """
    return "".join(
        ch for ch in text if ch.isalnum() or unicodedata.category(ch).startswith("M")
    )


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = f"""You are an expert audiobook dialogue editor at a professional studio.

You receive a complete story script (numbered paragraphs) and the CAST of the audiobook (speaker ids with descriptions). Split every paragraph into ordered UTTERANCES — one per speaker change — for a multi-voice narration.

HARD RULES
1. Never change, add, remove, translate or reorder a single spoken word. Concatenating all `original_text` values of a paragraph, in sequence order, must reproduce that paragraph exactly (punctuation may differ ONLY at the split points, e.g. quote marks or the comma before a quotation).
2. Every paragraph index from 0 to N-1 must appear. Inside each paragraph, `sequence_index` starts at 0 and increases by 1 per utterance. Output utterances already ordered by paragraph_index then sequence_index.
3. `speaker_id` MUST be one of the provided cast ids (the narrator is exactly "narrator"). `speaker_type` is "narrator" for the narrator and "character" otherwise.
4. Narration and dialogue stay DISTINCT: narrator keeps narration AND dialogue attribution ("he said", "she whispered"); quoted dialogue becomes its own utterance for the speaking character. NEVER turn narration into character dialogue. A pure-narration paragraph is ONE narrator utterance.
5. `tts_text`: copy of `original_text` with at most 1-2 RESTRAINED Eleven v4 audio tags when they clearly help — e.g. [whispers], [sad], [curious], [laughs], [sighs], [low, ominous voice], [quiet, tense delivery]. Prefer NO tags over weak ones; never exaggerate or overact; no SSML. Stripping every tag (and its single separating space) must reproduce `original_text` byte-for-byte.
6. `emotion` and `delivery_style` must each be EXACTLY one value from the closed ALLOWED VALUES lists below — never invent a new value, never use a synonym or a variant spelling; a value outside these lists invalidates the whole plan. `acting_intensity` 1..5 — keep it natural and restrained for audiobooks (mostly 1-3, higher only for genuine peaks). `narration_direction`: one short sentence (metadata, never spoken).
7. `pause_after_ms`: mid-paragraph utterances 0..300 (natural sentence rhythm); the LAST utterance of each paragraph 300..1200 to mark the paragraph break. Never insert long silence inside a paragraph.
8. Output ONLY one JSON object (no fences):
{{
  "utterances": [
    {{"utterance_id": "any unique label", "paragraph_index": 0, "sequence_index": 0,
     "speaker_id": "narrator", "speaker_type": "narrator",
     "original_text": "<exact words>", "tts_text": "<exact words + optional [tags]>",
     "emotion": "<exactly one allowed emotion>", "acting_intensity": 2,
     "delivery_style": "<exactly one allowed delivery style>", "narration_direction": "<short note>",
     "pause_after_ms": 400}}
  ]
}}
Do NOT include voice_id or model_id — the system assigns locked voices.

ALLOWED VALUES (closed lists — every utterance must use exactly these)
- emotion (exactly one): {", ".join(EMOTIONS)}
- delivery_style (exactly one): {", ".join(DELIVERY_STYLES)}"""


def build_user_prompt(
    paragraphs: Sequence[str],
    roster: Sequence[dict[str, str]],
    feedback: Optional[Sequence[str]] = None,
) -> str:
    numbered = [
        {"paragraph_index": index, "text": text}
        for index, text in enumerate(paragraphs)
    ]
    parts = [
        "Split the script below into utterances for this cast.",
        "",
        f"Total paragraphs: {len(paragraphs)}",
        "",
        "Cast (use only these speaker_id values):",
        json.dumps(list(roster), ensure_ascii=False, indent=2),
        "",
        "Script paragraphs (each becomes 1..k ordered utterances):",
        json.dumps(numbered, ensure_ascii=False, indent=2),
    ]
    if feedback:
        parts += [
            "",
            "Your previous attempt was REJECTED for these reasons:",
            *[f"- {problem}" for problem in feedback],
            "",
            "Fix every problem above and return the complete corrected JSON "
            "object only.",
        ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Validation (never trust the LLM)
# --------------------------------------------------------------------------- #
def strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _format_validation_error(exc: ValidationError) -> list[str]:
    """Turn Pydantic errors into targeted feedback for the next MiMo retry.

    Enum (`literal_error`) failures get the full treatment: exact utterance
    location, exact field, the ACTUAL invalid value, the complete accepted
    list and an instruction to regenerate only that field. Everything else
    keeps the location + message, with the offending value attached
    (truncated).
    """
    problems: list[str] = []
    for err in exc.errors()[:8]:
        loc = tuple(err.get("loc") or ())
        location = ".".join(str(part) for part in loc) or "<root>"
        field = str(loc[-1]) if loc else "<root>"

        actual = err.get("input", "<missing>")
        actual_repr = repr(actual)
        if len(actual_repr) > 120:
            actual_repr = actual_repr[:117] + "..."

        if err.get("type") == "literal_error":
            expected = (err.get("ctx") or {}).get("expected") or "the documented enum"
            where = ""
            if len(loc) >= 3 and str(loc[0]) == "utterances":
                where = f" (utterance {loc[1]})"
            problems.append(
                f"{location}: invalid value {actual_repr}{where} — "
                f"field '{field}'. Allowed values: {expected}. "
                f"Regenerate ONLY '{field}'{where} and keep every other plan "
                "value unchanged."
            )
        else:
            problems.append(f"{location}: {err['msg']} (actual value: {actual_repr})")
    return problems


def normalize_plan_payload(data: Any) -> tuple[Any, list[str]]:
    """Defensive pre-Pydantic cleanup of a decoded MiMo plan payload.

    Normalizes `utterances[].delivery_style` and `utterances[].emotion`
    (trim / casing / safe morphological aliases) and records a note for every
    change, preserving the original raw value in the note for debugging.
    Unknown values stay exactly as returned so validation rejects them
    loudly. Returns (data, notes).
    """
    notes: list[str] = []
    if not isinstance(data, dict):
        return data, notes
    utterances = data.get("utterances")
    if not isinstance(utterances, list):
        return data, notes

    for position, utterance in enumerate(utterances):
        if not isinstance(utterance, dict):
            continue
        index = utterance.get("utterance_id", position)
        for field, normalizer in (
            ("delivery_style", normalize_delivery_style),
            ("emotion", normalize_emotion),
        ):
            raw = utterance.get(field)
            if not isinstance(raw, str):
                continue
            normalized, changed = normalizer(raw)
            if not changed:
                continue  # unknown values stay exactly as returned -> rejected
            notes.append(
                f"utterance {index}: {field} {raw!r} normalized to "
                f"{normalized!r}"
            )
            utterance[field] = normalized
    return data, notes


def validate_plan_against_script(
    plan: MultiVoicePlan,
    paragraphs: Sequence[str],
    cast: CharacterCast,
) -> list[str]:
    """Cross-check utterances against the script and the cast."""
    problems: list[str] = []
    speaker_ids = {s.character_id for s in cast.required_speakers()}
    character_ids = {
        c.character_id for c in cast.story.characters if c.has_spoken_dialogue
    }

    # --- ordering: strictly increasing (paragraph, sequence) ------------- #
    keys = [(u.paragraph_index, u.sequence_index) for u in plan.utterances]
    if keys != sorted(keys):
        problems.append(
            "utterances must be ordered by paragraph_index then sequence_index"
        )
    if len(set(keys)) != len(keys):
        problems.append("duplicate (paragraph_index, sequence_index) pairs")

    # --- paragraph coverage ---------------------------------------------- #
    seen_paragraphs = {u.paragraph_index for u in plan.utterances}
    expected = set(range(len(paragraphs)))
    if seen_paragraphs != expected:
        missing = sorted(expected - seen_paragraphs)
        extra = sorted(seen_paragraphs - expected)
        if missing:
            problems.append(f"missing paragraph_index value(s): {missing}")
        if extra:
            problems.append(f"unknown paragraph_index value(s): {extra}")

    # --- per paragraph: sequence contiguity, speakers, word fidelity ----- #
    by_paragraph: dict[int, list[Utterance]] = {}
    for utterance in plan.utterances:
        by_paragraph.setdefault(utterance.paragraph_index, []).append(utterance)

    for index, original in enumerate(paragraphs):
        group = by_paragraph.get(index, [])
        if not group:
            continue
        sequences = [u.sequence_index for u in group]
        if sequences != list(range(len(group))):
            problems.append(
                f"paragraph {index}: sequence_index must be 0..{len(group) - 1} "
                f"without gaps (got {sequences})"
            )
        for utterance in group:
            if utterance.speaker_id not in speaker_ids:
                problems.append(
                    f"paragraph {index}: speaker_id '{utterance.speaker_id}' is "
                    "not part of the cast (narrator or a character with "
                    "spoken dialogue)"
                )
            if (
                utterance.speaker_type == "character"
                and utterance.speaker_id not in character_ids
            ):
                problems.append(
                    f"paragraph {index}: '{utterance.speaker_id}' has no spoken "
                    "dialogue in the character cast"
                )
            if len(group) > 1 and utterance.sequence_index < len(group) - 1:
                if utterance.pause_after_ms > MAX_MID_PARAGRAPH_PAUSE_MS:
                    problems.append(
                        f"paragraph {index} utterance {utterance.sequence_index}: "
                        f"pause_after_ms {utterance.pause_after_ms} would insert "
                        f"unnecessary long silence inside a paragraph "
                        f"(max {MAX_MID_PARAGRAPH_PAUSE_MS})"
                    )

        # Word-level fidelity: utterances must reconstruct the paragraph.
        rebuilt = "".join(
            _spoken_skeleton(u.original_text) for u in sorted(
                group, key=lambda u: u.sequence_index
            )
        )
        if rebuilt != _spoken_skeleton(original):
            problems.append(
                f"paragraph {index}: the utterances' words do not reproduce the "
                "original paragraph — spoken words may never be changed, "
                "added, removed, translated or reordered"
            )

    return problems


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
class MultiVoicePlanner:
    """Drives MiMo, validates the utterance plan and persists it."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        project_service: Optional[ProjectService] = None,
        mimo_client: Optional[MiMoClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.projects = project_service or ProjectService(self.settings)
        # DI seam: tests inject a fake MiMo (no credits).
        self.mimo = mimo_client or MiMoClient(self.settings)

    def create(
        self, project_id: str, *, joined: bool = False
    ) -> MultiVoicePlanResponse:
        """Build (or recover) the utterance plan for an existing character cast.

        Server-authoritative plan recovery: when a plan artefact on disk
        still describes the CURRENT script + cast + generation-relevant
        narration preferences (server-computed fingerprint), it is reused —
        no second MiMo call and no second workflow. `joined` tells the
        planner that this request joined a workflow whose allowance was
        already spent (the middleware decided so); whether the artefact may
        be reused is ALWAYS decided from the stored project data, never from
        anything the browser sent.
        """
        self.projects.get_project(project_id)

        fingerprint = plan_identity.compute_fingerprint(self.settings, project_id)
        state = plan_identity.read_state(self.settings, project_id)
        stored = self.projects.get_multi_voice_plan(project_id)
        if stored is not None and plan_identity.artefact_is_reusable(
            state, fingerprint
        ):
            # Same logical workflow (Case A/B) or unchanged inputs after a
            # finished/failed run (Case D/E): the saved plan stands.
            plan, saved_at = stored
            plan_identity.mark_ready(
                self.settings, project_id, current_generation_id(), fingerprint
            )
            return MultiVoicePlanResponse.from_plan(
                project_id, plan, saved=True, saved_at=saved_at
            )

        script = self.projects.read_script(project_id)
        paragraphs = split_paragraphs(script)
        if not paragraphs:
            raise EmptyScriptError(project_id)

        cast = self.projects.get_character_cast(project_id)
        if cast is None:
            raise CharacterCastNotFoundError(project_id)

        roster = [
            {
                "speaker_id": speaker.character_id,
                "name": speaker.character_name,
                "role": speaker.role,
                "voice_description": speaker.voice_description,
            }
            for speaker in cast.required_speakers()
        ]

        plan = self.build_plan(paragraphs, roster, cast)
        saved_at = self.projects.save_multi_voice_plan(project_id, plan)
        # The build succeeded — the plan is ready for `generate-audio`
        # (`creating -> ready`, same generation_id).
        plan_identity.mark_ready(
            self.settings, project_id, current_generation_id(), fingerprint
        )
        return MultiVoicePlanResponse.from_plan(
            project_id, plan, saved=True, saved_at=saved_at
        )

    def get_plan(self, project_id: str) -> MultiVoicePlanResponse:
        self.projects.get_project(project_id)
        stored = self.projects.get_multi_voice_plan(project_id)
        if stored is None:
            raise MultiVoicePlanNotFoundError(project_id)
        plan, saved_at = stored
        return MultiVoicePlanResponse.from_plan(
            project_id, plan, saved=True, saved_at=saved_at
        )

    # ------------------------------------------------------------------ #
    # MiMo round trip + retries
    # ------------------------------------------------------------------ #
    def build_plan(
        self,
        paragraphs: Sequence[str],
        roster: Sequence[dict[str, str]],
        cast: CharacterCast,
    ) -> MultiVoicePlan:
        attempts = self.settings.mimo_max_attempts
        problems: list[str] = ["no response from MiMo"]

        for attempt in range(1, attempts + 1):
            user = build_user_prompt(
                paragraphs, roster, feedback=problems if attempt > 1 else None
            )
            raw = self.mimo.complete_json(system=SYSTEM_PROMPT, user=user)

            try:
                data: Any = json.loads(strip_code_fences(raw))
            except json.JSONDecodeError as exc:
                problems = [f"response was not valid JSON: {exc}"]
                continue
            if not isinstance(data, dict):
                problems = ["response must be a single JSON object"]
                continue

            # Story metadata comes from the approved cast — never from MiMo.
            data["language"] = cast.story.language or data.get("language") or "en"
            data["overall_tone"] = cast.story.overall_tone

            # Defensive normalization (trim / casing / safe morphological
            # aliases for emotion + delivery_style) BEFORE validation.
            data, _notes = normalize_plan_payload(data)

            try:
                plan = MultiVoicePlan.model_validate(data)
            except ValidationError as exc:
                problems = _format_validation_error(exc)
                continue

            problems = validate_plan_against_script(plan, paragraphs, cast)
            if problems:
                continue

            self._stamp(plan, cast)
            return plan

        raise MultiVoicePlanError(problems, attempts)

    def _stamp(self, plan: MultiVoicePlan, cast: CharacterCast) -> None:
        """Canonical ids + locked voices — MiMo never assigns voices itself."""
        for utterance in plan.utterances:
            utterance.utterance_id = canonical_utterance_id(
                utterance.paragraph_index, utterance.sequence_index
            )
            lock = cast.casting_for(utterance.speaker_id).selected
            utterance.voice_id = lock.voice_id if lock else None
            utterance.model_id = self.settings.elevenlabs_model_id
