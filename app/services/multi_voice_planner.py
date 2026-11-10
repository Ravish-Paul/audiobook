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
from app.schemas.narration_preferences import NarrationPreferences
from app.security import current_generation_id
from app.services.errors import (
    CharacterCastNotFoundError,
    EmptyScriptError,
    MultiVoicePlanError,
    MultiVoicePlanNotFoundError,
)
from app.services.mimo_client import MiMoClient
from app.services.narration_planner import (
    MAX_BATCH_CHARACTERS,
    MAX_BATCH_PARAGRAPHS,
    _preferences_block,
    _story_analysis_block,
    effective_profile,
    normalize_delivery_style,
    normalize_emotion,
    preferences_for_profile,
    split_batches,
    split_paragraphs,
)
from app.services.project_service import ProjectService
from app.schemas.story_analysis import StoryDirectorAnalysis

# Mid-paragraph utterances must not inject long silences (spec §9).
MAX_MID_PARAGRAPH_PAUSE_MS = 1000

# Long scripts are planned in BATCHES (shared with the single-voice planner:
# `MAX_BATCH_PARAGRAPHS` / `MAX_BATCH_CHARACTERS` / `split_batches`) so no
# single response can overflow the completion budget. Each batch keeps the
# GLOBAL paragraph indices; batches are merged and re-validated afterwards.


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
4. Narration and dialogue stay DISTINCT — double-check every speaker assignment before you answer. The narrator keeps narration AND dialogue attribution ("he said", "she whispered", "उसने कहा", "Meera frowned.") — EVEN WHEN IT IS ONLY ONE OR TWO WORDS sitting between two dialogue lines. Quoted dialogue becomes its own utterance for the character who actually speaks it (match each quote to its speaker through the attribution and the story context); everything outside the quotes stays the narrator's. NEVER merge an attribution beat into a character's utterance, NEVER let a character speak the narrator's words and NEVER let the narrator speak a character's quoted words. A pure-narration paragraph is ONE narrator utterance.

SPLITTING EXAMPLE (dialogue with short narration beats in between)
Script: “The door was open,” said Rohan, wiping his hands. “I didn’t touch anything.” Meera frowned. “Then who did?”
Correct utterances:
  1. speaker rohan (character): "The door was open,"
  2. speaker narrator (narrator): "said Rohan, wiping his hands."
  3. speaker rohan (character): "I didn’t touch anything."
  4. speaker narrator (narrator): "Meera frowned."
  5. speaker meera (character): "Then who did?"
The beats "said Rohan, wiping his hands." and "Meera frowned." are one-to-four words of NARRATION — they are NEVER a character's line, no matter how short. The same holds in Hindi: "उसने कहा,", "वह बोली।", "राहुल हँसा।" are always narrator utterances.
5. `tts_text`: copy of `original_text` with at most 1-2 RESTRAINED Eleven v4 audio tags when they clearly help — e.g. [whispers], [sad], [curious], [laughs], [sighs], [low, ominous voice], [quiet, tense delivery]; mood tags like [softly], [gently], [smooth], [sensually], [intimate] when the narration preferences call for that mood. Prefer NO tags over weak ones; never exaggerate or overact; no SSML. Stripping every tag (and its single separating space) must reproduce `original_text` byte-for-byte.
6. `emotion` and `delivery_style` must each be EXACTLY one value from the closed ALLOWED VALUES lists below — never invent a new value, never use a synonym or a variant spelling (words like "hope", "peace" or "warm" are NOT valid values); if unsure which value fits, use `neutral` (emotion) or `calm` (delivery_style). `acting_intensity` 1..5 — keep it natural and restrained for audiobooks (mostly 1-3, higher only for genuine peaks). `narration_direction`: one short sentence (metadata, never spoken).
7. `pause_after_ms`: mid-paragraph utterances 0..300 (natural sentence rhythm); the LAST utterance of each paragraph 300..1200 to mark the paragraph break. Never insert long silence inside a paragraph.
8. CHARACTER IDENTITY IS STABLE: a character's voice never changes with emotion. The character's `description`, `speaking_style` and the locked voice stay fixed; only the per-utterance emotion, intensity, delivery and direction may change. Do not invent a character's personality beyond what the script and cast establish.
9. NARRATOR STAYS SEPARATE: the narrator follows the GLOBAL narration profile and the current scene, never a character's dialogue emotion. Narration beats between quotes belong to the narrator; never copy a character's emotional direction onto the narrator.
10. Do NOT apply one emotion or direction everywhere. Ordinary narration stays neutral; a horror story has calm passages, a romance has ordinary conversation and conflict, and a sensual story's non-intimate scenes remain natural. Where the script lets you identify a scene/beat, fill the optional `scene_id`, `scene_purpose` and `pacing` metadata (planning notes, never spoken).
11. Output ONLY one JSON object (no fences):
{{
  "utterances": [
    {{"utterance_id": "any unique label", "paragraph_index": 0, "sequence_index": 0,
     "speaker_id": "narrator", "speaker_type": "narrator",
     "original_text": "<exact words>", "tts_text": "<exact words + optional [tags]>",
     "emotion": "<exactly one allowed emotion>", "acting_intensity": 2,
     "delivery_style": "<exactly one allowed delivery style>", "narration_direction": "<short note>",
     "scene_id": "<optional>", "scene_purpose": "<optional>", "pacing": "<optional>",
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
    *,
    index_offset: int = 0,
    preferences: Optional[NarrationPreferences] = None,
    total_paragraphs: Optional[int] = None,
    story_analysis: Optional[StoryDirectorAnalysis] = None,
) -> str:
    """One batch of script paragraphs, numbered with their GLOBAL indices.

    `index_offset` keeps the numbering continuous across batches — MiMo must
    see (and return) the exact `paragraph_index` values of the full script.
    """
    numbered = [
        {"paragraph_index": index_offset + index, "text": text}
        for index, text in enumerate(paragraphs)
    ]
    parts = [
        "Split the script below into utterances for this cast.",
        "",
        f"Total paragraphs: {len(paragraphs)}",
    ]
    total = total_paragraphs or len(paragraphs)
    if index_offset > 0 or total != len(paragraphs):
        parts.append(
            f"This is one BATCH of a longer script: it covers paragraph_index "
            f"{index_offset}..{index_offset + len(paragraphs) - 1} out of "
            f"{total}. Use these exact paragraph_index values — never "
            "renumber them."
        )
    parts += [
        "",
        "Cast (use only these speaker_id values):",
        json.dumps(list(roster), ensure_ascii=False, indent=2),
        "",
        "Script paragraphs (each becomes 1..k ordered utterances):",
        json.dumps(numbered, ensure_ascii=False, indent=2),
    ]
    block = _preferences_block(preferences, total)
    if block:
        parts += ["", block]
    analysis_block = _story_analysis_block(
        story_analysis, preferences, effective_profile(preferences, story_analysis)
    )
    if analysis_block:
        parts += ["", analysis_block]
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


# Final-attempt safety net. `emotion` / `delivery_style` are metadata only —
# the TTS client drops them, only `tts_text` and the locked voice ever reach
# the provider — so an out-of-enum value on these two fields can be defaulted
# without changing the spoken output one bit.
METADATA_ENUM_DEFAULTS: dict[str, str] = {
    "emotion": "neutral",
    "delivery_style": "calm",
}


def _metadata_enum_errors_only(exc: ValidationError) -> bool:
    """True when EVERY validation failure is an enum (`literal_error`) failure
    on the metadata-only `emotion`/`delivery_style` fields.

    Any other problem type (missing field, wrong type, out-of-range number)
    means the plan is NOT salvageable this way and must fail loudly. The
    metadata fields are only ever defaulted as a LAST resort — attempts 1..n-1
    always reject strictly so MiMo gets the chance to fix the value itself.
    """
    errors = exc.errors()
    return bool(errors) and all(
        err.get("type") == "literal_error"
        and str((err.get("loc") or ("<root>",))[-1]) in METADATA_ENUM_DEFAULTS
        for err in errors
    )


def coerce_metadata_enums(data: Any) -> list[str]:
    """Default unknown `emotion`/`delivery_style` values (final attempt only).

    Called ONLY when `_metadata_enum_errors_only` approved the salvage. Every
    value still outside its closed list becomes the safe default
    ('neutral' / 'calm'); the original raw value is preserved in the returned
    note. Spoken words, speakers and everything else stay untouched.
    """
    notes: list[str] = []
    if not isinstance(data, dict):
        return notes
    utterances = data.get("utterances")
    if not isinstance(utterances, list):
        return notes

    for position, utterance in enumerate(utterances):
        if not isinstance(utterance, dict):
            continue
        index = utterance.get("utterance_id", position)
        for field, normalizer, allowed in (
            ("emotion", normalize_emotion, EMOTIONS),
            ("delivery_style", normalize_delivery_style, DELIVERY_STYLES),
        ):
            raw = utterance.get(field)
            if not isinstance(raw, str):
                continue
            normalized, _changed = normalizer(raw)
            if normalized in allowed:
                if normalized != raw:
                    utterance[field] = normalized
                continue
            default = METADATA_ENUM_DEFAULTS[field]
            notes.append(
                f"utterance {index}: {field} {raw!r} is not a known value — "
                f"replaced with the safe default {default!r} (metadata only)"
            )
            utterance[field] = default
    return notes


# --------------------------------------------------------------------------- #
# Speaker sanity: narration beats never belong to a character
# --------------------------------------------------------------------------- #
# The live bug this fixes: one or two words of narration between two dialogue
# lines ("he said.", "she asked,", "उसने कहा,") were spoken by a CHARACTER.
# Three high-precision checks reject that mechanically (they only ever fire
# for `speaker_type == "character"` utterances and are fed back to MiMo):
#   1. the whole utterance is a bare dialogue beat;
#   2. the utterance STARTS with a beat merged into the character's line;
#   3. the utterance ENDS with a beat merged into the character's line;
# plus a quoted-dialogue check: in paragraphs with quotation marks, a
# character's words must lie INSIDE the quotes.
_BEAT_VERBS_EN = (
    "said|asked|replied|whispered|shouted|yelled|muttered|added|answered|"
    "continued|cried|called|murmured|repeated|insisted|snapped|growled|"
    "sighed|laughed|smiled|nodded"
)
_BEAT_VERBS_HI = (
    "ने\\s+कहा|ने\\s+पूछा|कहा|पूछा|बोला|बोली|फुसफुसाया|फुसफुसाई|"
    "चिल्लाया|चिल्लाई|जवाब\\s+दिया|मुस्कुराया|मुस्कुराई|हँसा|हँसी|"
    "सिर\\s+हिलाया|चुप\\s+हो\\s+गया|चुप\\s+हो\\s+गई"
)
_EN_BEAT_SUBJ = (
    r"(?!(?:I|i|We|we|Me|me|My|my|Mine|mine|Us|us|Our|our)\b)"
    r"[A-Za-z][\w'’-]*(?:\s+[A-Za-z][\w'’-]*){0,3}"
)
_HI_BEAT_SUBJ = r"(?!मैं|मैंने|मुझे|हम|हमने)[\u0900-\u097F'’ ]{1,40}?"
_BEAT_END_PUNCT = r"[,.:;!?…\u0964\u0965\s]*"

# 1. the whole utterance is a bare beat ("she said." / "said Rohan." /
#    "उसने कहा," / "कहा।")
_BEAT_WHOLE_RE = re.compile(
    rf"^(?:"
    rf"{_EN_BEAT_SUBJ}\s+(?i:{_BEAT_VERBS_EN})\b{_BEAT_END_PUNCT}$"
    rf"|(?i:{_BEAT_VERBS_EN})\b\s+(?:[Hh]e|[Ss]he|[Tt]hey|[Ii]t|[A-Z][\w'’-]*)"
    rf"{_BEAT_END_PUNCT}$"
    rf"|{_HI_BEAT_SUBJ}\s+(?:{_BEAT_VERBS_HI}){_BEAT_END_PUNCT}$"
    rf"|(?:{_BEAT_VERBS_HI}){_BEAT_END_PUNCT}$"
    rf")"
)
# 2. a beat merged at the START of the character's line ("she said, go on.")
_BEAT_LEAD_RE = re.compile(
    rf"^(?:"
    rf"{_EN_BEAT_SUBJ}\s+(?i:{_BEAT_VERBS_EN})\b"
    rf"|{_HI_BEAT_SUBJ}\s+(?:{_BEAT_VERBS_HI})"
    rf")\s*,\s*\S"
)
# 3. a beat merged at the END of the character's line ("go on, she said.")
_BEAT_TRAIL_RE = re.compile(
    rf"\S\s*,\s*(?:"
    rf"{_EN_BEAT_SUBJ}\s+(?i:{_BEAT_VERBS_EN})\b"
    rf"|{_HI_BEAT_SUBJ}\s+(?:{_BEAT_VERBS_HI})"
    rf")\s*[.!?…\u0964\u0965]*$"
)
_NON_THIRD_PERSON_RE = re.compile(
    r"\b(?:I|i|me|my|mine|we|us|our|ours|you|your|yours)\b"
    r"|मैं|मैंने|मुझे|मेरा|मेरी|हम|हमने|हमारा|तुम|तुमने|तुम्हें|तेरा|तेरी|आप|आपने"
)
_QUOTE_CHARS = "“”‘’«»„\"'"

# Quote pairs: (opener, closer). ASCII single quotes only open/close a span at
# word boundaries, so apostrophes ("don't") never delimit anything.
_QUOTE_PAIRS = (
    ("\u201c", "\u201d"),  # “ ”
    ("\u2018", "\u2019"),  # ‘ ’
    ("\u00ab", "\u00bb"),  # « »
    ("\u201e", "\u201c"),  # „ “
    ('"', '"'),
    ("'", "'"),
)


def _quoted_skeleton(text: str) -> str:
    """Spoken-word skeleton of every QUOTED span of `text` ("" when none).

    Unbalanced quotation marks skip the quoted-dialogue check entirely — the
    validator must never guess where a quote ends.
    """
    spans: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        opener = text[index]
        pair = next((p for p in _QUOTE_PAIRS if p[0] == opener), None)
        if pair is None:
            index += 1
            continue
        if index > 0 and (text[index - 1].isalnum() or text[index - 1] == "'"):
            index += 1  # an apostrophe inside a word — not a delimiter
            continue
        closer = pair[1]
        end = -1
        for scan in range(index + 1, length):
            if text[scan] == closer and (
                scan + 1 >= length
                or not (text[scan + 1].isalnum() or text[scan + 1] == "'")
            ):
                end = scan
                break
        if end < 0:
            return ""  # unbalanced — never guess
        spans.append(text[index + 1 : end])
        index = end + 1
    return "".join(_spoken_skeleton(span) for span in spans)


def _has_quote(text: str) -> bool:
    return any(quote in text for quote in _QUOTE_CHARS)


def _narration_beat_problems(utterance: Utterance, quoted: str) -> list[str]:
    """Speaker-sanity problems for one CHARACTER utterance (never narrator)."""
    where = (
        f"paragraph {utterance.paragraph_index} utterance "
        f"{utterance.sequence_index}"
    )
    text = (utterance.original_text or "").strip()
    speaker = utterance.speaker_id

    # 1. The whole utterance is a bare narration beat ("she said." /
    #    "उसने कहा,"). First/second-person wording ("I said.", "You said.")
    #    may be genuine terse dialogue — never flag it.
    if _BEAT_WHOLE_RE.match(text) and not _NON_THIRD_PERSON_RE.search(text):
        return [
            f"{where}: {text!r} is narration (a dialogue beat like 'he said' "
            f"/ 'उसने कहा') but is assigned to character '{speaker}' — "
            "narration beats ALWAYS belong to the narrator; give the "
            "character only the spoken words"
        ]

    # 2. A beat merged at the START of the character's line. First/second
    #    person or quotes in the REST of the line mean the words may be
    #    genuine dialogue — never flag those.
    lead = _BEAT_LEAD_RE.match(text)
    if lead:
        rest = text[lead.end() :]
        if not _NON_THIRD_PERSON_RE.search(rest) and not _has_quote(rest):
            beat = text.split(",", 1)[0].strip()
            return [
                f"{where}: the narration beat {beat!r} is merged into "
                f"character '{speaker}'s line — split it out into its own "
                "narrator utterance; a character only speaks the dialogue "
                "words"
            ]
        return []

    # 3. A beat merged at the END of the character's line ("I'm fine, she
    #    said."). Quotes in the dialogue part mean it may be genuine speech.
    trail = _BEAT_TRAIL_RE.search(text)
    if trail:
        rest = text[: trail.start()]
        if not _has_quote(rest):
            beat = text[trail.start() :].split(",", 1)[-1].strip()
            return [
                f"{where}: the narration beat {beat!r} is merged into "
                f"character '{speaker}'s line — split it out into its own "
                "narrator utterance; a character only speaks the dialogue "
                "words"
            ]
        return []

    # 4. Quoted paragraphs: a character's words must lie INSIDE the quotes —
    #    attribution and action beats between them stay the narrator's.
    if quoted and _spoken_skeleton(utterance.original_text) not in quoted:
        return [
            f"{where}: '{speaker}' is assigned words outside the quoted "
            "dialogue — only the words inside the quotation marks belong to "
            "a character; everything outside them (attribution, action "
            "beats) is the narrator's"
        ]
    return []


def validate_plan_against_script(
    plan: MultiVoicePlan,
    paragraphs: Sequence[str],
    cast: CharacterCast,
    *,
    index_offset: int = 0,
) -> list[str]:
    """Cross-check utterances against the script and the cast.

    `index_offset` is the GLOBAL paragraph index of `paragraphs[0]` — used to
    validate one batch of a longer script; the utterances always carry their
    global `paragraph_index`.
    """
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
    expected = set(range(index_offset, index_offset + len(paragraphs)))
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

    for position, original in enumerate(paragraphs):
        index = index_offset + position
        group = by_paragraph.get(index, [])
        if not group:
            continue
        quoted = _quoted_skeleton(original)
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
            # Speaker sanity: narration beats ("he said.", "उसने कहा,") and
            # words outside the quoted dialogue NEVER belong to a character.
            if (
                utterance.speaker_type == "character"
                and utterance.speaker_id in character_ids
            ):
                problems += _narration_beat_problems(utterance, quoted)
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

        # The stored narration style drives the tags and delivery the plan may
        # use — it is part of the plan fingerprint too, so changing the style
        # always rebuilds the plan. In `auto` mode the whole-story analysis
        # resolves the concrete global profile.
        stored_preferences = self.projects.get_narration_preferences(project_id)
        preferences = stored_preferences[0] if stored_preferences else None
        analysis = self._story_analysis_for(preferences, project_id)

        plan = self.build_plan(
            paragraphs,
            roster,
            cast,
            preferences=preferences,
            story_analysis=analysis,
        )
        plan.story_analysis = analysis
        plan.effective_profile = effective_profile(preferences, analysis)
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
    def _story_analysis_for(
        self,
        preferences: Optional[NarrationPreferences],
        project_id: str,
    ) -> Optional[StoryDirectorAnalysis]:
        from app.services.story_analysis_service import StoryAnalysisService

        service = StoryAnalysisService(
            self.settings,
            project_service=self.projects,
            mimo_client=self.mimo,
        )
        stored = service.load(project_id)
        if stored is not None:
            return stored
        if preferences is not None and preferences.preset == "auto":
            return service.analyze(project_id).analysis
        return None

    def build_plan(
        self,
        paragraphs: Sequence[str],
        roster: Sequence[dict[str, str]],
        cast: CharacterCast,
        preferences: Optional[NarrationPreferences] = None,
        story_analysis: Optional[StoryDirectorAnalysis] = None,
    ) -> MultiVoicePlan:
        """Build the plan — batched for long scripts, then merged and checked.

        Each batch is planned and validated on its own (own retry loop, GLOBAL
        paragraph indices); afterwards the MERGED plan is re-validated against
        the whole script — the second, dual check — before any voice is
        stamped. Invalid output is retried with feedback, then rejected.
        """
        batches = split_batches(paragraphs)
        utterances: list[Utterance] = []
        for offset, batch in batches:
            utterances.extend(
                self._build_batch(
                    batch,
                    roster,
                    cast,
                    index_offset=offset,
                    preferences=preferences,
                    total_paragraphs=len(paragraphs),
                    story_analysis=story_analysis,
                )
            )

        merged = MultiVoicePlan.model_validate(
            {
                "language": cast.story.language or "en",
                "overall_tone": cast.story.overall_tone,
                "utterances": [u.model_dump() for u in utterances],
                "story_analysis": story_analysis.model_dump()
                if story_analysis is not None
                else None,
                "effective_profile": effective_profile(
                    preferences, story_analysis
                ),
            }
        )
        problems = validate_plan_against_script(merged, paragraphs, cast)
        if problems:
            raise MultiVoicePlanError(problems, self.settings.mimo_max_attempts)

        self._stamp(merged, cast)
        return merged

    def _build_batch(
        self,
        paragraphs: Sequence[str],
        roster: Sequence[dict[str, str]],
        cast: CharacterCast,
        *,
        index_offset: int,
        preferences: Optional[NarrationPreferences],
        total_paragraphs: int,
        story_analysis: Optional[StoryDirectorAnalysis] = None,
    ) -> list[Utterance]:
        attempts = self.settings.mimo_max_attempts
        problems: list[str] = ["no response from MiMo"]

        for attempt in range(1, attempts + 1):
            user = build_user_prompt(
                paragraphs,
                roster,
                feedback=problems if attempt > 1 else None,
                index_offset=index_offset,
                preferences=preferences,
                total_paragraphs=total_paragraphs,
                story_analysis=story_analysis,
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
                if attempt == attempts and _metadata_enum_errors_only(exc):
                    # FINAL attempt, and the only failures left are enum values
                    # on the metadata-only `emotion`/`delivery_style` fields.
                    # Degrade gracefully: default them (spoken output is
                    # unaffected) instead of failing the whole generation.
                    # Every other problem type still fails loudly below.
                    coerce_metadata_enums(data)
                    try:
                        plan = MultiVoicePlan.model_validate(data)
                    except ValidationError as retry_exc:
                        problems = _format_validation_error(retry_exc)
                        continue
                else:
                    problems = _format_validation_error(exc)
                    continue

            problems = validate_plan_against_script(
                plan, paragraphs, cast, index_offset=index_offset
            )
            if problems:
                continue

            return plan.utterances

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
