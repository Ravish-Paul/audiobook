"""NarrationPolicyValidator — enforces the user's narration preferences.

Where it runs (see `NarrationPlanner.build_plan`):

    MiMo JSON → json.loads → NarrationPlan (Pydantic schema)
              → validate_against_script + require_tts_text
              → NarrationPolicyValidator.validate        ← this module
              → save narration_plan.json

Pydantic only knows *ranges* (intensity 1..5, stability 0..1 …). This
validator knows what the USER asked for. Every violation comes back as a
targeted feedback string that the retry loop sends to MiMo; when the retries
are exhausted the plan is rejected (502) — a policy violation is NEVER
silently accepted.

Rules
-----
1. whisper        — `whisper_policy`: never / only user-selected paragraphs /
                    rarely (budgeted fraction of the plan).
2. tag density    — max inline tags per paragraph (minimal 1, standard 3,
                    expressive 8).
3. acting intensity — no paragraph may exceed `acting_intensity`.
4. dramatic / amplified instructions — shouting, sobbing, trembling,
                    breathless, panic … are budgeted per preset (natural = 0).
5. unnecessary delivery changes — an amplified tag on a paragraph that
                    declares `neutral` + `calm`/`conversational`.
6. voice settings — unsupported parameters for the configured model must stay
                    at their inert defaults, restrained presets keep stability
                    reasonably high, and common settings stay consistent
                    across paragraphs.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Sequence

from app.schemas.narration import NarrationPlan, ParagraphPlan
from app.schemas.narration_preferences import (
    TAG_DENSITY_LABELS,
    NarrationPreferences,
)
from app.services.eleven_models import ModelCapabilities, get_capabilities

# --------------------------------------------------------------------------- #
# Marker vocabularies
# --------------------------------------------------------------------------- #
# "whispering" delivery (tags or direction prose). Word boundaries keep
# negating prefixes ("unwhispered", "non-whispered") out of the vocabulary —
# those forbid whispering and are handled as explicit negations below.
# The breathy/breathless family belongs to the same delivery family
# ("speak breathlessly" whispers just as much as "whispered").
WHISPER_RE = re.compile(
    r"\b(?:whisper\w*|hush\w*|breath(?:y|ily|less\w*))\b", re.IGNORECASE
)
# delivery that amplifies a scene into overacting
AMPLIFIED_RE = re.compile(
    r"\b(?:shouts?|screams?|yells?|shrieks?|roars?|bellows?|sobs?|weeps?|"
    r"wails?|trembling|trembles?|trembled|quivers?|breathless\w*|panting|"
    r"panics?|panicked|hysterical\w*|devastated)\w*",
    re.IGNORECASE,
)
# theatrical instruction wording
DRAMATIC_RE = re.compile(
    r"\b(?:dramatic\w*|theatrical\w*|melodramatic\w*|exaggerated\w*|"
    r"over-the-top|larger than life)\b",
    re.IGNORECASE,
)

# How many amplified/dramatic markers each preset tolerates, as a fraction of
# the paragraphs being validated. `natural` tolerates none at all.
PRESET_MARKER_FRACTION: dict[str, float] = {
    "natural": 0.0,
    "restrained_cinematic": 0.15,
    "expressive": 0.5,
    "dramatic": 1.0,
}

# Minimum stability for the restrained presets (an unnecessarily low
# stability makes the voice unstable and "acted").
RESTRAINED_STABILITY_FLOOR = 0.45
RESTRAINED_PRESETS = ("natural", "restrained_cinematic")

# Voice identity must not drift between paragraphs.
MAX_STABILITY_SPREAD = 0.35
MAX_SIMILARITY_SPREAD = 0.25

# Inert values for parameters the configured model cannot accept.
INERT_SPEED = 1.0
INERT_STYLE = 0.0

_EPSILON = 1e-6


def max_tags_per_paragraph(tag_density: str) -> int:
    """Hard per-paragraph tag budget for a density preference."""
    return {"minimal": 1, "standard": 3, "expressive": 8}.get(tag_density, 1)


def marker_budget(preset: str, paragraph_count: int) -> int:
    """How many amplified/dramatic markers a preset allows for N paragraphs."""
    fraction = PRESET_MARKER_FRACTION.get(preset, 0.0)
    return math.ceil(paragraph_count * fraction)


def whisper_budget(paragraph_count: int) -> int:
    """`rare` may whisper in at most 10% of the paragraphs (always ≥ 1)."""
    return max(1, math.ceil(paragraph_count * 0.1))


def _contains(pattern: re.Pattern[str], text: str) -> bool:
    return bool(text) and bool(pattern.search(text))


# --------------------------------------------------------------------------- #
# Negation-aware instruction detection
# --------------------------------------------------------------------------- #
# A whisper term only ASKS for whispering when the wording does not explicitly
# forbid it: "never breathy or whispered" prohibits the delivery, while
# "quiet whispered delivery" requests it. Every whisper term is therefore
# checked against its left context before it counts as a violation.
_TOKEN_RE = re.compile(r"[\w']+|[^\w\s]")
_WORD_RE = re.compile(r"\w")

# Negators with wide scope: they negate the whole phrase that follows them,
# including coordinated lists ("never breathy or whispered").
WIDE_NEGATORS = frozenset(
    {
        "never",
        "not",
        "no",
        "none",
        "nothing",
        "without",
        "neither",
        "n't",
        "cannot",
        "can't",
        "dont",
        "don't",
        "doesnt",
        "doesn't",
        "didnt",
        "didn't",
        "isnt",
        "isn't",
        "arent",
        "aren't",
        "wasnt",
        "wasn't",
        "werent",
        "weren't",
        "wont",
        "won't",
        "wouldnt",
        "wouldn't",
        "shouldnt",
        "shouldn't",
        "couldnt",
        "couldn't",
        "mustnt",
        "mustn't",
        "neednt",
        "needn't",
    }
)

# Negators that only negate their own complement ("avoid whispering"): their
# scope does NOT reach over the next clause — "avoid whispering and speak
# breathlessly" still asks for breathless delivery.
NARROW_NEGATORS = frozenset(
    {
        "avoid",
        "avoids",
        "avoided",
        "avoiding",
        "skip",
        "skips",
        "skipped",
        "skipping",
        "stop",
        "stops",
        "stopped",
        "stopping",
        "prevent",
        "prevents",
        "prevented",
        "preventing",
        "refrain",
        "refrains",
        "refrained",
        "forbid",
        "forbids",
        "forbidden",
        "prohibit",
        "prohibits",
        "prohibited",
        "exclude",
        "excludes",
        "excluding",
        "omit",
        "omits",
        "omitted",
    }
)

COORDINATORS = frozenset({"or", "and", "nor", "either", "versus", "vs", "&"})
CONTRASTIVES = frozenset({"but", "however", "although", "though", "yet", "except"})
CLAUSE_BOUNDARIES = frozenset(".!?;:\n\u2014\u2013")  # sentence/clause ends + dashes
MAX_NEGATION_SCOPE_WORDS = 8
_NEGATING_PREFIX_RE = re.compile(r"\b(?:un|non)[-']?\s*$", re.IGNORECASE)


def _tokenize(text: str) -> list[tuple[str, bool]]:
    """(token, is_word) pairs — punctuation stays separate so scope rules
    can treat commas and clause boundaries differently."""
    return [
        (match.group(0), bool(_WORD_RE.search(match.group(0))))
        for match in _TOKEN_RE.finditer(text)
    ]


def is_negated_instruction(text: str, match_start: int) -> bool:
    """True when the term at `match_start` is explicitly FORBIDDEN.

    Scans the words to the left of the term and decides whether a negator
    covers it:

    * wide negators (`never`, `not`, `no`, `without`, …) cover the phrase
      that follows them, including coordinated lists;
    * narrow negators (`avoid`, `stop`, …) cover only their own complement,
      so they do not stretch over the next clause;
    * the scope ends at a clause boundary (`.`, `;`, `:`, `—`, newline), at a
      contrastive (`but`), at a plain clause comma and after
      `MAX_NEGATION_SCOPE_WORDS` ordinary words.
    """
    if not text:
        return False
    prefix = text[:match_start]
    if _NEGATING_PREFIX_RE.search(prefix):
        return True  # "un-whispered" / "non-whispered"

    opaque = 0
    coordinator_seen = False
    crossed_non_modifier = False
    narrow_allowed = True

    for token, is_word in reversed(_tokenize(prefix)):
        low = token.lower()
        if is_word:
            if low in WIDE_NEGATORS:
                return True
            if low in NARROW_NEGATORS and narrow_allowed:
                return True
            if low in COORDINATORS:
                coordinator_seen = True
                if crossed_non_modifier:
                    # Two clauses coordinated ("…and speak breathlessly"):
                    # a narrow negator no longer covers the second clause.
                    narrow_allowed = False
                continue
            if low in CONTRASTIVES:
                return False
            opaque += 1
            if opaque > MAX_NEGATION_SCOPE_WORDS:
                return False
            if not WHISPER_RE.fullmatch(token):
                crossed_non_modifier = True
        else:
            if token in CLAUSE_BOUNDARIES:
                return False
            if token == ",":
                if not coordinator_seen:
                    return False  # a clause comma ends the negation scope
                continue
            # quotes, brackets, hyphens are transparent
    return False


def whisper_terms(text: str) -> list[str]:
    """Whisper words that are INSTRUCTED (explicitly forbidden ones excluded)."""
    if not text:
        return []
    return [
        match.group(0)
        for match in WHISPER_RE.finditer(text)
        if not is_negated_instruction(text, match.start())
    ]


def has_whisper_instruction(text: str) -> bool:
    """True when `text` asks for whispered/breathy/hushed delivery."""
    return bool(whisper_terms(text))


def whisper_snippets(paragraph: ParagraphPlan) -> list[str]:
    """Every whisper-REQUESTING instruction in this paragraph.

    Scans the audio tags inside `tts_text` and `narration_direction`. Explicit
    prohibitions ("never breathy or whispered", "unwhispered") are NOT
    snippets — they forbid whispering instead of asking for it. The untagged
    prose of `tts_text` is the spoken script ("She whispered.") and is never
    treated as a delivery instruction.
    """
    snippets: list[str] = []
    for tag in _tags_of(paragraph):
        if has_whisper_instruction(tag):
            snippets.append(f"[{tag}]")
    if has_whisper_instruction(paragraph.narration_direction):
        snippets.append(f"narration_direction ({paragraph.narration_direction[:60]})")
    return snippets


def _tags_of(paragraph: ParagraphPlan) -> list[str]:
    from app.services.audio_tags import find_tags

    return find_tags(paragraph.tts_text or "")


def amplified_snippets(paragraph: ParagraphPlan) -> list[str]:
    """Shouting / trembling / breathless style instructions in this paragraph."""
    snippets: list[str] = []
    for tag in [f"[{tag}]" for tag in _tags_of(paragraph)]:
        if _contains(AMPLIFIED_RE, tag):
            snippets.append(tag)
    if _contains(AMPLIFIED_RE, paragraph.narration_direction):
        snippets.append("narration_direction")
    return snippets


def dramatic_snippets(paragraph: ParagraphPlan) -> list[str]:
    """Over-the-top theatrical wording in this paragraph."""
    snippets: list[str] = []
    for tag in [f"[{tag}]" for tag in _tags_of(paragraph)]:
        if _contains(DRAMATIC_RE, tag):
            snippets.append(tag)
    if _contains(DRAMATIC_RE, paragraph.narration_direction):
        snippets.append("narration_direction")
    return snippets


class NarrationPolicyValidator:
    """Check a Pydantic-valid plan against the user's narration preferences."""

    def __init__(
        self,
        preferences: NarrationPreferences,
        *,
        capabilities: Optional[ModelCapabilities] = None,
        index_map: Optional[dict[int, int]] = None,
        model_id: str = "",
    ) -> None:
        """
        preferences  — the user's preferences (enforced as a hard contract).
        capabilities — ElevenLabs model capabilities (defaults to `model_id`).
        index_map    — plan paragraph_index → SCRIPT paragraph_index. The
                       preview endpoint validates a window of the script, so
                       whisper allowances (which store script indexes) need
                       the mapping; identity when omitted.
        """
        self.preferences = preferences
        self.capabilities = capabilities or get_capabilities(model_id)
        self.index_map = index_map or {}

    # ------------------------------------------------------------------ #
    def validate(self, plan: NarrationPlan) -> list[str]:
        """Return every policy violation (empty when the plan is acceptable)."""
        paragraphs = plan.paragraphs
        problems: list[str] = []
        problems += self._check_intensity(paragraphs)
        problems += self._check_whisper(paragraphs)
        problems += self._check_tag_density(paragraphs)
        problems += self._check_marker_budgets(paragraphs)
        problems += self._check_unnecessary_delivery_changes(paragraphs)
        problems += self._check_voice_settings(paragraphs)
        return problems

    # ------------------------------------------------------------------ #
    def _script_index(self, paragraph: ParagraphPlan) -> int:
        return self.index_map.get(paragraph.paragraph_index, paragraph.paragraph_index)

    def _check_intensity(self, paragraphs: Sequence[ParagraphPlan]) -> list[str]:
        ceiling = self.preferences.acting_intensity
        problems: list[str] = []
        for paragraph in paragraphs:
            if paragraph.intensity > ceiling:
                problems.append(
                    f"paragraph {paragraph.paragraph_index}: intensity "
                    f"{paragraph.intensity} exceeds the user's acting_intensity "
                    f"{ceiling} — lower it to {ceiling} or below "
                    "(the user wants a controlled performance)"
                )
        return problems

    def _check_whisper(self, paragraphs: Sequence[ParagraphPlan]) -> list[str]:
        policy = self.preferences.whisper_policy
        allowed = set(self.preferences.whisper_paragraphs)
        problems: list[str] = []

        whispered = [
            (paragraph, snippet)
            for paragraph in paragraphs
            for snippet in whisper_snippets(paragraph)
        ]

        if policy == "never" and whispered:
            for paragraph, snippet in whispered:
                problems.append(
                    f"paragraph {paragraph.paragraph_index}: found whispering "
                    f"({snippet}) but whisper_policy is 'Never' — remove the "
                    "whisper/hushed instruction and play the line naturally"
                )
            return problems

        if policy == "user_selected":
            for paragraph, snippet in whispered:
                if self._script_index(paragraph) not in allowed:
                    allowed_text = sorted(allowed) or "none"
                    problems.append(
                        f"paragraph {paragraph.paragraph_index}: found whispering "
                        f"({snippet}) but whisper_policy is 'Only user-selected "
                        f"lines' and this line was not selected (selected "
                        f"paragraphs: {allowed_text}) — remove the whisper/hushed "
                        "instruction"
                    )
            return problems

        if policy == "rare":
            budget = whisper_budget(len(paragraphs))
            whispered_paragraphs = sorted(
                {paragraph.paragraph_index for paragraph, _ in whispered}
            )
            if len(whispered_paragraphs) > budget:
                problems.append(
                    f"whispering appears in {len(whispered_paragraphs)} "
                    f"paragraphs {whispered_paragraphs} but whisper_policy "
                    f"'AI may use rarely when essential' allows at most {budget} "
                    "— keep the narration unwhispered except where it is "
                    "genuinely essential"
                )
        return problems

    def _check_tag_density(self, paragraphs: Sequence[ParagraphPlan]) -> list[str]:
        density = self.preferences.tag_density
        cap = max_tags_per_paragraph(density)
        label = TAG_DENSITY_LABELS.get(density, density)
        problems: list[str] = []
        for paragraph in paragraphs:
            count = len(_tags_of(paragraph))
            if count > cap:
                problems.append(
                    f"paragraph {paragraph.paragraph_index}: {count} audio tags "
                    f"but audio tag density '{label}' allows at most {cap} — "
                    "remove the extra tags (prefer no tag over a weak one)"
                )
        return problems

    def _check_marker_budgets(self, paragraphs: Sequence[ParagraphPlan]) -> list[str]:
        preset = self.preferences.preset
        budget = marker_budget(preset, len(paragraphs))
        problems: list[str] = []

        for kind, snippets_for in (
            ("amplified", amplified_snippets),
            ("dramatic", dramatic_snippets),
        ):
            hits: list[tuple[ParagraphPlan, str]] = [
                (paragraph, snippet)
                for paragraph in paragraphs
                for snippet in snippets_for(paragraph)
            ]
            if kind == "dramatic":
                # `delivery_style: dramatic` is itself a dramatic instruction,
                # even when no tag or direction spells it out.
                hits += [
                    (paragraph, "delivery_style 'dramatic'")
                    for paragraph in paragraphs
                    if paragraph.delivery_style == "dramatic"
                ]
                hits.sort(key=lambda hit: hit[0].paragraph_index)
            if len(hits) <= budget:
                continue
            if kind == "amplified":
                reason = (
                    "shouting/sobbing/trembling/breathless-style delivery"
                )
            else:
                reason = "over-the-top dramatic instructions"
            for paragraph, snippet in hits[:5]:
                problems.append(
                    f"paragraph {paragraph.paragraph_index}: {reason} ({snippet}) "
                    f"is overused for the '{preset}' preset — at most {budget} "
                    "such instruction(s) are allowed; keep the narration a "
                    "natural, controlled human performance"
                )
            if len(hits) > 5:
                problems.append(
                    f"…and {len(hits) - 5} more overacting instructions were "
                    "added; strip them all"
                )
        return problems

    def _check_unnecessary_delivery_changes(
        self, paragraphs: Sequence[ParagraphPlan]
    ) -> list[str]:
        """An amplified tag on a paragraph that declares 'nothing changes'."""
        problems: list[str] = []
        allowed_whisper = set(self.preferences.whisper_paragraphs)
        for paragraph in paragraphs:
            if paragraph.emotion != "neutral":
                continue
            if paragraph.delivery_style not in ("calm", "conversational"):
                continue
            if paragraph.intensity > 2:
                continue
            for tag in _tags_of(paragraph):
                if not _contains(AMPLIFIED_RE, tag):
                    continue
                if has_whisper_instruction(tag) and (
                    self._script_index(paragraph) in allowed_whisper
                ):
                    continue  # the user explicitly asked for this line
                problems.append(
                    f"paragraph {paragraph.paragraph_index}: emotion 'neutral' + "
                    f"delivery_style '{paragraph.delivery_style}' says nothing "
                    f"changes, yet [{tag}] forces a delivery change — either "
                    "drop the tag or justify it with the declared delivery"
                )
        return problems

    def _check_voice_settings(self, paragraphs: Sequence[ParagraphPlan]) -> list[str]:
        problems: list[str] = []
        problems += self._check_unsupported_parameters(paragraphs)
        problems += self._check_stability_floor(paragraphs)
        problems += self._check_consistency(paragraphs)
        return problems

    def _check_unsupported_parameters(
        self, paragraphs: Sequence[ParagraphPlan]
    ) -> list[str]:
        """speed/style that the model never receives must stay inert."""
        caps = self.capabilities
        model = caps.model_id or "the configured model"
        problems: list[str] = []
        for paragraph in paragraphs:
            settings = paragraph.voice_settings
            if not caps.supports_speed and abs(settings.speed - INERT_SPEED) > _EPSILON:
                problems.append(
                    f"paragraph {paragraph.paragraph_index}: speed "
                    f"{settings.speed} is NOT supported by {model} and is never "
                    f"sent — set speed to {INERT_SPEED} and express pacing with "
                    "pause_after_ms / narration_direction instead"
                )
            if not caps.supports_style and abs(settings.style - INERT_STYLE) > _EPSILON:
                problems.append(
                    f"paragraph {paragraph.paragraph_index}: style "
                    f"{settings.style} is NOT supported by {model} and is never "
                    f"sent — set style to {INERT_STYLE} (style is not an "
                    "effective parameter for this model)"
                )
        return problems

    def _check_stability_floor(
        self, paragraphs: Sequence[ParagraphPlan]
    ) -> list[str]:
        if self.preferences.preset not in RESTRAINED_PRESETS:
            return []
        problems: list[str] = []
        for paragraph in paragraphs:
            stability = paragraph.voice_settings.stability
            if stability < RESTRAINED_STABILITY_FLOOR:
                problems.append(
                    f"paragraph {paragraph.paragraph_index}: stability "
                    f"{stability} is too low for the "
                    f"'{self.preferences.preset}' preset — keep stability at "
                    f"{RESTRAINED_STABILITY_FLOOR} or above so restrained "
                    "narration stays steady"
                )
        return problems

    def _check_consistency(self, paragraphs: Sequence[ParagraphPlan]) -> list[str]:
        if len(paragraphs) < 2:
            return []
        problems: list[str] = []
        stabilities = [p.voice_settings.stability for p in paragraphs]
        similarities = [p.voice_settings.similarity_boost for p in paragraphs]

        stability_spread = max(stabilities) - min(stabilities)
        if stability_spread > MAX_STABILITY_SPREAD + _EPSILON:
            problems.append(
                f"stability drifts from {min(stabilities)} to "
                f"{max(stabilities)} across paragraphs (spread "
                f"{round(stability_spread, 2)} > {MAX_STABILITY_SPREAD}) — keep "
                "the voice identity consistent; vary delivery through direction "
                "and pauses, not through large stability swings"
            )

        similarity_spread = max(similarities) - min(similarities)
        if similarity_spread > MAX_SIMILARITY_SPREAD + _EPSILON:
            problems.append(
                f"similarity_boost drifts from {min(similarities)} to "
                f"{max(similarities)} across paragraphs (spread "
                f"{round(similarity_spread, 2)} > {MAX_SIMILARITY_SPREAD}) — "
                "keep common voice settings consistent across the audiobook"
            )
        return problems
