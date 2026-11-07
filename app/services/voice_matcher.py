"""Character voice matching (spec §4): rank candidate voices per speaker.

Two raw payload shapes (account `VoiceResponseModel`, library
`LibraryVoiceResponseModel`) are normalised into one structure, then scored
against a speaker's profile on seven weighted criteria:

    language/locale 0.25 · accent 0.15 · vocal personality 0.20 ·
    age/gender 0.10 (only when the SCRIPT established them) ·
    audiobook suitability 0.15 · voice quality/previews 0.10 ·
    usage restrictions & cost 0.05

Hard exclusions (never returned as candidates):

* restricted voices (safety control bans);
* voices that declare their languages and do NOT support the required one;
* voices scoring below MIN_ACCEPT_SCORE — weak matches are dropped instead
  of being auto-selected just to fill the cast.

Top `limit` (default 3) survivors are returned with a full score breakdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.config import language_to_iso
from app.schemas.casting import CharacterProfile, VoiceCandidate

# Weighted criteria (sum == 1.0).
WEIGHTS: dict[str, float] = {
    "language": 0.25,
    "accent": 0.15,
    "personality": 0.20,
    "demographics": 0.10,
    "audiobook_fit": 0.15,
    "quality": 0.10,
    "cost_and_limits": 0.05,
}

# Below this a candidate is considered too weak to propose.
MIN_ACCEPT_SCORE = 0.55

AUDIOBOOK_HINTS = (
    "audiobook",
    "audio book",
    "narrative",
    "narration",
    "story",
    "storytelling",
    "reading",
    "meditation",
    "documentary",
)

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class RawVoice:
    """Normalised view of either account or library voice payload."""

    voice_id: str
    name: str = ""
    origin: str = "account"              # account | shared
    public_owner_id: Optional[str] = None
    languages: list[str] = field(default_factory=list)  # lowercase ISO codes
    locale: Optional[str] = None
    accent: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    category: Optional[str] = None
    use_case: Optional[str] = None
    description: Optional[str] = None
    preview_url: Optional[str] = None
    labels: dict[str, str] = field(default_factory=dict)
    restricted_reason: Optional[str] = None
    custom_rate: bool = False
    notice_period_days: Optional[int] = None
    free_users_allowed: Optional[bool] = None
    recording_quality: Optional[str] = None
    notes: list[str] = field(default_factory=list)


def _lower(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text.lower() if text else None


def _text(value: Any) -> Optional[str]:
    """Trim only — locales keep their conventional casing (hi-IN)."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _verified_languages(item: dict[str, Any]) -> tuple[list[str], Optional[str], Optional[str]]:
    """Return (languages, accent, locale) from `verified_languages` if any."""
    verified = item.get("verified_languages") or []
    languages, accents, locales = [], [], []
    for entry in verified:
        if not isinstance(entry, dict):
            continue
        lang = _lower(entry.get("language"))
        if lang:
            languages.append(lang)
        if entry.get("accent"):
            accents.append(str(entry["accent"]))
        if entry.get("locale"):
            locales.append(str(entry["locale"]))
    return languages, (accents[0] if accents else None), (locales[0] if locales else None)


def normalize_account_voice(item: dict[str, Any]) -> Optional[RawVoice]:
    """Shape of `GET /v2/voices` (and `GET /v1/voices/{id}`)."""
    voice_id = str(item.get("voice_id") or "").strip()
    if not voice_id:
        return None
    labels = {
        str(k): str(v) for k, v in (item.get("labels") or {}).items()
    }
    sharing = item.get("sharing") or {}

    languages, verified_accent, verified_locale = _verified_languages(item)
    label_language = _lower(labels.get("language"))
    if label_language and label_language not in languages:
        languages.append(label_language)

    safety = item.get("safety_control")
    restricted_reason = None
    if safety and str(safety).upper() not in ("NONE", ""):
        restricted_reason = f"safety control: {safety}"

    fiat_rate = sharing.get("fiat_rate")
    rate = sharing.get("rate")
    custom_rate = fiat_rate is not None or (rate is not None and float(rate) != 1.0)

    return RawVoice(
        voice_id=voice_id,
        name=str(item.get("name") or ""),
        origin="account",
        languages=languages,
        locale=verified_locale,
        accent=_lower(labels.get("accent")) or _text(verified_accent),
        gender=_lower(labels.get("gender")),
        age=_lower(labels.get("age")),
        category=_lower(item.get("category")),
        use_case=_lower(labels.get("use_case")),
        description=item.get("description") or sharing.get("description"),
        preview_url=item.get("preview_url"),
        labels={k.lower(): v for k, v in labels.items()},
        restricted_reason=restricted_reason,
        custom_rate=custom_rate,
        notice_period_days=sharing.get("notice_period"),
        recording_quality=_lower(item.get("recording_quality")),
    )


def normalize_shared_voice(item: dict[str, Any]) -> Optional[RawVoice]:
    """Shape of `GET /v1/shared-voices` (LibraryVoiceResponseModel)."""
    voice_id = str(item.get("voice_id") or "").strip()
    if not voice_id:
        return None

    languages, verified_accent, verified_locale = _verified_languages(item)
    direct_language = _lower(item.get("language"))
    if direct_language and direct_language not in languages:
        languages.insert(0, direct_language)

    return RawVoice(
        voice_id=voice_id,
        name=str(item.get("name") or ""),
        origin="shared",
        public_owner_id=item.get("public_owner_id"),
        languages=languages,
        locale=_text(item.get("locale")) or verified_locale,
        accent=_lower(item.get("accent")) or _text(verified_accent),
        gender=_lower(item.get("gender")),
        age=_lower(item.get("age")),
        category=_lower(item.get("category")),
        use_case=_lower(item.get("use_case")),
        description=item.get("description") or item.get("descriptive"),
        preview_url=item.get("preview_url"),
        labels={
            "descriptive": str(item.get("descriptive") or ""),
            "use_case": str(item.get("use_case") or ""),
        },
        custom_rate=item.get("fiat_rate") is not None,
        notice_period_days=item.get("notice_period"),
        free_users_allowed=item.get("free_users_allowed"),
    )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 2}


def _established(profile_value: Optional[str], attribute: str, uncertain: list[str]) -> bool:
    """True only when the SCRIPT established this attribute for the speaker."""
    return bool(profile_value) and attribute not in uncertain


def _fuzzy_equal(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    return a in b or b in a


def _norm_token(value: Any) -> str:
    """Lowercase + separator-insensitive ('Middle-Aged' == 'middle aged')."""
    return str(value or "").strip().lower().replace("-", " ").replace("_", " ")


# Canonical demographic vocabularies. Free-text values from the story
# analysis ("mature", "elderly", "old man") are folded onto the API's
# enum values so a script word never invalidates a search filter or counts
# as a voice mismatch. Unknown values become None (never sent, never
# penalised) instead of silently emptying the candidate pool.
_AGE_OLD = {"old", "older", "oldest", "elderly", "elder", "senior", "aged",
            "veteran", "grandfather", "grandmother", "grandpa", "grandma"}
_AGE_YOUNG = {"young", "younger", "youth", "teen", "teenage", "teenager",
              "child", "kid", "boy", "girl"}
_AGE_ADULT = {"mature", "adult", "grown"}
_GENDER_FEMALE = {"female", "woman", "women", "girl", "feminine", "f", "she",
                  "her", "hers", "mother", "mom", "sister", "daughter", "wife",
                  "aunt", "grandmother", "grandma", "queen", "princess",
                  "actress", "waitress", "goddess", "lady", "widow"}
_GENDER_MALE = {"male", "man", "men", "boy", "masculine", "m", "he", "him",
                "his", "father", "dad", "brother", "son", "uncle", "husband",
                "grandfather", "grandpa", "king", "prince", "actor", "waiter",
                "god", "lord", "sir", "gentleman", "widower"}
_GENDER_NEUTRAL = {"neutral", "other", "none", "non", "binary", "androgynous"}


def normalize_age(value: Optional[str]) -> Optional[str]:
    """Free-text age -> 'young' | 'middle_aged' | 'old' (the API enum) | None."""
    tokens = set(_norm_token(value).split())
    if not tokens:
        return None
    if "middle" in tokens:
        return "middle_aged"  # before 'aged' is read as old
    if tokens & _AGE_OLD:
        return "old"
    if tokens & _AGE_YOUNG:
        return "young"
    if tokens & _AGE_ADULT:
        return "middle_aged"
    return None


def normalize_gender(value: Optional[str]) -> Optional[str]:
    """Free-text gender -> 'male' | 'female' | 'neutral' (the API enum) | None."""
    tokens = set(_norm_token(value).split())
    if not tokens:
        return None
    if tokens & _GENDER_FEMALE:
        return "female"
    if tokens & _GENDER_MALE:
        return "male"
    if tokens & _GENDER_NEUTRAL:
        return "neutral"
    return None


def score_voice(
    profile: CharacterProfile,
    voice: RawVoice,
    *,
    required_language: Optional[str],
    required_locale: Optional[str] = None,
) -> tuple[Optional[dict[str, float]], list[str]]:
    """Score one voice for one profile.

    Returns `(scores, notes)` — or `(None, notes)` when the voice must be
    excluded (restricted or language-incompatible).
    """
    notes: list[str] = []

    # --- hard exclusion: restricted voices are never proposed ------------ #
    if voice.restricted_reason:
        return None, [f"excluded — {voice.restricted_reason}"]

    # --- hard exclusion: language compatibility -------------------------- #
    if required_language:
        required = language_to_iso(required_language) or required_language.lower()
        if voice.languages and required not in voice.languages:
            return None, [f"excluded — supports {', '.join(voice.languages)}, not {required}"]
        if not voice.languages:
            notes.append("language unverified (voice declares no languages)")
        language_score = 1.0 if voice.languages else 0.4

        # Locale compatibility (e.g. hi-IN): matching region keeps the full
        # score; a different region of the same language is slightly weaker;
        # an unknown locale stays neutral.
        if required_locale and voice.locale:
            if voice.locale.lower() == required_locale.lower():
                pass  # exact regional match — full score
            else:
                language_score *= 0.9
                notes.append(
                    f"locale differs ({voice.locale} vs {required_locale})"
                )
    else:
        language_score = 0.6
        notes.append("no required language — compatibility unverified")

    # --- accent ---------------------------------------------------------- #
    wanted_accent = profile.accent if _established(
        profile.accent, "accent", profile.uncertain_attributes
    ) else None
    if wanted_accent:
        if not voice.accent:
            accent_score = 0.5
            notes.append("voice accent unknown")
        elif _fuzzy_equal(wanted_accent, voice.accent):
            accent_score = 1.0
        else:
            accent_score = 0.3
            notes.append(f"accent mismatch ({voice.accent})")
    else:
        # The script did not establish an accent — never invent one.
        accent_score = 0.7

    # --- vocal personality ---------------------------------------------- #
    profile_tokens = _tokens(
        " ".join(
            [profile.voice_description, profile.speaking_style, profile.role,
             profile.description]
        )
    )
    voice_text = " ".join(
        str(x or "")
        for x in [voice.name, voice.description, voice.use_case, " ".join(
            voice.labels.values()
        )]
    )
    voice_tokens = _tokens(voice_text)
    if profile_tokens and voice_tokens:
        overlap = len(profile_tokens & voice_tokens)
        personality_score = min(1.0, (overlap / len(profile_tokens)) * 2.2)
    else:
        personality_score = 0.5

    # --- age / gender (only when the script established them) ------------ #
    demographic_scores: list[float] = []
    for attribute, voice_value in (("gender", voice.gender), ("age", voice.age)):
        if not _established(getattr(profile, attribute), attribute,
                            profile.uncertain_attributes):
            continue  # unstated by the script -> neutral, never penalised
        canonical = normalize_gender if attribute == "gender" else normalize_age
        wanted = canonical(getattr(profile, attribute))
        actual = canonical(voice_value)
        if not wanted or not actual:
            demographic_scores.append(0.5)
            notes.append(f"{attribute} of voice unknown")
        elif wanted == actual:
            demographic_scores.append(1.0)
        else:
            demographic_scores.append(0.0)
            notes.append(f"{attribute} mismatch ({voice_value})")
    demographics_score = (
        sum(demographic_scores) / len(demographic_scores)
        if demographic_scores else 0.7
    )

    # --- audiobook suitability ------------------------------------------ #
    haystack = " ".join(
        str(x or "").lower()
        for x in [voice.use_case, voice.description, " ".join(voice.labels.values())]
    )
    if any(hint in haystack for hint in AUDIOBOOK_HINTS):
        audiobook_score = 1.0
    elif voice.category in ("professional", "high_quality"):
        audiobook_score = 0.7
    elif voice.category in ("cloned", "generated", "premade", "famous"):
        audiobook_score = 0.5
    else:
        audiobook_score = 0.4

    # --- quality & previews --------------------------------------------- #
    quality_score = 0.3
    if voice.preview_url:
        quality_score += 0.4
    else:
        notes.append("no preview audio")
    if voice.description:
        quality_score += 0.15
    if voice.recording_quality in ("studio", "good"):
        quality_score += 0.15
    elif voice.recording_quality in ("poor", "bad"):
        quality_score = max(0.0, quality_score - 0.4)
        notes.append(f"recording quality: {voice.recording_quality}")
    quality_score = min(1.0, quality_score)

    # --- usage restrictions & cost -------------------------------------- #
    cost_score = 1.0
    if voice.custom_rate:
        cost_score -= 0.4
        notes.append("custom usage rate")
    if voice.notice_period_days:
        cost_score -= 0.2 if voice.notice_period_days > 30 else 0.1
        notes.append(f"notice period: {voice.notice_period_days} days")
    if voice.free_users_allowed is False:
        cost_score -= 0.3
        notes.append("requires a paid subscription plan")
    cost_score = max(0.0, cost_score)

    scores = {
        "language": round(language_score, 3),
        "accent": round(accent_score, 3),
        "personality": round(personality_score, 3),
        "demographics": round(demographics_score, 3),
        "audiobook_fit": round(audiobook_score, 3),
        "quality": round(quality_score, 3),
        "cost_and_limits": round(cost_score, 3),
    }
    return scores, notes


def total_score(scores: dict[str, float]) -> float:
    return round(sum(WEIGHTS[key] * scores[key] for key in WEIGHTS), 4)


def rank_candidates(
    profile: CharacterProfile,
    voices: list[RawVoice],
    *,
    required_language: Optional[str],
    required_locale: Optional[str] = None,
    limit: int = 3,
    min_accept_score: float = MIN_ACCEPT_SCORE,
) -> list[VoiceCandidate]:
    """Best `limit` candidates for `profile` — weak ones simply do not appear.

    `min_accept_score` is the progressive-relaxation knob used when a strict
    ranking finds nothing: every speaking role must end up with a voice in
    the story's language, but restricted voices are excluded at every level.
    """
    language = language_to_iso(required_language or profile.language or "") or None
    ranked: list[VoiceCandidate] = []
    seen: set[str] = set()

    for voice in voices:
        if not voice.voice_id or voice.voice_id in seen:
            continue
        seen.add(voice.voice_id)
        scores, notes = score_voice(
            profile,
            voice,
            required_language=language,
            required_locale=required_locale,
        )
        if scores is None:
            continue  # restricted / incompatible — never proposed
        total = total_score(scores)
        if total < min_accept_score:
            continue  # too weak — do not auto-select just to fill the cast
        ranked.append(
            VoiceCandidate(
                voice_id=voice.voice_id,
                name=voice.name,
                origin="account" if voice.origin == "account" else "shared",
                public_owner_id=voice.public_owner_id,
                language=(voice.languages[0] if voice.languages else None),
                locale=voice.locale,
                accent=voice.accent,
                gender=voice.gender,
                age=voice.age,
                category=voice.category,
                use_case=voice.use_case,
                description=voice.description,
                preview_url=voice.preview_url,
                labels=voice.labels,
                restricted=False,
                custom_rate=voice.custom_rate,
                notice_period_days=voice.notice_period_days,
                notes=notes,
                scores=scores,
                total_score=total,
                rank=1,  # reassigned below
            )
        )

    # Best first; account voices win ties; previews break remaining ties.
    ranked.sort(
        key=lambda c: (
            -c.total_score,
            c.origin != "account",
            c.preview_url is None,
            c.name,
        )
    )
    for index, candidate in enumerate(ranked[:limit], start=1):
        candidate.rank = index
    return ranked[:limit]
