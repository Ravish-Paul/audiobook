"""Eleven v4 inline audio tag parsing and validation.

Audio tags are short delivery instructions wrapped in square brackets that
Eleven v3/v4 understands natively:

    [whispers] [shouts] [curious] [sad] [laughs] [sighs]
    [low, ominous narrator voice] [quiet, tense delivery]

Hard rules enforced here (never trust the LLM):

* `original_text` stays byte-for-byte identical — `tts_text` may only *insert*
  bracketed tags, never change, drop or add spoken words;
* no narration prose outside the brackets (instructions must not be spoken);
* no SSML — Eleven v4 does not support `<break ...>` and friends;
* tag content must look like a delivery cue (letters/words/commas), not code.

`strip_tags()` is also used by the TTS client to remove tags before sending
them to models that do not support them (so a `[laughs]` tag is never spoken
aloud as literal words).
"""

from __future__ import annotations

import re
from typing import Iterable

# A tag body: 1..MAX_TAG_CHARS of anything except square brackets.
MAX_TAG_CHARS = 100
TAG_RE = re.compile(r"\[([^\[\]]{1,120})\]")

# SSML elements Eleven v4 does not support (docs: "SSML is not supported").
_SSML_TAGS = (
    "break",
    "emphasis",
    "lang",
    "p",
    "phoneme",
    "prosody",
    "s",
    "say-as",
    "speak",
    "sub",
    "voice",
)
_SSML_RE = re.compile(
    r"</?(?:" + "|".join(_SSML_TAGS) + r")\b[^<>]*>", re.IGNORECASE
)

# Characters allowed inside a delivery tag (words, commas, apostrophes…).
# Digits, quotes, angle brackets and slashes are rejected so that tags stay
# plain-English cues and never smuggle in code/SSML/numeric instructions
# (e.g. `[pause 3s]` is rejected — ElevenLabs has no SSML-style durations).
_TAG_BODY_RE = re.compile(
    r"^(?:[^\W\d]|[\s,.'’\-…!?:])+$", re.UNICODE
)

# Over-tagging degrades narration — keep tags sparse and purposeful.
MAX_TAGS_PER_PARAGRAPH = 8


def find_tags(text: str) -> list[str]:
    """Return the raw contents of every `[tag]` in `text` (in order)."""
    return [match.group(1).strip() for match in TAG_RE.finditer(text or "")]


def contains_ssml(text: str) -> bool:
    """True when the text carries SSML markup (unsupported by Eleven v4)."""
    return bool(_SSML_RE.search(text or ""))


def strip_tags(text: str, *, preserve: str = "") -> str:
    """Remove every `[tag]` from `text`, returning the spoken words only.

    Tags whose exact bracketed form also occurs in `preserve` (the original
    script) are kept: they are part of the spoken text, not director cues.
    """
    original = preserve or ""

    def _drop(match: re.Match[str]) -> str:
        chunk = match.group(0)
        if chunk in original:
            return chunk
        return ""

    return TAG_RE.sub(_drop, text or "")


def strip_tags_exact(text: str, *, preserve: str = "") -> str:
    """Remove every `[tag]` **and its single separating space**.

    This is the canonical inverse of tag insertion, used for the strict
    regression rule: stripping all audio tags from `tts_text` must reproduce
    `original_text` byte-for-byte.

        "[whispers] hello world"      -> "hello world"
        "hello world [sighs]"         -> "hello world"
        "one [laughs] two"            -> "one two"
        "[sad] a [curious] b"         -> "a b"

    The space that travels with a tag is consumed after the tag when one
    follows it; otherwise the space *before* the tag is consumed (end of
    text). Tags present verbatim in `preserve` are script words and are
    kept untouched.
    """
    source = text or ""
    original = preserve or ""
    pieces: list[str] = []
    cursor = 0
    for match in TAG_RE.finditer(source):
        chunk = match.group(0)
        start, end = match.span()
        if chunk in original:
            # Genuinely part of the script — keep it (and its spacing).
            pieces.append(source[cursor:end])
            cursor = end
            continue
        pieces.append(source[cursor:start])
        cursor = end
        if cursor < len(source) and source[cursor] == " ":
            cursor += 1  # the tag's trailing separator leaves with the tag
        elif pieces and pieces[-1].endswith(" "):
            pieces[-1] = pieces[-1][:-1]  # tag at the end: drop the gap before
    pieces.append(source[cursor:])
    return "".join(pieces)


def spoken_words_equal(original: str, tts_text: str) -> bool:
    """True when stripping all tags from `tts_text` yields `original` EXACTLY.

    Hard rule (never relax this): audio tags may be *added*, but the spoken
    words may never be changed, removed, translated or reordered — and no
    whitespace may differ either.
    """
    if tts_text is None:
        return (original or "") == ""
    return strip_tags_exact(tts_text, preserve=original) == (original or "")


def validate_tts_text(original: str, tts_text: str | None) -> list[str]:
    """Validate a MiMo-generated `tts_text` against `original_text`.

    Returns a list of human-readable problems (empty when valid). Used both
    by the Pydantic schema (fail fast) and by the planner (retry feedback).
    """
    if tts_text is None:
        return []  # optional field: legacy plans simply use original_text

    problems: list[str] = []
    stripped = tts_text.strip()
    if not stripped:
        return ["tts_text must not be empty"]

    if contains_ssml(tts_text):
        problems.append(
            "tts_text contains SSML markup; Eleven v4 does not support SSML — "
            "use inline audio tags like [whispers] instead"
        )

    for tag in find_tags(tts_text):
        if not tag:
            problems.append("tts_text contains an empty audio tag '[]'")
            continue
        if len(tag) > MAX_TAG_CHARS:
            problems.append(
                f"audio tag '[{tag[:40]}…]' is too long "
                f"(max {MAX_TAG_CHARS} characters)"
            )
            continue
        if not _TAG_BODY_RE.match(tag):
            problems.append(
                f"audio tag '[{tag}]' must be a plain delivery cue "
                "(letters, spaces and punctuation only — no digits, quotes "
                "or brackets)"
            )

    tag_count = len(find_tags(tts_text))
    if tag_count > MAX_TAGS_PER_PARAGRAPH:
        problems.append(
            f"tts_text uses {tag_count} audio tags; at most "
            f"{MAX_TAGS_PER_PARAGRAPH} are allowed — use tags only where they "
            "materially improve the narration"
        )

    if not problems and not spoken_words_equal(original, tts_text):
        problems.append(
            "tts_text must reproduce original_text EXACTLY once every "
            "[audio tag] (and its single separating space) is stripped — "
            "spoken words may never be changed, removed, translated or "
            "reordered, and narration instructions must never be written "
            "outside the brackets"
        )

    return problems


def sanitize_for_model(tts_text: str, original: str, *, supports_tags: bool) -> str:
    """Return the text to actually send for a given model.

    Models without audio-tag support would speak `[laughs]` out loud, so the
    tags are stripped there; `original` is passed through as `preserve` so
    bracketed words that are genuinely part of the script survive. Exact
    stripping keeps the spoken words byte-identical to `original`.
    """
    if supports_tags:
        return tts_text
    cleaned = strip_tags_exact(tts_text, preserve=original)
    # Trim only when a tag was actually removed (avoids touching scripts
    # that legitimately begin/end with whitespace).
    return cleaned if cleaned == tts_text else cleaned.strip()


def tag_preview(text: str, limit: int = 6) -> str:
    """Short, comma-joined preview of the tags in `text` (for logs/metadata)."""
    tags: Iterable[str] = find_tags(text)
    joined = ", ".join(f"[{tag}]" for tag in list(tags)[:limit])
    return joined
