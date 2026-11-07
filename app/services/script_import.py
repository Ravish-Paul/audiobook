"""Script import helpers: title + language detection for auto-created projects.

Used by `ProjectService.import_script` (the `POST /projects/from-script`
flow): the user uploads a plain .txt script and the backend derives

* the **title** from an explicit heading in the text
  (`# Room 207`, `Title: Room 207`, `शीर्षक: कमरा नंबर 207`, …) and
  falls back to the original filename without its extension;
* the **language** from the writing script (Devanagari, Bengali, Tamil, …)
  mapped onto the existing `LANGUAGE_NAME_TO_CODE` vocabulary in
  `app.config` — the same names MiMo reports and `normalize_language`
  understands. The narration plan's MiMo-detectated language stays the
  authoritative value after analysis.

Everything here is deterministic and offline: automatic project creation
never spends an LLM call just for naming.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import NamedTuple

from app.config import LANGUAGE_NAME_TO_CODE

# --------------------------------------------------------------------------- #
# Title detection
# --------------------------------------------------------------------------- #
# "# Room 207" / "## Room 207" (optionally closed with trailing hashes).
_TITLE_HEADING_RE = re.compile(r"^#{1,6}\s*(?P<title>.*?)\s*#*\s*$")
# "Title: Room 207" / "शीर्षक: कमरा नंबर 207" / "título: …" (case-insensitive).
_TITLE_LABEL_RE = re.compile(
    r"^(?:"
    r"title|titel|titre|t[ií]tulo|titolo|titel|"
    r"शीर्षक|عنوان|标题|標題|제목|タイトル"
    r")\s*[:：]\s*(?P<title>.*?)\s*$",
    re.IGNORECASE,
)

# Only the opening lines can declare the title (chapter headings deeper in the
# script must never win).
_TITLE_SCAN_LINES = 10
_MAX_TITLE_LENGTH = 120

DEFAULT_TITLE = "Untitled audiobook"


class ScriptLanguage(NamedTuple):
    """A detected language in the app's existing vocabulary."""

    name: str  # e.g. "Hindi" — the name MiMo reports too
    code: str  # ISO 639-1 from LANGUAGE_NAME_TO_CODE ("" when unknown)


def detect_title(text: str, filename: str) -> str:
    """Detect the book title: explicit heading first, filename as fallback.

    Priority (per product spec):
      1. an explicit title heading near the top of the script
         (`# Room 207`, `Title: Room 207`, `शीर्षक: कमरा नंबर 207`, …);
      2. the original filename without its extension.
    """
    heading = _heading_title(text)
    if heading:
        return heading

    stem = Path(filename or "").stem.strip()
    return stem or DEFAULT_TITLE


def _heading_title(text: str) -> str | None:
    scanned = 0
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        scanned += 1
        if scanned > _TITLE_SCAN_LINES:
            return None

        match = _TITLE_LABEL_RE.match(line) or (
            _TITLE_HEADING_RE.match(line) if line.startswith("#") else None
        )
        if not match:
            continue
        candidate = match.group("title").strip().strip("\"'“”‘’")
        candidate = re.sub(r"\s+", " ", candidate)
        if candidate:
            return candidate[:_MAX_TITLE_LENGTH]
    return None


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #
# Unicode block -> the language NAME used across the app (LANGUAGE_NAME_TO_CODE
# vocabulary where possible). Scripts shared by several languages map to the
# dominant one; the user can override the result in the UI.
_SCRIPT_BLOCKS: tuple[tuple[range, str], ...] = (
    (range(0x0900, 0x0980), "Hindi"),  # Devanagari (hi/mr/bhojpuri share it)
    (range(0x0980, 0x0A00), "Bengali"),
    (range(0x0A00, 0x0A80), "Punjabi"),  # Gurmukhi
    (range(0x0A80, 0x0B00), "Gujarati"),
    (range(0x0B00, 0x0B80), "Odia"),
    (range(0x0B80, 0x0C00), "Tamil"),
    (range(0x0C00, 0x0C80), "Telugu"),
    (range(0x0C80, 0x0D00), "Kannada"),
    (range(0x0D00, 0x0D80), "Malayalam"),
    (range(0x0D80, 0x0E00), "Sinhala"),
    (range(0x0E00, 0x0E80), "Thai"),
    (range(0x0E80, 0x0F00), "Lao"),
    (range(0x0530, 0x0590), "Armenian"),
    (range(0x0590, 0x0600), "Hebrew"),
    (range(0x10A0, 0x1100), "Georgian"),
    (range(0x1000, 0x10A0), "Burmese"),  # Myanmar
    (range(0x1200, 0x1380), "Amharic"),  # Ethiopic
    (
        range(0x0600, 0x0700),
        "Urdu",  # Arabic script (ur/ar/fa share it) — Urdu is the app's vocabulary
    ),
    (range(0x0750, 0x0780), "Urdu"),
    (range(0x08A0, 0x0900), "Urdu"),
    (range(0x3040, 0x30A0), "Japanese"),  # Hiragana
    (range(0x30A0, 0x3100), "Japanese"),  # Katakana
    (range(0x3400, 0xA000), "Chinese"),  # CJK ideographs
    (range(0xAC00, 0xD800), "Korean"),  # Hangul syllables
    (range(0x0400, 0x0500), "Russian"),  # Cyrillic
    (range(0x0370, 0x0400), "Greek"),
    (range(0x1E00, 0x1F00), "English"),  # Latin extended (default Latin)
)

_LATIN_RANGES: tuple[range, ...] = (
    range(0x0041, 0x005B),
    range(0x0061, 0x007B),
    range(0x00C0, 0x0250),
)

_FALLBACK_LANGUAGE = "English"


def detect_script_language(text: str) -> ScriptLanguage:
    """Detect the script language from its writing script (offline, instant).

    This is the upload-time preview of the language; the narration plan's
    MiMo-detection (`plan.language`) refines it after analysis. The UI may
    always override the value manually.
    """
    counts: dict[str, int] = {}
    for char in text or "":
        if not unicodedata.category(char).startswith("L"):
            continue  # only letters carry script information
        name = _block_of(ord(char))
        if name:
            counts[name] = counts.get(name, 0) + 1

    if not counts:
        return _language(_FALLBACK_LANGUAGE)

    name = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    return _language(name)


def _block_of(code_point: int) -> str | None:
    for block in _LATIN_RANGES:
        if code_point in block:
            return _FALLBACK_LANGUAGE
    for block, name in _SCRIPT_BLOCKS:
        if code_point in block:
            return name
    return None


def _language(name: str) -> ScriptLanguage:
    code = LANGUAGE_NAME_TO_CODE.get(name.lower(), "")
    return ScriptLanguage(name=name, code=code)


# --------------------------------------------------------------------------- #
# Preview
# --------------------------------------------------------------------------- #
_PREVIEW_CHARS = 240


def script_preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    """Short plain-text preview of the script for the summary card."""
    flattened = re.sub(r"\s+", " ", (text or "")).strip()
    if len(flattened) <= limit:
        return flattened
    return flattened[:limit].rstrip() + "…"
