"""`POST /projects/from-script` — automatic project creation for the simple UI.

Covers the product spec:

* title detection priority: explicit heading (`# Room 207`, `Title: …`,
  `शीर्षक: …`) first, original filename (without extension) as fallback;
* language detection from the writing script, mapped onto the app's
  existing `LANGUAGE_NAME_TO_CODE` vocabulary;
* automatic project creation + script storage (no manual project step);
* the same validation as `upload-script` (413 / 400 / 422);
* full backward compatibility of the existing create + upload flow.

Everything is offline and hermetic — title/language detection never call
MiMo or ElevenLabs.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import language_to_iso
from app.main import app
from app.services.script_import import (
    detect_script_language,
    detect_title,
    script_preview,
)

SCRIPT = (
    "The lane was empty at midnight.\n\n"
    "Somewhere a dog barked twice.\n\n"
    "She counted her steps to the iron gate."
)


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _import(
    client: TestClient,
    content: str | bytes = SCRIPT,
    filename: str = "script.txt",
):
    payload = content.encode("utf-8") if isinstance(content, str) else content
    return client.post(
        "/projects/from-script",
        files={"file": (filename, payload, "text/plain")},
    )


# --------------------------------------------------------------------------- #
# Title detection (priority chain)
# --------------------------------------------------------------------------- #
def test_title_from_hash_heading(client: TestClient) -> None:
    body = _import(client, content=f"# Room 207\n\n{SCRIPT}").json()
    assert body["title"] == "Room 207"


def test_title_from_english_label(client: TestClient) -> None:
    body = _import(client, content=f"Title: Room 207\n\n{SCRIPT}").json()
    assert body["title"] == "Room 207"


def test_title_from_hindi_label(client: TestClient) -> None:
    body = _import(
        client, content=f"शीर्षक: कमरा नंबर 207\n\n{SCRIPT}"
    ).json()
    assert body["title"] == "कमरा नंबर 207"


def test_title_label_is_case_insensitive(client: TestClient) -> None:
    body = _import(client, content=f"TITLE: The Long Way Home\n\n{SCRIPT}").json()
    assert body["title"] == "The Long Way Home"


def test_title_heading_may_follow_blank_lines(client: TestClient) -> None:
    body = _import(client, content=f"\n\n   # Room 207   \n\n{SCRIPT}").json()
    assert body["title"] == "Room 207"


def test_title_falls_back_to_the_filename(client: TestClient) -> None:
    body = _import(client, content=SCRIPT, filename="my_book_final.txt").json()
    assert body["title"] == "my_book_final"


def test_title_ignores_chapter_headings_deeper_in_the_script() -> None:
    text = "\n\n".join(["Once upon a time."] * 12 + ["# Chapter 1", "It began."])
    assert detect_title(text, "tale.txt") == "tale"


def test_title_is_capped_and_stripped() -> None:
    long_title = "x" * 300
    assert detect_title(f"# {long_title}\n\nbody", "a.txt") == "x" * 120
    assert detect_title('# "Quoted" #\n\nbody', "a.txt") == "Quoted"


# --------------------------------------------------------------------------- #
# Language detection (existing vocabulary, offline)
# --------------------------------------------------------------------------- #
def test_language_devanagari_is_hindi(client: TestClient) -> None:
    hindi = "रात बहुत अंधेरी थी और चारों तरफ सन्नाटा पसरा हुआ था।\n\nवह चलता गया।"
    body = _import(client, content=hindi).json()
    assert body["language"] == "Hindi"
    assert body["language_code"] == "hi"


def test_language_latin_is_english(client: TestClient) -> None:
    body = _import(client, content=SCRIPT).json()
    assert body["language"] == "English"
    assert body["language_code"] == "en"


def test_language_tamil_script() -> None:
    detected = detect_script_language("இரவு மிக இருண்டதாக இருந்தது")
    assert detected.name == "Tamil"
    assert detected.code == "ta"


def test_language_japanese_script() -> None:
    detected = detect_script_language("夜はとても暗かった。彼は静かに歩いた。")
    assert detected.name == "Japanese"
    assert detected.code == "ja"


def test_language_korean_script() -> None:
    detected = detect_script_language("밤은 매우 어두웠다. 그는 조용히 걸었다.")
    assert detected.name == "Korean"
    assert detected.code == "ko"


def test_language_armenian_script() -> None:
    detected = detect_script_language("Գիշերը շատ մութ էր։ Նա դանդաղ քայլեց։")
    assert detected.name == "Armenian"
    assert detected.code == "hy"


def test_language_amharic_script() -> None:
    detected = detect_script_language("ሌሊቱ በጣም ጨለማ ነበር። ቀስ ብሎ ተራመደ።")
    assert detected.name == "Amharic"
    assert detected.code == "am"


# name -> provider code: the official Eleven v4 supported vocabulary.
ELEVEN_V4_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("Afrikaans", "af"), ("Amharic", "am"), ("Arabic", "ar"),
    ("Armenian", "hy"), ("Assamese", "as"), ("Asturian", "ast"),
    ("Azerbaijani", "az"), ("Belarusian", "be"), ("Bengali", "bn"),
    ("Bosnian", "bs"), ("Bulgarian", "bg"), ("Burmese", "my"),
    ("Cantonese", "yue"), ("Catalan", "ca"), ("Cebuano", "ceb"),
    ("Croatian", "hr"), ("Czech", "cs"), ("Danish", "da"),
    ("Dutch", "nl"), ("English", "en"), ("Estonian", "et"),
    ("Filipino", "fil"), ("Finnish", "fi"), ("French", "fr"),
    ("Fula (Pulaar)", "ff"), ("Galician", "gl"), ("Georgian", "ka"),
    ("German", "de"), ("Greek", "el"), ("Gujarati", "gu"),
    ("Hausa", "ha"), ("Hebrew", "he"), ("Hindi", "hi"),
    ("Hungarian", "hu"), ("Icelandic", "is"), ("Indonesian", "id"),
    ("Italian", "it"), ("Japanese", "ja"), ("Javanese", "jv"),
    ("Kamba", "kam"), ("Kannada", "kn"), ("Kazakh", "kk"),
    ("Korean", "ko"), ("Kyrgyz", "ky"), ("Lao", "lo"),
    ("Latvian", "lv"), ("Lingala", "ln"), ("Lithuanian", "lt"),
    ("Luganda", "lg"), ("Luxembourgish", "lb"), ("Macedonian", "mk"),
    ("Malay", "ms"), ("Malayalam", "ml"), ("Maltese", "mt"),
    ("Mandarin Chinese", "zh"), ("Māori", "mi"), ("Marathi", "mr"),
    ("Mongolian", "mn"), ("Nepali", "ne"), ("Norwegian", "no"),
    ("Occitan", "oc"), ("Odia", "or"), ("Pashto", "ps"),
    ("Persian", "fa"), ("Polish", "pl"), ("Portuguese", "pt"),
    ("Punjabi", "pa"), ("Romanian", "ro"), ("Russian", "ru"),
    ("Serbian", "sr"), ("Shona", "sn"), ("Sindhi", "sd"),
    ("Slovak", "sk"), ("Slovenian", "sl"), ("Somali", "so"),
    ("Sorani Kurdish", "ckb"), ("Spanish", "es"), ("Swahili", "sw"),
    ("Swedish", "sv"), ("Tajik", "tg"), ("Tamil", "ta"),
    ("Telugu", "te"), ("Thai", "th"), ("Turkish", "tr"),
    ("Ukrainian", "uk"), ("Urdu", "ur"), ("Uzbek", "uz"),
    ("Vietnamese", "vi"), ("Welsh", "cy"), ("Wolof", "wo"),
    ("Zulu", "zu"),
)


def test_every_eleven_v4_language_folds_and_appears_in_the_dropdown() -> None:
    """A script in ANY supported language must produce its audiobook.

    For every Eleven v4 language: the name folds to the provider code (so
    voice casting searches and matches in that language) AND the UI override
    offers it.
    """
    html = (
        Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"
    ).read_text(encoding="utf-8")
    for name, code in ELEVEN_V4_LANGUAGES:
        assert language_to_iso(name) == code, name
        assert f'<option value="{name}">' in html, name


# --------------------------------------------------------------------------- #
# Automatic project creation + storage
# --------------------------------------------------------------------------- #
def test_project_is_created_automatically(client: TestClient) -> None:
    response = _import(client, content=f"# Room 207\n\n{SCRIPT}")

    assert response.status_code == 201, response.text
    body = response.json()
    uuid.UUID(body["project_id"])
    assert body["upload_status"] == "success"
    assert body["filename"] == "script.txt"
    assert body["character_count"] == len(f"# Room 207\n\n{SCRIPT}")
    assert body["paragraph_count"] == 4  # heading line + 3 paragraphs

    # The project exists, is named after the detected title and holds the script.
    project = client.get(f"/projects/{body['project_id']}").json()
    assert project["name"] == "Room 207"
    assert project["has_script"] is True

    stored = client.get(f"/media/projects/{body['project_id']}/script.txt")
    assert stored.status_code == 200
    assert stored.text.replace("\r\n", "\n") == f"# Room 207\n\n{SCRIPT}"


def test_paragraph_count_matches_the_backend_split(client: TestClient) -> None:
    body = _import(client, content=SCRIPT).json()
    assert body["paragraph_count"] == 3


def test_script_preview_is_short(client: TestClient) -> None:
    body = _import(client, content=SCRIPT).json()
    assert body["script_preview"].startswith("The lane was empty at midnight.")
    assert len(body["script_preview"]) <= 241  # 240 chars + ellipsis

    assert script_preview("short") == "short"
    assert script_preview("word " * 200).endswith("…")


# --------------------------------------------------------------------------- #
# Validation mirrors upload-script
# --------------------------------------------------------------------------- #
def test_rejects_non_txt(client: TestClient) -> None:
    response = _import(client, content="# notes", filename="notes.md")
    assert response.status_code == 400
    assert ".txt" in response.json()["detail"]


def test_rejects_non_utf8(client: TestClient) -> None:
    response = _import(client, content=b"\xff\xfe\x00\x81broken")
    assert response.status_code == 400
    assert "UTF-8" in response.json()["detail"]


def test_rejects_missing_file(client: TestClient) -> None:
    response = client.post("/projects/from-script")
    assert response.status_code == 422


def test_rejected_upload_reports_no_project(client: TestClient) -> None:
    # import_script validates BEFORE creating the project (service order), so
    # a rejected upload never returns — or leaves behind — a project id.
    response = _import(client, content=b"\xff\xfe", filename="script.txt")
    assert response.status_code == 400
    assert "project_id" not in response.json()


# --------------------------------------------------------------------------- #
# Backward compatibility — the manual flow keeps working unchanged
# --------------------------------------------------------------------------- #
def test_existing_create_and_upload_flow_still_works(client: TestClient) -> None:
    created = client.post("/projects", json={"name": "Manual"})
    assert created.status_code == 201
    project_id = created.json()["project_id"]

    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", SCRIPT.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201
    assert upload.json()["upload_status"] == "success"

    project = client.get(f"/projects/{project_id}").json()
    assert project["name"] == "Manual"
    assert project["has_script"] is True
