"""Simple-creator frontend tests — hermetic, no real MiMo/ElevenLabs/pyannote.

Covers:

* static asset delivery (`/`, `/static/style.css`, `/static/app.js`) and the
  absence of any root catch-all;
* the consumer UI markup (upload card, three style cards, ONE generate
  action, result screen) and the absence of the removed technical controls;
* the three narration presets mapped onto `NarrationPreferences`, their
  persistence, and the mapping staying in sync between app.js and the tests;
* the complete Generate-Audiobook workflow using fakes (auto-import ->
  preferences -> MiMo plan -> ElevenLabs/FFmpeg audio -> restore);
* error and loading states (upload validation, unknown projects, busy flag,
  honest stage progress — no invented percentages);
* responsive/accessibility markers.

No audio is really generated and no API credits are spent.
"""

from __future__ import annotations

import dataclasses
import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import DEFAULT_ELEVENLABS_BASE_URL, get_settings
from app.main import app
from app.routes import audio as audio_routes
from app.routes import narration as narration_routes
from app.services import AudioService, ElevenLabsClient, NarrationPlanner
from fakes import FAKE_MP3, FakeComposer, FakeMiMo

SCRIPT = (
    "The lane was empty at midnight.\n\n"
    "Somewhere a dog barked twice.\n\n"
    "She counted her steps to the iron gate."
)
PARAGRAPHS = [p for p in SCRIPT.split("\n\n")]  # 3 paragraphs

BROWSER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
UNKNOWN_ID = "00000000-0000-0000-0000-000000000000"

# The EXACT mapping app.js uses (STYLE_PRESETS) — asserted both ways below.
STYLE_PRESETS = {
    "natural": {
        "preset": "natural",
        "acting_intensity": 2,
        "whisper_policy": "never",
        "whisper_paragraphs": [],
        "tag_density": "minimal",
        "custom_direction": "",
    },
    "cinematic": {
        "preset": "restrained_cinematic",
        "acting_intensity": 2,
        "whisper_policy": "user_selected",
        "whisper_paragraphs": [],
        "tag_density": "minimal",
        "custom_direction": "",
    },
    "expressive": {
        "preset": "expressive",
        "acting_intensity": 3,
        "whisper_policy": "user_selected",
        "whisper_paragraphs": [],
        "tag_density": "standard",
        "custom_direction": "",
    },
}


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _import(client: TestClient, content: str | bytes = SCRIPT, filename: str = "script.txt"):
    payload = content.encode("utf-8") if isinstance(content, str) else content
    return client.post(
        "/projects/from-script",
        files={"file": (filename, payload, "text/plain")},
    )


def compliant_plan(texts: list[str], language: str = "English") -> dict:
    """A plan that satisfies every policy rule for the Natural preset."""
    styles = ("calm", "conversational", "soft", "reflective", "slow", "mysterious")
    paragraphs = []
    for index, text in enumerate(texts):
        paragraphs.append(
            {
                "paragraph_index": index,
                "original_text": text,
                "tts_text": text,
                "emotion": "neutral",
                "delivery_style": styles[index % len(styles)],
                "intensity": 2,
                "narration_direction": (
                    f"Read paragraph {index} as natural, controlled storytelling."
                ),
                "voice_settings": {
                    "speed": 1.0,
                    "stability": 0.6,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
                "pause_after_ms": 300 + 100 * index,
            }
        )
    return {
        "overall_story_tone": "quiet suspense",
        "language": language,
        "paragraphs": paragraphs,
    }


class _Recorder:
    """Fake ElevenLabs endpoint (records request bodies, never bills)."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=FAKE_MP3,
            headers={"content-type": "audio/mpeg"},
            request=request,
        )


def _use_fake_audio(monkeypatch: pytest.MonkeyPatch) -> tuple[_Recorder, FakeComposer]:
    """Wire a fully mocked AudioService into the generate-audio route."""
    settings = dataclasses.replace(
        get_settings(),
        elevenlabs_api_key="sk-frontend-test",
        elevenlabs_voice_id="voice-frontend",
        elevenlabs_base_url=DEFAULT_ELEVENLABS_BASE_URL,
    )
    recorder = _Recorder()
    composer = FakeComposer()
    service = AudioService(
        settings,
        tts_client=ElevenLabsClient(settings, transport=httpx.MockTransport(recorder)),
        composer=composer,
    )
    monkeypatch.setattr(audio_routes, "_audio_service", lambda: service)
    return recorder, composer


# --------------------------------------------------------------------------- #
# Static assets + routing
# --------------------------------------------------------------------------- #
def test_root_serves_the_panel_to_browsers(client: TestClient) -> None:
    response = client.get("/", headers={"Accept": BROWSER_ACCEPT})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert 'href="/static/style.css"' in html
    assert '<script src="/static/app.js" defer></script>' in html


def test_root_serves_json_to_api_clients(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "message" in response.json()


def test_css_and_js_are_served_under_static(client: TestClient) -> None:
    css = client.get("/static/style.css")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert len(css.content) > 0

    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert len(js.content) > 0
    assert "DOMContentLoaded" in js.text


def test_every_local_reference_in_the_html_is_served(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": BROWSER_ACCEPT}).text

    references = re.findall(r'(?:href|src)="(/[^"]+)"', html)
    assert references
    for url in references:
        assert client.get(url).status_code == 200, url


def test_no_catch_all_unknown_paths_stay_json_404(client: TestClient) -> None:
    response = client.get("/definitely-not-a-route")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")

    for legacy in ("/style.css", "/app.js"):
        assert client.get(legacy).status_code == 404, legacy

    assert client.get("/health").status_code == 200


# --------------------------------------------------------------------------- #
# Consumer UI markup
# --------------------------------------------------------------------------- #
def test_ui_is_the_simple_creator(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": BROWSER_ACCEPT}).text

    # Header
    assert "Create your audiobook" in html
    assert "Turn your script into a professionally narrated audiobook." in html

    # Upload card (drag & drop + choose file + file states)
    assert 'id="drop-zone"' in html
    assert "Choose file" in html
    for element_id in ("file-name", "file-size", "file-state", "btn-replace", "btn-remove"):
        assert f'id="{element_id}"' in html, element_id

    # Summary (title, language, paragraphs, preview)
    for element_id in ("book-title", "language-name", "paragraph-count", "script-preview"):
        assert f'id="{element_id}"' in html, element_id

    # Three style cards only
    for style_id in ("style-natural", "style-cinematic", "style-expressive"):
        assert f'id="{style_id}"' in html, style_id
    assert "Clear, conversational, realistic narration." in html
    assert "Immersive and emotionally controlled storytelling." in html
    assert "More emotional variation when the story requires it." in html

    # ONE main action + result screen
    assert "Generate Audiobook" in html
    assert 'id="audio-player"' in html
    assert "Download MP3" in html
    assert "Regenerate audiobook" in html

    # Progress stages (honest ones)
    assert "Understanding your story" in html
    assert "Generating voices" in html
    assert "Completed" in html


def test_ui_exposes_no_technical_controls(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": BROWSER_ACCEPT}).text

    removed = (
        "project-name",  # manual project name
        "btn-create-project",  # manual create
        "pref-intensity",  # acting intensity slider
        "whisper-policy",  # whisper radios
        "whisper-line-list",  # per-paragraph whisper checkboxes
        "tag-density",  # tag density
        "pref-custom-direction",  # custom direction textarea
        "btn-save-preferences",  # save preferences
        "btn-reset-preferences",  # reset preferences
        "preview-paragraph",  # paragraph preview dropdown
        "btn-preview",
        "btn-generate-audio",  # split generate buttons
        "project_id",  # internal id never shown
    )
    for marker in removed:
        assert marker not in html, marker


def test_responsive_and_accessible_markup(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": BROWSER_ACCEPT}).text
    css = client.get("/static/style.css").text

    assert 'name="viewport"' in html
    assert 'role="radiogroup"' in html
    assert 'aria-live="polite"' in html
    assert 'role="status"' in html
    assert "@media (max-width: 720px)" in css
    assert "prefers-reduced-motion" in css
    assert ":focus-visible" in css


# --------------------------------------------------------------------------- #
# Narration presets -> NarrationPreferences
# --------------------------------------------------------------------------- #
def test_style_preset_mappings_are_the_documented_ones() -> None:
    assert STYLE_PRESETS["natural"] == {
        "preset": "natural",
        "acting_intensity": 2,
        "whisper_policy": "never",
        "whisper_paragraphs": [],
        "tag_density": "minimal",
        "custom_direction": "",
    }
    assert STYLE_PRESETS["cinematic"] == {
        "preset": "restrained_cinematic",
        "acting_intensity": 2,
        "whisper_policy": "user_selected",
        "whisper_paragraphs": [],
        "tag_density": "minimal",
        "custom_direction": "",
    }
    assert STYLE_PRESETS["expressive"] == {
        "preset": "expressive",
        "acting_intensity": 3,
        "whisper_policy": "user_selected",
        "whisper_paragraphs": [],
        "tag_density": "standard",
        "custom_direction": "",
    }


@pytest.mark.parametrize("style", ["natural", "cinematic", "expressive"])
def test_style_presets_are_valid_and_persisted(client: TestClient, style: str) -> None:
    project_id = _import(client).json()["project_id"]
    preferences = STYLE_PRESETS[style]

    saved = client.put(
        f"/projects/{project_id}/narration-preferences", json=preferences
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["saved"] is True
    assert saved.json()["preferences"] == preferences

    loaded = client.get(f"/projects/{project_id}/narration-preferences").json()
    assert loaded["saved"] is True
    assert loaded["preferences"] == preferences


def test_app_js_maps_the_same_presets(client: TestClient) -> None:
    js = client.get("/static/app.js").text

    for snippet in (
        "STYLE_PRESETS",
        'preset: "natural"',
        'preset: "restrained_cinematic"',
        'preset: "expressive"',
        'whisper_policy: "never"',
        'whisper_policy: "user_selected"',
        "acting_intensity: 2",
        "acting_intensity: 3",
        'tag_density: "minimal"',
        'tag_density: "standard"',
    ):
        assert snippet in js, snippet


# --------------------------------------------------------------------------- #
# Complete Generate-Audiobook workflow (mocked externals)
# --------------------------------------------------------------------------- #
def test_full_generate_button_workflow_with_mocks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1) upload -> automatic project + title + language
    imported = _import(client, content=f"# Room 207\n\n{SCRIPT}")
    assert imported.status_code == 201
    data = imported.json()
    project_id = data["project_id"]
    assert data["title"] == "Room 207"
    assert data["paragraph_count"] == 4

    # 2) preferences saved BEFORE planning (the audio stage checks them)
    preferences = STYLE_PRESETS["natural"]
    saved = client.put(
        f"/projects/{project_id}/narration-preferences", json=preferences
    )
    assert saved.status_code == 200

    # 3) narration plan through a fake MiMo (the heading line is the first
    #    of the 4 stored paragraphs, so the plan covers all of them)
    fake = FakeMiMo(
        [json.dumps(compliant_plan(["# Room 207"] + PARAGRAPHS), ensure_ascii=False)]
    )
    monkeypatch.setattr(
        narration_routes,
        "_planner",
        lambda: NarrationPlanner(get_settings(), mimo_client=fake),
    )
    plan = client.post(
        f"/projects/{project_id}/analyze-narration", json={"preferences": preferences}
    )
    assert plan.status_code == 200, plan.text
    assert plan.json()["saved"] is True
    assert plan.json()["paragraph_count"] == 4
    assert plan.json()["language"] == "English"
    assert fake.call_count == 1

    # 4) audio through a fake ElevenLabs + fake FFmpeg
    recorder, composer = _use_fake_audio(monkeypatch)
    audio = client.post(f"/projects/{project_id}/generate-audio")
    assert audio.status_code == 200, audio.text
    body = audio.json()
    assert body["status"] == "completed"
    assert body["audio_url"].endswith("/audio/audiobook.mp3")
    assert body["generation_mode"] == "narration_plan"
    assert body["segment_count"] == 4
    assert len(recorder.requests) >= 1  # voices synthesised via the fake API
    assert composer.calls  # assembled via the fake composer

    # 5) the result screen restores after a refresh
    restored = client.get(f"/projects/{project_id}/audio")
    assert restored.status_code == 200
    assert restored.json()["audio_url"] == body["audio_url"]

    # The title flow end to end: project named after the detected heading.
    assert client.get(f"/projects/{project_id}").json()["name"] == "Room 207"


def test_regenerate_without_style_change_needs_no_new_plan(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged preferences keep the plan hash — segments stay cached."""
    project_id = _import(client).json()["project_id"]
    preferences = STYLE_PRESETS["natural"]
    client.put(f"/projects/{project_id}/narration-preferences", json=preferences)

    fake = FakeMiMo(
        [json.dumps(compliant_plan(PARAGRAPHS), ensure_ascii=False)]
    )
    monkeypatch.setattr(
        narration_routes,
        "_planner",
        lambda: NarrationPlanner(get_settings(), mimo_client=fake),
    )
    assert (
        client.post(
            f"/projects/{project_id}/analyze-narration",
            json={"preferences": preferences},
        ).status_code
        == 200
    )
    assert fake.call_count == 1

    recorder, composer = _use_fake_audio(monkeypatch)
    first = client.post(f"/projects/{project_id}/generate-audio")
    assert first.status_code == 200
    tts_calls_after_first = len(recorder.requests)

    # Second run with the SAME plan: the plan hash matches, so already
    # generated segments are reused instead of re-billed.
    second = client.post(f"/projects/{project_id}/generate-audio")
    assert second.status_code == 200
    assert second.json()["audio_url"] == first.json()["audio_url"]
    assert fake.call_count == 1  # MiMo was NOT called again
    assert len(recorder.requests) == tts_calls_after_first  # no new TTS spend
    assert composer.calls


# --------------------------------------------------------------------------- #
# Error and loading states
# --------------------------------------------------------------------------- #
def test_upload_error_states(client: TestClient) -> None:
    bad_type = _import(client, content="# notes", filename="notes.md")
    assert bad_type.status_code == 400
    assert ".txt" in bad_type.json()["detail"]

    bad_encoding = _import(client, content=b"\xff\xfe\x00\x81broken")
    assert bad_encoding.status_code == 400

    missing = client.post("/projects/from-script")
    assert missing.status_code == 422


def test_generation_error_states(client: TestClient) -> None:
    # Unknown project -> clean JSON errors the UI can display.
    plan = client.post(f"/projects/{UNKNOWN_ID}/analyze-narration")
    assert plan.status_code == 404
    assert "detail" in plan.json()

    audio = client.post(f"/projects/{UNKNOWN_ID}/generate-audio")
    assert audio.status_code == 404

    prefs = client.put(
        f"/projects/{UNKNOWN_ID}/narration-preferences",
        json=STYLE_PRESETS["natural"],
    )
    assert prefs.status_code == 404


def test_generation_without_script_fails_fast(client: TestClient) -> None:
    project_id = client.post("/projects", json={"name": "Empty"}).json()["project_id"]

    plan = client.post(f"/projects/{project_id}/analyze-narration")
    assert plan.status_code == 404  # no script — before any external call

    audio = client.post(f"/projects/{project_id}/generate-audio")
    assert audio.status_code == 404


# --------------------------------------------------------------------------- #
# JavaScript wiring (what the browser executes)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def panel_js(client: TestClient) -> str:
    response = client.get("/static/app.js")
    assert response.status_code == 200
    return response.text


def test_js_drives_the_full_pipeline(panel_js: str) -> None:
    # auto project creation on upload
    assert "/projects/from-script" in panel_js
    assert 'body.append("file"' in panel_js
    # preferences -> plan -> audio
    assert "narration-preferences" in panel_js
    assert "analyze-narration" in panel_js
    assert "generate-audio" in panel_js
    assert "narration-plan" in panel_js  # plan reuse check
    # result screen wiring
    assert "audio-player" in panel_js
    assert "btn-download" in panel_js
    assert "loadedmetadata" in panel_js  # real duration, no fake numbers


def test_js_reports_loading_success_and_error_states(panel_js: str) -> None:
    for message in (
        "Uploading your script…",
        "Understanding your story",
        "Generating voices and assembling your audiobook",
        "Your audiobook is ready",
    ):
        assert message in panel_js, message

    # Duplicate submissions are blocked and failures are logged (no secrets).
    assert "state.busy" in panel_js
    assert "Ignored click: another action is still running." in panel_js
    assert "console.error" in panel_js
    assert "console.warn" in panel_js

    # No invented percentages anywhere in the progress code.
    assert re.search(r"\d+\s*%", panel_js) is None


def test_js_persists_state_across_refreshes(panel_js: str) -> None:
    assert "localStorage" in panel_js
    assert 'STORAGE_KEY = "audiobook.state.v1"' in panel_js
    assert "restore" in panel_js


def test_frontend_contains_no_secrets(panel_js: str, client: TestClient) -> None:
    html = client.get("/", headers={"Accept": BROWSER_ACCEPT}).text
    for source in (panel_js, html):
        lowered = source.lower()
        assert "api_key" not in lowered
        assert "authorization" not in lowered
        assert "bearer" not in lowered
        assert "sk-" not in lowered
