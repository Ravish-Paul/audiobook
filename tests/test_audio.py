"""Audio generation tests.

**ElevenLabs is always mocked here.** Two mocking strategies are used:

* most tests replace `ElevenLabsClient.synthesize` with an async fake;
* `test_request_contract_...` injects an `httpx.MockTransport`, which lets us
  assert the exact URL / headers / JSON body sent to ElevenLabs — still
  without any network call or API credit.

Run with:  pytest -q
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import DEFAULT_ELEVENLABS_BASE_URL, get_settings
from app.main import app
from app.routes import audio as audio_routes
from app.services import AudioService, ElevenLabsClient
from app.services.errors import (
    ElevenLabsAPIError,
    ElevenLabsRateLimitError,
    ElevenLabsTimeoutError,
    ElevenLabsVoiceError,
)

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"

# Not a real MP3 — just enough to prove bytes round-trip through storage/HTTP.
FAKE_MP3 = b"ID3\x04\x00\x00fake-mp3-bytes-for-tests"
SCRIPT_TEXT = "Hello from the audiobook script. This line is spoken aloud."


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def project_id(client: TestClient) -> str:
    response = client.post("/projects", json={"name": "Audio Test Project"})
    assert response.status_code == 201
    return response.json()["project_id"]


@pytest.fixture()
def project_with_script(client: TestClient, project_id: str) -> str:
    """Project with a non-empty script.txt already stored."""
    response = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", SCRIPT_TEXT.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 201
    return project_id


def _patch_synthesize(
    monkeypatch: pytest.MonkeyPatch,
    *,
    audio: bytes = FAKE_MP3,
    error: Optional[Exception] = None,
) -> list[str]:
    """Replace `ElevenLabsClient.synthesize` with a fake. Returns seen texts."""
    seen: list[str] = []

    async def fake_synthesize(
        self: ElevenLabsClient,
        text: str,
        voice_settings: Optional[dict] = None,
        **kwargs: object,
    ) -> bytes:
        seen.append(text)
        if error is not None:
            raise error
        return audio

    monkeypatch.setattr(ElevenLabsClient, "synthesize", fake_synthesize)
    return seen


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_generate_audio_success(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_synthesize(monkeypatch)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert body["message"] == "Audio generated successfully"
    assert body["project_id"] == project_with_script
    assert body["audio_filename"] == "audiobook.mp3"
    assert body["audio_url"] == (
        f"/media/projects/{project_with_script}/audio/audiobook.mp3"
    )
    assert body["status"] == "completed"
    assert body["size_bytes"] == len(FAKE_MP3)
    assert body["character_count"] == len(SCRIPT_TEXT)
    assert body["model_id"] == get_settings().elevenlabs_model_id

    # The script came from storage — no second upload happened.
    assert seen == [SCRIPT_TEXT]

    # The MP3 really exists where the spec says it should.
    mp3_path = TEST_STORAGE / project_with_script / "audio" / "audiobook.mp3"
    assert mp3_path.is_file()
    assert mp3_path.read_bytes() == FAKE_MP3

    # ...and the metadata is recorded next to the existing project data.
    meta = json.loads(
        (TEST_STORAGE / project_with_script / "project.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["audio"]["filename"] == "audiobook.mp3"
    assert meta["audio"]["size_bytes"] == len(FAKE_MP3)
    assert meta["script"]["filename"] == "script.txt"  # untouched


def test_generated_audio_url_is_playable(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The returned audio_url must actually serve the MP3 through the browser."""
    _patch_synthesize(monkeypatch)
    body = client.post(f"/projects/{project_with_script}/generate-audio").json()

    media = client.get(body["audio_url"])
    assert media.status_code == 200
    assert media.content == FAKE_MP3
    assert media.headers["content-type"].startswith("audio/")


# --------------------------------------------------------------------------- #
# GET /projects/{id}/audio
# --------------------------------------------------------------------------- #
def test_get_audio_before_generation_is_404(
    client: TestClient, project_with_script: str
) -> None:
    response = client.get(f"/projects/{project_with_script}/audio")
    assert response.status_code == 404
    assert "not generated yet" in response.json()["detail"]


def test_get_audio_after_generation(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_synthesize(monkeypatch)
    generated = client.post(f"/projects/{project_with_script}/generate-audio").json()

    response = client.get(f"/projects/{project_with_script}/audio")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["status"] == "completed"
    assert body["audio_url"] == generated["audio_url"]
    assert body["audio_filename"] == "audiobook.mp3"
    assert body["generated_at"] is not None


def test_get_audio_unknown_project_is_404(client: TestClient) -> None:
    response = client.get("/projects/00000000-0000-0000-0000-000000000000/audio")
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# Script / project validation
# --------------------------------------------------------------------------- #
def test_generate_audio_unknown_project_is_404(client: TestClient) -> None:
    response = client.post(
        "/projects/00000000-0000-0000-0000-000000000000/generate-audio"
    )
    assert response.status_code == 404


def test_generate_audio_without_script_is_404(
    client: TestClient, project_id: str
) -> None:
    response = client.post(f"/projects/{project_id}/generate-audio")
    assert response.status_code == 404
    assert "No script has been uploaded" in response.json()["detail"]


def test_generate_audio_with_empty_script_is_400(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_synthesize(monkeypatch)
    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", b"   \n\n  ", "text/plain")},
    )
    assert upload.status_code == 201

    response = client.post(f"/projects/{project_id}/generate-audio")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
    assert seen == []  # ElevenLabs was never called


def test_generate_audio_script_too_long_is_400(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ELEVENLABS_MAX_CHARS", "10")
    seen = _patch_synthesize(monkeypatch)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 400
    assert "per-request limit" in response.json()["detail"]
    assert seen == []


# --------------------------------------------------------------------------- #
# Configuration errors (fail before any network call)
# --------------------------------------------------------------------------- #
def test_missing_api_key_is_reported_clearly(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 500
    assert "ELEVENLABS_API_KEY" in response.json()["detail"]


def test_missing_voice_id_is_reported_clearly(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test-placeholder")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "")

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 500
    assert "ELEVENLABS_VOICE_ID" in response.json()["detail"]


def test_api_key_is_never_exposed_in_responses(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "sk-THIS-IS-A-SECRET-KEY-DO-NOT-LEAK"
    monkeypatch.setenv("ELEVENLABS_API_KEY", sentinel)
    _patch_synthesize(
        monkeypatch, error=ElevenLabsAPIError("ElevenLabs API failed with HTTP 500")
    )

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 502
    assert sentinel not in response.text
    # The key must not appear in the app's repr either.
    assert sentinel not in repr(get_settings())


# --------------------------------------------------------------------------- #
# ElevenLabs failures -> correct HTTP status codes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ElevenLabsRateLimitError("ElevenLabs rate limit reached; retry after 10s"), 429),
        (ElevenLabsTimeoutError("ElevenLabs did not respond within 60 seconds"), 504),
        (ElevenLabsAPIError("Could not reach the ElevenLabs API"), 502),
        (ElevenLabsVoiceError("ElevenLabs could not find the configured voice"), 400),
    ],
    ids=["rate-limit", "timeout", "api-failure", "invalid-voice"],
)
def test_elevenlabs_failures_map_to_http_statuses(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: int,
) -> None:
    _patch_synthesize(monkeypatch, error=error)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == expected_status
    assert response.json()["detail"]


def test_audio_save_failure_is_500(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_synthesize(monkeypatch)

    def boom(self: Path, data: bytes) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", boom)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 500
    assert "Could not save the generated audio" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Request contract: real httpx call, intercepted by httpx.MockTransport
# --------------------------------------------------------------------------- #
def test_request_contract_matches_official_elevenlabs_endpoint(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = get_settings()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["xi_api_key"] = request.headers.get("xi-api-key")
        captured["accept"] = request.headers.get("accept")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=FAKE_MP3, headers={"content-type": "audio/mpeg"}
        )

    # Explicit, non-secret credentials for this request only. The base URL is
    # pinned to the real production endpoint so the contract is asserted
    # against the official ElevenLabs host (no traffic: MockTransport).
    fake_settings = dataclasses.replace(
        settings,
        elevenlabs_api_key="sk-contract-test",
        elevenlabs_voice_id="voice-abc123",
        elevenlabs_base_url=DEFAULT_ELEVENLABS_BASE_URL,
    )
    tts = ElevenLabsClient(fake_settings, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        audio_routes,
        "_audio_service",
        lambda: AudioService(fake_settings, tts_client=tts),
    )

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text

    assert captured["method"] == "POST"
    assert captured["url"] == (
        "https://api.elevenlabs.io/v1/text-to-speech/voice-abc123"
        "?output_format=mp3_44100_128"
    )
    assert captured["xi_api_key"] == "sk-contract-test"  # sent as xi-api-key header
    assert captured["accept"] == "audio/mpeg"

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model_id"] == get_settings().elevenlabs_model_id
    assert body["text"] == SCRIPT_TEXT
    assert body["voice_settings"]["use_speaker_boost"] is True
    # eleven_v4 capability filtering: speed/style are not part of v4.
    assert "speed" not in body["voice_settings"]
    assert "style" not in body["voice_settings"]
    # Text normalization is always explicit (`auto` by default).
    assert body["apply_text_normalization"] == "auto"
    # No language hint configured -> language_code is not sent.
    assert "language_code" not in body
    # No pronunciation dictionaries / audio effects configured -> absent.
    assert "pronunciation_dictionary_locators" not in body
    assert "audio_effects" not in body
    # Legacy single pass has no paragraph neighbours -> no context fields.
    assert "previous_text" not in body
    assert "next_text" not in body
    assert "previous_request_ids" not in body
    assert "next_request_ids" not in body


# --------------------------------------------------------------------------- #
# Optional live integration test — disabled unless explicitly enabled.
# Costs a tiny amount of ElevenLabs credit, so it never runs by default.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.getenv("RUN_ELEVENLABS_LIVE_TEST") != "1",
    reason="set RUN_ELEVENLABS_LIVE_TEST=1 with valid .env credentials to run",
)
def test_live_elevenlabs_roundtrip(client: TestClient) -> None:
    settings = get_settings()
    assert settings.elevenlabs_configured, (
        "RUN_ELEVENLABS_LIVE_TEST=1 requires ELEVENLABS_API_KEY and "
        "ELEVENLABS_VOICE_ID in .env"
    )

    project_id = client.post("/projects", json={"name": "Live TTS"}).json()[
        "project_id"
    ]
    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={
            "file": ("script.txt", b"Hello from the live integration test.", "text/plain")
        },
    )
    assert upload.status_code == 201

    generated = client.post(f"/projects/{project_id}/generate-audio")
    assert generated.status_code == 200, generated.text
    assert generated.json()["status"] == "completed"

    media = client.get(generated.json()["audio_url"])
    assert media.status_code == 200
    # Real MP3 files start with an ID3 tag or a frame sync word.
    assert media.content[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xfa")
