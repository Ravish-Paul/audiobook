"""Plan-driven audio generation tests.

Everything is mocked: ElevenLabs via `httpx.MockTransport`, FFmpeg via
`FakeComposer`, MiMo via a directly saved narration plan. No API credits and
no real ffmpeg run here (see `test_composer.py` for the real ffmpeg test).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import DEFAULT_ELEVENLABS_BASE_URL, get_settings
from app.main import app
from app.routes import audio as audio_routes
from app.schemas.narration import NarrationPlan, ParagraphPlan
from app.services import AudioService, ElevenLabsClient, ProjectService
from app.services.narration_planner import split_paragraphs
from fakes import FAKE_MP3, FAKE_MERGED_MP3, FakeComposer, make_plan_dict

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"

SCRIPT = (
    "रात बहुत अंधेरी थी और चारों तरफ सन्नाटा पसरा हुआ था।\n\n"
    "अचानक दरवाजे पर किसी ने दस्तक दी।\n\n"
    "वह धीरे-धीरे दरवाजे की ओर बढ़ा।"
)
PARAGRAPHS = split_paragraphs(SCRIPT)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tts_settings():
    """Settings with credentials, pointing at the official endpoint.

    `httpx.MockTransport` intercepts the request, so no traffic is generated.
    """
    return dataclasses.replace(
        get_settings(),
        elevenlabs_api_key="sk-plan-test",
        elevenlabs_voice_id="voice-plan",
        elevenlabs_base_url=DEFAULT_ELEVENLABS_BASE_URL,
    )


class ElevenLabsRecorder:
    """Fake ElevenLabs endpoint that records every request body."""

    def __init__(self, fail_text: Optional[str] = None, status: int = 500) -> None:
        self.requests: list[dict] = []
        self.fail_text = fail_text
        self.status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.fail_text is not None and body.get("text") == self.fail_text:
            return httpx.Response(
                self.status,
                json={"detail": "simulated upstream failure"},
                request=request,
            )
        return httpx.Response(
            200,
            content=FAKE_MP3,
            headers={"content-type": "audio/mpeg"},
            request=request,
        )


def _service(
    monkeypatch: pytest.MonkeyPatch,
    recorder: ElevenLabsRecorder,
    composer: Optional[FakeComposer] = None,
) -> FakeComposer:
    """Wire an AudioService with mocked ElevenLabs + composer into the route."""
    settings = _tts_settings()
    composer = composer or FakeComposer()
    service = AudioService(
        settings,
        tts_client=ElevenLabsClient(settings, transport=httpx.MockTransport(recorder)),
        composer=composer,
    )
    monkeypatch.setattr(audio_routes, "_audio_service", lambda: service)
    return composer


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def project_with_script(client: TestClient) -> str:
    project_id = client.post("/projects", json={"name": "Plan Audio Test"}).json()[
        "project_id"
    ]
    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", SCRIPT.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201
    return project_id


def _save_plan(project_id: str, texts=None) -> NarrationPlan:
    """Persist a valid narration plan (as /analyze-narration would)."""
    texts = texts if texts is not None else PARAGRAPHS
    plan = NarrationPlan.model_validate(make_plan_dict(texts))
    ProjectService(get_settings()).save_narration_plan(project_id, plan)
    return plan


# --------------------------------------------------------------------------- #
# Plan driven generation
# --------------------------------------------------------------------------- #
def test_generate_audio_from_narration_plan(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _save_plan(project_with_script)
    recorder = ElevenLabsRecorder()
    composer = _service(monkeypatch, recorder)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["success"] is True
    assert body["generation_mode"] == "narration_plan"
    assert body["segment_count"] == len(PARAGRAPHS)
    assert body["audio_filename"] == "audiobook.mp3"
    assert body["audio_url"] == (
        f"/media/projects/{project_with_script}/audio/audiobook.mp3"
    )
    assert body["status"] == "completed"
    assert "3 narration segments" in body["message"]

    # One ElevenLabs request per paragraph, original text untouched.
    assert len(recorder.requests) == len(PARAGRAPHS)
    for index, paragraph in enumerate(plan.paragraphs):
        request = recorder.requests[index]
        assert request["text"] == PARAGRAPHS[index]
        assert request["model_id"] == get_settings().elevenlabs_model_id
        # Text normalization defaults to `auto` (configurable via .env).
        assert request["apply_text_normalization"] == "auto"
        # MiMo's per-paragraph voice settings reach ElevenLabs — but only
        # the ones eleven_v4 supports: no speed / style / clone_strength.
        voice_settings = request["voice_settings"]
        assert (
            voice_settings["stability"] == paragraph.voice_settings.stability
        )
        assert (
            voice_settings["similarity_boost"]
            == paragraph.voice_settings.similarity_boost
        )
        assert voice_settings["use_speaker_boost"] is True
        assert "speed" not in voice_settings  # unsupported by eleven_v4
        assert "style" not in voice_settings  # unsupported by eleven_v4
        assert "clone_strength" not in voice_settings  # not a v4 PVC here
        # ...and narration metadata is NOT sent as an API parameter.
        assert "emotion" not in request
        assert "delivery_style" not in request
        assert "narration_direction" not in request
        assert "pause_after_ms" not in request
        assert "paragraph_index" not in request

    # Individual segments are stored where the spec says.
    segments_dir = TEST_STORAGE / project_with_script / "audio" / "segments"
    for index in range(len(PARAGRAPHS)):
        assert (segments_dir / f"paragraph_{index + 1:04d}.mp3").read_bytes() == FAKE_MP3

    # FFmpeg was asked to merge them, in order, with MiMo's pauses.
    assert composer.segment_counts == [len(PARAGRAPHS)]
    merged_segments = composer.calls[0]
    assert [s.path.name for s in merged_segments] == [
        f"paragraph_{i + 1:04d}.mp3" for i in range(len(PARAGRAPHS))
    ]
    assert [s.pause_after_ms for s in merged_segments] == [
        p.pause_after_ms for p in plan.paragraphs
    ]


def test_generation_metadata_is_saved(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _save_plan(project_with_script)
    _service(monkeypatch, ElevenLabsRecorder())
    client.post(f"/projects/{project_with_script}/generate-audio")

    metadata = json.loads(
        (TEST_STORAGE / project_with_script / "audio" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["mode"] == "narration_plan"
    assert metadata["status"] == "completed"
    assert metadata["segment_count"] == len(PARAGRAPHS)
    assert metadata["language"] == "Hindi"
    assert metadata["overall_story_tone"] == "suspenseful"
    assert metadata["audiobook"]["filename"] == "audiobook.mp3"
    assert metadata["total_pause_ms"] == sum(p.pause_after_ms for p in plan.paragraphs)
    assert len(metadata["segments"]) == len(PARAGRAPHS)

    first = metadata["segments"][0]
    assert first["filename"] == "paragraph_0001.mp3"
    assert first["emotion"] == plan.paragraphs[0].emotion
    assert first["delivery_style"] == plan.paragraphs[0].delivery_style
    assert first["narration_direction"] == plan.paragraphs[0].narration_direction
    assert first["voice_settings"] == plan.paragraphs[0].voice_settings.model_dump()
    assert first["reused"] is False


def test_generated_audiobook_is_playable(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_plan(project_with_script)
    _service(monkeypatch, ElevenLabsRecorder())

    audio_url = client.post(f"/projects/{project_with_script}/generate-audio").json()[
        "audio_url"
    ]
    media = client.get(audio_url)
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("audio/")
    assert media.content == FAKE_MERGED_MP3

    # ...and the audio info endpoint reports it too.
    info = client.get(f"/projects/{project_with_script}/audio").json()
    assert info["generation_mode"] == "narration_plan"
    assert info["segment_count"] == len(PARAGRAPHS)
    assert info["audio_url"] == audio_url


def test_original_script_is_untouched_after_generation(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_plan(project_with_script)
    _service(monkeypatch, ElevenLabsRecorder())
    client.post(f"/projects/{project_with_script}/generate-audio")

    stored = (TEST_STORAGE / project_with_script / "script.txt").read_text(
        encoding="utf-8"
    )
    assert stored == SCRIPT


# --------------------------------------------------------------------------- #
# Backwards compatible fallback
# --------------------------------------------------------------------------- #
def test_without_plan_the_legacy_single_pass_still_works(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not (
        TEST_STORAGE / project_with_script / "narration_plan.json"
    ).exists(), "this project must have no plan"

    recorder = ElevenLabsRecorder()
    composer = _service(monkeypatch, recorder)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200
    body = response.json()

    assert body["generation_mode"] == "single_pass"
    assert body["segment_count"] is None
    assert len(recorder.requests) == 1
    assert recorder.requests[0]["text"] == SCRIPT  # whole script, one request
    assert composer.segment_counts == []  # FFmpeg not needed in single-pass


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #
def test_failed_paragraph_preserves_segments_and_reports_partial(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_plan(project_with_script)

    failing = ElevenLabsRecorder(fail_text=PARAGRAPHS[1], status=500)
    _service(monkeypatch, failing)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "Paragraph 1 failed" in detail
    assert "1/3 segment(s)" in detail
    assert "preserved" in detail

    segments_dir = TEST_STORAGE / project_with_script / "audio" / "segments"
    assert (segments_dir / "paragraph_0001.mp3").is_file()  # completed work kept
    assert not (segments_dir / "paragraph_0002.mp3").exists()
    assert not (segments_dir / "paragraph_0003.mp3").exists()

    # No final audiobook is ever claimed on failure.
    assert not (
        TEST_STORAGE / project_with_script / "audio" / "audiobook.mp3"
    ).exists()
    assert client.get(f"/projects/{project_with_script}/audio").status_code == 404

    metadata = json.loads(
        (TEST_STORAGE / project_with_script / "audio" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["status"] == "partial"
    assert metadata["failed_index"] == 1
    assert len(metadata["segments"]) == 1

    # --- recovery: the next run reuses segment 1 and only redoes the rest --
    healthy = ElevenLabsRecorder()
    _service(monkeypatch, healthy)

    retry = client.post(f"/projects/{project_with_script}/generate-audio")
    assert retry.status_code == 200, retry.text

    # segment 1 was NOT re-billed: only paragraphs 2 and 3 were requested.
    assert len(healthy.requests) == 2
    assert healthy.requests[0]["text"] == PARAGRAPHS[1]
    assert healthy.requests[1]["text"] == PARAGRAPHS[2]

    segments_dir = TEST_STORAGE / project_with_script / "audio" / "segments"
    for index in range(3):
        assert (segments_dir / f"paragraph_{index + 1:04d}.mp3").is_file()

    metadata = json.loads(
        (TEST_STORAGE / project_with_script / "audio" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["status"] == "completed"
    assert [s["reused"] for s in metadata["segments"]] == [True, False, False]

    assert retry.json()["segment_count"] == 3
    assert client.get(retry.json()["audio_url"]).status_code == 200


def test_composition_failure_is_reported_without_final_audio(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_plan(project_with_script)
    _service(monkeypatch, ElevenLabsRecorder(), composer=FakeComposer(fail=True))

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 500
    assert "FFmpeg" in response.json()["detail"]
    assert not (
        TEST_STORAGE / project_with_script / "audio" / "audiobook.mp3"
    ).exists()

    metadata = json.loads(
        (TEST_STORAGE / project_with_script / "audio" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["status"] == "composition_failed"
    assert len(metadata["segments"]) == 3  # segments themselves are preserved


def test_stale_plan_is_rejected_when_script_changes(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Plan built for a DIFFERENT script than the one stored.
    _save_plan(project_with_script, texts=["different", "paragraphs", "entirely"])
    recorder = ElevenLabsRecorder()
    _service(monkeypatch, recorder)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 409
    assert "analyze-narration" in response.json()["detail"]
    assert recorder.requests == []  # ElevenLabs was never called
