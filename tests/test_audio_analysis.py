"""pyannoteAI audio analysis tests (Stage 1) — FULLY MOCKED.

Every pyannote call is intercepted with `httpx.MockTransport`: the suite
NEVER reaches api.pyannote.ai and NEVER spends diarization credits
(pyannote bills a 20-second minimum per successful job).

Covered (spec §12):
 1. local audio upload flow (declare media key -> PUT bytes)
 2. temporary media key generation (unique per analysis, pattern-valid)
 3. successful job submission (exact request body)
 4. job status retrieval
 5. processing and completed statuses
 6. failed and canceled jobs
 7. API timeout and retry behavior
 8. fractional timestamp conversion
 9. millisecond rounding using Decimal
10. sample-rate-dependent sample conversion
11. multiple speakers
12. overlapping speaker turns
13. missing optional arrays
14. invalid timestamps
15. audio duration validation
16. raw response preservation
17. normalized output persistence
18. duplicate analysis prevention
19. audio hash change -> new analysis version
20. existing audiobook routes keep working
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.routes import audio_analysis as analysis_routes
from app.schemas.audio_analysis import AudioAnalysisNormalized
from app.services import AudioAnalysisService, ProjectService, PyannoteClient
from app.services.audio_analysis_normalizer import (
    seconds_to_ms,
    seconds_to_samples,
)
from app.services.audio_probe import AudioProbe
from app.services.errors import (
    AudioAnalysisNormalizationError,
    PyannoteConfigError,
    PyannoteTimeoutError,
)

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"

FAKE_AUDIO = b"ID3\x04\x00\x00fake-voice-only-audiobook-for-analysis"
FAKE_AUDIO_V2 = b"ID3\x04\x00\x00fake-voice-only-audiobook-REGENERATED"

SIGNED_URL = (
    "https://storage.example.com/upload?X-Amz-Signature=SUPER-SECRET-VALUE"
)
JOB_ID = "3c8a89a5-dcc6-4edb-a75d-ffd64739674d"


# --------------------------------------------------------------------------- #
# Provider output fixtures (shape per the official OpenAPI spec)
# --------------------------------------------------------------------------- #
def make_output() -> dict[str, Any]:
    """A realistic succeeded-job output: 2 speakers + an overlap."""
    return {
        "diarization": [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5},
            # SPEAKER_01 starts while SPEAKER_00 is still talking -> overlap
            {"speaker": "SPEAKER_01", "start": 2.4, "end": 4.0},
            {"speaker": "SPEAKER_00", "start": 4.2, "end": 6.0},
        ],
        "wordLevelTranscription": [
            {"start": 0.005, "end": 0.405, "text": "Namaste",
             "speaker": "SPEAKER_00"},
            {"start": 0.748932, "end": 1.305678, "text": "dosto",
             "speaker": "SPEAKER_00"},
            {"start": 2.45, "end": 2.9, "text": "kya",
             "speaker": "SPEAKER_01"},
            {"start": 3.1, "end": 3.6, "text": "haal",
             "speaker": "SPEAKER_01"},
            {"start": 4.2, "end": 4.7, "text": "theek",
             "speaker": "SPEAKER_00"},
        ],
        "turnLevelTranscription": [
            {"start": 0.0, "end": 2.5, "text": "Namaste dosto",
             "speaker": "SPEAKER_00"},
            {"start": 2.4, "end": 4.0, "text": "kya haal",
             "speaker": "SPEAKER_01"},
        ],
    }


# --------------------------------------------------------------------------- #
# Fake pyannote endpoint (records every request)
# --------------------------------------------------------------------------- #
class FakePyannote:
    """Scripted pyannoteAI endpoint behind httpx.MockTransport."""

    def __init__(
        self,
        job_script: Optional[list[dict[str, Any]]] = None,
        output: Any = None,
        media_status: int = 201,
        diarize_status: int = 200,
        fail_upload: bool = False,
    ) -> None:
        self.media_requests: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.diarize_requests: list[dict[str, Any]] = []
        self.job_requests: list[str] = []
        self.job_script = list(job_script or [{"jobId": JOB_ID, "status": "running"}])
        self.output = output
        self.media_status = media_status
        self.diarize_status = diarize_status
        self.fail_upload = fail_upload

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if path == "/v1/media/input" and request.method == "POST":
            self.media_requests.append(json.loads(request.content))
            if self.media_status >= 400:
                return httpx.Response(
                    self.media_status,
                    json={"message": "simulated media failure"},
                    request=request,
                )
            return httpx.Response(
                self.media_status,
                json={"url": SIGNED_URL},
                request=request,
            )

        if request.method == "PUT":
            self.uploads.append(
                {
                    "url": str(request.url),
                    "body": bytes(request.content),
                    "content_type": request.headers.get("content-type"),
                    "authorization": request.headers.get("authorization"),
                }
            )
            if self.fail_upload:
                return httpx.Response(500, request=request)
            return httpx.Response(200, request=request)

        if path == "/v1/diarize" and request.method == "POST":
            self.diarize_requests.append(json.loads(request.content))
            if self.diarize_status >= 400:
                return httpx.Response(
                    self.diarize_status,
                    json={"message": "simulated diarize failure"},
                    request=request,
                )
            return httpx.Response(
                200,
                json={"jobId": JOB_ID, "status": "created"},
                request=request,
            )

        if path.startswith("/v1/jobs/"):
            self.job_requests.append(path)
            item = (
                self.job_script.pop(0)
                if len(self.job_script) > 1
                else self.job_script[0]
            )
            return httpx.Response(200, json=item, request=request)

        return httpx.Response(404, json={"message": "not found"}, request=request)


class FlakyTransport:
    """Raises `error` for the first N matching requests, then delegates."""

    def __init__(
        self, inner: FakePyannote, *, fail_times: int, match_job: bool = True
    ) -> None:
        self.inner = inner
        self.fail_times = fail_times
        self.match_job = match_job
        self.failures = 0
        self.attempts = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.match_job and request.url.path.startswith("/v1/jobs/"):
            self.attempts += 1
            if self.failures < self.fail_times:
                self.failures += 1
                raise httpx.ConnectTimeout("simulated timeout")
        return self.inner(request)


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _settings(**overrides: Any):
    """Test settings: real base URL (transport is mocked), polling immediate."""
    base = dataclasses.replace(
        get_settings(),
        pyannote_api_key="test-pyannote-key",
        pyannote_base_url="https://api.pyannote.ai",
        pyannote_poll_interval_seconds=0.0,
        pyannote_poll_max_interval_seconds=0.0,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def _fake_probe(path: Path, settings: Any) -> AudioProbe:
    """Fixed probe result: 44.1 kHz, 30 seconds (tests never run ffprobe)."""
    return AudioProbe(sample_rate=44100, duration_ms=30_000)


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakePyannote,
    *,
    probe: Any = None,
    settings: Any = None,
) -> None:
    """Inject a mocked client + probe into the route's service factory."""
    settings = settings or _settings()
    client = PyannoteClient(settings, transport=httpx.MockTransport(fake))
    service = AudioAnalysisService(
        settings, client=client, probe=probe or _fake_probe
    )
    monkeypatch.setattr(analysis_routes, "_service", lambda: service)


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def project_id(client: TestClient) -> str:
    """A project with a finished voice-only audiobook on disk."""
    pid = client.post("/projects", json={"name": "Audio Analysis Test"}).json()[
        "project_id"
    ]
    audio_dir = TEST_STORAGE / pid / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "audiobook.mp3").write_bytes(FAKE_AUDIO)
    ProjectService(get_settings()).save_audio_metadata(
        pid,
        {
            "filename": "audiobook.mp3",
            "size_bytes": len(FAKE_AUDIO),
            "model_id": "eleven_v4",
            "output_format": "mp3_44100_128",
            "generated_at": "2026-11-05T15:35:05+00:00",
            "mode": "multi_voice",
            "segment_count": 5,
        },
    )
    return pid


def _start(client: TestClient, pid: str) -> dict[str, Any]:
    response = client.post(f"/projects/{pid}/audio-analysis")
    assert response.status_code == 200, response.text
    return response.json()


def _status(client: TestClient, pid: str) -> dict[str, Any]:
    response = client.get(f"/projects/{pid}/audio-analysis/status")
    assert response.status_code == 200, response.text
    return response.json()


def _run_to_completion(
    client: TestClient, pid: str, fake: FakePyannote, *, steps: int = 4
) -> dict[str, Any]:
    """Poll until the scripted job reaches a terminal state."""
    body: dict[str, Any] = {}
    for _ in range(steps):
        body = _status(client, pid)
        if body["status"] in ("completed", "failed", "canceled"):
            return body
    raise AssertionError(f"analysis never finished: {body}")


# --------------------------------------------------------------------------- #
# 1-3. Upload flow, media keys, submission
# --------------------------------------------------------------------------- #
def test_local_audio_upload_flow(client: TestClient, project_id: str,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """media:// declared first, then the local bytes are PUT to the URL."""
    fake = FakePyannote()
    _wire(monkeypatch, fake)

    body = _start(client, project_id)

    # Step 1: declare the media object key.
    assert len(fake.media_requests) == 1
    declared = fake.media_requests[0]["url"]
    assert declared.startswith("media://")
    assert body["analysis_id"] in declared

    # Step 2: the pre-signed URL receives the ORIGINAL audiobook bytes.
    assert len(fake.uploads) == 1
    upload = fake.uploads[0]
    assert upload["url"] == SIGNED_URL
    assert upload["body"] == FAKE_AUDIO
    assert upload["content_type"] == "application/octet-stream"
    # The API key is NOT sent to the storage host (the signature is auth).
    assert upload["authorization"] is None

    # The original file must be untouched on disk.
    audio_path = TEST_STORAGE / project_id / "audio" / "audiobook.mp3"
    assert audio_path.read_bytes() == FAKE_AUDIO

    # Pre-signed URLs never leak into API responses.
    assert SIGNED_URL not in json.dumps(body)


def test_media_object_key_is_unique_per_analysis(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No fixed key: each analysis gets its own pattern-valid media key."""
    fake_a = FakePyannote()
    _wire(monkeypatch, fake_a)
    _start(client, project_id)

    # A second project with different audio -> different key.
    other = client.post("/projects", json={"name": "Second"}).json()["project_id"]
    audio_dir = TEST_STORAGE / other / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "audiobook.mp3").write_bytes(FAKE_AUDIO_V2)
    _start(client, other)

    keys = [item["url"] for item in fake_a.media_requests]
    assert len(keys) == 2
    assert keys[0] != keys[1]
    # Provider pattern: media://[a-zA-Z0-9\-_./]+
    import re

    for key in keys:
        assert re.fullmatch(r"media://[a-zA-Z0-9\-_./]+", key), key
    # And never the same literal key for every audiobook.
    assert "media://audiobook" not in keys


def test_successful_job_submission_body(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /v1/diarize carries exactly the documented payload."""
    fake = FakePyannote()
    _wire(monkeypatch, fake)

    body = _start(client, project_id)

    assert len(fake.diarize_requests) == 1
    payload = fake.diarize_requests[0]
    assert payload["url"].startswith("media://")
    assert payload["model"] == "precision-3"
    assert payload["transcription"] is True
    assert payload["transcriptionConfig"] == {
        "model": "faster-whisper-large-v3-turbo"
    }

    assert body["success"] is True
    assert body["reused"] is False
    assert body["status"] == "processing"
    assert body["job_status"] == "created"
    assert body["model_id"] == "precision-3"
    assert body["transcription_model"] == "faster-whisper-large-v3-turbo"
    assert len(body["analysis_id"]) > 10


def test_metadata_records_required_fields(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persisted job record holds every field required by the spec."""
    _wire(monkeypatch, FakePyannote())
    _start(client, project_id)

    meta_path = (
        TEST_STORAGE / project_id / "audio" / "analysis"
        / "audio_analysis_metadata.json"
    )
    assert meta_path.is_file()
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    record = data["analyses"][-1]

    for field in (
        "project_id", "analysis_id", "pyannote_job_id", "audio_hash",
        "audio_filename", "model_id", "transcription_model", "job_status",
        "created_at", "updated_at",
    ):
        assert field in record, f"missing {field}"
    assert record["project_id"] == project_id
    assert record["audio_filename"] == "audiobook.mp3"
    assert record["pyannote_job_id"] == JOB_ID
    assert record["sample_rate"] == 44100
    assert record["audio_duration_ms"] == 30_000
    assert data["current_analysis_id"] == record["analysis_id"]


# --------------------------------------------------------------------------- #
# 4-5. Status polling, processing and completion
# --------------------------------------------------------------------------- #
def test_job_status_retrieval_uses_stored_job_id(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "running"}]
    )
    _wire(monkeypatch, fake)
    _start(client, project_id)

    body = _status(client, project_id)

    assert fake.job_requests == [f"/v1/jobs/{JOB_ID}"]
    assert body["status"] == "processing"
    assert body["job_status"] == "running"
    assert body["poll_count"] == 1


def test_processing_then_completed(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = make_output()
    fake = FakePyannote(
        job_script=[
            {"jobId": JOB_ID, "status": "created"},
            {"jobId": JOB_ID, "status": "running"},
            {
                "jobId": JOB_ID,
                "status": "succeeded",
                "createdAt": "2026-11-06T10:00:00Z",
                "updatedAt": "2026-11-06T10:01:00Z",
                "output": output,
            },
        ]
    )
    _wire(monkeypatch, fake)
    _start(client, project_id)

    first = _status(client, project_id)
    assert first["status"] == "processing"
    assert first["job_status"] == "created"

    second = _status(client, project_id)
    assert second["status"] == "processing"
    assert second["job_status"] == "running"

    third = _status(client, project_id)
    assert third["status"] == "completed"
    assert third["job_status"] == "succeeded"
    assert third["word_count"] == 5
    assert third["speaker_count"] == 2
    assert third["completed_at"] is not None

    # Both artifacts exist immediately after success.
    root = TEST_STORAGE / project_id / "audio" / "analysis"
    analysis_id = third["analysis_id"]
    assert (root / analysis_id / "audio_analysis_raw.json").is_file()
    assert (root / analysis_id / "audio_analysis_normalized.json").is_file()


def test_get_analysis_returns_normalized_timeline(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "succeeded",
                     "output": make_output()}]
    )
    _wire(monkeypatch, fake)
    _start(client, project_id)
    _status(client, project_id)

    response = client.get(f"/projects/{project_id}/audio-analysis")
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "completed"
    assert body["time_unit"] == "milliseconds"
    assert body["sample_rate"] == 44100
    assert body["audio_duration_ms"] == 30_000
    assert [s["speaker_id"] for s in body["speakers"]] == [
        "SPEAKER_00", "SPEAKER_01",
    ]
    word = body["words"][1]
    assert word["text"] == "dosto"
    assert word["speaker_id"] == "SPEAKER_00"
    assert word["raw_start_seconds"] == 0.748932
    assert word["start_ms"] == 749          # 0.748932 s -> 749 ms (HALF_UP)
    assert word["end_ms"] == 1306           # 1.305678 s -> 1306 ms
    assert word["start_sample"] == 33028    # 0.748932 * 44100, HALF_UP
    # Pydantic model validates the same structure the file holds.
    AudioAnalysisNormalized.model_validate(body)


def test_polling_backoff_limits_provider_calls(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second status call inside the backoff window must NOT hit the API."""
    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "running"}]
    )
    _wire(
        monkeypatch,
        fake,
        settings=_settings(
            pyannote_poll_interval_seconds=30.0,
            pyannote_poll_max_interval_seconds=60.0,
        ),
    )
    _start(client, project_id)

    first = _status(client, project_id)
    assert first["poll_count"] == 1
    assert len(fake.job_requests) == 1

    second = _status(client, project_id)  # immediately after -> cached
    assert second["poll_count"] == 1
    assert len(fake.job_requests) == 1
    assert "not due" in second["message"]


# --------------------------------------------------------------------------- #
# 6. Failed and canceled jobs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("provider_status", "local_status"),
    [("failed", "failed"), ("canceled", "canceled")],
)
def test_failed_and_canceled_jobs(
    client: TestClient,
    project_id: str,
    monkeypatch: pytest.MonkeyPatch,
    provider_status: str,
    local_status: str,
) -> None:
    fake = FakePyannote(
        job_script=[
            {"jobId": JOB_ID, "status": provider_status,
             "error": "model crashed on this audio"}
        ]
    )
    _wire(monkeypatch, fake)
    _start(client, project_id)

    body = _status(client, project_id)
    assert body["status"] == local_status
    assert body["job_status"] == provider_status
    assert "model crashed" in (body["error"] or "")

    # No normalized result for a non-successful job.
    response = client.get(f"/projects/{project_id}/audio-analysis")
    assert response.status_code == 409
    assert local_status in response.json()["detail"]


# --------------------------------------------------------------------------- #
# 7. Timeout and retry behavior
# --------------------------------------------------------------------------- #
def test_status_timeout_is_retried_within_budget(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /v1/jobs/{id} is idempotent: a timeout is retried once."""
    inner = FakePyannote(job_script=[{"jobId": JOB_ID, "status": "running"}])
    flaky = FlakyTransport(inner, fail_times=1)
    settings = _settings(pyannote_max_attempts=2)
    service = AudioAnalysisService(
        settings,
        client=PyannoteClient(settings, transport=httpx.MockTransport(flaky)),
        probe=_fake_probe,
    )
    monkeypatch.setattr(analysis_routes, "_service", lambda: service)

    _start(client, project_id)
    body = _status(client, project_id)

    assert flaky.attempts == 2  # one failure + one success
    assert body["status"] == "processing"
    assert body["job_status"] == "running"


def test_submit_timeout_is_never_retried(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A /v1/diarize timeout surfaces as 504 after exactly ONE attempt."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/diarize":
            calls["count"] += 1
            raise httpx.ConnectTimeout("submit timed out")
        if request.url.path == "/v1/media/input":
            return httpx.Response(201, json={"url": SIGNED_URL},
                                  request=request)
        if request.method == "PUT":
            return httpx.Response(200, request=request)
        return httpx.Response(404, json={"message": "nope"}, request=request)

    settings = _settings()
    service = AudioAnalysisService(
        settings,
        client=PyannoteClient(settings, transport=httpx.MockTransport(handler)),
        probe=_fake_probe,
    )
    monkeypatch.setattr(analysis_routes, "_service", lambda: service)

    response = client.post(f"/projects/{project_id}/audio-analysis")
    assert response.status_code == 504
    assert calls["count"] == 1  # duplicate jobs would cost money
    detail = str(response.json()["detail"]).lower()
    assert "timeout" in detail or "did not respond" in detail


def test_missing_api_key_is_a_500(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = AudioAnalysisService(
        _settings(pyannote_api_key=""), client=PyannoteClient(_settings()),
        probe=_fake_probe,
    )
    monkeypatch.setattr(analysis_routes, "_service", lambda: service)

    response = client.post(f"/projects/{project_id}/audio-analysis")
    assert response.status_code == 500
    assert "PYANNOTE_API_KEY" in response.json()["detail"]
    with pytest.raises(PyannoteConfigError):
        PyannoteClient(_settings(pyannote_api_key=""))._require_configuration()


def test_upload_failure_surfaces_as_502(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakePyannote(media_status=500)
    _wire(monkeypatch, fake)

    response = client.post(f"/projects/{project_id}/audio-analysis")
    assert response.status_code == 502
    # The job was never submitted -> no credits spent.
    assert fake.diarize_requests == []
    assert SIGNED_URL not in response.text


def test_job_poll_transport_failure_maps_to_5xx(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/v1/jobs/"):
            raise httpx.ConnectError("network down")
        if request.url.path == "/v1/media/input":
            return httpx.Response(201, json={"url": SIGNED_URL},
                                  request=request)
        if request.method == "PUT":
            return httpx.Response(200, request=request)
        return httpx.Response(
            200, json={"jobId": JOB_ID, "status": "created"}, request=request
        )

    settings = _settings(pyannote_max_attempts=2)
    service = AudioAnalysisService(
        settings,
        client=PyannoteClient(settings, transport=httpx.MockTransport(handler)),
        probe=_fake_probe,
    )
    monkeypatch.setattr(analysis_routes, "_service", lambda: service)
    _start(client, project_id)

    response = client.get(f"/projects/{project_id}/audio-analysis/status")
    assert response.status_code == 502


# --------------------------------------------------------------------------- #
# 8-10. Timestamp math (Decimal ROUND_HALF_UP, real sample rate)
# --------------------------------------------------------------------------- #
def test_fractional_timestamp_conversion() -> None:
    assert seconds_to_ms(0.748932) == 749
    assert seconds_to_ms(1.305678) == 1306
    assert seconds_to_ms(10.434376) == 10434
    assert seconds_to_ms(0.005) == 5
    assert seconds_to_ms(4.005) == 4005  # exact half -> ROUND_HALF_UP


def test_millisecond_rounding_uses_decimal_not_binary_float() -> None:
    # float arithmetic would produce 1000.4999999999999 -> 1000.
    assert seconds_to_ms(1.0005) == 1001
    # 2.675 * 1000 is 2674.9999999999995 as a binary float.
    assert seconds_to_ms(2.675) == 2675
    assert seconds_to_ms(0.0005) == 1  # .5 boundary always rounds UP
    # Non-numeric input is rejected, never guessed.
    with pytest.raises(ValueError):
        seconds_to_ms("not-a-number")
    with pytest.raises(ValueError):
        seconds_to_ms(None)


def test_sample_conversion_uses_the_real_sample_rate() -> None:
    # The same timestamp maps to DIFFERENT samples at different rates.
    assert seconds_to_samples(0.748932, 44100) == 33028
    assert seconds_to_samples(0.748932, 16000) == 11983
    assert seconds_to_samples(1.0, 48000) == 48000
    with pytest.raises(ValueError):
        seconds_to_samples(1.0, 0)  # 0 Hz is never assumed valid


def test_normalized_sample_rate_comes_from_the_probe(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """16 kHz source -> 16 kHz timeline (44100 must NOT be hardcoded)."""
    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "succeeded",
                     "output": make_output()}]
    )
    _wire(
        monkeypatch,
        fake,
        probe=lambda path, settings: AudioProbe(
            sample_rate=16000, duration_ms=12_000
        ),
    )
    _start(client, project_id)
    _status(client, project_id)

    body = client.get(f"/projects/{project_id}/audio-analysis").json()
    assert body["sample_rate"] == 16000
    assert body["audio_duration_ms"] == 12_000
    # 0.748932 s * 16000 Hz = 11982.912 -> 11983 samples
    word = next(w for w in body["words"] if w["text"] == "dosto")
    assert word["start_sample"] == 11983


# --------------------------------------------------------------------------- #
# 11-13. Speakers, overlaps, optional arrays
# --------------------------------------------------------------------------- #
def test_multiple_speakers_are_attributed(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "succeeded",
                     "output": make_output()}]
    )
    _wire(monkeypatch, fake)
    _start(client, project_id)
    _status(client, project_id)

    body = client.get(f"/projects/{project_id}/audio-analysis").json()
    speakers = {s["speaker_id"]: s for s in body["speakers"]}
    assert set(speakers) == {"SPEAKER_00", "SPEAKER_01"}
    assert speakers["SPEAKER_00"]["word_count"] == 3
    assert speakers["SPEAKER_01"]["word_count"] == 2
    assert speakers["SPEAKER_00"]["turn_count"] == 2
    assert speakers["SPEAKER_01"]["turn_count"] == 1
    # No confidence is ever invented.
    assert set(speakers["SPEAKER_00"]) == {
        "speaker_id", "turn_count", "word_count",
        "first_start_ms", "last_end_ms",
    }


def test_overlapping_speaker_turns_are_preserved(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two speakers talking at once: BOTH intervals must survive."""
    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "succeeded",
                     "output": make_output()}]
    )
    _wire(monkeypatch, fake)
    _start(client, project_id)
    _status(client, project_id)

    body = client.get(f"/projects/{project_id}/audio-analysis").json()
    turns = body["speaker_turns"]
    assert len(turns) == 3  # the overlapping pair is not merged or dropped

    first = turns[0]
    second = turns[1]
    assert first["speaker_id"] == "SPEAKER_00"
    assert second["speaker_id"] == "SPEAKER_01"
    # 2.4 < 2.5 -> genuine overlap, both kept.
    assert second["raw_start_seconds"] < first["raw_end_seconds"]
    assert body["validation_issues"] == []


def test_missing_optional_arrays_are_tolerated(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exclusiveDiarization / probabilities / turn text may be absent."""
    output = make_output()
    del output["turnLevelTranscription"]  # optional
    # exclusiveDiarization, speakerProbability, speechProbability,
    # crosstalkProbability were never requested -> simply not present.

    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "succeeded",
                     "output": output}]
    )
    _wire(monkeypatch, fake)
    _start(client, project_id)
    _status(client, project_id)

    response = client.get(f"/projects/{project_id}/audio-analysis")
    assert response.status_code == 200
    body = response.json()
    assert body["turn_level_transcriptions"] == []
    assert len(body["words"]) == 5
    assert len(body["speaker_turns"]) == 3


def test_turn_level_transcription_is_normalized_when_present(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "succeeded",
                     "output": make_output()}]
    )
    _wire(monkeypatch, fake)
    _start(client, project_id)
    _status(client, project_id)

    body = client.get(f"/projects/{project_id}/audio-analysis").json()
    turns = body["turn_level_transcriptions"]
    assert len(turns) == 2
    assert turns[0]["text"] == "Namaste dosto"
    assert turns[0]["start_ms"] == 0
    assert turns[0]["end_ms"] == 2500


# --------------------------------------------------------------------------- #
# 14-15. Invalid timestamps + duration validation
# --------------------------------------------------------------------------- #
def test_invalid_timestamps_are_excluded_and_reported(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = make_output()
    output["wordLevelTranscription"] = [
        # valid control record
        {"start": 0.005, "end": 0.405, "text": "Namaste",
         "speaker": "SPEAKER_00"},
        # negative start
        {"start": -0.5, "end": 1.0, "text": "bad", "speaker": "SPEAKER_00"},
        # end <= start
        {"start": 2.0, "end": 2.0, "text": "zero", "speaker": "SPEAKER_00"},
        # missing end timestamp
        {"start": 1.0, "text": "notail", "speaker": "SPEAKER_00"},
        # non-numeric timestamp
        {"start": "x", "end": 2.0, "text": "nan", "speaker": "SPEAKER_00"},
        # empty text
        {"start": 1.0, "end": 1.5, "text": "   ", "speaker": "SPEAKER_00"},
        # missing speaker
        {"start": 1.0, "end": 1.5, "text": "speakerless"},
        # malformed record (not an object)
        "not-a-dict",
    ]

    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "succeeded",
                     "output": output}]
    )
    _wire(monkeypatch, fake)
    _start(client, project_id)
    _status(client, project_id)

    body = client.get(f"/projects/{project_id}/audio-analysis").json()
    assert [w["text"] for w in body["words"]] == ["Namaste"]

    reasons = [issue["reason"] for issue in body["validation_issues"]]
    assert "negative_start_timestamp" in reasons
    assert "end_not_greater_than_start" in reasons
    assert "missing_timestamp" in reasons
    assert "non_numeric_timestamp" in reasons
    assert "empty_text" in reasons
    assert "missing_speaker_id" in reasons
    assert "malformed_record" in reasons
    # Invalid records are excluded from the timeline but never rewritten.
    assert len(body["words"]) == 1


def test_timestamps_beyond_audio_duration_are_rejected(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """30 s audio (+500 ms tolerance): 31 s fails, 30.4 s is allowed."""
    output = make_output()
    output["wordLevelTranscription"] = [
        {"start": 0.005, "end": 0.405, "text": "inside",
         "speaker": "SPEAKER_00"},
        {"start": 30.1, "end": 30.4, "text": "within-tolerance",
         "speaker": "SPEAKER_00"},
        {"start": 31.0, "end": 31.5, "text": "way-past-the-end",
         "speaker": "SPEAKER_00"},
    ]

    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "succeeded",
                     "output": output}]
    )
    _wire(monkeypatch, fake)
    _start(client, project_id)
    _status(client, project_id)

    body = client.get(f"/projects/{project_id}/audio-analysis").json()
    texts = [w["text"] for w in body["words"]]
    assert texts == ["inside", "within-tolerance"]
    duration_issues = [
        issue for issue in body["validation_issues"]
        if issue["reason"] == "exceeds_audio_duration"
    ]
    assert len(duration_issues) == 1
    assert duration_issues[0]["index"] == 2
    assert body["duration_tolerance_ms"] == 500


# --------------------------------------------------------------------------- #
# 16-17. Raw preservation + normalized persistence
# --------------------------------------------------------------------------- #
def test_raw_provider_output_is_preserved_verbatim(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """audio_analysis_raw.json keeps the provider words byte-for-byte."""
    output = make_output()
    output["speakerProbability"] = {  # optional field, if ever returned
        "score": {"SPEAKER_00": [95, 90]}, "resolution": 0.02
    }
    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "succeeded",
                     "output": output}]
    )
    _wire(monkeypatch, fake)
    _start(client, project_id)
    analysis_id = _status(client, project_id)["analysis_id"]

    raw_path = (
        TEST_STORAGE / project_id / "audio" / "analysis" / analysis_id
        / "audio_analysis_raw.json"
    )
    assert raw_path.is_file()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    # Every provider field survives, unaltered, at the top level.
    for key in output:
        assert raw[key] == output[key], f"provider field {key} was altered"
    assert raw["wordLevelTranscription"] == output["wordLevelTranscription"]
    assert raw["speakerProbability"] == output["speakerProbability"]
    # Our bookkeeping lives under its own key.
    meta = raw["_analysis_metadata"]
    assert meta["analysis_id"] == analysis_id
    assert meta["pyannote_job_id"] == JOB_ID
    assert meta["job_status"] == "succeeded"
    assert meta["fetched_at"]


def test_normalization_failure_still_preserves_raw_output(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Succeeded job without usable words -> failed status, raw kept."""
    broken = {"diarization": [{"speaker": "SPEAKER_00",
                               "start": 0.0, "end": 1.0}]}
    # wordLevelTranscription missing entirely.
    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "succeeded",
                     "output": broken}]
    )
    _wire(monkeypatch, fake)
    _start(client, project_id)
    body = _status(client, project_id)

    assert body["status"] == "failed"
    assert "wordLevelTranscription" in (body["error"] or "")

    raw_path = (
        TEST_STORAGE / project_id / "audio" / "analysis"
        / body["analysis_id"] / "audio_analysis_raw.json"
    )
    assert raw_path.is_file()  # evidence survives the failure
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw["diarization"] == broken["diarization"]

    # No normalized file was written for a failed analysis.
    normalized = raw_path.with_name("audio_analysis_normalized.json")
    assert not normalized.is_file()
    assert client.get(
        f"/projects/{project_id}/audio-analysis"
    ).status_code == 409


def test_normalized_file_shape_matches_spec(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "succeeded",
                     "output": make_output()}]
    )
    _wire(monkeypatch, fake)
    _start(client, project_id)
    analysis_id = _status(client, project_id)["analysis_id"]

    path = (
        TEST_STORAGE / project_id / "audio" / "analysis" / analysis_id
        / "audio_analysis_normalized.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "project_id", "analysis_id", "status", "model_id",
        "transcription_model", "time_unit", "sample_rate",
        "audio_duration_ms", "speakers", "words", "speaker_turns",
    ):
        assert key in data
    word = data["words"][0]
    for key in (
        "text", "speaker_id", "raw_start_seconds", "raw_end_seconds",
        "start_ms", "end_ms", "start_sample", "end_sample",
    ):
        assert key in word
    assert data["project_id"] == project_id
    assert data["analysis_id"] == analysis_id


# --------------------------------------------------------------------------- #
# 18-19. Duplicate prevention + new versions
# --------------------------------------------------------------------------- #
def test_identical_analysis_is_reused_without_new_credits(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "succeeded",
                     "output": make_output()}]
    )
    _wire(monkeypatch, fake)

    first = _start(client, project_id)
    _status(client, project_id)  # completes -> raw + normalized on disk

    second = client.post(f"/projects/{project_id}/audio-analysis")
    assert second.status_code == 200
    body = second.json()

    assert body["reused"] is True
    assert body["analysis_id"] == first["analysis_id"]
    assert body["status"] == "completed"
    # Exactly ONE upload + ONE diarization job for the whole test.
    assert len(fake.diarize_requests) == 1
    assert len(fake.uploads) == 1


def test_in_flight_analysis_is_never_submitted_twice(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakePyannote()  # job stays "running" forever
    _wire(monkeypatch, fake)

    first = _start(client, project_id)
    second = _start(client, project_id)

    assert second["reused"] is True
    assert second["analysis_id"] == first["analysis_id"]
    assert len(fake.diarize_requests) == 1
    assert len(fake.uploads) == 1


def test_audio_change_creates_a_new_analysis_version(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakePyannote(
        job_script=[{"jobId": JOB_ID, "status": "succeeded",
                     "output": make_output()}]
    )
    _wire(monkeypatch, fake)

    first = _start(client, project_id)
    _status(client, project_id)

    # The audiobook is regenerated -> different SHA-256 -> new version.
    audio_path = TEST_STORAGE / project_id / "audio" / "audiobook.mp3"
    audio_path.write_bytes(FAKE_AUDIO_V2)

    second = _start(client, project_id)
    assert second["reused"] is False
    assert second["analysis_id"] != first["analysis_id"]
    assert second["audio_hash"] != first["audio_hash"]
    assert len(fake.diarize_requests) == 2

    meta = json.loads(
        (
            TEST_STORAGE / project_id / "audio" / "analysis"
            / "audio_analysis_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert len(meta["analyses"]) == 2
    assert meta["current_analysis_id"] == second["analysis_id"]


# --------------------------------------------------------------------------- #
# Error handling: projects / audiobook / incomplete analysis
# --------------------------------------------------------------------------- #
def test_unknown_project_is_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire(monkeypatch, FakePyannote())
    assert client.post(
        "/projects/does-not-exist/audio-analysis"
    ).status_code == 404
    assert client.get(
        "/projects/does-not-exist/audio-analysis/status"
    ).status_code == 404
    assert client.get(
        "/projects/does-not-exist/audio-analysis"
    ).status_code == 404


def test_missing_audiobook_is_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = client.post("/projects", json={"name": "No Audio"}).json()[
        "project_id"
    ]
    fake = FakePyannote()
    _wire(monkeypatch, fake)

    response = client.post(f"/projects/{pid}/audio-analysis")
    assert response.status_code == 404
    assert "generated" in response.json()["detail"]
    # Nothing was uploaded or submitted.
    assert fake.media_requests == []
    assert fake.diarize_requests == []


def test_status_before_start_is_404(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire(monkeypatch, FakePyannote())
    response = client.get(f"/projects/{project_id}/audio-analysis/status")
    assert response.status_code == 404
    assert "audio-analysis" in response.json()["detail"]


def test_analysis_while_processing_is_409(
    client: TestClient, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire(monkeypatch, FakePyannote())  # never reaches succeeded
    _start(client, project_id)

    response = client.get(f"/projects/{project_id}/audio-analysis")
    assert response.status_code == 409
    assert "processing" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# 20. Existing routes keep working (regression smoke)
# --------------------------------------------------------------------------- #
def test_existing_routes_still_work(client: TestClient) -> None:
    welcome = client.get("/")
    assert welcome.status_code == 200
    assert "audio-analysis" not in welcome.json().get("endpoints", [])

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    created = client.post("/projects", json={"name": "Regression"})
    assert created.status_code == 201
    pid = created.json()["project_id"]

    assert client.get(f"/projects/{pid}").status_code == 200
    assert client.get(f"/projects/{pid}/audio").status_code == 404
    assert client.get(f"/projects/{pid}/narration-plan").status_code == 404


# --------------------------------------------------------------------------- #
# Optional: REAL ffprobe (no credits, local file only) — skipped if absent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    shutil.which("ffprobe") is None,
    reason="ffprobe is not installed",
)
def test_probe_audio_reads_the_real_sample_rate(tmp_path: Path) -> None:
    from app.services.audio_probe import probe_audio

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    tone = tmp_path / "tone.mp3"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
         "-c:a", "libmp3lame", "-q:a", "5", str(tone)],
        check=True, capture_output=True, timeout=60,
    )

    probe = probe_audio(tone, get_settings())
    assert probe.sample_rate == 44100
    assert 1400 <= probe.duration_ms <= 1600
