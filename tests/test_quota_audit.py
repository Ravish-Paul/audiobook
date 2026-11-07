"""Final quota-semantics audit — auxiliary operations vs. audiobook allowance.

Everything is mocked: NO real MiMo / ElevenLabs / pyannote / FFmpeg calls,
no provider credits, no real tester usage or tokens (isolated
`tests/_test_storage` + test identities only).

Coverage:

* auxiliary operations (narration preview, character analysis, voice
  audition, diarization/audio analysis) NEVER reserve or consume the
  complete-audiobook allowance, never open a generation transaction, and
  stay available at 0 remaining — while keeping their rate limiting and
  duplicate-submission protection;
* one complete audiobook workflow (planning + interleaved auxiliary calls
  + generate-audio + post-generation diarization) still consumes EXACTLY
  ONE allowance under a single shared `generation_id`;
* usage persistence is verified across REAL OS processes: a separate
  Python process writes the counter to a temp storage directory, exits,
  and a second process on the same directory reads it back unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import DEFAULT_ELEVENLABS_BASE_URL, get_settings
from app.main import app
from app.routes import audio as audio_routes
from app.routes import casting as casting_routes
from app.routes import narration as narration_routes
from app.services import (
    AudioService,
    CharacterAnalyzer,
    ElevenLabsClient,
    NarrationPlanner,
    ProjectService,
)
from app.services.narration_planner import split_paragraphs
from fakes import FAKE_MP3, MV_PARAGRAPHS, MV_SCRIPT, FakeComposer, FakeMiMo
from fakes import plan_json, story_json
from test_audio_analysis import FAKE_AUDIO, FakePyannote, _wire as _wire_pyannote
from test_multi_voice import _wire_casting

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"
REPO_ROOT = Path(__file__).resolve().parents[1]

ALICE = {"Authorization": "Bearer alice-secret-1"}
BOB = {"Authorization": "Bearer bob-secret-2"}

# Auxiliary endpoints under audit (all stay in GENERATION_SUFFIXES so the
# rate limiter and duplicate guard keep applying to them).
AUX_PATHS = (
    "/analyze-narration/preview",
    "/analyze-characters",
    "/voice-casting/audition",
    "/audio-analysis",
)


# --------------------------------------------------------------------------- #
# Fixtures + helpers (same conventions as test_generation_limits.py)
# --------------------------------------------------------------------------- #
def _reset_runtime(clear_storage: bool = True) -> None:
    """Clear in-memory middleware state (and optionally the ledger files)."""
    from app import security as security_mod

    with security_mod._RATE_LOCK:
        security_mod._RATE_WINDOWS.clear()
    with security_mod._DEDUP_LOCK:
        security_mod._IN_FLIGHT.clear()
        security_mod._LAST_COMPLETED.clear()
    if clear_storage:
        shutil.rmtree(TEST_STORAGE / "_beta", ignore_errors=True)


@pytest.fixture()
def beta_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Private beta mode with two testers and tight, deterministic limits."""
    monkeypatch.setenv("BETA_ACCESS_TOKENS", "alice:alice-secret-1,bob:bob-secret-2")
    monkeypatch.setenv("BETA_TESTER_GENERATION_LIMITS", "")
    monkeypatch.setenv("BETA_GENERATION_LIMIT", "2")
    monkeypatch.setenv("BETA_RATE_LIMIT_PER_MINUTE", "200")
    monkeypatch.setenv("BETA_LOGIN_RATE_LIMIT_PER_MINUTE", "10")
    monkeypatch.setenv("BETA_DUPLICATE_WINDOW_SECONDS", "10")
    _reset_runtime()
    yield
    _reset_runtime()


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _create_project(
    client: TestClient, headers: dict, content: str = MV_SCRIPT
) -> str:
    response = client.post(
        "/projects/from-script",
        files={"file": ("s.txt", content.encode("utf-8"), "text/plain")},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()["project_id"]


def _usage(client: TestClient, headers: dict) -> dict[str, Any]:
    response = client.get("/usage", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def _generations() -> dict[str, Any]:
    path = TEST_STORAGE / "_beta" / "generations.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _quota_record() -> dict[str, Any]:
    """The persisted per-tester usage ledger ({} when never charged)."""
    path = TEST_STORAGE / "_beta" / "quota.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class _Recorder:
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


def _use_fake_audio(
    monkeypatch: pytest.MonkeyPatch, composer: Optional[FakeComposer] = None
) -> AudioService:
    """Mocked ElevenLabs + FFmpeg for generate-audio (never bills)."""
    import dataclasses

    settings = dataclasses.replace(
        get_settings(),
        elevenlabs_api_key="sk-beta-test",
        elevenlabs_voice_id="voice-beta",
        elevenlabs_base_url=DEFAULT_ELEVENLABS_BASE_URL,
    )
    service = AudioService(
        settings,
        tts_client=ElevenLabsClient(settings, transport=httpx.MockTransport(_Recorder())),
        composer=composer or FakeComposer(),
    )
    monkeypatch.setattr(audio_routes, "_audio_service", lambda: service)
    return service


def _use_fake_mimo(monkeypatch: pytest.MonkeyPatch, fake: FakeMiMo) -> FakeMiMo:
    monkeypatch.setattr(
        narration_routes,
        "_planner",
        lambda: NarrationPlanner(get_settings(), mimo_client=fake),
    )
    return fake


def _use_fake_analyzer(monkeypatch: pytest.MonkeyPatch, fake: FakeMiMo) -> FakeMiMo:
    monkeypatch.setattr(
        casting_routes,
        "_analyzer",
        lambda: CharacterAnalyzer(get_settings(), mimo_client=fake),
    )
    return fake


def _add_audiobook_on_disk(project_id: str) -> None:
    """Give the project a finished (fake) audiobook for diarization."""
    audio_dir = TEST_STORAGE / project_id / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "audiobook.mp3").write_bytes(FAKE_AUDIO)
    ProjectService(get_settings()).save_audio_metadata(
        project_id,
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


def _preview(client: TestClient, project_id: str) -> httpx.Response:
    return client.post(
        f"/projects/{project_id}/analyze-narration/preview",
        json={"paragraph_index": 1},
        headers=ALICE,
    )


def _analyze_characters(client: TestClient, project_id: str) -> httpx.Response:
    return client.post(
        f"/projects/{project_id}/analyze-characters", headers=ALICE
    )


def _audition(client: TestClient, project_id: str) -> httpx.Response:
    return client.post(
        f"/projects/{project_id}/voice-casting/audition",
        json={
            "speaker_id": "suman",
            "text": "नमस्ते दोस्तों, कैसे हो?",
            "voice_id": "voice-suman-02",
        },
        headers=ALICE,
    )


def _start_analysis(client: TestClient, project_id: str) -> httpx.Response:
    return client.post(f"/projects/{project_id}/audio-analysis", headers=ALICE)


# --------------------------------------------------------------------------- #
# 1. Classification: aux paths are protected but never audiobook generations
# --------------------------------------------------------------------------- #
def test_auxiliary_paths_are_protected_but_never_classified_as_generations() -> None:
    from app.security import _generation_kind, _is_generation_path

    uuid = "00000000-0000-0000-0000-000000000000"
    for suffix in AUX_PATHS:
        path = f"/projects/{uuid}{suffix}"
        # Still a protected generation path -> rate limit + duplicate guard.
        assert _is_generation_path(path), suffix
        # …but classified `aux` -> never charges/blocks the allowance.
        assert _generation_kind(path) == "aux", suffix

    # Planning and generation keep their workflow semantics.
    assert _generation_kind(f"/projects/{uuid}/analyze-narration") == "plan"
    assert _generation_kind(f"/projects/{uuid}/create-multi-voice-plan") == "plan"
    assert _generation_kind(f"/projects/{uuid}/generate-audio") == "generate"
    # Unknown billable paths stay conservative (one allowance).
    assert _generation_kind(f"/projects/{uuid}/something-else") == "other"


# --------------------------------------------------------------------------- #
# 2. Auxiliary operations never consume or reserve the allowance
# --------------------------------------------------------------------------- #
def test_auxiliary_operations_never_consume_the_audiobook_allowance(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_mimo(monkeypatch, FakeMiMo([plan_json(MV_PARAGRAPHS)]))
    _use_fake_analyzer(monkeypatch, FakeMiMo([story_json()]))
    _wire_casting(monkeypatch)
    _wire_pyannote(monkeypatch, FakePyannote())

    with TestClient(app) as client:
        project = _create_project(client, ALICE)

        # a) narration preview (MiMo mocked)
        preview = _preview(client, project)
        assert preview.status_code == 200, preview.text
        assert _usage(client, ALICE)["generations_used"] == 0

        # b) character analysis (MiMo mocked)
        analysis = _analyze_characters(client, project)
        assert analysis.status_code == 200, analysis.text
        assert _usage(client, ALICE)["generations_used"] == 0

        # c) voice audition (ElevenLabs mocked)
        audition = _audition(client, project)
        assert audition.status_code == 200, audition.text
        assert _usage(client, ALICE)["generations_used"] == 0

        # d) diarization / audio analysis (pyannote mocked)
        _add_audiobook_on_disk(project)
        diarization = _start_analysis(client, project)
        assert diarization.status_code == 200, diarization.text
        assert _usage(client, ALICE)["generations_used"] == 0

        usage = _usage(client, ALICE)
        assert usage["generations_used"] == 0  # never charged
        assert usage["generations_remaining"] == 2  # full allowance intact

        # The quota file itself was never touched for this tester.
        assert "alice" not in _quota_record()


def test_auxiliary_operations_never_open_a_generation_transaction(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_mimo(monkeypatch, FakeMiMo([plan_json(MV_PARAGRAPHS)]))
    _use_fake_analyzer(monkeypatch, FakeMiMo([story_json()]))

    with TestClient(app) as client:
        project = _create_project(client, ALICE)
        assert _preview(client, project).status_code == 200
        assert _analyze_characters(client, project).status_code == 200

        ledger = _generations()
        assert ledger.get("workflows") == {}  # never opened a workflow
        history = ledger.get("history", [])
        assert history  # the requests were still recorded…
        for entry in history:
            assert entry["kind"] == "aux"
            assert entry["charged"] is False  # …but never charged
            assert entry["tester"] == "alice"


def test_auxiliary_operations_keep_the_duplicate_submission_guard(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_mimo(monkeypatch, FakeMiMo([plan_json(MV_PARAGRAPHS)]))

    with TestClient(app) as client:
        project = _create_project(client, ALICE)
        first = _preview(client, project)
        assert first.status_code == 200, first.text

        # Same endpoint inside the duplicate window -> refused, not re-run.
        second = _preview(client, project)
        assert second.status_code == 409
        assert "Duplicate" in second.json()["detail"]

        assert _usage(client, ALICE)["generations_used"] == 0
        # Only the first (successful) call reached the ledger.
        assert len(_generations().get("history", [])) == 1


def test_auxiliary_operations_stay_available_when_the_allowance_is_exhausted(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BETA_GENERATION_LIMIT", "0")  # zero allowance
    _use_fake_mimo(monkeypatch, FakeMiMo([plan_json(MV_PARAGRAPHS)]))
    _use_fake_analyzer(monkeypatch, FakeMiMo([story_json()]))
    _wire_casting(monkeypatch)
    _wire_pyannote(monkeypatch, FakePyannote())

    with TestClient(app) as client:
        project = _create_project(client, ALICE)

        # The complete-audiobook allowance is still enforced…
        refused = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
        assert refused.status_code == 429

        # …while every auxiliary operation keeps working (never blocked).
        assert _preview(client, project).status_code == 200
        assert _analyze_characters(client, project).status_code == 200
        assert _audition(client, project).status_code == 200
        _add_audiobook_on_disk(project)
        assert _start_analysis(client, project).status_code == 200

        usage = _usage(client, ALICE)
        assert usage["generations_used"] == 0  # nothing was ever charged
        assert usage["generations_remaining"] == 0  # and none is left


# --------------------------------------------------------------------------- #
# 3. The complete workflow still costs exactly ONE allowance
# --------------------------------------------------------------------------- #
def test_complete_workflow_with_interleaved_aux_calls_costs_exactly_one(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_planner = _use_fake_mimo(
        monkeypatch, FakeMiMo([plan_json(MV_PARAGRAPHS)])
    )
    _use_fake_analyzer(monkeypatch, FakeMiMo([story_json()]))
    _wire_casting(monkeypatch)
    _wire_pyannote(monkeypatch, FakePyannote())
    service = _use_fake_audio(monkeypatch)

    with TestClient(app) as client:
        project = _create_project(client, ALICE)

        # 1) planning opens the workflow and consumes the ONE allowance.
        plan = client.post(f"/projects/{project}/analyze-narration", headers=ALICE)
        assert plan.status_code == 200, plan.text
        assert _usage(client, ALICE)["generations_used"] == 1

        # 2) auxiliary calls in between never add an allowance.
        assert _preview(client, project).status_code == 200
        assert _usage(client, ALICE)["generations_used"] == 1
        assert _analyze_characters(client, project).status_code == 200
        assert _usage(client, ALICE)["generations_used"] == 1
        assert _audition(client, project).status_code == 200
        assert _usage(client, ALICE)["generations_used"] == 1

        # 3) generate-audio JOINS the open workflow (same generation_id).
        audio = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
        assert audio.status_code == 200, audio.text
        assert _usage(client, ALICE)["generations_used"] == 1

        # 4) post-generation diarization is still free of allowance cost.
        assert _start_analysis(client, project).status_code == 200
        assert _usage(client, ALICE)["generations_used"] == 1

        usage = _usage(client, ALICE)
        assert usage["generations_used"] == 1  # EXACTLY one allowance
        assert usage["generations_remaining"] == 1

        # ElevenLabs segments + FFmpeg ran inside that one generation.
        assert service.composer.segment_counts == [3]
        assert fake_planner.call_count >= 1

        ledger = _generations()
        entries = [
            entry
            for entry in ledger["history"]
            if entry.get("tester") == "alice"
        ]
        charged = [entry for entry in entries if entry["charged"]]
        assert len(charged) == 1  # one reservation in the whole workflow

        plan_ids = {e["generation_id"] for e in entries if e["kind"] == "plan"}
        generate_ids = {e["generation_id"] for e in entries if e["kind"] == "generate"}
        assert len(plan_ids) == 1
        assert plan_ids == generate_ids  # planning joined by generate-audio

        aux_entries = [e for e in entries if e["kind"] == "aux"]
        assert len(aux_entries) == 4  # preview, cast, audition, diarization
        assert all(not e["charged"] for e in aux_entries)
        aux_ids = {e["generation_id"] for e in aux_entries}
        assert aux_ids.isdisjoint(plan_ids)  # aux never joins the workflow

        # The workflow was closed by generate-audio.
        assert ledger.get("workflows") == {}

        # The stored audiobook carries the workflow's generation_id.
        metadata = ProjectService(get_settings()).get_audio_metadata(project)
        assert metadata is not None
        assert metadata["generation_id"] in plan_ids


# --------------------------------------------------------------------------- #
# 4. Real OS-process restart persistence (isolated temp storage)
# --------------------------------------------------------------------------- #
_PROBE_SOURCE = '''\
"""Cross-process quota persistence probe — run twice as a fresh OS process.

Usage: python quota_probe.py charge | read
Prints one JSON line with the tester's usage counter.
"""
import json
import sys

from app.config import get_settings
from app.security import _ledger_charge, _ledger_complete, generation_usage

TESTER = "restart-probe-tester"


def main() -> int:
    settings = get_settings()
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "charge":
        # Two complete ledger transactions (write/update the counter).
        for _ in range(2):
            error, generation_id, charged = _ledger_charge(
                settings, TESTER, None, "plan"
            )
            if error or not charged:
                print(json.dumps({"error": error, "charged": charged}))
                return 1
            _ledger_complete(
                settings, TESTER, None, "plan", generation_id, True, True
            )
    elif mode != "read":
        print(json.dumps({"error": "unknown mode: %r" % mode}))
        return 1
    print(json.dumps(generation_usage(settings, TESTER)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _run_probe(tmp_path: Path, mode: str) -> dict[str, Any]:
    """Execute the probe script in a SEPARATE OS process."""
    storage = tmp_path / "storage"
    probe = tmp_path / "quota_probe.py"
    if not probe.exists():
        probe.write_text(_PROBE_SOURCE, encoding="utf-8")

    env = os.environ.copy()
    env["STORAGE_DIR"] = str(storage)  # isolated temp storage, same for both runs
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, str(probe), mode],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, f"probe printed nothing: {result.stdout!r} {result.stderr!r}"
    return json.loads(lines[-1])


def test_usage_counter_survives_a_real_os_process_restart(tmp_path: Path) -> None:
    """Process A writes usage, exits; process B on the same storage reads it."""
    # Process 1: create/update persistent usage for a test identity, exit.
    written = _run_probe(tmp_path, "charge")
    assert "error" not in written, written
    assert written["generations_used"] == 2  # two ledger transactions

    # Process 2: brand-new OS process, EXACTLY the same storage directory.
    read_back = _run_probe(tmp_path, "read")
    assert "error" not in read_back, read_back
    assert read_back["generations_used"] == written["generations_used"]  # no reset
    assert read_back["generations_used"] == 2
    assert (
        read_back["generations_remaining"] == written["generations_remaining"]
    )

    # The persisted file on disk agrees with what both processes reported.
    quota_file = tmp_path / "storage" / "_beta" / "quota.json"
    on_disk = json.loads(quota_file.read_text(encoding="utf-8"))
    assert on_disk["restart-probe-tester"]["used"] == 2
