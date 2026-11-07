"""Durable multi-voice plan recovery — server-authoritative (spec §14).

The browser is never trusted for plan existence or validity: a plan is the
same logical workflow only when the SERVER recomputes the same fingerprint
from the stored project data (script + character cast + generation-relevant
narration preferences) and the persisted plan record still matches it.

Everything is mocked: NO real MiMo / ElevenLabs / pyannote / FFmpeg calls,
no provider credits, no production FFmpeg, no real tester tokens or data
(isolated `tests/_test_storage` + test identities only).

The 16 scenarios (§14):

 1. unchanged project + cast reuse the same plan;
 2. repeated create-plan calls do not double-charge;
 3. browser localStorage loss does not cause duplicate plan creation;
 4. browser reload restores the existing valid plan (GET plan status);
 5. the same tester recovers the project from another browser session;
 6. a different tester cannot retrieve that plan;
 7. a changed script invalidates the previous plan;
 8. a changed approved voice mapping invalidates the previous plan;
 9. changed generation-relevant narration preferences invalidate it;
10. concurrent matching plan requests create one billable workflow;
11. generate-audio joins the recovered generation_id (no second charge);
12. a completed audiobook remains available after reload;
13. an explicit new regeneration follows the separate-generation policy;
14. provider failures keep the reservation (no refunds, conservative policy);
15. existing single-voice generation still works unchanged;
16. multi-voice generation uses the actual locked voice ids in every TTS call.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.routes import audio as audio_routes
from app.security import _ledger_charge
from app.services import AudioService, ElevenLabsClient, ProjectService
from fakes import (
    FakeComposer,
    FakeMiMo,
    MV_PARAGRAPHS,
    MV_SCRIPT,
    NARRATOR_VOICE_ID,
    SUMAN_VOICE_ID,
    make_multi_voice_plan_dict,
    multi_voice_plan_json,
    story_json,
)
from test_multi_voice import _default_mock, _settings, _use_analyzer, _use_planner, _wire_casting

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"

ALICE = {"Authorization": "Bearer alice-secret-1"}
BOB = {"Authorization": "Bearer bob-secret-2"}

# A replacement script (scenario 7) — different words, same shape.
NEW_SCRIPT = (
    "The corridor lamp flickered once.\n\n"
    "Someone had left the door wide open."
)
NEW_PARAGRAPHS = [
    "The corridor lamp flickered once.",
    "Someone had left the door wide open.",
]


# --------------------------------------------------------------------------- #
# Fixtures (same conventions as test_casting_ui / test_generation_limits)
# --------------------------------------------------------------------------- #
def _reset_runtime() -> None:
    """Clear in-memory middleware state + the isolated quota ledger files."""
    from app import security as security_mod

    with security_mod._RATE_LOCK:
        security_mod._RATE_WINDOWS.clear()
    with security_mod._DEDUP_LOCK:
        security_mod._IN_FLIGHT.clear()
        security_mod._LAST_COMPLETED.clear()
    import shutil

    shutil.rmtree(TEST_STORAGE / "_beta", ignore_errors=True)


@pytest.fixture()
def beta_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Private beta mode with two testers and a deterministic allowance."""
    monkeypatch.setenv("BETA_ACCESS_TOKENS", "alice:alice-secret-1,bob:bob-secret-2")
    monkeypatch.setenv("BETA_TESTER_GENERATION_LIMITS", "")
    monkeypatch.setenv("BETA_GENERATION_LIMIT", "10")
    monkeypatch.setenv("BETA_RATE_LIMIT_PER_MINUTE", "500")
    monkeypatch.setenv("BETA_LOGIN_RATE_LIMIT_PER_MINUTE", "10")
    # Window 0 = retries may land immediately (requests "spaced beyond the
    # window"): the QUOTA behaviour under test, not the duplicate guard.
    monkeypatch.setenv("BETA_DUPLICATE_WINDOW_SECONDS", "0")
    _reset_runtime()
    yield
    _reset_runtime()


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def other_session() -> TestClient:
    """A second, independent browser session for the same tester."""
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Helpers (mocked providers only)
# --------------------------------------------------------------------------- #
def _project(client: TestClient, headers: dict, content: str = MV_SCRIPT) -> str:
    response = client.post(
        "/projects",
        json={"name": "Recovery Test"},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    project_id = response.json()["project_id"]
    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", content.encode("utf-8"), "text/plain")},
        headers=headers,
    )
    assert upload.status_code == 201, upload.text
    return project_id


def _used(client: TestClient, headers: dict = ALICE) -> int:
    response = client.get("/usage", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()["generations_used"]


def _status(client: TestClient, project_id: str, headers: dict = ALICE) -> dict[str, Any]:
    response = client.get(
        f"/projects/{project_id}/multi-voice-plan/status", headers=headers
    )
    return response


def _analyze(client: TestClient, monkeypatch: pytest.MonkeyPatch, project_id: str) -> None:
    _use_analyzer(monkeypatch, FakeMiMo([story_json()]))
    response = client.post(
        f"/projects/{project_id}/analyze-characters", headers=ALICE
    )
    assert response.status_code == 200, response.text


def _setup_cast(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    project_id: str,
    *,
    mock: Any = None,
    suman_voice: str = SUMAN_VOICE_ID,
    approve: bool = True,
) -> Any:
    """Character analysis + casting + locks (+ approval) — zero allowances."""
    _analyze(client, monkeypatch, project_id)
    mock = _wire_casting(monkeypatch, mock)
    built = client.post(f"/projects/{project_id}/voice-casting", headers=ALICE)
    assert built.status_code == 200, built.text
    for speaker, voice in (("narrator", NARRATOR_VOICE_ID), ("suman", suman_voice)):
        selected = client.post(
            f"/projects/{project_id}/voice-casting/select",
            json={"speaker_id": speaker, "voice_id": voice},
            headers=ALICE,
        )
        assert selected.status_code == 200, selected.text
    if approve:
        approved = client.post(
            f"/projects/{project_id}/voice-casting/approve", headers=ALICE
        )
        assert approved.status_code == 200, approved.text
    return mock


def _create_plan(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    project_id: str,
    *,
    responses: Optional[list[str]] = None,
) -> tuple[Any, FakeMiMo]:
    fake = FakeMiMo(responses or [multi_voice_plan_json()])
    _use_planner(monkeypatch, fake)
    response = client.post(
        f"/projects/{project_id}/create-multi-voice-plan", headers=ALICE
    )
    return response, fake


def _generate(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    project_id: str,
    mock: Any,
    *,
    composer: Optional[FakeComposer] = None,
) -> Any:
    settings = _settings()
    composer = composer or FakeComposer()
    service = AudioService(
        settings,
        tts_client=ElevenLabsClient(
            settings, transport=httpx.MockTransport(mock)
        ),
        composer=composer,
    )
    monkeypatch.setattr(audio_routes, "_audio_service", lambda: service)
    return client.post(f"/projects/{project_id}/generate-audio", headers=ALICE)


def _plan_json_for(paragraphs: list[str]) -> str:
    """A valid word-exact utterance plan for arbitrary paragraphs."""
    utterances = []
    for index, text in enumerate(paragraphs):
        utterances.append(
            {
                "utterance_id": f"x-{index}",
                "paragraph_index": index,
                "sequence_index": 0,
                "speaker_id": "narrator",
                "speaker_type": "narrator",
                "original_text": text,
                "tts_text": text,
                "emotion": "neutral",
                "acting_intensity": 2,
                "delivery_style": "calm",
                "narration_direction": "Narrate plainly.",
                "pause_after_ms": 700,
            }
        )
    return json.dumps(
        {
            "language": "hi",
            "overall_tone": "suspenseful",
            "utterances": utterances,
        },
        ensure_ascii=False,
    )


def _metadata(project_id: str) -> dict[str, Any]:
    """The stored audiobook's own metadata (project.json `audio` section)."""
    info = ProjectService(get_settings()).get_audio_metadata(project_id)
    assert info is not None
    return info


def _state_record(project_id: str) -> dict[str, Any]:
    return json.loads(
        (TEST_STORAGE / project_id / "multi_voice_plan_state.json").read_text(
            encoding="utf-8"
        )
    )


# --------------------------------------------------------------------------- #
# 1–2. Reuse of the same plan; repeated create-plan calls never double-charge
# --------------------------------------------------------------------------- #
def test_unchanged_inputs_reuse_the_same_plan(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    _setup_cast(client, monkeypatch, project)

    first, fake_first = _create_plan(client, monkeypatch, project)
    assert first.status_code == 200, first.text
    assert fake_first.call_count == 1
    assert _used(client) == 1  # Case A: exactly one allowance

    second, fake_second = _create_plan(client, monkeypatch, project)
    assert second.status_code == 200, second.text
    # Same logical workflow: identical plan, NO second MiMo call (Case B).
    assert second.json()["utterances"] == first.json()["utterances"]
    assert fake_second.call_count == 0
    assert _used(client) == 1  # never double-charged

    status = _status(client, project)
    assert status.status_code == 200, status.text
    body = status.json()
    assert body["exists"] and body["valid"] and body["ready_for_generate"]
    assert body["status"] == "ready"
    assert body["plan_id"]
    assert body["generation_id"]


def test_repeated_create_plan_calls_do_not_double_charge(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    _setup_cast(client, monkeypatch, project)

    for _ in range(3):
        response, _fake = _create_plan(client, monkeypatch, project)
        assert response.status_code == 200, response.text
    assert _used(client) == 1

    generations = json.loads(
        (TEST_STORAGE / "_beta" / "generations.json").read_text(encoding="utf-8")
    )
    charged_entries = [
        entry
        for entry in generations["history"]
        if entry["kind"] == "plan" and entry["charged"]
    ]
    assert len(charged_entries) == 1  # exactly one billable plan workflow


# --------------------------------------------------------------------------- #
# 3–5. localStorage loss / reload / another session — server-side recovery
# --------------------------------------------------------------------------- #
def test_localstorage_loss_does_not_duplicate_plan_creation(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    _setup_cast(client, monkeypatch, project)
    first, _ = _create_plan(client, monkeypatch, project)
    assert first.status_code == 200
    assert _used(client) == 1

    # The browser lost ALL its local state and simply asks for a plan again.
    recovered, fake = _create_plan(client, monkeypatch, project)
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["utterances"] == first.json()["utterances"]
    assert fake.call_count == 0  # no new MiMo call
    assert _used(client) == 1  # no duplicate plan creation, no second charge
    assert _state_record(project)["generation_id"]  # workflow identity kept


def test_reload_restores_the_valid_plan_from_the_server(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    _setup_cast(client, monkeypatch, project)
    before = _used(client)
    _create_plan(client, monkeypatch, project)

    # A reload asks the server — GET is never charged and never mutates.
    status = _status(client, project)
    body = status.json()
    assert body["exists"] is True
    assert body["valid"] is True
    assert body["ready_for_generate"] is True
    assert body["status"] == "ready"
    assert body["plan_id"] and body["generation_id"] and body["plan_fingerprint"]
    assert body["created_at"] and body["updated_at"]
    assert _used(client) == before + 1  # only the original plan call


def test_same_tester_recovers_from_another_browser_session(
    client: TestClient,
    other_session: TestClient,
    beta_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(client, ALICE)
    _setup_cast(client, monkeypatch, project)
    first, _ = _create_plan(client, monkeypatch, project)

    # A different browser session of the SAME authenticated tester.
    status = _status(other_session, project)
    assert status.status_code == 200
    body = status.json()
    assert body["valid"] and body["ready_for_generate"]
    assert body["generation_id"] == _state_record(project)["generation_id"]

    recovered, fake = _create_plan(other_session, monkeypatch, project)
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["utterances"] == first.json()["utterances"]
    assert fake.call_count == 0
    assert _used(client) == 1


# --------------------------------------------------------------------------- #
# 6. Tester isolation — another tester never sees the plan
# --------------------------------------------------------------------------- #
def test_different_tester_cannot_retrieve_the_plan(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    _setup_cast(client, monkeypatch, project)
    _create_plan(client, monkeypatch, project)

    # Bob asking for Alice's project: same 404 as a missing project.
    stolen = _status(client, project, headers=BOB)
    assert stolen.status_code == 404

    # Bob's own project list never contains Alice's project.
    own = client.get("/projects", headers=BOB)
    assert own.status_code == 200
    assert all(p["project_id"] != project for p in own.json())

    # …and his own status endpoint stays empty of foreign state.
    bob_project = _project(client, BOB)
    bob_status = _status(client, bob_project, headers=BOB).json()
    assert bob_status["exists"] is False
    assert bob_status["generation_id"] is None
    assert _used(client, headers=BOB) == 0


# --------------------------------------------------------------------------- #
# 7–9. Changed inputs invalidate the plan (Case C)
# --------------------------------------------------------------------------- #
def test_changed_script_invalidates_the_previous_plan(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    _setup_cast(client, monkeypatch, project)
    first, _ = _create_plan(client, monkeypatch, project)
    first_plan_id = _status(client, project).json()["plan_id"]

    replaced = client.post(
        f"/projects/{project}/upload-script",
        files={"file": ("script.txt", NEW_SCRIPT.encode("utf-8"), "text/plain")},
        headers=ALICE,
    )
    assert replaced.status_code == 201, replaced.text

    stale = _status(client, project).json()
    assert stale["exists"] is True
    assert stale["valid"] is False  # the old plan must not be reused silently
    assert stale["ready_for_generate"] is False

    second, fake = _create_plan(
        client, monkeypatch, project, responses=[_plan_json_for(NEW_PARAGRAPHS)]
    )
    assert second.status_code == 200, second.text
    assert fake.call_count == 1  # a genuine new plan workflow (Case C)
    assert _used(client) == 2
    fresh = _status(client, project).json()
    assert fresh["valid"] and fresh["ready_for_generate"]
    assert fresh["plan_id"] != first_plan_id  # new workflow identity
    # The old workflow's audit record is preserved (history, never deleted).
    assert len(_state_record(project)["history"]) >= 1


def test_changed_voice_mapping_invalidates_the_previous_plan(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    mock = _setup_cast(client, monkeypatch, project)
    _create_plan(client, monkeypatch, project)

    # Replace one locked voice (invalidates approval), then re-approve.
    changed = client.post(
        f"/projects/{project}/voice-casting/select",
        json={"speaker_id": "suman", "voice_id": "voice-hi-3"},
        headers=ALICE,
    )
    assert changed.status_code == 200, changed.text
    approved = client.post(
        f"/projects/{project}/voice-casting/approve", headers=ALICE
    )
    assert approved.status_code == 200, approved.text

    stale = _status(client, project).json()
    assert stale["valid"] is False and stale["ready_for_generate"] is False

    second, fake = _create_plan(client, monkeypatch, project)
    assert second.status_code == 200, second.text
    assert fake.call_count == 1  # rebuilt for the new approved mapping
    assert _used(client) == 2
    assert _status(client, project).json()["ready_for_generate"] is True
    assert mock is not None


def test_changed_narration_preferences_invalidate_the_plan(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    _setup_cast(client, monkeypatch, project)
    _create_plan(client, monkeypatch, project)
    assert _status(client, project).json()["ready_for_generate"] is True

    prefs = {
        "preset": "expressive",
        "acting_intensity": 4,
        "whisper_policy": "never",
        "whisper_paragraphs": [],
        "tag_density": "minimal",
        "custom_direction": "Read slower.",
    }
    saved = client.put(
        f"/projects/{project}/narration-preferences", json=prefs, headers=ALICE
    )
    assert saved.status_code == 200, saved.text

    stale = _status(client, project).json()
    assert stale["valid"] is False and stale["ready_for_generate"] is False

    second, fake = _create_plan(client, monkeypatch, project)
    assert second.status_code == 200, second.text
    assert fake.call_count == 1
    assert _used(client) == 2
    assert _status(client, project).json()["ready_for_generate"] is True


# --------------------------------------------------------------------------- #
# 10. Atomic check-and-create under _QUOTA_LOCK
# --------------------------------------------------------------------------- #
def test_concurrent_plan_requests_create_one_billable_workflow(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    _setup_cast(client, monkeypatch, project)

    settings = get_settings()
    results: list[tuple[Optional[str], Optional[str], bool]] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        results.append(
            _ledger_charge(settings, "alice", project, "plan", mv_plan_request=True)
        )

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert sorted(charged for _err, _gid, charged in results) == [False, True]
    # Both requests describe ONE workflow: same generation_id.
    assert len({gid for _err, gid, _c in results}) == 1
    assert _used(client) == 1  # exactly one allowance for both requests


# --------------------------------------------------------------------------- #
# 11. generate-audio joins the recovered workflow
# --------------------------------------------------------------------------- #
def test_generate_audio_joins_the_recovered_generation_id(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    mock = _setup_cast(client, monkeypatch, project)
    _create_plan(client, monkeypatch, project)
    generation_id = _status(client, project).json()["generation_id"]
    assert _used(client) == 1

    audio = _generate(client, monkeypatch, project, mock)
    assert audio.status_code == 200, audio.text
    assert audio.json()["generation_mode"] == "multi_voice"

    # SAME generation_id end-to-end and still exactly ONE allowance.
    assert _metadata(project)["generation_id"] == generation_id
    assert _used(client) == 1
    assert _state_record(project)["status"] == "completed"


# --------------------------------------------------------------------------- #
# 12–13. Completed audiobook + explicit regeneration (Case D)
# --------------------------------------------------------------------------- #
def test_completed_audiobook_remains_available_after_reload(
    client: TestClient,
    other_session: TestClient,
    beta_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(client, ALICE)
    mock = _setup_cast(client, monkeypatch, project)
    _create_plan(client, monkeypatch, project)
    audio = _generate(client, monkeypatch, project, mock)
    assert audio.status_code == 200
    audio_url = audio.json()["audio_url"]

    # "Reload": a fresh session still finds the finished audiobook.
    restored = other_session.get(f"/projects/{project}/audio", headers=ALICE)
    assert restored.status_code == 200
    assert restored.json()["audio_url"] == audio_url

    status = _status(other_session, project).json()
    assert status["exists"] is True
    assert status["valid"] is True
    assert status["status"] == "completed"
    # A completed workflow is NOT presented as an unfinished plan.
    assert status["ready_for_generate"] is False
    assert _used(client) == 1


def test_explicit_regeneration_follows_the_separate_generation_policy(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    mock = _setup_cast(client, monkeypatch, project)
    _create_plan(client, monkeypatch, project)
    first = _generate(client, monkeypatch, project, mock)
    assert first.status_code == 200
    assert _used(client) == 1

    # Explicit new audiobook after completion = a NEW generation (Case D).
    second = _generate(client, monkeypatch, project, mock)
    assert second.status_code == 200, second.text
    assert second.json()["generation_mode"] == "multi_voice"
    assert _used(client) == 2  # existing separate-generation quota policy
    assert _state_record(project)["status"] == "completed"


# --------------------------------------------------------------------------- #
# 14. Provider failure: conservative policy, never a refund (Case E)
# --------------------------------------------------------------------------- #
def test_provider_failure_keeps_the_reservation_without_refund(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    failing_text = make_multi_voice_plan_dict()["utterances"][0]["tts_text"]
    mock = _setup_cast(
        client,
        monkeypatch,
        project,
        mock=_default_mock(tts_fail_text=failing_text),
    )
    _create_plan(client, monkeypatch, project)
    assert _used(client) == 1

    failed = _generate(client, monkeypatch, project, mock)
    assert failed.status_code >= 400
    assert _used(client) == 1  # the reservation is NEVER refunded
    assert _state_record(project)["status"] == "failed"

    # A genuine retry is a new workflow → follows the existing quota policy.
    retry = _generate(client, monkeypatch, project, _default_mock())
    assert retry.status_code == 200, retry.text
    assert retry.json()["generation_mode"] == "multi_voice"
    assert _used(client) == 2


# --------------------------------------------------------------------------- #
# 15. Single-voice workflow is unchanged
# --------------------------------------------------------------------------- #
def test_single_voice_generation_still_works(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    mock = _default_mock()
    audio = _generate(client, monkeypatch, project, mock)
    assert audio.status_code == 200, audio.text
    assert audio.json()["generation_mode"] != "multi_voice"
    assert _used(client) == 1  # one complete workflow = one allowance

    status = _status(client, project).json()
    assert status["exists"] is False  # no multi-voice plan was ever created
    assert _used(client) == 1  # the status read is free


# --------------------------------------------------------------------------- #
# 16. Multi-voice generation really uses the locked voice ids
# --------------------------------------------------------------------------- #
def test_multi_voice_generation_uses_locked_voice_ids_in_tts(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(client, ALICE)
    mock = _setup_cast(client, monkeypatch, project)
    plan = _create_plan(client, monkeypatch, project)[0].json()
    audio = _generate(client, monkeypatch, project, mock)
    assert audio.status_code == 200, audio.text
    assert audio.json()["generation_mode"] == "multi_voice"

    expected = [
        NARRATOR_VOICE_ID if u["speaker_id"] == "narrator" else SUMAN_VOICE_ID
        for u in plan["utterances"]
    ]
    used_voices = [
        request["path"].split("/text-to-speech/")[1].split("/")[0]
        for request in mock.tts_requests
    ]
    assert used_voices == expected  # every utterance in its locked voice

    metadata = _metadata(project)
    assert metadata["mode"] == "multi_voice"
    assert metadata["generation_id"] == _state_record(project)["generation_id"]
