"""Per-tester configurable generation limits — automated tests.

Everything is mocked: NO real MiMo / ElevenLabs / pyannote / FFmpeg calls,
no provider credits, no real tester usage or tokens are touched (isolated
`tests/_test_storage` + test identities only).

Coverage (matching the implementation spec):

* §1 configuration      — per-tester limits, defaults, startup validation
                          (unknown identities, duplicates, malformed entries,
                          negative/decimal values, zero allowance, no token
                          leakage in errors);
* §2 persistent usage   — survives restart, limit changes never reset `used`,
                          raising a limit restores allowance, lowering it
                          below usage blocks generation, tester isolation;
* §3 generation transactions — one complete audiobook workflow (planning +
                          MiMo retries + validation + ElevenLabs segments +
                          FFmpeg + storage) consumes EXACTLY ONE allowance
                          under ONE generation_id; duplicates, concurrency,
                          provider failures;
* §4 authentication     — login redemption and generation allowances are
                          independent; usage is per-tester only;
* §5 script limits      — 3000 accepted / 3001 rejected, Unicode code points
                          (Hindi, emoji), newline parity with the frontend;
* §6 frontend           — node-driven tests of the real app.js/limits.js.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import (
    DEFAULT_ELEVENLABS_BASE_URL,
    get_settings,
    parse_generation_limit,
    parse_tester_generation_limits,
)
from app.main import app
from app.routes import audio as audio_routes
from app.routes import narration as narration_routes
from app.security import effective_generation_limit
from app.services import AudioService, ElevenLabsClient, NarrationPlanner
from app.services.project_service import ProjectService
from fakes import FAKE_MP3, FakeComposer, FakeMiMo, plan_json

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"
JS_DIR = Path(__file__).resolve().parent / "js"
NODE = shutil.which("node")

SCRIPT = (
    "The lane was empty at midnight.\n\n"
    "Somewhere a dog barked twice.\n\n"
    "She counted her steps to the iron gate."
)

ALICE = {"Authorization": "Bearer alice-secret-1"}
BOB = {"Authorization": "Bearer bob-secret-2"}


# --------------------------------------------------------------------------- #
# Fixtures + helpers
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


def _import(client: TestClient, headers: dict, content: str = SCRIPT):
    return client.post(
        "/projects/from-script",
        files={"file": ("s.txt", content.encode("utf-8"), "text/plain")},
        headers=headers,
    )


def _create_project(client: TestClient, headers: dict, content: str = SCRIPT) -> str:
    response = _import(client, headers, content=content)
    assert response.status_code == 201, response.text
    return response.json()["project_id"]


def _usage(client: TestClient, headers: dict) -> dict[str, Any]:
    response = client.get("/usage", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def _generations() -> dict[str, Any]:
    path = TEST_STORAGE / "_beta" / "generations.json"
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


def _fake_audio_service(composer: Optional[FakeComposer] = None) -> AudioService:
    settings = dataclasses.replace(
        get_settings(),
        elevenlabs_api_key="sk-beta-test",
        elevenlabs_voice_id="voice-beta",
        elevenlabs_base_url=DEFAULT_ELEVENLABS_BASE_URL,
    )
    return AudioService(
        settings,
        tts_client=ElevenLabsClient(settings, transport=httpx.MockTransport(_Recorder())),
        composer=composer or FakeComposer(),
    )


def _use_fake_audio(
    monkeypatch: pytest.MonkeyPatch, composer: Optional[FakeComposer] = None
) -> AudioService:
    """Mocked ElevenLabs + FFmpeg for generate-audio (never bills)."""
    service = _fake_audio_service(composer)
    monkeypatch.setattr(audio_routes, "_audio_service", lambda: service)
    return service


def _use_fake_mimo(monkeypatch: pytest.MonkeyPatch, fake: FakeMiMo) -> FakeMiMo:
    monkeypatch.setattr(
        narration_routes,
        "_planner",
        lambda: NarrationPlanner(get_settings(), mimo_client=fake),
    )
    return fake


def _plan_texts() -> list[str]:
    from app.services.narration_planner import split_paragraphs

    return split_paragraphs(SCRIPT)


# --------------------------------------------------------------------------- #
# §1 Configuration parsing + startup validation
# --------------------------------------------------------------------------- #
def test_per_tester_limits_override_the_default() -> None:
    limits = parse_tester_generation_limits(
        "client1:1,client2:3,client3:5,internal:10",
        ["client1", "client2", "client3", "internal", "client9"],
    )
    assert limits == {"client1": 1, "client2": 3, "client3": 5, "internal": 10}

    settings = dataclasses.replace(
        get_settings(), beta_generation_limit=2, beta_tester_generation_limits=limits
    )
    assert effective_generation_limit(settings, "client1") == 1
    assert effective_generation_limit(settings, "internal") == 10


def test_default_limit_applies_to_unlisted_testers() -> None:
    limits = parse_tester_generation_limits("client1:1", ["client1", "client9"])
    settings = dataclasses.replace(
        get_settings(), beta_generation_limit=4, beta_tester_generation_limits=limits
    )
    assert effective_generation_limit(settings, "client9") == 4  # no override
    assert effective_generation_limit(settings, "someone-else") == 4


def test_zero_allowance_parses_and_means_no_generations() -> None:
    assert parse_tester_generation_limits("client1:0", ["client1"]) == {"client1": 0}
    settings = dataclasses.replace(
        get_settings(),
        beta_generation_limit=5,
        beta_tester_generation_limits={"client1": 0},
    )
    assert effective_generation_limit(settings, "client1") == 0


def test_unknown_tester_identity_is_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_tester_generation_limits("ghost:2", ["client1", "client2"])
    assert "ghost" in str(excinfo.value)


def test_duplicate_tester_entries_are_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_tester_generation_limits("client1:1,client1:2", ["client1"])
    assert "duplicate" in str(excinfo.value)
    assert "client1" in str(excinfo.value)


@pytest.mark.parametrize(
    "raw",
    [
        "client1",  # no limit
        ":2",  # no name
        "client1:",  # no value
        "client1:2:3",  # two colons
        "client1=2",  # wrong separator
        "client1:2,,client2:3",  # empty chunk is malformed
    ],
)
def test_malformed_entries_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_tester_generation_limits(raw, ["client1", "client2"])


def test_negative_values_are_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_tester_generation_limits("client1:-1", ["client1"])
    assert "negative" in str(excinfo.value)


def test_decimal_values_are_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_tester_generation_limits("client1:1.5", ["client1"])
    assert "non-negative integers" in str(excinfo.value)


def test_non_numeric_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        parse_tester_generation_limits("client1:many", ["client1"])


def test_configuration_errors_never_expose_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BETA_ACCESS_TOKENS", "alice:alice-secret-1,bob:bob-secret-2")
    monkeypatch.setenv("BETA_TESTER_GENERATION_LIMITS", "ghost:1")
    with pytest.raises(ValueError) as excinfo:
        get_settings()
    message = str(excinfo.value)
    assert "ghost" in message  # identifies the invalid tester…
    assert "alice-secret-1" not in message  # …but never a token
    assert "bob-secret-2" not in message


def test_startup_configuration_validation_rejects_bad_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unknown identity in BETA_TESTER_GENERATION_LIMITS -> startup error.
    monkeypatch.setenv("BETA_ACCESS_TOKENS", "alice:alice-secret-1")
    monkeypatch.setenv("BETA_TESTER_GENERATION_LIMITS", "mallory:1")
    with pytest.raises(ValueError) as excinfo:
        get_settings()
    assert "mallory" in str(excinfo.value)

    # Invalid default limits -> startup error (never silently reinterpreted).
    monkeypatch.setenv("BETA_TESTER_GENERATION_LIMITS", "")
    for bad in ("-1", "1.5", "unlimited"):
        monkeypatch.setenv("BETA_GENERATION_LIMIT", bad)
        with pytest.raises(ValueError):
            get_settings()

    # Non-negative integers (including 0) are accepted.
    monkeypatch.setenv("BETA_GENERATION_LIMIT", "0")
    assert get_settings().beta_generation_limit == 0


def test_default_limit_parser_rejects_negatives_and_decimals() -> None:
    assert parse_generation_limit("7", "X") == 7
    assert parse_generation_limit("0", "X") == 0
    with pytest.raises(ValueError):
        parse_generation_limit("-2", "X")
    with pytest.raises(ValueError):
        parse_generation_limit("2.5", "X")


# --------------------------------------------------------------------------- #
# §2 Persistent per-tester usage
# --------------------------------------------------------------------------- #
def test_usage_survives_an_application_restart(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_audio(monkeypatch)
    with TestClient(app) as client:
        project = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{project}/generate-audio", headers=ALICE).status_code
            == 200
        )
        before = _usage(client, ALICE)
        assert before["generations_used"] == 1

    # Restart: in-memory middleware state is reset, storage is kept.
    _reset_runtime(clear_storage=False)
    with TestClient(app) as client:
        after = _usage(client, ALICE)
        assert after["generations_used"] == 1  # never reset by a restart
        assert after["generations_remaining"] == before["generations_remaining"]


def test_increasing_a_limit_restores_allowance_without_resetting_usage(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BETA_GENERATION_LIMIT", "1")
    _use_fake_audio(monkeypatch)
    with TestClient(app) as client:
        p1 = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{p1}/generate-audio", headers=ALICE).status_code
            == 200
        )
        p2 = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{p2}/generate-audio", headers=ALICE).status_code
            == 429
        )

        monkeypatch.setenv("BETA_GENERATION_LIMIT", "3")  # config reload
        usage = _usage(client, ALICE)
        assert usage["generations_used"] == 1  # usage NOT reset
        assert usage["generations_remaining"] == 2  # new allowance available

        p3 = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{p3}/generate-audio", headers=ALICE).status_code
            == 200
        )
        assert _usage(client, ALICE)["generations_used"] == 2


def test_decreasing_a_limit_below_usage_blocks_new_generations(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BETA_GENERATION_LIMIT", "2")
    _use_fake_audio(monkeypatch)
    with TestClient(app) as client:
        for _ in range(2):
            project = _create_project(client, ALICE)
            assert (
                client.post(f"/projects/{project}/generate-audio", headers=ALICE).status_code
                == 200
            )
        assert _usage(client, ALICE)["generations_used"] == 2

        monkeypatch.setenv("BETA_GENERATION_LIMIT", "1")  # below usage
        usage = _usage(client, ALICE)
        assert usage["generations_remaining"] == 0
        blocked = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{blocked}/generate-audio", headers=ALICE).status_code
            == 429
        )

        # Still blocked while the limit does not exceed current usage…
        monkeypatch.setenv("BETA_GENERATION_LIMIT", "2")
        assert _usage(client, ALICE)["generations_remaining"] == 0
        # …and unblocked as soon as it does.
        monkeypatch.setenv("BETA_GENERATION_LIMIT", "3")
        assert _usage(client, ALICE)["generations_remaining"] == 1
        allowed = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{allowed}/generate-audio", headers=ALICE).status_code
            == 200
        )


def test_usage_is_isolated_between_testers(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_audio(monkeypatch)
    with TestClient(app) as client:
        project = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{project}/generate-audio", headers=ALICE).status_code
            == 200
        )
        assert _usage(client, ALICE)["generations_used"] == 1
        assert _usage(client, BOB)["generations_used"] == 0
        assert _usage(client, BOB)["generations_remaining"] == 2


def test_individual_override_is_used_at_request_time(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BETA_TESTER_GENERATION_LIMITS", "alice:1")
    _use_fake_audio(monkeypatch)
    with TestClient(app) as client:
        assert _usage(client, ALICE)["generation_limit"] == 1
        assert _usage(client, BOB)["generation_limit"] == 2  # default
        p1 = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{p1}/generate-audio", headers=ALICE).status_code
            == 200
        )
        p2 = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{p2}/generate-audio", headers=ALICE).status_code
            == 429
        )
        # Bob is unaffected by Alice's tighter limit.
        pb = _create_project(client, BOB)
        assert (
            client.post(f"/projects/{pb}/generate-audio", headers=BOB).status_code
            == 200
        )


# --------------------------------------------------------------------------- #
# §3 Generation transactions
# --------------------------------------------------------------------------- #
def test_exhausted_allowance_blocks_generation(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BETA_GENERATION_LIMIT", "1")
    _use_fake_audio(monkeypatch)
    with TestClient(app) as client:
        p1 = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{p1}/generate-audio", headers=ALICE).status_code
            == 200
        )
        p2 = _create_project(client, ALICE)
        refused = client.post(f"/projects/{p2}/generate-audio", headers=ALICE)
        assert refused.status_code == 429
        detail = refused.json()["detail"].lower()
        assert "limit" in detail
        assert "administrator" in detail


def test_duplicate_generation_requests_do_not_double_charge(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BETA_GENERATION_LIMIT", "2")
    _use_fake_audio(monkeypatch)
    with TestClient(app) as client:
        project = _create_project(client, ALICE)
        first = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
        assert first.status_code == 200
        duplicate = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
        assert duplicate.status_code == 409  # duplicate guard
        assert _usage(client, ALICE)["generations_used"] == 1  # no double charge


def test_concurrent_generation_attempts_cannot_exceed_the_allowance(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BETA_GENERATION_LIMIT", "1")
    _use_fake_audio(monkeypatch)
    barrier = threading.Barrier(2, timeout=30)
    results: list[int] = []

    def worker() -> None:
        with TestClient(app) as worker_client:
            project = _create_project(worker_client, ALICE)
            barrier.wait()
            response = worker_client.post(
                f"/projects/{project}/generate-audio", headers=ALICE
            )
            results.append(response.status_code)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert sorted(results) == [200, 429]  # exactly one spent the last allowance
    with TestClient(app) as client:
        assert _usage(client, ALICE)["generations_used"] == 1


def test_full_pipeline_consumes_exactly_one_allowance(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planning + MiMo retries + validation + segments + FFmpeg + storage = 1."""
    fake_mimo = _use_fake_mimo(monkeypatch, FakeMiMo([plan_json(_plan_texts())]))
    service = _use_fake_audio(monkeypatch)
    with TestClient(app) as client:
        project = _create_project(client, ALICE)
        plan = client.post(f"/projects/{project}/analyze-narration", headers=ALICE)
        assert plan.status_code == 200, plan.text
        audio = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
        assert audio.status_code == 200, audio.text

        usage = _usage(client, ALICE)
        assert usage["generations_used"] == 1  # exactly ONE allowance
        assert usage["generations_remaining"] == 1

        # ElevenLabs segments and FFmpeg ran inside that one generation.
        assert service.composer.segment_counts == [3]
        assert fake_mimo.call_count == 1

        # One generation_id shared by planning and generation (and the
        # stored audiobook metadata).
        entries = [
            entry
            for entry in _generations()["history"]
            if entry.get("project_id") == project
        ]
        plan_ids = {e["generation_id"] for e in entries if e["kind"] == "plan"}
        generate_ids = {e["generation_id"] for e in entries if e["kind"] == "generate"}
        assert len(plan_ids) == 1
        assert plan_ids == generate_ids
        assert sum(1 for e in entries if e["charged"]) == 1  # one reservation

        metadata = ProjectService(get_settings()).get_audio_metadata(project)
        assert metadata is not None
        assert metadata["generation_id"] in plan_ids


def test_mimo_retries_do_not_consume_additional_allowances(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = json.dumps({"paragraphs": []})  # invalid plan -> targeted retry
    fake = _use_fake_mimo(monkeypatch, FakeMiMo([bad, plan_json(_plan_texts())]))
    with TestClient(app) as client:
        project = _create_project(client, ALICE)
        response = client.post(f"/projects/{project}/analyze-narration", headers=ALICE)
        assert response.status_code == 200, response.text
        assert fake.call_count == 2  # one internal retry
        usage = _usage(client, ALICE)
        assert usage["generations_used"] == 1  # retries are free


def test_elevenlabs_segment_count_does_not_affect_allowance(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_mimo(monkeypatch, FakeMiMo([plan_json(_plan_texts())]))
    service = _use_fake_audio(monkeypatch)
    with TestClient(app) as client:
        project = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{project}/analyze-narration", headers=ALICE).status_code
            == 200
        )
        assert (
            client.post(f"/projects/{project}/generate-audio", headers=ALICE).status_code
            == 200
        )
        assert service.composer.segment_counts == [3]  # 3 segments…
        assert _usage(client, ALICE)["generations_used"] == 1  # …one allowance


def test_ffmpeg_assembly_does_not_consume_another_allowance(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_mimo(monkeypatch, FakeMiMo([plan_json(_plan_texts())]))
    service = _use_fake_audio(monkeypatch)
    with TestClient(app) as client:
        project = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{project}/analyze-narration", headers=ALICE).status_code
            == 200
        )
        assert (
            client.post(f"/projects/{project}/generate-audio", headers=ALICE).status_code
            == 200
        )
        assert len(service.composer.calls) == 1  # FFmpeg ran once…
        assert _usage(client, ALICE)["generations_used"] == 1  # …no extra charge


def test_provider_failure_keeps_the_allowance(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conservative policy: never refund after billable work may have run."""
    _use_fake_mimo(monkeypatch, FakeMiMo([plan_json(_plan_texts())]))
    _use_fake_audio(monkeypatch, composer=FakeComposer(fail=True))
    with TestClient(app) as client:
        project = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{project}/analyze-narration", headers=ALICE).status_code
            == 200
        )
        failed = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
        assert failed.status_code == 500  # simulated FFmpeg failure
        assert _usage(client, ALICE)["generations_used"] == 1  # NOT refunded

        # A retry is allowed (failed attempts never block) and is its own
        # generation — the pipeline starts a fresh transaction.
        _use_fake_audio(monkeypatch)  # working providers again
        retry = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
        assert retry.status_code == 200
        assert _usage(client, ALICE)["generations_used"] == 2


def test_regenerate_after_success_costs_one_more_allowance(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_audio(monkeypatch)
    with TestClient(app) as client:
        project = _create_project(client, ALICE)
        assert (
            client.post(f"/projects/{project}/generate-audio", headers=ALICE).status_code
            == 200
        )
        assert (
            client.post(f"/projects/{project}/generate-audio", headers=ALICE).status_code
            == 409  # immediate duplicate -> protected by the duplicate guard
        )
        # After the guard window a new generation is a new allowance.
        from app import security as security_mod

        with security_mod._DEDUP_LOCK:
            security_mod._LAST_COMPLETED.clear()
        assert (
            client.post(f"/projects/{project}/generate-audio", headers=ALICE).status_code
            == 200
        )
        assert _usage(client, ALICE)["generations_used"] == 2


# --------------------------------------------------------------------------- #
# §4 Authentication independence + usage privacy
# --------------------------------------------------------------------------- #
def test_usage_endpoint_requires_authentication(
    client: TestClient, beta_env
) -> None:
    assert client.get("/usage").status_code == 401
    assert client.get("/usage", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_usage_response_contains_only_the_documented_fields(
    client: TestClient, beta_env
) -> None:
    usage = _usage(client, ALICE)
    assert set(usage) == {
        "generation_limit",
        "generations_used",
        "generations_remaining",
        "max_script_characters",
    }
    assert usage["generation_limit"] == 2
    assert usage["generations_used"] == 0
    assert usage["generations_remaining"] == 2


def test_one_tester_cannot_read_another_testers_usage(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_audio(monkeypatch)
    project = _create_project(client, ALICE)
    assert (
        client.post(f"/projects/{project}/generate-audio", headers=ALICE).status_code
        == 200
    )
    alice_usage = _usage(client, ALICE)
    bob_usage = _usage(client, BOB)
    assert alice_usage["generations_used"] == 1
    assert bob_usage["generations_used"] == 0  # Alice's usage never leaks
    assert "alice" not in json.dumps(bob_usage).lower()
    assert "bob" not in json.dumps(alice_usage).lower()


def test_login_redemption_does_not_consume_a_generation(
    client: TestClient, beta_env
) -> None:
    response = client.post(
        "/auth/login", data={"token": "alice-secret-1"}, follow_redirects=False
    )
    assert response.status_code in (302, 303)
    assert _usage(client, ALICE)["generations_used"] == 0  # login is free
    assert _usage(client, {})["generations_used"] == 0  # cookie session too


def test_generation_does_not_invalidate_the_login_token(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_audio(monkeypatch)
    project = _create_project(client, ALICE)
    assert (
        client.post(f"/projects/{project}/generate-audio", headers=ALICE).status_code
        == 200
    )
    # The token still redeems a fresh session after generation…
    response = client.post(
        "/auth/login", data={"token": "alice-secret-1"}, follow_redirects=False
    )
    assert response.status_code in (302, 303)
    assert "beta_session" in response.headers.get("set-cookie", "")
    # …and still works as a bearer token (unchanged one-time login flow).
    assert client.get("/health", headers=ALICE).status_code == 200
    # Redeeming never refunds or consumes an allowance.
    assert _usage(client, ALICE)["generations_used"] == 1


# --------------------------------------------------------------------------- #
# §5 Script character limits (Unicode code points)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("content", "ok"),
    [
        ("a" * 3000, True),  # exactly 3000 -> accepted
        ("a" * 3001, False),  # 3001 -> rejected
        ("क" * 3000, True),  # Hindi: code points, not the 9000 UTF-8 bytes
        ("क" * 3001, False),
        ("😀" * 3000, True),  # supplementary plane: 1 code point per emoji
        ("😀" * 3001, False),
        ("a" * 2998 + "\r\n", True),  # newlines counted as-is (parity)
        ("a" * 2999 + "\r\n", False),
    ],
)
def test_character_limit_counts_unicode_code_points(
    monkeypatch: pytest.MonkeyPatch, content: str, ok: bool
) -> None:
    monkeypatch.setenv("MAX_SCRIPT_CHARACTERS", "3000")
    with TestClient(app) as client:
        response = _import(client, {}, content=content)
        if ok:
            assert response.status_code == 201, response.text
            assert response.json()["character_count"] == 3000
        else:
            assert response.status_code == 400
            assert "too long" in response.json()["detail"]


def test_hindi_script_is_never_measured_in_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "क" * 3000  # 3000 code points = 9000 UTF-8 bytes
    assert len(content) == 3000
    assert len(content.encode("utf-8")) == 9000
    monkeypatch.setenv("MAX_SCRIPT_CHARACTERS", "3000")
    with TestClient(app) as client:
        assert _import(client, {}, content=content).status_code == 201


def test_generation_rechecks_the_character_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_SCRIPT_CHARACTERS", "100000")
    with TestClient(app) as client:
        project = _create_project(client, {}, content="a" * 4000)
        monkeypatch.setenv("MAX_SCRIPT_CHARACTERS", "3000")  # tightened later
        _use_fake_audio(monkeypatch)
        response = client.post(f"/projects/{project}/generate-audio")
        assert response.status_code == 400
        assert "too long" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# §6 Frontend (node-driven: real app.js + limits.js, fake DOM, no network)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_frontend_counter_matches_python_counts() -> None:
    """JavaScript and Python must count characters identically."""
    result = subprocess.run(
        [NODE, str(JS_DIR / "limits.test.js")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    prefix = "SAMPLES="
    line = next(
        line for line in result.stdout.splitlines() if line.startswith(prefix)
    )
    js_counts = json.loads(line[len(prefix) :])
    python_counts = {
        "ascii_3000": len("a" * 3000),
        "hindi_3000": len("क" * 3000),
        "emoji_3000": len("😀" * 3000),
        "mixed": len("नमस्ते 😀 world — कहानी"),
        "crlf_3000": len("a" * 2998 + "\r\n"),
        "lf_2999": len("a" * 2998 + "\n"),
        "bom_3000": len("a" * 3000),  # backend strips the BOM (utf-8-sig)
        "zwj_family": len("\U0001F468\u200D\U0001F469\u200D\U0001F467"),
    }
    assert js_counts == python_counts


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_frontend_app_behavior() -> None:
    """Usage display, disabled controls and the exhaustion message."""
    result = subprocess.run(
        [NODE, str(JS_DIR / "app_behavior.test.js")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all frontend app behavior checks passed" in result.stdout


def test_page_contains_usage_counter_and_quota_elements(
    client: TestClient, beta_env
) -> None:
    page = client.get("/", headers={**ALICE, "Accept": "text/html"})
    assert page.status_code == 200
    for needle in (
        'id="usage-line"',
        'id="char-count"',
        'id="quota-note"',
        "/static/limits.js",
    ):
        assert needle in page.text
