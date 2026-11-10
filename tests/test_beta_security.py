"""Private-beta security tests — hermetic (providers always mocked).

Covers the beta security layer end to end:

* unauthenticated requests (401 for API clients, login redirect for
  browsers) across the frontend, API, project routes and media downloads;
* login/logout session cookies + bearer/header tokens;
* per-tester project isolation (a tester can never see or download another
  tester's projects — even via /media or path traversal);
* generation quotas per tester;
* upload limits (file size, script length, estimated duration);
* duplicate generation submissions (in-flight + recency guard) and API rate
  limiting;
* /docs + /openapi.json disabled for beta users;
* privacy notice + project deletion (single and "delete my data");
* the existing audiobook flow keeps working with and without beta mode.

No real MiMo / ElevenLabs / pyannoteAI calls, no credits are spent.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import security as security_mod
from app.config import DEFAULT_ELEVENLABS_BASE_URL, get_settings
from app.main import app
from app.routes import audio as audio_routes
from app.routes import casting as casting_routes
from app.services import AudioService, ElevenLabsClient
from fakes import FAKE_MP3, FakeComposer

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"

SCRIPT = (
    "The lane was empty at midnight.\n\n"
    "Somewhere a dog barked twice.\n\n"
    "She counted her steps to the iron gate."
)

ALICE = {"Authorization": "Bearer alice-secret-1"}
BOB = {"Authorization": "Bearer bob-secret-2"}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _reset_beta_state() -> None:
    with security_mod._RATE_LOCK:
        security_mod._RATE_WINDOWS.clear()
    with security_mod._DEDUP_LOCK:
        security_mod._IN_FLIGHT.clear()
        security_mod._LAST_COMPLETED.clear()
    shutil.rmtree(TEST_STORAGE / "_beta", ignore_errors=True)


@pytest.fixture()
def beta_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Private beta mode with two testers and tight, deterministic limits."""
    monkeypatch.setenv("BETA_ACCESS_TOKENS", "alice:alice-secret-1,bob:bob-secret-2")
    monkeypatch.setenv("BETA_GENERATION_LIMIT", "2")
    monkeypatch.setenv("BETA_GENERATION_WINDOW_HOURS", "24")
    monkeypatch.setenv("BETA_RATE_LIMIT_PER_MINUTE", "200")
    monkeypatch.setenv("BETA_LOGIN_RATE_LIMIT_PER_MINUTE", "5")
    monkeypatch.setenv("BETA_DUPLICATE_WINDOW_SECONDS", "10")
    _reset_beta_state()
    yield
    _reset_beta_state()


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _import(client: TestClient, headers: dict, content: str = SCRIPT, name: str = "s.txt"):
    return client.post(
        "/projects/from-script",
        files={"file": (name, content.encode("utf-8"), "text/plain")},
        headers=headers,
    )


def _create_project(
    client: TestClient, headers: dict, content: str = SCRIPT
) -> str:
    response = _import(client, headers, content=content)
    assert response.status_code == 201, response.text
    return response.json()["project_id"]


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(
            200, content=FAKE_MP3, headers={"content-type": "audio/mpeg"},
            request=request,
        )


def _use_fake_audio(monkeypatch: pytest.MonkeyPatch):
    """Mocked ElevenLabs + FFmpeg for generate-audio (never bills)."""
    settings = dataclasses.replace(
        get_settings(),
        elevenlabs_api_key="sk-beta-test",
        elevenlabs_voice_id="voice-beta",
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
# Unauthenticated access
# --------------------------------------------------------------------------- #
def test_401_is_reserved_for_the_callers_own_session() -> None:
    """Provider auth failures are UPSTREAM errors (502), never 401.

    Regression: while the ElevenLabs credit quota was exhausted, the voice
    preview (audition) endpoint answered 401 — the frontend treats any 401
    as "session expired" and redirects to /auth/login, so every voice
    preview logged the user out over a billing problem.
    """
    from app.services.errors import (
        ElevenLabsAuthError,
        MiMoAuthError,
        PyannoteAuthError,
    )

    for error in (
        ElevenLabsAuthError("upstream rejected our key"),
        MiMoAuthError("upstream rejected our key"),
        PyannoteAuthError("upstream rejected our key"),
    ):
        assert error.status_code == 502
        assert error.status_code != 401


def test_unauthenticated_api_requests_get_401(client: TestClient, beta_env) -> None:
    assert client.get("/").status_code == 401
    assert client.get("/health").status_code == 401
    assert client.post("/projects", json={"name": "x"}).status_code == 401
    assert client.get("/projects/00000000-0000-0000-0000-000000000000").status_code == 401
    assert (
        client.get("/media/projects/00000000-0000-0000-0000-000000000000/audio/audiobook.mp3").status_code
        == 401
    )
    # The app bundle is protected too — only the login stylesheet is public.
    assert client.get("/static/app.js").status_code == 401


def test_unauthenticated_browser_requests_redirect_to_login(
    client: TestClient, beta_env
) -> None:
    response = client.get(
        "/", headers={"Accept": "text/html"}, follow_redirects=False
    )
    assert response.status_code == 302
    assert response.headers["location"].startswith("/auth/login")


def test_login_page_stylesheet_is_public(client: TestClient, beta_env) -> None:
    assert client.get("/static/style.css").status_code == 200
    page = client.get("/auth/login")
    assert page.status_code == 200
    assert "access token" in page.text
    assert "Private beta" in page.text  # privacy notice on the login page too


def test_wrong_token_is_rejected(client: TestClient, beta_env) -> None:
    response = client.post(
        "/auth/login", data={"token": "wrong"}, follow_redirects=False
    )
    assert response.status_code == 401
    assert "beta_session" not in response.headers.get("set-cookie", "")

    assert client.get("/", headers={"Authorization": "Bearer nope"}).status_code == 401
    # A tampered session cookie is rejected as well.
    assert (
        client.get(
            "/", headers={"Cookie": "beta_session=alice.99999999999.deadbeef"}
        ).status_code
        == 401
    )


# --------------------------------------------------------------------------- #
# Authenticated access (cookie + bearer)
# --------------------------------------------------------------------------- #
def test_login_sets_a_session_cookie_and_grants_access(
    client: TestClient, beta_env
) -> None:
    response = client.post(
        "/auth/login", data={"token": "alice-secret-1"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert "beta_session=" in response.headers.get("set-cookie", "")
    assert "HttpOnly" in response.headers.get("set-cookie", "")

    # The cookie now authenticates the whole app (API + frontend).
    home = client.get("/", headers={"Accept": "text/html"})
    assert home.status_code == 200
    assert "Create your audiobook" in home.text
    assert client.get("/health").status_code == 200

    # Logout clears the session again.
    client.post("/auth/logout", follow_redirects=False)
    assert client.get("/health").status_code == 401


def test_bearer_and_header_tokens_work_for_api_clients(
    client: TestClient, beta_env
) -> None:
    assert client.get("/", headers=ALICE).status_code == 200
    assert client.get("/", headers={"X-Beta-Token": "bob-secret-2"}).status_code == 200


def test_api_docs_are_disabled_for_beta_users(client: TestClient, beta_env) -> None:
    for path in ("/docs", "/openapi.json", "/redoc"):
        assert client.get(path, headers=ALICE).status_code == 404, path
        assert client.get(path).status_code == 404, path


def test_api_docs_stay_available_without_beta_mode(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 200


# --------------------------------------------------------------------------- #
# Project isolation + media protection
# --------------------------------------------------------------------------- #
def test_testers_cannot_see_each_others_projects(
    client: TestClient, beta_env
) -> None:
    alice_project = _create_project(client, ALICE)

    # Owner: full access.
    assert client.get(f"/projects/{alice_project}", headers=ALICE).status_code == 200
    assert (
        client.get(f"/media/projects/{alice_project}/script.txt", headers=ALICE).status_code
        == 200
    )

    # The other tester gets the same 404 as a missing project (no leak).
    as_bob = client.get(f"/projects/{alice_project}", headers=BOB)
    assert as_bob.status_code == 404
    assert client.get(f"/projects/{alice_project}", headers=ALICE).status_code == 200
    assert (
        client.get(f"/media/projects/{alice_project}/script.txt", headers=BOB).status_code
        == 404
    )
    # …and cannot upload, generate into or delete it either.
    assert (
        client.post(
            f"/projects/{alice_project}/upload-script",
            files={"file": ("x.txt", b"hi", "text/plain")},
            headers=BOB,
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/projects/{alice_project}/analyze-narration", headers=BOB
        ).status_code
        == 404
    )
    assert (
        client.delete(f"/projects/{alice_project}", headers=BOB).status_code == 404
    )
    assert client.get(f"/projects/{alice_project}", headers=ALICE).status_code == 200


def test_media_path_traversal_is_blocked(client: TestClient, beta_env) -> None:
    alice_project = _create_project(client, ALICE)
    bob_project = _create_project(client, BOB)

    # %2e%2e = ".." — try to reach bob's script through alice's media URL.
    response = client.get(
        f"/media/projects/{alice_project}/%2e%2e%2f{bob_project}%2fscript.txt",
        headers=ALICE,
    )
    assert response.status_code == 404
    assert (
        client.get(
            f"/media/projects/{bob_project}/script.txt", headers=ALICE
        ).status_code
        == 404
    )


def test_projects_created_without_an_owner_are_not_exposed_in_beta(
    client: TestClient, beta_env
) -> None:
    from app.services import ProjectService
    from app.schemas.project import ProjectCreateRequest

    legacy = ProjectService(get_settings()).create_project(
        ProjectCreateRequest(name="legacy")  # no owner (created outside the API)
    )
    assert client.get(f"/projects/{legacy.project_id}", headers=ALICE).status_code == 404
    assert client.get(f"/projects/{legacy.project_id}", headers=BOB).status_code == 404


# --------------------------------------------------------------------------- #
# Generation quotas
# --------------------------------------------------------------------------- #
def test_generation_quota_per_tester(client: TestClient, beta_env, monkeypatch) -> None:
    _use_fake_audio(monkeypatch)
    limit = 2  # BETA_GENERATION_LIMIT in beta_env

    for _ in range(limit):
        project = _create_project(client, ALICE)
        response = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
        assert response.status_code == 200, response.text

    # The next generation is refused with a clear 429…
    project = _create_project(client, ALICE)
    refused = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
    assert refused.status_code == 429
    assert "limit" in refused.json()["detail"].lower()

    # …while the other tester still has their own budget.
    bob_project = _create_project(client, BOB)
    assert (
        client.post(f"/projects/{bob_project}/generate-audio", headers=BOB).status_code
        == 200
    )


def test_zero_generation_limit_blocks_all_generations(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`BETA_GENERATION_LIMIT=0` means ZERO allowance (not "unlimited").

    (Semantic change agreed for the per-tester limits stage: limits are
    non-negative integers and 0 blocks generation entirely.)
    """
    monkeypatch.setenv("BETA_ACCESS_TOKENS", "alice:alice-secret-1")
    monkeypatch.setenv("BETA_GENERATION_LIMIT", "0")
    _use_fake_audio(monkeypatch)
    _reset_beta_state()

    project = _create_project(client, ALICE)
    refused = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
    assert refused.status_code == 429
    assert "limit" in refused.json()["detail"].lower()

    # The tester can still see the project and any existing audiobook.
    assert client.get(f"/projects/{project}", headers=ALICE).status_code == 200


# --------------------------------------------------------------------------- #
# Upload limits
# --------------------------------------------------------------------------- #
def test_script_length_limit(client: TestClient, beta_env, monkeypatch) -> None:
    monkeypatch.setenv("MAX_SCRIPT_CHARACTERS", "50")
    long_script = "a" * 60
    response = _import(client, ALICE, content=long_script)
    assert response.status_code == 400
    assert "too long" in response.json()["detail"]


def test_estimated_duration_limit(client: TestClient, beta_env, monkeypatch) -> None:
    monkeypatch.setenv("MAX_AUDIO_DURATION_MINUTES", "1")
    monkeypatch.setenv("SPEECH_CHARS_PER_MINUTE", "100")
    response = _import(client, ALICE, content="a" * 300)  # ~3 minutes estimated
    assert response.status_code == 400
    assert "minutes" in response.json()["detail"]


def test_upload_file_size_limit_still_applies_in_beta(
    client: TestClient, beta_env
) -> None:
    oversized = b"a" * (5 * 1024 * 1024 + 1)
    response = client.post(
        "/projects/from-script",
        files={"file": ("big.txt", oversized, "text/plain")},
        headers=ALICE,
    )
    assert response.status_code == 413


def test_generation_rechecks_the_duration_limit(
    client: TestClient, beta_env, monkeypatch
) -> None:
    project = _create_project(client, ALICE)
    # Tighten the limit AFTER the upload — generation must refuse too.
    monkeypatch.setenv("MAX_AUDIO_DURATION_MINUTES", "1")
    monkeypatch.setenv("SPEECH_CHARS_PER_MINUTE", "1")
    _use_fake_audio(monkeypatch)
    response = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
    assert response.status_code == 400
    assert "minutes" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Duplicate submissions + rate limiting
# --------------------------------------------------------------------------- #
def test_duplicate_generation_submission_is_rejected(
    client: TestClient, beta_env, monkeypatch
) -> None:
    _use_fake_audio(monkeypatch)
    project = _create_project(client, ALICE)

    first = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
    assert first.status_code == 200

    duplicate = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
    assert duplicate.status_code == 409
    assert "duplicate" in duplicate.json()["detail"].lower()


def test_in_flight_generation_is_rejected_and_failures_can_be_retried() -> None:
    _reset_beta_state()
    # First request is now running…
    assert security_mod._duplicate_begin("alice:/projects/p/generate-audio", 10) is None
    # …a concurrent duplicate is refused…
    busy = security_mod._duplicate_begin("alice:/projects/p/generate-audio", 10)
    assert busy is not None and "already being processed" in busy
    # …a FAILED attempt does not block the retry…
    security_mod._duplicate_end("alice:/projects/p/generate-audio", succeeded=False)
    assert security_mod._duplicate_begin("alice:/projects/p/generate-audio", 10) is None
    # …but a successful one does (recency window).
    security_mod._duplicate_end("alice:/projects/p/generate-audio", succeeded=True)
    recent = security_mod._duplicate_begin("alice:/projects/p/generate-audio", 10)
    assert recent is not None and "Duplicate submission" in recent
    # A different project is unaffected.
    assert security_mod._duplicate_begin("alice:/projects/q/generate-audio", 10) is None


def test_the_duplicate_window_compares_the_whole_submission(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching a voice preview must work immediately (live bug: previewing
    a second voice within the window got a 409 "Duplicate submission"), while
    an identical double submission is still refused."""
    _reset_beta_state()
    project = _create_project(client, ALICE)

    class _StubCasting:
        async def audition(self, project_id: str, payload) -> bytes:  # noqa: ANN001
            return b"fake-audio-bytes"

    monkeypatch.setattr(casting_routes, "_casting_service", lambda: _StubCasting())
    url = f"/projects/{project}/voice-casting/audition"

    first = client.post(
        url,
        json={"speaker_id": "narrator", "voice_id": "voice-a", "text": "Hello there."},
        headers=ALICE,
    )
    assert first.status_code == 200, first.text

    # Another voice right away = a DIFFERENT submission — never blocked.
    second = client.post(
        url,
        json={"speaker_id": "narrator", "voice_id": "voice-b", "text": "Hello there."},
        headers=ALICE,
    )
    assert second.status_code == 200, second.text

    # The exact same submission again = a real duplicate — still refused.
    third = client.post(
        url,
        json={"speaker_id": "narrator", "voice_id": "voice-a", "text": "Hello there."},
        headers=ALICE,
    )
    assert third.status_code == 409
    assert "duplicate" in third.json()["detail"].lower()


def test_api_rate_limiting(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BETA_ACCESS_TOKENS", "alice:alice-secret-1")
    monkeypatch.setenv("BETA_RATE_LIMIT_PER_MINUTE", "3")
    _reset_beta_state()

    for _ in range(3):
        assert client.get("/health", headers=ALICE).status_code == 200
    limited = client.get("/health", headers=ALICE)
    assert limited.status_code == 429
    assert limited.headers.get("retry-after")


def test_login_attempts_are_rate_limited(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BETA_ACCESS_TOKENS", "alice:alice-secret-1")
    monkeypatch.setenv("BETA_LOGIN_RATE_LIMIT_PER_MINUTE", "2")
    _reset_beta_state()

    for _ in range(2):
        assert (
            client.post(
                "/auth/login", data={"token": "wrong"}, follow_redirects=False
            ).status_code
            == 401
        )
    assert (
        client.post("/auth/login", data={"token": "wrong"}, follow_redirects=False).status_code
        == 429
    )


# --------------------------------------------------------------------------- #
# Privacy: notice + deletion
# --------------------------------------------------------------------------- #
def test_privacy_notice_and_delete_action_are_in_the_page(
    client: TestClient, beta_env
) -> None:
    html = client.get(
        "/", headers={**ALICE, "Accept": "text/html"}
    ).text
    assert "Private beta" in html
    assert "stored" in html and "testing purposes" in html
    assert 'id="btn-delete-data"' in html
    assert "Delete my projects" in html


def test_delete_project_removes_the_files(
    client: TestClient, beta_env
) -> None:
    project = _create_project(client, ALICE)
    assert (TEST_STORAGE / project / "script.txt").exists()

    response = client.delete(f"/projects/{project}", headers=ALICE)
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert not (TEST_STORAGE / project).exists()
    assert client.get(f"/projects/{project}", headers=ALICE).status_code == 404
    assert (
        client.get(f"/media/projects/{project}/script.txt", headers=ALICE).status_code
        == 404
    )


def test_delete_my_data_only_deletes_own_projects(
    client: TestClient, beta_env
) -> None:
    # Start from a clean slate (the test storage is shared across the suite).
    client.delete("/projects", headers=ALICE)
    client.delete("/projects", headers=BOB)

    alice_one = _create_project(client, ALICE)
    alice_two = _create_project(client, ALICE, content=SCRIPT + "\n\nExtra.")
    bob_project = _create_project(client, BOB)

    response = client.delete("/projects", headers=ALICE)
    assert response.status_code == 200
    assert response.json()["deleted_count"] == 2

    for project in (alice_one, alice_two):
        assert not (TEST_STORAGE / project).exists()
    assert (TEST_STORAGE / bob_project / "script.txt").exists()
    assert client.get(f"/projects/{bob_project}", headers=BOB).status_code == 200


# --------------------------------------------------------------------------- #
# End to end with a token (providers mocked)
# --------------------------------------------------------------------------- #
def test_full_authenticated_pipeline_with_mocks(
    client: TestClient, beta_env, monkeypatch
) -> None:
    imported = _import(client, ALICE, content=f"# Room 207\n\n{SCRIPT}")
    assert imported.status_code == 201
    project = imported.json()["project_id"]

    prefs = {
        "preset": "natural",
        "acting_intensity": 2,
        "whisper_policy": "never",
        "whisper_paragraphs": [],
        "tag_density": "minimal",
        "custom_direction": "",
    }
    assert (
        client.put(
            f"/projects/{project}/narration-preferences", json=prefs, headers=ALICE
        ).status_code
        == 200
    )

    _use_fake_audio(monkeypatch)
    audio = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
    assert audio.status_code == 200
    assert audio.json()["audio_url"].endswith("audiobook.mp3")

    # The player/download URL works for the owner…
    media = client.get(audio.json()["audio_url"], headers=ALICE)
    assert media.status_code == 200
    assert media.content == FAKE_MP3
    # …and for nobody else.
    assert client.get(audio.json()["audio_url"], headers=BOB).status_code == 404
    assert client.get(audio.json()["audio_url"]).status_code == 401

    assert client.delete(f"/projects/{project}", headers=ALICE).status_code == 200


# --------------------------------------------------------------------------- #
# Development mode (no tokens configured) is unchanged
# --------------------------------------------------------------------------- #
def test_without_tokens_the_app_stays_open(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    imported = _import(client, headers={})
    assert imported.status_code == 201
    project = imported.json()["project_id"]
    assert client.get(f"/projects/{project}").status_code == 200
    assert (
        client.get(f"/media/projects/{project}/script.txt").status_code == 200
    )
    assert client.delete(f"/projects/{project}").status_code == 200
