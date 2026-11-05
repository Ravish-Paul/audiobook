"""AI character casting + multi-voice pipeline tests (spec §1–§12) — zero credits.

All ElevenLabs traffic goes through `httpx.MockTransport` (discovery, voice
details, shared-voice import, TTS), MiMo through `FakeMiMo`, FFmpeg through
`FakeComposer`. No real TTS credits are ever spent and the full story is
never generated live.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import DEFAULT_ELEVENLABS_BASE_URL, get_settings
from app.main import app
from app.routes import audio as audio_routes
from app.routes import casting as casting_routes
from app.schemas.casting import CharacterCast, StoryAnalysis
from app.schemas.multi_voice import MultiVoicePlan
from app.services import (
    AudioService,
    CharacterAnalyzer,
    ElevenLabsClient,
    ElevenLabsVoiceDiscovery,
    MultiVoicePlanner,
    ProjectService,
    VoiceCastingService,
)
from app.services.multi_voice_planner import validate_plan_against_script
from fakes import (
    FAKE_MP3,
    FakeComposer,
    FakeMiMo,
    MV_PARAGRAPHS,
    MV_SCRIPT,
    NARRATOR_VOICE_ID,
    SUMAN_VOICE_ID,
    make_multi_voice_plan_dict,
    make_story_analysis_dict,
    multi_voice_plan_json,
    story_json,
)
from test_voice_discovery import DiscoveryMock, account_voice, shared_voice

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"

# Account pool: three natural Hindi audiobook voices + one English-only voice
# + one restricted voice (both must NEVER be proposed).
HI_VOICE_IDS = ["voice-narrator-01", "voice-suman-02", "voice-hi-3"]
ACCOUNT_PAGE = {
    "voices": [
        account_voice(
            "voice-narrator-01", name="Narrator Deep",
            description="Deep, calm, measured narrator voice for audiobooks",
        ),
        account_voice(
            "voice-suman-02", name="Warm Soft",
            description="Soft warm grounded voice, natural and restrained for stories",
        ),
        account_voice(
            "voice-hi-3", name="Warm Storyteller",
            description="Calm narrator voice with warm storytelling delivery",
        ),
        account_voice(
            "voice-en-1", name="American Podcast", languages=("en",),
            accent="american",
            description="American english podcast voice",
        ),
        account_voice(
            "voice-banned", name="Banned Voice", safety_control="BAN",
        ),
    ],
    "has_more": False,
}
EMPTY_PAGE = {"voices": [], "has_more": False}
VOICE_DETAILS = {
    vid: account_voice(vid, name=f"Detail {vid}") for vid in HI_VOICE_IDS
}
VOICE_DETAILS["voice-banned"] = account_voice(
    "voice-banned", safety_control="BAN"
)
VOICE_DETAILS["voice-shared-1"] = account_voice("voice-shared-1")


# --------------------------------------------------------------------------- #
# Mock plumbing
# --------------------------------------------------------------------------- #
class CastingMock(DiscoveryMock):
    """Discovery + voice-detail + shared-import + TTS in one transport."""

    def __init__(self, *, tts_fail_text: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tts_requests: list[dict[str, Any]] = []
        self.tts_fail_text = tts_fail_text

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/text-to-speech/" in path:
            body = json.loads(request.content)
            self.tts_requests.append(
                {
                    "path": path,
                    "body": body,
                    "params": dict(request.url.params),
                }
            )
            if self.tts_fail_text is not None and body.get("text") == self.tts_fail_text:
                return httpx.Response(
                    500, json={"detail": "simulated upstream failure"},
                    request=request,
                )
            return httpx.Response(
                200,
                content=FAKE_MP3,
                headers={
                    "content-type": "audio/mpeg",
                    "request-id": f"req-{len(self.tts_requests):04d}",
                },
                request=request,
            )
        return super().__call__(request)


def _settings(**overrides: Any):
    base = dict(
        elevenlabs_api_key="sk-mv-test",
        elevenlabs_voice_id="voice-project-default",
        elevenlabs_base_url=DEFAULT_ELEVENLABS_BASE_URL,
        elevenlabs_model_id="eleven_v4",
        elevenlabs_output_format="mp3_44100_128",
        elevenlabs_intermediate_output_format="",
        elevenlabs_language_code="",
        elevenlabs_pronunciation_dictionaries=(),
        elevenlabs_audio_effects=None,
        elevenlabs_voice_is_pvc=False,
        elevenlabs_clone_strength=None,
    )
    base.update(overrides)
    return dataclasses.replace(get_settings(), **base)


def _default_mock(**overrides: Any) -> CastingMock:
    kwargs: dict[str, Any] = {
        "account_default": ACCOUNT_PAGE,
        "shared_default": EMPTY_PAGE,
        "details": dict(VOICE_DETAILS),
    }
    kwargs.update(overrides)
    return CastingMock(**kwargs)


def _wire_casting(
    monkeypatch: pytest.MonkeyPatch, mock: Optional[CastingMock] = None
) -> CastingMock:
    """Route casting endpoints through mocked discovery + TTS."""
    mock = mock or _default_mock()
    settings = _settings()
    transport = httpx.MockTransport(mock)
    service = VoiceCastingService(
        settings,
        discovery=ElevenLabsVoiceDiscovery(settings, transport=transport),
        tts_client=ElevenLabsClient(settings, transport=transport),
    )
    monkeypatch.setattr(casting_routes, "_casting_service", lambda: service)
    return mock


def _wire_audio(
    monkeypatch: pytest.MonkeyPatch, mock: CastingMock
) -> FakeComposer:
    """Route generate-audio through the mocked TTS + a fake composer."""
    settings = _settings()
    composer = FakeComposer()
    service = AudioService(
        settings,
        tts_client=ElevenLabsClient(
            settings, transport=httpx.MockTransport(mock)
        ),
        composer=composer,
    )
    monkeypatch.setattr(audio_routes, "_audio_service", lambda: service)
    return composer


def _use_analyzer(monkeypatch: pytest.MonkeyPatch, fake: FakeMiMo) -> None:
    monkeypatch.setattr(
        casting_routes,
        "_analyzer",
        lambda: CharacterAnalyzer(get_settings(), mimo_client=fake),
    )


def _use_planner(monkeypatch: pytest.MonkeyPatch, fake: FakeMiMo) -> None:
    monkeypatch.setattr(
        casting_routes,
        "_planner",
        lambda: MultiVoicePlanner(get_settings(), mimo_client=fake),
    )


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def project_with_script(client: TestClient) -> str:
    project_id = client.post("/projects", json={"name": "Multi Voice Test"}).json()[
        "project_id"
    ]
    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", MV_SCRIPT.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201
    return project_id


def _analyze(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, project_id: str,
    *, responses: Optional[list[str]] = None,
) -> httpx.Response:
    _use_analyzer(monkeypatch, FakeMiMo(responses or [story_json()]))
    response = client.post(f"/projects/{project_id}/analyze-characters")
    assert response.status_code == 200, response.text
    return response


def _select_both(client: TestClient, project_id: str, *, narrator: str = "voice-narrator-01",
                 suman: str = "voice-suman-02") -> None:
    for speaker, voice in (("narrator", narrator), ("suman", suman)):
        response = client.post(
            f"/projects/{project_id}/voice-casting/select",
            json={"speaker_id": speaker, "voice_id": voice},
        )
        assert response.status_code == 200, response.text


def _approve(client: TestClient, project_id: str) -> None:
    response = client.post(f"/projects/{project_id}/voice-casting/approve")
    assert response.status_code == 200, response.text


def _create_plan(client: TestClient, monkeypatch: pytest.MonkeyPatch,
                 project_id: str, *, responses: Optional[list[str]] = None) -> httpx.Response:
    _use_planner(monkeypatch, FakeMiMo(responses or [multi_voice_plan_json()]))
    response = client.post(f"/projects/{project_id}/create-multi-voice-plan")
    assert response.status_code == 200, response.text
    return response


def _cast_file(project_id: str) -> dict[str, Any]:
    return json.loads(
        (TEST_STORAGE / project_id / "character_cast.json").read_text(
            encoding="utf-8"
        )
    )


def _metadata(project_id: str) -> dict[str, Any]:
    return json.loads(
        (TEST_STORAGE / project_id / "audio" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )


# --------------------------------------------------------------------------- #
# 1. Story analysis + character discovery (§1/§2)
# --------------------------------------------------------------------------- #
def test_get_casting_before_analysis_is_404(
    client: TestClient, project_with_script: str
) -> None:
    response = client.post(f"/projects/{project_with_script}/voice-casting")
    assert response.status_code == 404
    assert "analyze-characters" in response.json()["detail"]


def test_analyze_characters_discovers_cast_and_narrator(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _analyze(client, monkeypatch, project_with_script).json()

    # narrator + speaking characters only — no voice for the silent Rahul.
    assert body["speaker_ids"] == ["narrator", "suman"]
    assert body["saved"] is True
    assert body["story"]["language"] == "hi"
    assert body["story"]["locale"] == "hi-IN"

    # Profiles keep the spec §2 shape; attributes stay null (never invented).
    characters = {c["character_id"]: c for c in body["story"]["characters"]}
    suman = characters["suman"]
    assert suman["gender"] is None and suman["age"] is None and suman["accent"] is None
    assert suman["uncertain_attributes"] == []
    assert suman["has_spoken_dialogue"] is True
    assert suman["voice_description"]

    rahul = characters["rahul"]
    assert rahul["has_spoken_dialogue"] is False
    assert rahul["voice_description"] == ""   # no voice requirement

    # narrator is a separate profile with the reserved id.
    assert body["story"]["narrator"]["character_id"] == "narrator"
    assert body["story"]["narrator"]["role"] == "narrator"

    # Permanent file with locked=false / voice_id=null until selection.
    cast = _cast_file(project_with_script)
    assert cast["project_id"] == project_with_script
    assert cast["approved"] is False
    for profile in [cast["story"]["narrator"], *cast["story"]["characters"]]:
        assert profile["voice_id"] is None
        assert profile["voice_locked"] is False
    assert set(cast["casting"]) == {"narrator", "suman"}


def test_analyze_characters_normalizes_language_names(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _analyze(
        client, monkeypatch, project_with_script,
        responses=[story_json(language="Hindi", narrator={
            **make_story_analysis_dict()["narrator"], "language": "",
        })],
    ).json()
    assert body["story"]["language"] == "hi"                 # "Hindi" -> "hi"
    assert body["story"]["narrator"]["language"] == "hi"     # inherits story lang


def test_analyze_rejects_invented_dialogue_character(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = make_story_analysis_dict()
    bad["characters"].append({
        "character_id": "ghost",
        "character_name": "भूत",              # never appears in the script
        "role": "minor_character",
        "language": "hi",
        "gender": None, "age": None, "accent": None,
        "uncertain_attributes": [],
        "voice_description": "Eerie whispering voice",
        "speaking_style": "Haunting",
        "has_spoken_dialogue": True,
    })
    fake = FakeMiMo([json.dumps(bad, ensure_ascii=False), story_json()])
    _use_analyzer(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/analyze-characters")
    assert response.status_code == 200, response.text

    assert fake.call_count == 2
    assert "never appears in the script" in fake.calls[1]["user"]
    character_ids = [
        c["character_id"] for c in response.json()["story"]["characters"]
    ]
    assert character_ids == ["suman", "rahul"]   # the invented ghost is gone


# --------------------------------------------------------------------------- #
# 2. Voice discovery + ranking (§3/§4)
# --------------------------------------------------------------------------- #
def test_voice_casting_builds_top_candidates_per_speaker(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)
    mock = _wire_casting(monkeypatch)

    response = client.post(f"/projects/{project_with_script}/voice-casting")
    assert response.status_code == 200, response.text
    body = response.json()

    # Account voices first — the library was not needed for this pool …
    assert "/v2/voices" in mock.paths
    assert "/v1/shared-voices" not in mock.paths

    for entry in (body["narrator"], *body["characters"].values()):
        assert 1 <= len(entry["candidates"]) <= 3
        assert entry["insufficient_candidates"] is False
        for candidate in entry["candidates"]:
            assert candidate["voice_id"] in HI_VOICE_IDS  # en + banned excluded
            assert candidate["name"]
            assert candidate["preview_url"]               # preview player data
            assert candidate["rank"] >= 1
            assert candidate["total_score"] >= 0.55
            assert "language" in candidate["scores"]
        scores = [c["total_score"] for c in entry["candidates"]]
        assert scores == sorted(scores, reverse=True)

    assert body["approved"] is False
    assert set(body["missing_speakers"]) == {"narrator", "suman"}

    # GET returns the same casting.
    fetched = client.get(f"/projects/{project_with_script}/voice-casting")
    assert fetched.status_code == 200
    assert fetched.json()["narrator"]["candidates"] == body["narrator"]["candidates"]


def test_voice_casting_falls_back_to_shared_library(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)
    thin_account = {
        "voices": [
            account_voice("voice-narrator-01", description="Deep calm measured narrator "
                          "voice for audiobooks"),
            account_voice("voice-en-1", languages=("en",)),
        ],
        "has_more": False,
    }
    library = {
        "voices": [
            shared_voice("voice-shared-1"),
            shared_voice("voice-shared-2", owner="owner-2"),
        ],
        "has_more": False,
    }
    mock = _wire_casting(
        monkeypatch,
        _default_mock(account_default=thin_account, shared_default=library),
    )
    details = dict(VOICE_DETAILS)
    details["voice-shared-1"] = account_voice("voice-shared-1")
    details["voice-shared-2"] = account_voice("voice-shared-2")
    mock.details = details

    body = client.post(
        f"/projects/{project_with_script}/voice-casting"
    ).json()

    # Account pool was thin (1 usable) -> the library was searched.
    assert "/v1/shared-voices" in mock.paths
    narrator_ids = [c["voice_id"] for c in body["narrator"]["candidates"]]
    assert "voice-narrator-01" in narrator_ids                      # account candidate
    origins = {
        c["voice_id"]: c["origin"] for c in body["narrator"]["candidates"]
    }
    assert origins.get("voice-shared-1") == "shared"         # library candidates
    assert len(body["narrator"]["candidates"]) == 3


def test_voice_casting_never_picks_unsupported_language(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Hindi story must get no English-only voices, even if that means none."""
    _analyze(client, monkeypatch, project_with_script)
    english_only = {
        "voices": [account_voice(f"en-{i}", languages=("en",)) for i in range(3)],
        "has_more": False,
    }
    _wire_casting(
        monkeypatch,
        _default_mock(account_default=english_only, shared_default=english_only),
    )

    body = client.post(f"/projects/{project_with_script}/voice-casting").json()

    for entry in (body["narrator"], *body["characters"].values()):
        assert entry["candidates"] == []            # no weak candidates proposed
        assert entry["insufficient_candidates"] is True


# --------------------------------------------------------------------------- #
# 3. Selection + permanent lock (§6)
# --------------------------------------------------------------------------- #
def test_select_locks_voice_with_full_metadata(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)
    _wire_casting(monkeypatch)

    response = client.post(
        f"/projects/{project_with_script}/voice-casting/select",
        json={"speaker_id": "narrator", "voice_id": "voice-narrator-01"},
    )
    assert response.status_code == 200, response.text
    lock = response.json()["narrator"]["selected"]

    assert lock["character_id"] == "narrator"
    assert lock["voice_id"] == "voice-narrator-01"
    assert lock["original_voice_id"] is None
    assert lock["provider"] == "elevenlabs"
    assert lock["language"] == "hi"
    assert lock["model_id"] == "eleven_v4"
    assert lock["locked"] is True
    assert lock["selected_at"]
    # Supported settings reflect the model capabilities (v4: no speed/style).
    assert "speed" not in lock["supported_voice_settings"]
    assert "style" not in lock["supported_voice_settings"]
    assert "stability" in lock["supported_voice_settings"]
    assert lock["voice_metadata"]["origin"] == "account"

    # Profile mirrors the lock; approval was invalidated.
    cast = _cast_file(project_with_script)
    narrator = cast["story"]["narrator"]
    assert narrator["voice_id"] == "voice-narrator-01"
    assert narrator["voice_locked"] is True
    assert cast["approved"] is False
    assert response.json()["missing_speakers"] == ["suman"]


def test_select_unknown_or_non_speaker_is_400(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)
    _wire_casting(monkeypatch)

    for speaker in ("ghost", "rahul"):  # unknown / character without dialogue
        response = client.post(
            f"/projects/{project_with_script}/voice-casting/select",
            json={"speaker_id": speaker, "voice_id": "voice-narrator-01"},
        )
        assert response.status_code == 400
        assert "rahul" in response.json()["detail"] or speaker == "ghost"


def test_select_unknown_voice_is_400(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)
    _wire_casting(monkeypatch)   # details contain no "nope-voice"

    response = client.post(
        f"/projects/{project_with_script}/voice-casting/select",
        json={"speaker_id": "narrator", "voice_id": "nope-voice"},
    )
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]


def test_select_restricted_voice_is_400(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)
    _wire_casting(monkeypatch)

    response = client.post(
        f"/projects/{project_with_script}/voice-casting/select",
        json={"speaker_id": "narrator", "voice_id": "voice-banned"},
    )
    assert response.status_code == 400
    assert "restricted" in response.json()["detail"]


def test_select_shared_voice_imports_it_to_the_account(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)
    mock = _wire_casting(monkeypatch)

    response = client.post(
        f"/projects/{project_with_script}/voice-casting/select",
        json={
            "speaker_id": "suman",
            "voice_id": "voice-shared-1",
            "public_owner_id": "owner-abc",
            "new_name": "Suman Cast Voice",
        },
    )
    assert response.status_code == 200, response.text
    lock = response.json()["characters"]["suman"]["selected"]

    assert "/v1/voices/add/owner-abc/voice-shared-1" in mock.paths
    assert lock["voice_id"] == "voice-shared-1"
    assert lock["original_voice_id"] == "voice-shared-1"
    assert lock["origin"] == "shared"


def test_changing_one_voice_does_not_affect_others(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)
    _wire_casting(monkeypatch)
    _select_both(client, project_with_script)
    _approve(client, project_with_script)
    assert client.get(
        f"/projects/{project_with_script}/voice-casting"
    ).json()["approved"] is True

    # Replace ONLY suman's voice.
    response = client.post(
        f"/projects/{project_with_script}/voice-casting/select",
        json={"speaker_id": "suman", "voice_id": "voice-hi-3"},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["characters"]["suman"]["selected"]["voice_id"] == "voice-hi-3"
    assert body["narrator"]["selected"]["voice_id"] == "voice-narrator-01"  # untouched
    assert body["approved"] is False  # any change requires re-approval


def test_approve_requires_every_speaking_role(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)
    _wire_casting(monkeypatch)
    response = client.post(
        f"/projects/{project_with_script}/voice-casting/select",
        json={"speaker_id": "narrator", "voice_id": "voice-narrator-01"},
    )
    assert response.status_code == 200

    response = client.post(f"/projects/{project_with_script}/voice-casting/approve")
    assert response.status_code == 409
    assert "suman" in response.json()["detail"]


def test_approve_marks_cast_approved(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)
    _wire_casting(monkeypatch)
    _select_both(client, project_with_script)

    body = _approve_and_get(client, project_with_script)
    assert body["approved"] is True
    assert body["approved_at"] is not None
    assert body["missing_speakers"] == []
    assert body["narrator"]["status"] == "approved"
    assert body["characters"]["suman"]["status"] == "approved"
    assert _cast_file(project_with_script)["approved"] is True


def _approve_and_get(client: TestClient, project_id: str) -> dict[str, Any]:
    response = client.post(f"/projects/{project_id}/voice-casting/approve")
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------- #
# 4. Audition (§5) — short line only, never the full story
# --------------------------------------------------------------------------- #
def test_audition_renders_short_line_without_touching_the_cast(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)
    mock = _wire_casting(monkeypatch)
    _select_both(client, project_with_script)
    before = client.get(f"/projects/{project_with_script}/voice-casting").json()

    response = client.post(
        f"/projects/{project_with_script}/voice-casting/audition",
        json={"speaker_id": "suman", "text": "नमस्ते दोस्तों, कैसे हो?"},
    )
    assert response.status_code == 200
    assert response.content == FAKE_MP3
    assert response.headers["content-type"].startswith("audio/mpeg")

    assert len(mock.tts_requests) == 1
    assert mock.tts_requests[0]["path"] == f"/v1/text-to-speech/{SUMAN_VOICE_ID}"

    # The cast itself was not modified by the audition.
    after = client.get(f"/projects/{project_with_script}/voice-casting").json()
    assert after["narrator"] == before["narrator"]
    assert after["characters"] == before["characters"]


def test_audition_text_length_is_capped(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)
    _wire_casting(monkeypatch)
    _select_both(client, project_with_script)

    response = client.post(
        f"/projects/{project_with_script}/voice-casting/audition",
        json={"speaker_id": "suman", "text": "x" * 401},
    )
    assert response.status_code == 422  # never a full-story generation


# --------------------------------------------------------------------------- #
# 5. Multi-voice (utterance) plan (§7)
# --------------------------------------------------------------------------- #
def test_create_multi_voice_plan_success(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)
    _wire_casting(monkeypatch)
    _select_both(client, project_with_script)
    _approve(client, project_with_script)

    body = _create_plan(client, monkeypatch, project_with_script).json()

    assert body["utterance_count"] == 6
    assert body["paragraph_count"] == 3
    assert body["speakers"] == ["narrator", "suman"]
    assert body["saved"] is True

    keys = [(u["paragraph_index"], u["sequence_index"]) for u in body["utterances"]]
    assert keys == sorted(keys)  # exact story order
    assert [u["utterance_id"] for u in body["utterances"]] == [
        f"p{p:04d}-u{s:04d}" for p, s in keys  # canonical ids stamped by us
    ]

    # Voices stamped from the LOCKS, never chosen by MiMo.
    for utterance in body["utterances"]:
        expected = (
            NARRATOR_VOICE_ID
            if utterance["speaker_id"] == "narrator"
            else SUMAN_VOICE_ID
        )
        assert utterance["voice_id"] == expected
        assert utterance["model_id"] == "eleven_v4"

    # Persisted file round-trips.
    saved = json.loads(
        (TEST_STORAGE / project_with_script / "multi_voice_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["project_id"] == project_with_script
    assert len(saved["utterances"]) == 6


def test_create_plan_without_cast_is_404(
    client: TestClient, project_with_script: str
) -> None:
    response = client.post(f"/projects/{project_with_script}/create-multi-voice-plan")
    assert response.status_code == 404
    assert "analyze-characters" in response.json()["detail"]


def test_plan_word_change_is_rejected_and_retried(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)

    # Both original_text AND tts_text lose a word — internally consistent so
    # the schema passes, but the paragraph reconstruction check must catch it.
    bad = make_multi_voice_plan_dict()
    bad["utterances"][1]["original_text"] = "आईने में मत देखना।"
    bad["utterances"][1]["tts_text"] = "[quiet, tense delivery] आईने में मत देखना।"
    fake = FakeMiMo([
        json.dumps(bad, ensure_ascii=False),
        multi_voice_plan_json(),
    ])
    _use_planner(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/create-multi-voice-plan")
    assert response.status_code == 200, response.text
    assert fake.call_count == 2
    assert "do not reproduce the original paragraph" in fake.calls[1]["user"]


def test_plan_rejects_utterance_for_non_speaker(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)

    bad = make_multi_voice_plan_dict()
    # Rahul exists in the cast but has NO spoken dialogue -> not a speaker.
    bad["utterances"][2]["speaker_id"] = "rahul"
    bad["utterances"][2]["speaker_type"] = "character"
    fake = FakeMiMo([
        json.dumps(bad, ensure_ascii=False),
        multi_voice_plan_json(),
    ])
    _use_planner(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/create-multi-voice-plan")
    assert response.status_code == 200, response.text
    assert fake.call_count == 2
    assert "not part of the cast" in fake.calls[1]["user"]


def test_plan_rejects_long_mid_paragraph_silence(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _analyze(client, monkeypatch, project_with_script)

    bad = make_multi_voice_plan_dict()
    bad["utterances"][0]["pause_after_ms"] = 2500  # silence INSIDE a paragraph
    fake = FakeMiMo([
        json.dumps(bad, ensure_ascii=False),
        multi_voice_plan_json(),
    ])
    _use_planner(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/create-multi-voice-plan")
    assert response.status_code == 200, response.text
    assert "unnecessary long silence" in fake.calls[1]["user"]


def test_plan_validator_detects_out_of_order_utterances() -> None:
    cast = CharacterCast(
        project_id="x",
        story=StoryAnalysis.model_validate(make_story_analysis_dict()),
    )
    data = make_multi_voice_plan_dict()
    data["utterances"][0], data["utterances"][3] = (
        data["utterances"][3],
        data["utterances"][0],
    )
    plan = MultiVoicePlan.model_validate(data)

    problems = validate_plan_against_script(plan, MV_PARAGRAPHS, cast)
    assert any("ordered by paragraph_index" in p for p in problems)


def test_tts_text_rules_apply_to_utterances() -> None:
    """SSML / word changes inside an utterance are rejected by the schema."""
    data = make_multi_voice_plan_dict()
    data["utterances"][1]["tts_text"] = (
        MV_PARAGRAPHS[0] + '<break time="1s" />'
    )
    with pytest.raises(Exception) as exc:
        MultiVoicePlan.model_validate(data)
    assert "tts_text" in str(exc.value)


# --------------------------------------------------------------------------- #
# 6. Multi-voice generation (§8/§9/§10)
# --------------------------------------------------------------------------- #
def _prepare_generable_project(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, project_id: str,
    *, approve: bool = True,
) -> CastingMock:
    _analyze(client, monkeypatch, project_id)
    mock = _wire_casting(monkeypatch)
    _select_both(client, project_id)
    if approve:
        _approve(client, project_id)
    _create_plan(client, monkeypatch, project_id)
    return mock


def test_multi_voice_generation_uses_locked_voice_per_utterance(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_generable_project(client, monkeypatch, project_with_script)

    # Also keep a paragraph plan around: multi-voice mode must win.
    from app.schemas.narration import NarrationPlan

    ProjectService(get_settings()).save_narration_plan(
        project_with_script,
        NarrationPlan.model_validate({
            "overall_story_tone": "suspenseful",
            "language": "Hindi",
            "paragraphs": [
                {
                    "paragraph_index": i,
                    "original_text": text,
                    "emotion": "neutral",
                    "delivery_style": "calm",
                    "intensity": 1,
                    "narration_direction": "n",
                    "voice_settings": {},
                    "pause_after_ms": 0,
                }
                for i, text in enumerate(MV_PARAGRAPHS)
            ],
        }),
    )

    discovery_mock = _default_mock()
    composer = _wire_audio(monkeypatch, discovery_mock)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["generation_mode"] == "multi_voice"
    assert body["segment_count"] == 6

    # Voices follow the LOCKED cast in exact story order: N S N N S N.
    paths = [r["path"] for r in discovery_mock.tts_requests]
    assert paths == [
        f"/v1/text-to-speech/{NARRATOR_VOICE_ID}",
        f"/v1/text-to-speech/{SUMAN_VOICE_ID}",
        f"/v1/text-to-speech/{NARRATOR_VOICE_ID}",
        f"/v1/text-to-speech/{NARRATOR_VOICE_ID}",
        f"/v1/text-to-speech/{SUMAN_VOICE_ID}",
        f"/v1/text-to-speech/{NARRATOR_VOICE_ID}",
    ]

    # tts_text (with restrained tags) is what ElevenLabs receives.
    texts = [r["body"]["text"] for r in discovery_mock.tts_requests]
    assert texts[1] == "[quiet, tense delivery] कभी आईने में मत देखना।"
    assert texts[4] == "[sad] डरो मत।"
    assert texts[0] == MV_PARAGRAPHS[0][: len(texts[0])]  # narrator starts the para
    for request in discovery_mock.tts_requests:
        voice_settings = request["body"]["voice_settings"]
        assert "speed" not in voice_settings       # eleven_v4 capability filter
        assert "style" not in voice_settings
        assert "stability" in voice_settings
        assert request["body"]["language_code"] == "hi"

    # FFmpeg received every utterance in story order with planned pauses.
    assert composer.segment_counts == [6]
    merged = composer.calls[0]
    assert [s.path.name for s in merged] == [
        "utterance_p0000_u0000.mp3",
        "utterance_p0000_u0001.mp3",
        "utterance_p0001_u0000.mp3",
        "utterance_p0002_u0000.mp3",
        "utterance_p0002_u0001.mp3",
        "utterance_p0002_u0002.mp3",
    ]
    assert [s.pause_after_ms for s in merged] == [200, 700, 500, 150, 200, 700]


def test_metadata_records_utterance_speakers_voices_and_tags(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_generable_project(client, monkeypatch, project_with_script)
    discovery_mock = _default_mock()
    _wire_audio(monkeypatch, discovery_mock)

    assert (
        client.post(f"/projects/{project_with_script}/generate-audio").status_code
        == 200
    )

    metadata = _metadata(project_with_script)
    assert metadata["mode"] == "multi_voice"
    assert metadata["status"] == "completed"
    assert metadata["language"] == "hi"
    assert metadata["voice_id"] is None              # mixed voices
    assert metadata["total_pause_ms"] == 200 + 700 + 500 + 150 + 200 + 700
    assert len(metadata["segments"]) == 6

    for segment in metadata["segments"]:
        assert segment["utterance_id"].startswith("p0")
        assert segment["speaker_type"] in ("narrator", "character")
        expected_voice = (
            NARRATOR_VOICE_ID
            if segment["speaker_id"] == "narrator"
            else SUMAN_VOICE_ID
        )
        assert segment["voice_id"] == expected_voice  # persisted lock usage
        assert segment["model_id"] == "eleven_v4"
        assert segment["tts_text"]
        assert "speed" not in segment["effective_voice_settings"]
        assert "style" not in segment["effective_voice_settings"]
        assert segment["request_id"]
        assert segment["status"] == "generated"

    tagged = [
        s["has_audio_tags"] for s in sorted(
            metadata["segments"], key=lambda s: (s["paragraph_index"], s["sequence_index"])
        )
    ]
    assert tagged == [False, True, False, False, True, False]

    # Segment files exist per utterance.
    segments_dir = TEST_STORAGE / project_with_script / "audio" / "segments"
    for segment in metadata["segments"]:
        assert (segments_dir / segment["filename"]).is_file()


def test_changing_a_locked_voice_regenerates_only_that_speaker(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §6: a replaced voice must actually take effect on regeneration."""
    _prepare_generable_project(client, monkeypatch, project_with_script)

    # First full run: all 6 utterances synthesised in the original cast.
    first = _default_mock()
    _wire_audio(monkeypatch, first)
    assert (
        client.post(f"/projects/{project_with_script}/generate-audio").status_code
        == 200
    )
    assert len(first.tts_requests) == 6

    # Replace ONLY suman's voice (select invalidates approval -> re-approve).
    select = client.post(
        f"/projects/{project_with_script}/voice-casting/select",
        json={"speaker_id": "suman", "voice_id": "voice-hi-3"},
    )
    assert select.status_code == 200, select.text
    _approve(client, project_with_script)

    # Second run: the plan is unchanged, but the LOCK changed for suman.
    second = _default_mock()
    _wire_audio(monkeypatch, second)
    assert (
        client.post(f"/projects/{project_with_script}/generate-audio").status_code
        == 200
    )

    # Only suman's 2 utterances regenerate — in the NEW voice; the narrator's
    # 4 utterances are reused (old audio kept, not re-billed).
    assert len(second.tts_requests) == 2
    assert {r["path"] for r in second.tts_requests} == {
        f"/v1/text-to-speech/voice-hi-3"
    }

    metadata = _metadata(project_with_script)
    by_speaker: dict[str, list[dict]] = {}
    for segment in metadata["segments"]:
        by_speaker.setdefault(segment["speaker_id"], []).append(segment)

    narrator_segments = by_speaker["narrator"]
    assert {s["voice_id"] for s in narrator_segments} == {NARRATOR_VOICE_ID}
    assert all(s["reused"] is True for s in narrator_segments)

    suman_segments = by_speaker["suman"]
    assert {s["voice_id"] for s in suman_segments} == {"voice-hi-3"}  # new voice
    assert all(s["reused"] is False for s in suman_segments)


def test_generation_without_approval_is_409(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_generable_project(
        client, monkeypatch, project_with_script, approve=False
    )
    mock = _default_mock()
    _wire_audio(monkeypatch, mock)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 409
    assert "approved" in response.json()["detail"]
    assert mock.tts_requests == []  # not a single TTS credit was spent


def test_generation_without_cast_is_404(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_generable_project(client, monkeypatch, project_with_script)
    (TEST_STORAGE / project_with_script / "character_cast.json").unlink()
    mock = _default_mock()
    _wire_audio(monkeypatch, mock)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 404
    assert "analyze-characters" in response.json()["detail"]
    assert mock.tts_requests == []


def test_failed_utterance_preserves_completed_work_and_recovers(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_generable_project(client, monkeypatch, project_with_script)

    failing = CastingMock(tts_fail_text="[sad] डरो मत।")
    failing.details = dict(VOICE_DETAILS)
    _wire_audio(monkeypatch, failing)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "Utterance p0002-u0001 failed" in detail
    assert "4/6 utterance(s)" in detail
    assert "preserved" in detail

    metadata = _metadata(project_with_script)
    assert metadata["status"] == "partial"
    assert metadata["failed_utterance_id"] == "p0002-u0001"
    assert len(metadata["segments"]) == 4            # completed work kept
    assert not (
        TEST_STORAGE / project_with_script / "audio" / "audiobook.mp3"
    ).exists()

    # Recovery: only the failed utterance + the last one are re-requested.
    healthy = _default_mock()
    _wire_audio(monkeypatch, healthy)
    retry = client.post(f"/projects/{project_with_script}/generate-audio")
    assert retry.status_code == 200, retry.text
    assert len(healthy.tts_requests) == 2
    assert [r["body"]["text"] for r in healthy.tts_requests] == [
        "[sad] डरो मत।",
        "फिर वह चुप हो गई।",
    ]

    metadata = _metadata(project_with_script)
    assert metadata["status"] == "completed"
    assert [s["reused"] for s in metadata["segments"]] == [
        True, True, True, True, False, False,
    ]


def test_project_without_multi_voice_plan_keeps_existing_modes(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Back-compat: a cast alone must NOT hijack paragraph-plan generation."""
    _analyze(client, monkeypatch, project_with_script)

    from app.schemas.narration import NarrationPlan

    ProjectService(get_settings()).save_narration_plan(
        project_with_script,
        NarrationPlan.model_validate({
            "overall_story_tone": "suspenseful",
            "language": "Hindi",
            "paragraphs": [
                {
                    "paragraph_index": i,
                    "original_text": text,
                    "tts_text": text,
                    "emotion": "neutral",
                    "delivery_style": "calm",
                    "intensity": 1,
                    "narration_direction": "note",
                    "voice_settings": {"stability": 0.5},
                    "pause_after_ms": 100,
                }
                for i, text in enumerate(MV_PARAGRAPHS)
            ],
        }),
    )

    mock = _default_mock()
    composer = _wire_audio(monkeypatch, mock)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text
    # Paragraph-plan mode (not multi_voice) because no utterance plan exists.
    assert response.json()["generation_mode"] == "narration_plan"
    # …and the default project voice is used (no per-utterance casting).
    paths = {r["path"] for r in mock.tts_requests}
    assert paths == {"/v1/text-to-speech/voice-project-default"}
    assert composer.segment_counts == [len(MV_PARAGRAPHS)]
