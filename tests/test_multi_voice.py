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
    director_analysis_json,
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

    def __init__(
        self,
        *,
        tts_fail_text: Optional[str] = None,
        tts_fail_status: int = 500,
        tts_fail_detail: str = "simulated upstream failure",
        tts_fail_always: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tts_requests: list[dict[str, Any]] = []
        self.tts_fail_text = tts_fail_text
        self.tts_fail_status = tts_fail_status
        self.tts_fail_detail = tts_fail_detail
        self.tts_fail_always = tts_fail_always

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
            if self.tts_fail_always or (
                self.tts_fail_text is not None
                and body.get("text") == self.tts_fail_text
            ):
                return httpx.Response(
                    self.tts_fail_status,
                    json={"detail": self.tts_fail_detail},
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


def _project_with(client: TestClient, script: str) -> str:
    project_id = client.post(
        "/projects", json={"name": "Name Grounding"}
    ).json()["project_id"]
    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", script.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201
    return project_id


def _speaking_character(character_id: str, name: str) -> dict:
    return {
        "character_id": character_id,
        "character_name": name,
        "role": "minor_character",
        "language": "hi",
        "gender": None, "age": None, "accent": None,
        "uncertain_attributes": [],
        "voice_description": "Eerie whispering voice",
        "speaking_style": "Haunting",
        "has_spoken_dialogue": True,
    }


def test_analyze_accepts_inflected_and_wrapped_character_names(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Name grounding tolerates real-script spelling variants (not inventions).

    Live failure this pins: MiMo wrote 'बूढ़ा आदमी' (direct case) for a script
    that says 'बूढ़े आदमी' (oblique), and 'आईने की आवाज़' where the script wraps
    the phrase across a line break and spells it without the nukta — all three
    describe speakers that ARE named in the script.
    """
    script = (
        "रिसेप्शन पर बैठे बूढ़े आदमी ने चाबी दी।\n\n"
        "रात को आईने की\nआवाज़ गूंजी। समीर डर गया।"
    )
    project = _project_with(client, script)

    story = make_story_analysis_dict()
    story["characters"] = [
        _speaking_character("old_receptionist", "बूढ़ा आदमी"),  # case variant
        _speaking_character("mirror_voice", "आईने की आवाज़"),    # wrap + nukta
    ]
    fake = FakeMiMo([json.dumps(story, ensure_ascii=False)])
    _use_analyzer(monkeypatch, fake)

    response = client.post(f"/projects/{project}/analyze-characters")
    assert response.status_code == 200, response.text
    assert fake.call_count == 1  # grounded names — no rejection loop
    names = [
        c["character_name"] for c in response.json()["story"]["characters"]
    ]
    assert names == ["बूढ़ा आदमी", "आईने की आवाज़"]


def test_analyze_still_rejects_names_with_ungrounded_words(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name containing words that are NOWHERE in the script stays rejected.

    The mirror voice below is never named in this script (only 'फुसफुसाहट'):
    composing 'आईने की आवाज़' would invent a character credited with dialogue.
    """
    script = "समीर कमरे में गया।\n\nफुसफुसाहट गूंजी, \"कौन है?\""
    project = _project_with(client, script)

    bad = make_story_analysis_dict()
    bad["characters"] = [_speaking_character("mirror_voice", "आईने की आवाज़")]
    good = make_story_analysis_dict()
    good["characters"] = []  # narrator-only correction
    fake = FakeMiMo([json.dumps(bad, ensure_ascii=False),
                     json.dumps(good, ensure_ascii=False)])
    _use_analyzer(monkeypatch, fake)

    response = client.post(f"/projects/{project}/analyze-characters")
    assert response.status_code == 200, response.text

    assert fake.call_count == 2
    feedback = fake.calls[1]["user"]
    assert "never appears in the script" in feedback
    assert "never invent a name" in feedback  # actionable retry guidance
    assert response.json()["story"]["characters"] == []


def test_analyze_accepts_null_attributes_marked_uncertain(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`gender: null` + `uncertain_attributes: ["gender"]` is a VALID way to
    say "the script never established it" (live bug: MiMo's natural encoding
    was rejected three times and the whole analysis failed).
    """
    script = "सुमन ने चाबी सौंपी और चेतावनी दी, कभी आईने में मत देखना।"
    project = _project_with(client, script)

    character = _speaking_character("suman", "सुमन")
    character["uncertain_attributes"] = ["gender", "age"]
    story = make_story_analysis_dict()
    story["characters"] = [character]
    fake = FakeMiMo([json.dumps(story, ensure_ascii=False)])
    _use_analyzer(monkeypatch, fake)

    response = client.post(f"/projects/{project}/analyze-characters")
    assert response.status_code == 200, response.text
    assert fake.call_count == 1  # accepted first try — no rejection loop
    saved = response.json()["story"]["characters"][0]
    assert saved["gender"] is None  # nothing invented
    assert saved["uncertain_attributes"] == ["gender", "age"]


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

    # Account voices first — every candidate of this pool is an account
    # voice (the library only tops thin pools up and adds nothing here) …
    assert "/v2/voices" in mock.paths

    for entry in (body["narrator"], *body["characters"].values()):
        assert 1 <= len(entry["candidates"]) <= 10  # up to 10 choices per role
        assert entry["insufficient_candidates"] is False
        for candidate in entry["candidates"]:
            assert candidate["voice_id"] in HI_VOICE_IDS  # en + banned excluded
            assert candidate["origin"] == "account"
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


class FilterSensitiveCastingMock(CastingMock):
    """Discovery that finds NOTHING whenever a demographic filter is sent.

    Simulates an account where the strict (gender/age/accent) searches come
    back empty — the cast must still end up with candidates.
    """

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if request.url.path in ("/v2/voices", "/v1/shared-voices") and any(
            key in params for key in ("gender", "age", "accent")
        ):
            self.requests.append(
                {"method": request.method, "path": request.url.path,
                 "params": params, "body": request.content}
            )
            return httpx.Response(
                200, json={"voices": [], "has_more": False}, request=request
            )
        return super().__call__(request)


def test_every_speaking_role_always_gets_candidates(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"No candidate available" must never block the cast (live bug).

    Even when every filtered search comes back empty, each speaking role is
    offered usable voices by the progressive fallback (restricted voices are
    still excluded at every level).
    """
    _analyze(client, monkeypatch, project_with_script)
    _wire_casting(
        monkeypatch,
        FilterSensitiveCastingMock(
            account_default=ACCOUNT_PAGE,
            shared_default=EMPTY_PAGE,
            details=dict(VOICE_DETAILS),
        ),
    )

    response = client.post(f"/projects/{project_with_script}/voice-casting")
    assert response.status_code == 200, response.text
    body = response.json()
    entries = [("narrator", body["narrator"])] + list(body["characters"].items())
    for speaker, entry in entries:
        assert entry["candidates"], f"{speaker} was left without any voice"
        assert entry["insufficient_candidates"] is False
        assert all(
            c["voice_id"] != "voice-banned" for c in entry["candidates"]
        ), "restricted voices are never proposed, even in the fallback"


def test_demographic_aliases_match_without_emptying_the_pool(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Script words like 'mature'/'elderly' match the voices' enum ages.

    Live bug: the free-text age 'elderly' was sent verbatim as a search
    filter and compared verbatim against 'old' — the pool came back empty
    and the character got "No candidate available".
    """
    old_man = _speaking_character("rahul", "राहुल")
    old_man["gender"] = "male"
    old_man["age"] = "elderly"
    story = make_story_analysis_dict()
    story["characters"] = [old_man]
    fake = FakeMiMo([json.dumps(story, ensure_ascii=False)])
    _use_analyzer(monkeypatch, fake)
    assert client.post(
        f"/projects/{project_with_script}/analyze-characters"
    ).status_code == 200

    page = {
        "voices": [
            account_voice(
                "voice-old", name="Gravel Elder", gender="male", age="old",
                description="Deep gravelly elderly man voice for audiobooks",
            ),
            account_voice(
                "voice-young", name="Bright Youth", gender="female", age="young",
                description="Bright youthful voice for audiobooks",
            ),
        ],
        "has_more": False,
    }
    mock = _wire_casting(
        monkeypatch,
        _default_mock(account_default=page, shared_default=EMPTY_PAGE),
    )

    response = client.post(f"/projects/{project_with_script}/voice-casting")
    assert response.status_code == 200, response.text
    candidates = response.json()["characters"]["rahul"]["candidates"]
    assert candidates, "the elderly male speaker must get candidates"
    assert candidates[0]["voice_id"] == "voice-old"  # the matching voice wins

    # The search filter uses the provider's enum vocabulary.
    search = next(
        r for r in mock.requests
        if r["path"] == "/v2/voices" and "age" in r["params"]
    )
    assert search["params"]["age"] == "old"
    assert search["params"]["gender"] == "male"


def test_japanese_story_gets_only_japanese_voices(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extended languages work end-to-end with the SAME hard guarantees.

    The story language NAME ('Japanese') is folded to its ISO code ('ja') for
    search and matching, every speaking role gets candidates, and an
    English-only voice is never proposed for a Japanese story.
    """
    script = "夜はとても暗かった。ユウジは静かにドアを閉めた。"
    project = _project_with(client, script)

    yuji = _speaking_character("yuji", "ユウジ")
    yuji["language"] = "Japanese"
    story = make_story_analysis_dict()
    story["language"] = "Japanese"
    story["locale"] = "ja-JP"
    story["narrator"]["language"] = "Japanese"
    story["characters"] = [yuji]
    fake = FakeMiMo([json.dumps(story, ensure_ascii=False)])
    _use_analyzer(monkeypatch, fake)

    response = client.post(f"/projects/{project}/analyze-characters")
    assert response.status_code == 200, response.text
    assert fake.call_count == 1  # grounded name — no rejection loop
    assert response.json()["story"]["language"] == "ja"  # name -> ISO code

    page = {
        "voices": [
            account_voice(
                "voice-ja", name="Yuki Calm", languages=("ja",),
                description="Calm Japanese audiobook narration voice",
            ),
            account_voice("voice-en", name="English Only", languages=("en",)),
        ],
        "has_more": False,
    }
    mock = _wire_casting(
        monkeypatch,
        _default_mock(account_default=page, shared_default=EMPTY_PAGE),
    )

    casting = client.post(f"/projects/{project}/voice-casting")
    assert casting.status_code == 200, casting.text
    body = casting.json()
    entries = [("narrator", body["narrator"])] + list(body["characters"].items())
    for speaker, entry in entries:
        assert entry["candidates"], f"{speaker} must get Japanese candidates"
        assert all(
            c["voice_id"] == "voice-ja" for c in entry["candidates"]
        ), "an English-only voice must never be proposed for a Japanese story"

    # The search filter carries the folded ISO code, never the raw name.
    search = next(
        r for r in mock.requests
        if r["path"] == "/v2/voices" and "language" in r["params"]
    )
    assert search["params"]["language"] == '["ja"]'


def test_shared_voice_keeps_its_own_name_when_locked(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locked voice is named after the VOICE, never after the character.

    Live bug: imported library voices were renamed to the character's name,
    so the UI showed "Selected: Narrator" instead of the chosen voice.
    """
    _analyze(client, monkeypatch, project_with_script)
    mock = _wire_casting(
        monkeypatch,
        _default_mock(
            account_default={
                "voices": [
                    account_voice(
                        "voice-narrator-01", name="Narrator Deep",
                        description="Deep, calm, measured narrator voice for audiobooks",
                    )
                ],
                "has_more": False,
            },
            shared_default={
                "voices": [shared_voice("voice-shared-1", name="Library Whisper")],
                "has_more": False,
            },
        ),
    )
    built = client.post(f"/projects/{project_with_script}/voice-casting")
    assert built.status_code == 200, built.text

    body = built.json()
    entries = [("narrator", body["narrator"])] + list(body["characters"].items())
    speaker, _entry = next(
        (sid, entry)
        for sid, entry in entries
        if any(c["voice_id"] == "voice-shared-1" for c in entry["candidates"])
    )

    response = client.post(
        f"/projects/{project_with_script}/voice-casting/select",
        json={
            "speaker_id": speaker,
            "voice_id": "voice-shared-1",
            "public_owner_id": "owner-abc",
        },  # NO new_name — the import must keep the voice's own name
    )
    assert response.status_code == 200, response.text
    lock = (
        response.json()["narrator"]
        if speaker == "narrator"
        else response.json()["characters"][speaker]
    )["selected"]
    assert lock["voice_name"] == "Library Whisper"

    imported = next(
        r for r in mock.requests if r["path"].startswith("/v1/voices/add/")
    )
    assert json.loads(imported["body"])["new_name"] == "Library Whisper"


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


def test_provider_401_quota_never_logs_the_user_out(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ElevenLabs answering 401 (exhausted credit quota) → HTTP 502, not 401.

    Regression: the frontend treats ANY 401 as "session expired" and
    redirects to /auth/login — so while the ElevenLabs quota was spent,
    every voice preview logged the user out instead of showing the error.
    """
    _analyze(client, monkeypatch, project_with_script)
    _wire_casting(
        monkeypatch,
        _default_mock(
            tts_fail_always=True,
            tts_fail_status=401,
            tts_fail_detail=(
                "This request exceeds your API key (Dauntless Komodo Dragon) "
                "quota of 10000. You have 4 credits remaining, while 8 "
                "credits are required for this request."
            ),
        ),
    )
    _select_both(client, project_with_script)

    response = client.post(
        f"/projects/{project_with_script}/voice-casting/audition",
        json={"speaker_id": "suman", "text": "नमस्ते दोस्तों, कैसे हो?"},
    )

    assert response.status_code == 502  # NEVER 401 — that logs the UI out
    detail = response.json()["detail"]
    assert "ELEVENLABS_API_KEY" in detail
    assert "quota of 10000" in detail  # the provider's real reason survives


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


def test_plan_prompt_lists_the_closed_vocabularies() -> None:
    """MiMo cannot obey an enum it never sees (live bug: it invented styles).

    Every allowed emotion and delivery style must appear in the prompt.
    """
    from app.schemas.narration import DELIVERY_STYLES, EMOTIONS
    from app.services.multi_voice_planner import SYSTEM_PROMPT

    for value in (*EMOTIONS, *DELIVERY_STYLES):
        assert value in SYSTEM_PROMPT, f"{value!r} missing from the prompt"
    assert "closed lists" in SYSTEM_PROMPT


def test_plan_alias_values_are_normalized_not_rejected(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Safe morphological variants fold onto the enums instead of failing.

    Live bug: free-text values like 'Soft'/'conversation'/'sad' failed
    Pydantic's Literal check on all 3 attempts and the whole generation
    failed after casting approval.
    """
    _analyze(client, monkeypatch, project_with_script)

    aliased = make_multi_voice_plan_dict()
    aliased["utterances"][0]["delivery_style"] = "Conversation"
    aliased["utterances"][1]["emotion"] = "sad"
    aliased["utterances"][1]["delivery_style"] = "Softly"
    aliased["utterances"][2]["emotion"] = "Excited"
    fake = FakeMiMo([json.dumps(aliased, ensure_ascii=False)])
    _use_planner(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/create-multi-voice-plan")
    assert response.status_code == 200, response.text
    assert fake.call_count == 1  # normalized on the first try — no rejection

    utterances = response.json()["utterances"]
    assert utterances[0]["delivery_style"] == "conversational"
    assert utterances[1]["emotion"] == "sadness"
    assert utterances[1]["delivery_style"] == "soft"
    assert utterances[2]["emotion"] == "excitement"


def test_plan_unknown_style_is_rejected_with_actionable_feedback(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown values are never guessed at — and the retry names them.

    'cinematic' is a preset name, not a delivery style: it must stay exactly
    as returned, be rejected loudly, and the feedback must show MiMo the
    actual value plus the complete accepted list.
    """
    _analyze(client, monkeypatch, project_with_script)

    bad = make_multi_voice_plan_dict()
    bad["utterances"][0]["delivery_style"] = "cinematic"
    bad["utterances"][0]["emotion"] = "euphoric"  # not an allowed emotion
    fake = FakeMiMo([
        json.dumps(bad, ensure_ascii=False),
        multi_voice_plan_json(),
    ])
    _use_planner(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/create-multi-voice-plan")
    assert response.status_code == 200, response.text
    assert fake.call_count == 2

    feedback = fake.calls[1]["user"]
    assert "'cinematic'" in feedback  # the ACTUAL value MiMo wrote
    assert "'euphoric'" in feedback
    assert "Allowed values:" in feedback  # the complete closed list
    assert "Regenerate ONLY 'delivery_style'" in feedback


def test_plan_invented_metadata_degrades_on_the_final_attempt(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invented emotion/delivery_style values must not kill the generation.

    Live bug: MiMo kept inventing semantic values ('wistful', 'warm', 'peace')
    on all 3 attempts and NO audio was generated. Those two fields are
    metadata only (never sent to TTS), so the final attempt degrades them to
    safe defaults — after two strict rejections with actionable feedback.
    """
    _analyze(client, monkeypatch, project_with_script)

    bad1 = make_multi_voice_plan_dict()
    bad1["utterances"][0]["emotion"] = "wistful"
    bad2 = make_multi_voice_plan_dict()
    bad2["utterances"][1]["delivery_style"] = "warm"
    bad3 = make_multi_voice_plan_dict()  # the live-bug shape: several at once
    bad3["utterances"][0]["emotion"] = "wistful"
    bad3["utterances"][1]["delivery_style"] = "warm"
    bad3["utterances"][2]["emotion"] = "peace"
    fake = FakeMiMo([
        json.dumps(bad1, ensure_ascii=False),
        json.dumps(bad2, ensure_ascii=False),
        json.dumps(bad3, ensure_ascii=False),
    ])
    _use_planner(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/create-multi-voice-plan")
    assert response.status_code == 200, response.text
    assert fake.call_count == 3

    # Attempts 1-2 still reject STRICTLY and name the offending value.
    assert "'wistful'" in fake.calls[1]["user"]
    assert "'warm'" in fake.calls[2]["user"]

    utterances = response.json()["utterances"]
    # Only the invented values degrade — to the safe defaults.
    assert utterances[0]["emotion"] == "neutral"
    assert utterances[1]["delivery_style"] == "calm"
    assert utterances[2]["emotion"] == "neutral"
    # Every valid planned value stays exactly as MiMo returned it.
    assert utterances[0]["delivery_style"] == "calm"
    assert utterances[1]["emotion"] == "fear"
    assert utterances[2]["delivery_style"] == "slow"


def test_plan_word_changes_are_never_masked_by_metadata_coercion(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coercion only ever fixes metadata — changed story words still fail."""
    _analyze(client, monkeypatch, project_with_script)

    bad = make_multi_voice_plan_dict()
    bad["utterances"][0]["emotion"] = "hope"  # salvageable metadata ...
    # ... but BOTH texts also lose a word: that must fail loudly.
    bad["utterances"][1]["original_text"] = "आईने में मत देखना।"
    bad["utterances"][1]["tts_text"] = "[quiet, tense delivery] आईने में मत देखना।"
    fake = FakeMiMo([json.dumps(bad, ensure_ascii=False)])
    _use_planner(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/create-multi-voice-plan")
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "do not reproduce the original paragraph" in detail
    assert "no audio was generated" in detail
    assert fake.call_count == 3


def test_plan_non_enum_failures_are_never_coerced(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the metadata enums degrade — a broken structure still fails."""
    _analyze(client, monkeypatch, project_with_script)

    bad = make_multi_voice_plan_dict()
    bad["utterances"][0]["emotion"] = "hope"  # salvageable on its own ...
    bad["utterances"][0]["acting_intensity"] = 99  # ... but this is not
    fake = FakeMiMo([json.dumps(bad, ensure_ascii=False)])
    _use_planner(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/create-multi-voice-plan")
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "acting_intensity" in detail
    assert "no audio was generated" in detail
    assert fake.call_count == 3


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


# --------------------------------------------------------------------------- #
# 13. Ten candidates per speaker (the voice dropdown)
# --------------------------------------------------------------------------- #
def test_voice_casting_offers_ten_candidates_for_a_large_pool(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rich pool yields TEN ranked choices per role — the user asked for
    enough candidates to find the right voice for story, script and language.
    """
    _analyze(client, monkeypatch, project_with_script)
    description = (
        "Deep, calm, measured narrator voice for audiobooks; soft warm "
        "grounded voice, natural and restrained for stories"
    )
    pool = {
        "voices": [
            account_voice(
                f"voice-pool-{index:02d}",
                name=f"Pool Voice {index:02d}",
                description=description,
            )
            for index in range(14)
        ],
        "has_more": False,
    }
    _wire_casting(monkeypatch, _default_mock(account_default=pool))

    body = client.post(f"/projects/{project_with_script}/voice-casting").json()

    for entry in (body["narrator"], *body["characters"].values()):
        candidates = entry["candidates"]
        assert len(candidates) == 10, "ten voices in the dropdown"
        ids = [c["voice_id"] for c in candidates]
        assert len(set(ids)) == 10  # ten DIFFERENT voices
        scores = [c["total_score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)  # best first
        assert [c["rank"] for c in candidates] == list(range(1, 11))
        for candidate in candidates:
            assert candidate["voice_id"].startswith("voice-pool-")
            assert candidate["preview_url"]
            assert candidate["total_score"] >= 0.55


# --------------------------------------------------------------------------- #
# 14. Narration style drives the multi-voice prompt (tags + mood)
# --------------------------------------------------------------------------- #
def test_multi_voice_prompt_follows_the_selected_narration_style(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Natural / Cinematic / Expressive / Soft / … / AI Automatic really
    change the plan prompt — so the tag and mood differences are audible.
    """
    _analyze(client, monkeypatch, project_with_script)
    _wire_casting(monkeypatch)
    _select_both(client, project_with_script)
    _approve(client, project_with_script)

    def prompt_for(preferences: dict[str, Any], *, analysis: bool = False) -> str:
        saved = client.put(
            f"/projects/{project_with_script}/narration-preferences",
            json=preferences,
        )
        assert saved.status_code == 200, saved.text
        responses: list[str] = []
        if analysis:
            # Automatic mode first runs the whole-story analysis (aux MiMo).
            responses.append(director_analysis_json())
        responses.append(multi_voice_plan_json())
        fake = FakeMiMo(responses)
        _use_planner(monkeypatch, fake)
        response = client.post(
            f"/projects/{project_with_script}/create-multi-voice-plan"
        )
        assert response.status_code == 200, response.text
        assert fake.call_count == len(responses)
        return fake.calls[-1]["user"]

    expressive = prompt_for({
        "preset": "cinematic", "acting_intensity": 2,
        "whisper_policy": "user_selected", "whisper_paragraphs": [],
        "tag_density": "minimal", "custom_direction": "",
    })
    assert "NARRATION PREFERENCES" in expressive
    assert "Cinematic &amp; Immersive" in expressive or "Cinematic" in expressive
    assert "at most 1 audio tag(s) per paragraph" in expressive

    natural = prompt_for({
        "preset": "natural", "acting_intensity": 2,
        "whisper_policy": "never", "whisper_paragraphs": [],
        "tag_density": "minimal", "custom_direction": "",
    })
    assert "Natural" in natural
    assert "at most 1 audio tag(s) per paragraph" in natural
    assert "no whisper or hushed delivery anywhere" in natural

    sensual = prompt_for({
        "preset": "sensual", "acting_intensity": 2,
        "whisper_policy": "rare", "whisper_paragraphs": [],
        "tag_density": "standard", "custom_direction": "",
    })
    assert "Sensual" in sensual
    assert "sensual, intimate, velvety delivery" in sensual

    automatic = prompt_for({
        "preset": "auto", "acting_intensity": 3,
        "whisper_policy": "user_selected", "whisper_paragraphs": [],
        "tag_density": "standard", "custom_direction": "",
    }, analysis=True)
    assert "AI Automatic" in automatic
    assert "AI-automatic mood" in automatic
    assert "choose each paragraph's mood" in automatic


# --------------------------------------------------------------------------- #
# 15. Voice exchange: context never crosses speakers (the dual check)
# --------------------------------------------------------------------------- #
def test_utterance_context_never_crosses_speakers() -> None:
    """Neighbour context is SAME-SPEAKER ONLY, text and request IDs alike.

    Live bug: `previous_text`/`next_text` and the stored request IDs came
    from ANY neighbouring utterance, so ElevenLabs bled one voice's delivery
    into another's — the narrator's lines came back spoken by a character.
    """
    from datetime import datetime, timezone

    from app.schemas.multi_voice import Utterance

    service = AudioService(_settings(), composer=FakeComposer())
    ordered = [
        Utterance.model_validate(raw)
        for raw in make_multi_voice_plan_dict()["utterances"]
    ]  # N S N N S N
    now = datetime.now(timezone.utc).isoformat()

    def record(voice_id: str, request_id: str) -> dict[str, Any]:
        return {
            "status": "generated",
            "voice_id": voice_id,
            "model_id": "eleven_v4",
            "request_id": request_id,
            "generated_at": now,
        }

    prior = {
        "custom-0": record("voice-narrator-01", "req-n0"),
        "custom-1": record("voice-suman-02", "req-s1"),
        "custom-2": record("voice-narrator-01", "req-n2"),
        "custom-3": record("voice-narrator-01", "req-n3"),
        "custom-4": record("voice-suman-02", "req-s4"),
        "custom-5": record("voice-narrator-01", "req-n5"),
    }

    # First-time generation: TEXT context, same speaker only.
    context = service._build_utterance_context(
        position=1, ordered=ordered, prior_records={}, using_prior_ids=False,
        voice_id="voice-suman-02",
    )
    assert context is not None
    assert context.previous_text is None  # custom-0 belongs to the narrator
    assert context.next_text == service._spoken(ordered[4])  # her own line

    context = service._build_utterance_context(
        position=0, ordered=ordered, prior_records={}, using_prior_ids=False,
        voice_id="voice-narrator-01",
    )
    assert context is not None
    assert context.previous_text is None
    assert context.next_text == service._spoken(ordered[2])  # never custom-1

    # Regeneration: REQUEST IDs, same speaker AND same locked voice only.
    context = service._build_utterance_context(
        position=1, ordered=ordered, prior_records=prior, using_prior_ids=True,
        voice_id="voice-suman-02",
    )
    assert context.previous_request_ids == ()  # the narrator's id is excluded
    assert context.next_request_ids == ("req-s4",)

    # A stored ID generated in a DIFFERENT locked voice is never reused.
    changed = dict(prior)
    changed["custom-0"] = record("voice-old-replacement", "req-n0")
    context = service._build_utterance_context(
        position=2, ordered=ordered, prior_records=changed, using_prior_ids=True,
        voice_id="voice-narrator-01",
    )
    assert context.previous_request_ids == ()  # voice mismatch -> dropped
    assert context.previous_text == service._spoken(ordered[0])  # text instead


def test_generated_requests_never_carry_another_speakers_context(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wire-level dual check: on the way to ElevenLabs, no request ever
    contains another speaker's line as its continuity context."""
    mock = _prepare_generable_project(client, monkeypatch, project_with_script)
    _wire_audio(monkeypatch, mock)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text

    spoken_by_voice: dict[str, set[str]] = {
        NARRATOR_VOICE_ID: set(),
        SUMAN_VOICE_ID: set(),
    }
    for raw in make_multi_voice_plan_dict()["utterances"]:
        voice = SUMAN_VOICE_ID if raw["speaker_id"] == "suman" else NARRATOR_VOICE_ID
        spoken_by_voice[voice].add(raw["tts_text"] or raw["original_text"])

    requests = mock.tts_requests
    assert len(requests) == 6  # one per utterance, story order
    for request in requests:
        voice = request["path"].rsplit("/", 1)[-1]
        body = request["body"]
        other = spoken_by_voice[SUMAN_VOICE_ID if voice == NARRATOR_VOICE_ID
                                else NARRATOR_VOICE_ID]
        for field in ("previous_text", "next_text"):
            value = body.get(field)
            if value:
                assert value not in other, f"{field} leaked another speaker's line"
        assert not body.get("previous_request_ids")
        assert not body.get("next_request_ids")

    # The character's line still carries its audio tags and its own voice.
    suman = [r for r in requests if r["path"].endswith(SUMAN_VOICE_ID)]
    assert len(suman) == 2
    assert suman[0]["body"]["text"] == "[quiet, tense delivery] कभी आईने में मत देखना।"


# --------------------------------------------------------------------------- #
# 16. Long scripts (5k words) are planned in batches and merged
# --------------------------------------------------------------------------- #
class _ParagraphEchoMiMo:
    """Answers every batch with one narrator utterance per numbered paragraph.

    Reads the numbered paragraphs out of the prompt, so each response covers
    exactly the batch the planner asked for (GLOBAL paragraph indices).
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def complete_json(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        marker = "Script paragraphs (each becomes 1..k ordered utterances):\n"
        remainder = user.split(marker, 1)[1]
        numbered, _ = json.JSONDecoder().raw_decode(remainder.lstrip())
        utterances = [
            {
                "utterance_id": f"echo-{item['paragraph_index']:04d}",
                "paragraph_index": item["paragraph_index"],
                "sequence_index": 0,
                "speaker_id": "narrator",
                "speaker_type": "narrator",
                "original_text": item["text"],
                "tts_text": item["text"],
                "emotion": "neutral",
                "acting_intensity": 2,
                "delivery_style": "calm",
                "narration_direction": "Steady narration.",
                "pause_after_ms": 500,
            }
            for item in numbered
        ]
        return json.dumps({"utterances": utterances}, ensure_ascii=False)


def test_five_thousand_word_scripts_are_planned_in_batches_and_merged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 5k-word manuscript plans in small batches (GLOBAL indices) and the
    merged plan covers every paragraph — never a truncated/invalid response.
    """
    from app.services.multi_voice_planner import (
        MAX_BATCH_CHARACTERS,
        MAX_BATCH_PARAGRAPHS,
    )

    filler = (
        "The old lamp flickered on the platform while the night train waited "
        "in the rain, carrying tired passengers and their quiet untold stories "
        "onward through the dark hours of the long journey home again, while "
        "somewhere behind the windows a slow song kept playing softly on."
    )
    paragraphs: list[str] = []
    for index in range(165):
        if index % 5 == 2:
            paragraphs.append(MV_PARAGRAPHS[index % len(MV_PARAGRAPHS)])
        else:
            paragraphs.append(f"{filler} Segment {index}.")
    script = "\n\n".join(paragraphs)
    assert len(script.split()) >= 5000, "the fixture really is a 5k-word script"

    project = client.post("/projects", json={"name": "Long Script"}).json()[
        "project_id"
    ]
    upload = client.post(
        f"/projects/{project}/upload-script",
        files={"file": ("script.txt", script.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201, upload.text

    _analyze(client, monkeypatch, project)
    _wire_casting(monkeypatch)
    _select_both(client, project)
    _approve(client, project)

    fake = _ParagraphEchoMiMo()
    _use_planner(monkeypatch, fake)
    response = client.post(f"/projects/{project}/create-multi-voice-plan")
    assert response.status_code == 200, response.text
    body = response.json()

    # The merged plan covers every paragraph, in order, with global indices.
    assert body["paragraph_count"] == len(paragraphs)
    assert body["utterance_count"] == len(paragraphs)
    assert body["speakers"] == ["narrator"]
    assert [u["paragraph_index"] for u in body["utterances"]] == list(
        range(len(paragraphs))
    )

    # …and no single MiMo call was ever asked for the whole script.
    assert fake.calls, "the planner called MiMo"
    assert len(fake.calls) > 1, "long scripts are split into batches"
    seen: set[int] = set()
    for call in fake.calls:
        assert "This is one BATCH of a longer script" in call["user"]
        marker = "Script paragraphs (each becomes 1..k ordered utterances):\n"
        numbered, _ = json.JSONDecoder().raw_decode(
            call["user"].split(marker, 1)[1].lstrip()
        )
        assert len(numbered) <= MAX_BATCH_PARAGRAPHS
        assert sum(len(item["text"]) for item in numbered) <= MAX_BATCH_CHARACTERS
        for item in numbered:
            index = item["paragraph_index"]
            assert index not in seen, "batches never overlap"
            seen.add(index)
            assert item["text"] == paragraphs[index], "global numbering is exact"
    assert seen == set(range(len(paragraphs))), "every paragraph was planned"


# --------------------------------------------------------------------------- #
# 17. Narration beats between dialogue never belong to a character
# --------------------------------------------------------------------------- #
def _utterance(
    uid: str,
    seq: int,
    speaker: str,
    speaker_type: str,
    text: str,
    *,
    paragraph_index: int = 2,
    pause_after_ms: int = 200,
) -> dict[str, Any]:
    return {
        "utterance_id": uid,
        "paragraph_index": paragraph_index,
        "sequence_index": seq,
        "speaker_id": speaker,
        "speaker_type": speaker_type,
        "original_text": text,
        "tts_text": text,
        "emotion": "neutral",
        "acting_intensity": 2,
        "delivery_style": "calm",
        "narration_direction": "Steady delivery.",
        "pause_after_ms": pause_after_ms,
    }


def test_narration_beats_between_dialogue_stay_with_the_narrator(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live bug: 1-2 narration words between two dialogue lines ("he
    said.", "उसने कहा,") were spoken by a character. Both the separate-beat
    and the merged-beat misassignments must be rejected with targeted
    feedback so MiMo splits them out for the narrator.
    """
    _analyze(client, monkeypatch, project_with_script)
    _wire_casting(monkeypatch)
    _select_both(client, project_with_script)
    _approve(client, project_with_script)

    # BAD A — the beat "उसने कहा," is its own utterance but spoken by SUMAN.
    bad_beat = [
        _utterance("a0", 0, "narrator", "narrator", "सुमन ने चाबी सौंपी और चेतावनी दी",
                   paragraph_index=0),
        _utterance("a1", 1, "suman", "character", "कभी आईने में मत देखना।",
                   paragraph_index=0, pause_after_ms=700),
        _utterance("a2", 0, "narrator", "narrator", "अचानक दरवाजे पर राहुल ने दस्तक दी।",
                   paragraph_index=1, pause_after_ms=500),
        _utterance("a3", 0, "narrator", "narrator", "सुमन मुस्कुराई।", pause_after_ms=150),
        _utterance("a4", 1, "suman", "character", "उसने कहा,"),
        _utterance("a5", 2, "suman", "character", "डरो मत।"),
        _utterance("a6", 3, "narrator", "narrator", "फिर वह चुप हो गई।",
                   pause_after_ms=700),
    ]
    # BAD B — the beat "उसने कहा," is MERGED into SUMAN's dialogue line.
    bad_merged = [
        _utterance("b0", 0, "narrator", "narrator", "सुमन ने चाबी सौंपी और चेतावनी दी",
                   paragraph_index=0),
        _utterance("b1", 1, "suman", "character", "कभी आईने में मत देखना।",
                   paragraph_index=0, pause_after_ms=700),
        _utterance("b2", 0, "narrator", "narrator", "अचानक दरवाजे पर राहुल ने दस्तक दी।",
                   paragraph_index=1, pause_after_ms=500),
        _utterance("b3", 0, "narrator", "narrator", "सुमन मुस्कुराई।", pause_after_ms=150),
        _utterance("b4", 1, "suman", "character", "उसने कहा, डरो मत।"),
        _utterance("b5", 2, "narrator", "narrator", "फिर वह चुप हो गई।",
                   pause_after_ms=700),
    ]
    good = make_multi_voice_plan_dict()["utterances"]

    fake = FakeMiMo([
        json.dumps({"utterances": bad_beat}, ensure_ascii=False),
        json.dumps({"utterances": bad_merged}, ensure_ascii=False),
        json.dumps({"utterances": good}, ensure_ascii=False),
    ])
    _use_planner(monkeypatch, fake)
    response = client.post(f"/projects/{project_with_script}/create-multi-voice-plan")
    assert response.status_code == 200, response.text
    assert fake.call_count == 3, "both misassignments rejected, then accepted"

    first_feedback = fake.calls[1]["user"]
    assert "dialogue beat" in first_feedback
    assert "उसने कहा," in first_feedback
    assert "ALWAYS belong to the narrator" in first_feedback

    second_feedback = fake.calls[2]["user"]
    assert "merged into" in second_feedback
    assert "split it out into its own narrator utterance" in second_feedback


def test_quoted_dialogue_words_never_leak_to_characters(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In quoted paragraphs a character may only speak the words INSIDE the
    quotation marks — narration between the quotes is the narrator's.
    """
    script = "सुमन ने राहुल से कहा, “मैं ठीक हूँ।” और फिर वह चली गई।"
    project = client.post("/projects", json={"name": "Quote Test"}).json()[
        "project_id"
    ]
    upload = client.post(
        f"/projects/{project}/upload-script",
        files={"file": ("script.txt", script.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201, upload.text

    _analyze(client, monkeypatch, project)
    _wire_casting(monkeypatch)
    _select_both(client, project)
    _approve(client, project)

    # BAD — SUMAN also speaks the unquoted narration "और फिर वह चली गई।".
    bad = [
        _utterance("q0", 0, "narrator", "narrator", "सुमन ने राहुल से कहा,",
                   paragraph_index=0),
        _utterance("q1", 1, "suman", "character", "मैं ठीक हूँ। और फिर वह चली गई।",
                   paragraph_index=0, pause_after_ms=700),
    ]
    # GOOD — only the quoted words belong to SUMAN.
    good = [
        _utterance("g0", 0, "narrator", "narrator", "सुमन ने राहुल से कहा,",
                   paragraph_index=0),
        _utterance("g1", 1, "suman", "character", "मैं ठीक हूँ।", paragraph_index=0),
        _utterance("g2", 2, "narrator", "narrator", "और फिर वह चली गई।",
                   paragraph_index=0, pause_after_ms=700),
    ]

    fake = FakeMiMo([
        json.dumps({"utterances": bad}, ensure_ascii=False),
        json.dumps({"utterances": good}, ensure_ascii=False),
    ])
    _use_planner(monkeypatch, fake)
    response = client.post(f"/projects/{project}/create-multi-voice-plan")
    assert response.status_code == 200, response.text
    assert fake.call_count == 2, "the leaked narration rejected, then accepted"
    assert "outside the quoted dialogue" in fake.calls[1]["user"]


def test_plan_prompt_teaches_the_narration_beat_split() -> None:
    """MiMo cannot obey a split rule it never sees — the prompt must teach
    that one-or-two-word narration beats are narrator lines (the live bug).
    """
    from app.services.multi_voice_planner import SYSTEM_PROMPT

    assert "SPLITTING EXAMPLE" in SYSTEM_PROMPT
    assert "EVEN WHEN IT IS ONLY ONE OR TWO WORDS" in SYSTEM_PROMPT
    assert "said Rohan, wiping his hands." in SYSTEM_PROMPT
    assert "NEVER merge an attribution beat" in SYSTEM_PROMPT
    assert "उसने कहा," in SYSTEM_PROMPT
