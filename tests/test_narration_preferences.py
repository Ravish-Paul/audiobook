"""Narration Preferences tests — MiMo and ElevenLabs are always mocked.

Covers the task requirements:

* §1/§2 — GET/PUT preferences (persistence, validation, defaults), the
  preferences object travelling with `analyze-narration` and being embedded
  in the saved plan, plus the static Narration Preferences panel;
* §3    — the anti-overacting rules in the system prompt;
* §4    — `NarrationPolicyValidator` for all three whisper policies, acting
  intensity, tag density and dramatic instructions, with targeted retry
  feedback and a hard rejection when the retries run out;
* §5    — capability-aware voice settings (eleven_v4: speed/style are
  planning metadata only) and consistent voice identity;
* §7    — plan hashing (preferences included, legacy plans unchanged),
  the paragraph preview endpoint and staleness on generate-audio.

No full audio is generated and no real API credits are spent.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import DEFAULT_ELEVENLABS_BASE_URL, get_settings
from app.main import app
from app.routes import audio as audio_routes
from app.routes import narration as narration_routes
from app.schemas.narration import EMOTIONS, NarrationPlan
from app.schemas.narration_preferences import NarrationPreferences
from app.services import AudioService, ElevenLabsClient, NarrationPlanner
from app.services.audio_service import plan_hash
from app.services.narration_planner import build_system_prompt, split_paragraphs
from fakes import FAKE_MP3, FakeComposer, FakeMiMo, make_plan_dict

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"

SCRIPT = (
    "रात बहुत अंधेरी थी और चारों तरफ सन्नाटा पसरा हुआ था।\n\n"
    "अचानक दरवाजे पर किसी ने दस्तक दी।\n\n"
    "वह धीरे-धीरे दरवाजे की ओर बढ़ा और दरवाजा खोला।"
)
PARAGRAPHS = split_paragraphs(SCRIPT)  # 3 paragraphs

# Delivery styles a *restrained* plan may use (no "dramatic": that one is
# counted against the preset's dramatic-instruction budget).
RESTRICTED_STYLES = ("calm", "conversational", "soft", "reflective", "slow", "mysterious")

# Eight plain delivery cues — within the schema's hard cap of 8, so only the
# tag-density policy can reject them.
EIGHT_TAGS = (
    "curious",
    "softly",
    "warm tone",
    "low voice",
    "gentle",
    "intimate",
    "grounded",
    "quiet urgency",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def prefs(**overrides: Any) -> dict[str, Any]:
    """The complete preferences object the Narration Preferences panel sends."""
    base: dict[str, Any] = {
        "preset": "natural",
        "acting_intensity": 2,
        "whisper_policy": "user_selected",
        "whisper_paragraphs": [],
        "tag_density": "minimal",
        "custom_direction": "",
    }
    base.update(overrides)
    return base


def tagged(text: str, tags: Sequence[str]) -> str:
    """`text` with inline Eleven v4 tags in front (stripping them restores it)."""
    if not tags:
        return text
    return " ".join(f"[{tag}]" for tag in tags) + f" {text}"


def compliant_plan(
    texts: Sequence[str],
    *,
    intensity: int = 2,
    tts_texts: Optional[Sequence[str]] = None,
    stability: float = 0.6,
    delivery_styles: Optional[Sequence[str]] = None,
    emotions: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """A plan that satisfies every policy for `preset=natural` by default.

    `tests.fakes.make_plan_dict` deliberately drifts intensity and stability
    (to exercise the no-preferences path), so policy tests need this
    builder instead: inert speed/style for eleven_v4, stable voice identity,
    intensity within the default ceiling and no overacting markers.
    """
    paragraphs: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        style = (
            delivery_styles[index]
            if delivery_styles is not None
            else RESTRICTED_STYLES[index % len(RESTRICTED_STYLES)]
        )
        emotion = (
            emotions[index]
            if emotions is not None
            else ("neutral", "suspense", "fear", "sadness", "curiosity")[index % 5]
        )
        paragraphs.append(
            {
                "paragraph_index": index,
                "original_text": text,
                "tts_text": tts_texts[index] if tts_texts is not None else text,
                "emotion": emotion if emotion in EMOTIONS else "neutral",
                "delivery_style": style,
                "intensity": intensity,
                "narration_direction": (
                    f"Read paragraph {index} as natural, controlled storytelling."
                ),
                "voice_settings": {
                    "speed": 1.0,  # not supported by eleven_v4 -> inert
                    "stability": stability,
                    "similarity_boost": 0.75,
                    "style": 0.0,  # not supported by eleven_v4 -> inert
                    "use_speaker_boost": True,
                },
                "pause_after_ms": 300 + 100 * index,
            }
        )
    return {
        "overall_story_tone": "quiet suspense",
        "language": "Hindi",
        "paragraphs": paragraphs,
    }


def compliant_json(texts: Sequence[str], **kwargs: Any) -> str:
    """JSON string of `compliant_plan` (what a well-behaved MiMo returns)."""
    return json.dumps(compliant_plan(texts, **kwargs), ensure_ascii=False)


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def project_with_script(client: TestClient) -> str:
    project_id = client.post("/projects", json={"name": "Preferences Test"}).json()[
        "project_id"
    ]
    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", SCRIPT.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201
    return project_id


def _use_fake_mimo(monkeypatch: pytest.MonkeyPatch, fake: FakeMiMo) -> FakeMiMo:
    """Route every narration endpoint through a planner with a fake MiMo."""
    monkeypatch.setattr(
        narration_routes,
        "_planner",
        lambda: NarrationPlanner(get_settings(), mimo_client=fake),
    )
    return fake


def _analyze(
    client: TestClient,
    project_id: str,
    preferences: Optional[dict[str, Any]] = None,
):
    """POST analyze-narration — with the panel's object, or body-less (legacy)."""
    url = f"/projects/{project_id}/analyze-narration"
    if preferences is None:
        return client.post(url)
    return client.post(url, json={"preferences": preferences})


def _prefs_url(project_id: str) -> str:
    return f"/projects/{project_id}/narration-preferences"


def _plan_path(project_id: str) -> Path:
    return TEST_STORAGE / project_id / "narration_plan.json"


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
        elevenlabs_api_key="sk-plan-test",
        elevenlabs_voice_id="voice-plan",
        elevenlabs_base_url=DEFAULT_ELEVENLABS_BASE_URL,
    )
    recorder = _Recorder()
    composer = FakeComposer()
    service = AudioService(
        settings,
        tts_client=ElevenLabsClient(
            settings, transport=httpx.MockTransport(recorder)
        ),
        composer=composer,
    )
    monkeypatch.setattr(audio_routes, "_audio_service", lambda: service)
    return recorder, composer


# --------------------------------------------------------------------------- #
# §2 — preferences endpoints (persisted per project)
# --------------------------------------------------------------------------- #
def test_get_preferences_returns_documented_defaults(
    client: TestClient, project_with_script: str
) -> None:
    body = client.get(_prefs_url(project_with_script)).json()

    assert body["saved"] is False
    assert body["saved_at"] is None
    assert body["preferences"] == {
        "preset": "natural",
        "acting_intensity": 2,
        "whisper_policy": "user_selected",
        "whisper_paragraphs": [],
        "tag_density": "minimal",
        "custom_direction": "",
    }


def test_put_preferences_are_persisted_per_project(
    client: TestClient, project_with_script: str
) -> None:
    saved = client.put(
        _prefs_url(project_with_script),
        json=prefs(
            preset="restrained_cinematic",
            acting_intensity=3,
            whisper_policy="never",
            tag_density="standard",
            custom_direction="Warm, close microphone.",
        ),
    )
    assert saved.status_code == 200, saved.text
    body = saved.json()
    assert body["saved"] is True
    assert body["saved_at"] is not None
    assert body["preferences"]["preset"] == "restrained_cinematic"
    assert body["preferences"]["whisper_policy"] == "never"

    # On disk, nested under "preferences" (the schema forbids extra keys).
    stored = json.loads(
        (TEST_STORAGE / project_with_script / "narration_preferences.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["project_id"] == project_with_script
    assert stored["preferences"]["custom_direction"] == "Warm, close microphone."

    # …and read back for the panel.
    again = client.get(_prefs_url(project_with_script)).json()
    assert again["saved"] is True
    assert again["preferences"] == body["preferences"]


def test_put_preferences_rejects_invalid_values(
    client: TestClient, project_with_script: str
) -> None:
    url = _prefs_url(project_with_script)

    assert client.put(url, json=prefs(preset="nope")).status_code == 422
    assert client.put(url, json=prefs(acting_intensity=0)).status_code == 422
    assert client.put(url, json=prefs(acting_intensity=6)).status_code == 422
    assert client.put(url, json=prefs(whisper_policy="sometimes")).status_code == 422
    assert client.put(url, json=prefs(tag_density="lots")).status_code == 422
    assert client.put(url, json=prefs(whisper_paragraphs=[-1])).status_code == 422
    # extra="forbid": unknown fields never reach storage.
    assert client.put(url, json=prefs(volume=11)).status_code == 422


def test_put_preferences_rejects_whisper_index_outside_the_script(
    client: TestClient, project_with_script: str
) -> None:
    response = client.put(
        _prefs_url(project_with_script), json=prefs(whisper_paragraphs=[5])
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "[5]" in detail
    assert "0..2" in detail


# --------------------------------------------------------------------------- #
# §4 — whisper policy (all three variants)
# --------------------------------------------------------------------------- #
def test_whisper_policy_never_accepts_a_plain_plan(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _use_fake_mimo(monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS)]))

    response = _analyze(
        client, project_with_script, prefs(whisper_policy="never")
    )

    assert response.status_code == 200, response.text
    assert fake.call_count == 1
    assert response.json()["preferences"]["whisper_policy"] == "never"


def test_whisper_policy_never_rejects_a_whisper_and_retries_with_feedback(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    whispering = compliant_json(
        PARAGRAPHS, tts_texts=[tagged(PARAGRAPHS[0], ["whispers"]), PARAGRAPHS[1], PARAGRAPHS[2]]
    )
    fake = _use_fake_mimo(
        monkeypatch, FakeMiMo([whispering, compliant_json(PARAGRAPHS)])
    )

    response = _analyze(client, project_with_script, prefs(whisper_policy="never"))

    assert response.status_code == 200, response.text
    # One violation, one targeted correction — never silently accepted.
    assert fake.call_count == 2
    feedback = fake.calls[1]["user"]
    assert "REJECTED" in feedback
    assert "whisper_policy is 'Never'" in feedback
    assert "[whispers]" in feedback


def test_whisper_policy_never_never_silently_accepts_a_whisper(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    whispering = compliant_json(
        PARAGRAPHS, tts_texts=[tagged(PARAGRAPHS[0], ["whispers"]), PARAGRAPHS[1], PARAGRAPHS[2]]
    )
    fake = _use_fake_mimo(monkeypatch, FakeMiMo([whispering]))

    response = _analyze(client, project_with_script, prefs(whisper_policy="never"))

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "whisper_policy is 'Never'" in detail
    assert "attempt(s)" in detail
    assert fake.call_count == 3  # mimo_max_attempts
    assert not _plan_path(project_with_script).exists()


def test_whisper_policy_user_selected_allows_the_selected_line(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts_texts = [PARAGRAPHS[0], tagged(PARAGRAPHS[1], ["whispers"]), PARAGRAPHS[2]]
    fake = _use_fake_mimo(
        monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS, tts_texts=tts_texts)])
    )

    response = _analyze(
        client,
        project_with_script,
        prefs(whisper_policy="user_selected", whisper_paragraphs=[1]),
    )

    assert response.status_code == 200, response.text
    assert fake.call_count == 1
    assert response.json()["paragraphs"][1]["tts_text"].startswith("[whispers]")


def test_whisper_policy_user_selected_rejects_an_unselected_line(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts_texts = [tagged(PARAGRAPHS[0], ["whispers"]), PARAGRAPHS[1], PARAGRAPHS[2]]
    _use_fake_mimo(
        monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS, tts_texts=tts_texts)])
    )

    response = _analyze(
        client,
        project_with_script,
        prefs(whisper_policy="user_selected", whisper_paragraphs=[1]),
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "not selected" in detail
    assert "[1]" in detail
    assert not _plan_path(project_with_script).exists()


def test_whisper_policy_rare_budgets_whispered_paragraphs(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Rare = max(1, ceil(3 * 0.1)) = 1 whispered paragraph out of three.
    one_whisper = [tagged(PARAGRAPHS[0], ["whispers"]), PARAGRAPHS[1], PARAGRAPHS[2]]
    _use_fake_mimo(
        monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS, tts_texts=one_whisper)])
    )

    response = _analyze(client, project_with_script, prefs(whisper_policy="rare"))

    assert response.status_code == 200, response.text

    two_whispers = [
        tagged(PARAGRAPHS[0], ["whispers"]),
        tagged(PARAGRAPHS[1], ["whispers"]),
        PARAGRAPHS[2],
    ]
    _use_fake_mimo(
        monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS, tts_texts=two_whispers)])
    )

    rejected = _analyze(client, project_with_script, prefs(whisper_policy="rare"))

    assert rejected.status_code == 502
    assert "at most 1" in rejected.json()["detail"]


# --------------------------------------------------------------------------- #
# §4 — acting intensity
# --------------------------------------------------------------------------- #
def test_intensity_above_the_preference_is_rejected_then_corrected(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _use_fake_mimo(
        monkeypatch,
        FakeMiMo(
            [
                compliant_json(PARAGRAPHS, intensity=4),
                compliant_json(PARAGRAPHS, intensity=1),
            ]
        ),
    )

    response = _analyze(client, project_with_script, prefs(acting_intensity=1))

    assert response.status_code == 200, response.text
    assert fake.call_count == 2
    assert all(p["intensity"] == 1 for p in response.json()["paragraphs"])
    feedback = fake.calls[1]["user"]
    assert "REJECTED" in feedback
    assert "exceeds the user's acting_intensity 1" in feedback


def test_intensity_violation_exhausts_retries_and_saves_nothing(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _use_fake_mimo(
        monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS, intensity=4)])
    )

    response = _analyze(client, project_with_script, prefs(acting_intensity=1))

    assert response.status_code == 502
    assert "exceeds the user's acting_intensity 1" in response.json()["detail"]
    assert fake.call_count == 3
    assert not _plan_path(project_with_script).exists()


# --------------------------------------------------------------------------- #
# §4 — audio tag density
# --------------------------------------------------------------------------- #
def test_tag_density_minimal_rejects_two_tags_in_one_paragraph(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts_texts = [tagged(PARAGRAPHS[0], ["curious", "softly"]), PARAGRAPHS[1], PARAGRAPHS[2]]
    _use_fake_mimo(
        monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS, tts_texts=tts_texts)])
    )

    response = _analyze(client, project_with_script, prefs(tag_density="minimal"))

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "audio tag density 'Minimal' allows at most 1" in detail


def test_tag_density_standard_accepts_three_tags(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts_texts = [
        tagged(PARAGRAPHS[0], ["curious", "softly", "warm tone"]),
        PARAGRAPHS[1],
        PARAGRAPHS[2],
    ]
    _use_fake_mimo(
        monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS, tts_texts=tts_texts)])
    )

    response = _analyze(client, project_with_script, prefs(tag_density="standard"))

    assert response.status_code == 200, response.text


def test_tag_density_decides_whether_eight_tags_are_allowed(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts_texts = [tagged(PARAGRAPHS[0], EIGHT_TAGS), PARAGRAPHS[1], PARAGRAPHS[2]]

    # Same plan: rejected under Minimal…
    _use_fake_mimo(
        monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS, tts_texts=tts_texts)])
    )
    rejected = _analyze(client, project_with_script, prefs(tag_density="minimal"))
    assert rejected.status_code == 502
    assert "allows at most 1" in rejected.json()["detail"]

    # …accepted under Expressive (the schema's hard cap of 8).
    second_project = client.post("/projects", json={"name": "Second"}).json()[
        "project_id"
    ]
    assert (
        client.post(
            f"/projects/{second_project}/upload-script",
            files={"file": ("script.txt", SCRIPT.encode("utf-8"), "text/plain")},
        ).status_code
        == 201
    )
    _use_fake_mimo(
        monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS, tts_texts=tts_texts)])
    )
    allowed = _analyze(client, second_project, prefs(tag_density="expressive"))
    assert allowed.status_code == 200, allowed.text


# --------------------------------------------------------------------------- #
# §4 — dramatic instructions
# --------------------------------------------------------------------------- #
def test_dramatic_delivery_style_counts_against_the_preset_budget(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    styles = ["dramatic", "calm", "calm"]

    # Natural tolerates no dramatic staging at all …
    _use_fake_mimo(
        monkeypatch,
        FakeMiMo([compliant_json(PARAGRAPHS, delivery_styles=styles)]),
    )
    rejected = _analyze(client, project_with_script, prefs(preset="natural"))
    assert rejected.status_code == 502
    assert "delivery_style 'dramatic'" in rejected.json()["detail"]

    # … while the Dramatic preset allows it.
    _use_fake_mimo(
        monkeypatch,
        FakeMiMo([compliant_json(PARAGRAPHS, delivery_styles=styles)]),
    )
    allowed = _analyze(client, project_with_script, prefs(preset="dramatic"))
    assert allowed.status_code == 200, allowed.text


def test_overacting_wording_in_the_direction_is_rejected(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = compliant_json(PARAGRAPHS)
    data = json.loads(plan)
    data["paragraphs"][0]["narration_direction"] = (
        "Deliver this with breathless, trembling panic."
    )
    _use_fake_mimo(monkeypatch, FakeMiMo([json.dumps(data, ensure_ascii=False)]))

    response = _analyze(client, project_with_script, prefs(preset="natural"))

    assert response.status_code == 502
    assert "breathless" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# §2/§7 — plan hashing
# --------------------------------------------------------------------------- #
def test_plan_hash_includes_preferences_and_keeps_legacy_hash_stable() -> None:
    base = make_plan_dict(PARAGRAPHS)  # no "preferences" key (legacy shape)

    legacy = NarrationPlan.model_validate(base)
    with_a = NarrationPlan.model_validate({**base, "preferences": prefs()})
    with_b = NarrationPlan.model_validate(
        {**base, "preferences": prefs(acting_intensity=4)}
    )

    # Changing preferences changes the fingerprint (fresh plan, no reuse).
    assert plan_hash(with_a) != plan_hash(with_b)
    assert plan_hash(legacy) != plan_hash(with_a)

    # Plans written before preferences existed keep their exact old hash,
    # so no already-paid-for segment is silently re-billed.
    data = legacy.model_dump(mode="json")
    assert data["preferences"] is None
    data.pop("preferences")
    expected = hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    assert plan_hash(legacy) == expected


# --------------------------------------------------------------------------- #
# §7 — compatibility with the existing (no-preferences) flow
# --------------------------------------------------------------------------- #
def test_analyze_without_preferences_keeps_the_legacy_behaviour(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _use_fake_mimo(monkeypatch, FakeMiMo([json.dumps(make_plan_dict(PARAGRAPHS))]))

    response = _analyze(client, project_with_script)  # no body at all

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["preferences"] is None
    assert body["saved"] is True
    assert len(body["paragraphs"]) == 3
    assert body["voice_model"]["model_id"] == "eleven_v4"
    assert len(body["tts_voice_settings"]) == 3

    # The anti-overacting rules are in force even without preferences…
    assert "Fear does not automatically mean whispering" in fake.calls[0]["system"]
    # …but the preference block itself is not rendered.
    assert "NARRATION PREFERENCES (set by the user" not in fake.calls[0]["system"]

    # The saved plan round-trips through GET too.
    stored = client.get(f"/projects/{project_with_script}/narration-plan").json()
    assert stored["preferences"] is None
    assert stored["paragraph_count"] == 3


# --------------------------------------------------------------------------- #
# §1 — preview one paragraph before generating everything
# --------------------------------------------------------------------------- #
def test_preview_returns_the_target_paragraph_without_saving_a_plan(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _use_fake_mimo(monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS)]))

    response = client.post(
        f"/projects/{project_with_script}/analyze-narration/preview",
        json={
            "paragraph_index": 1,
            "preferences": prefs(custom_direction="Keep it intimate."),
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["paragraph_index"] == 1
    assert body["saved"] is False
    assert body["paragraph"]["original_text"] == PARAGRAPHS[1]
    assert body["context_paragraph_indices"] == [0, 1, 2]
    assert body["preferences"]["custom_direction"] == "Keep it intimate."
    assert body["tts_voice_settings"]["paragraph_index"] == 1
    # The model was asked for exactly the 3-paragraph context window.
    assert "Total paragraphs: 3" in fake.calls[0]["user"]

    # A preview never overwrites the plan…
    assert not _plan_path(project_with_script).exists()
    # …but the preferences it was given are persisted for the project.
    stored = client.get(_prefs_url(project_with_script)).json()
    assert stored["saved"] is True
    assert stored["preferences"]["custom_direction"] == "Keep it intimate."


def test_preview_rejects_a_paragraph_outside_the_script(
    client: TestClient, project_with_script: str
) -> None:
    response = client.post(
        f"/projects/{project_with_script}/analyze-narration/preview",
        json={"paragraph_index": 7, "preferences": prefs()},
    )

    assert response.status_code == 400
    assert "outside this script" in response.json()["detail"]


def test_preview_maps_whisper_selection_onto_script_indexes(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Previewing the LAST line gives a window of script paragraphs 1..2, so
    # window index 1 is script index 2. Without the index mapping the
    # selection below (script 2) would not match the whispered window index.
    window = PARAGRAPHS[1:3]
    _use_fake_mimo(
        monkeypatch,
        FakeMiMo(
            [
                compliant_json(
                    window,
                    tts_texts=[window[0], tagged(window[1], ["whispers"])],
                )
            ]
        ),
    )

    response = client.post(
        f"/projects/{project_with_script}/analyze-narration/preview",
        json={
            "paragraph_index": 2,
            "preferences": prefs(
                whisper_policy="user_selected", whisper_paragraphs=[2]
            ),
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["paragraph_index"] == 2
    assert body["paragraph"]["tts_text"].startswith("[whispers]")

    # The same whisper on a line the user did NOT select is rejected.
    _use_fake_mimo(
        monkeypatch,
        FakeMiMo(
            [
                compliant_json(
                    window,
                    tts_texts=[tagged(window[0], ["whispers"]), window[1]],
                )
            ]
        ),
    )
    rejected = client.post(
        f"/projects/{project_with_script}/analyze-narration/preview",
        json={
            "paragraph_index": 2,
            "preferences": prefs(
                whisper_policy="user_selected", whisper_paragraphs=[2]
            ),
        },
    )
    assert rejected.status_code == 502
    assert "not selected" in rejected.json()["detail"]


# --------------------------------------------------------------------------- #
# §2 — changing preferences invalidates the generated plan
# --------------------------------------------------------------------------- #
def test_generate_audio_is_rejected_when_preferences_changed(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_fake_mimo(monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS)]))
    analysed = _analyze(client, project_with_script, prefs(preset="natural"))
    assert analysed.status_code == 200, analysed.text
    assert analysed.json()["preferences"]["preset"] == "natural"

    # The user tweaks the panel afterwards — the old plan no longer applies.
    updated = client.put(
        _prefs_url(project_with_script),
        json=prefs(preset="dramatic", acting_intensity=5),
    )
    assert updated.status_code == 200

    recorder, composer = _use_fake_audio(monkeypatch)
    response = client.post(f"/projects/{project_with_script}/generate-audio")

    assert response.status_code == 409
    assert "narration preferences" in response.json()["detail"]
    assert "analyze-narration" in response.json()["detail"]
    # Nothing was synthesised or composed.
    assert recorder.requests == []
    assert composer.calls == []
    assert not (TEST_STORAGE / project_with_script / "audio" / "audiobook.mp3").exists()


# --------------------------------------------------------------------------- #
# §5 — capability-aware voice settings
# --------------------------------------------------------------------------- #
def test_response_separates_planning_metadata_from_effective_tts_settings(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_fake_mimo(monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS)]))
    body = _analyze(client, project_with_script, prefs()).json()

    # What the model CAN accept…
    model = body["voice_model"]
    assert model["model_id"] == "eleven_v4"
    assert model["supports_speed"] is False
    assert model["supports_style"] is False
    # …speed/style are declared as planning-only, never as effective.
    assert "speed" in model["planning_only_parameters"]
    assert "style" in model["planning_only_parameters"]

    # Per-paragraph request settings: only supported keys, consistent identity.
    assert len(body["tts_voice_settings"]) == 3
    for item in body["tts_voice_settings"]:
        settings = item["settings"]
        assert "speed" not in settings
        assert "style" not in settings
        assert "clone_strength" not in settings
        assert {"stability", "similarity_boost", "use_speaker_boost"} <= set(settings)

    # Descriptive metadata stays on the paragraph (separate from the request).
    assert body["paragraphs"][0]["voice_settings"]["speed"] == 1.0
    assert body["paragraphs"][0]["emotion"] in EMOTIONS
    assert body["paragraphs"][0]["narration_direction"]


# --------------------------------------------------------------------------- #
# §3 — prompt rules
# --------------------------------------------------------------------------- #
def test_system_prompt_contains_the_anti_overacting_rules() -> None:
    prompt = build_system_prompt(None)

    required = (
        "Fear does not automatically mean whispering.",
        "Suspense does not automatically mean breathless delivery.",
        "Sadness does not automatically mean crying.",
        "Anger does not automatically mean shouting.",
        "Most of the narration must sound like a natural human being telling a story",
        "Emotional performance stays SUBTLE",
        "Do NOT insert an audio tag merely because a paragraph has an emotion label.",
        "Avoid excessive whisper, hushed, trembling, breathless, shouting and dramatic instructions",
        "Create an audio tag ONLY when there is an intentional, audible delivery change",
    )
    for rule in required:
        assert rule in prompt, rule


def test_system_prompt_renders_the_preferences_and_capabilities() -> None:
    preferences = NarrationPreferences(
        preset="restrained_cinematic",
        acting_intensity=3,
        whisper_policy="user_selected",
        whisper_paragraphs=[1],
        tag_density="standard",
        custom_direction="Never raise the voice.",
    )
    prompt = build_system_prompt(preferences, paragraph_count=3)

    assert "NARRATION PREFERENCES" in prompt
    assert "Restrained Cinematic" in prompt
    assert "acting_intensity: 3/5" in prompt
    assert "ONLY in paragraph index(es) [1]" in prompt
    assert "audio tag density: Standard" in prompt
    assert "at most 3 audio tag(s) per paragraph" in prompt
    assert "Never raise the voice." in prompt
    # eleven_v4: unsupported knobs are never presented as effective controls.
    assert "speed is NOT supported by eleven_v4" in prompt
    assert "style is NOT supported by eleven_v4" in prompt
    # …while the ones that ARE sent are spelled out.
    assert "stability 0.0..1.0" in prompt


# --------------------------------------------------------------------------- #
# §1 — the static Narration Preferences panel
# --------------------------------------------------------------------------- #
def test_frontend_panel_is_served_to_browsers(client: TestClient) -> None:
    response = client.get(
        "/", headers={"Accept": "text/html,application/xhtml+xml"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    for marker in (
        "Create your audiobook",
        "Your script",
        "Narration style",
        "Generate Audiobook",
        "Natural",
        "Cinematic",
        "Expressive",
    ):
        assert marker in html, marker

    # API clients keep the documented JSON welcome message on the same URL.
    api = client.get("/")
    assert api.headers["content-type"].startswith("application/json")
    assert "message" in api.json()
    assert "docs_url" in api.json()

    # Static assets are served under /static (mounted in app.main).
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_frontend_sends_the_complete_preferences_object(client: TestClient) -> None:
    script = client.get("/static/app.js").text

    # Persisted per project and sent with the plan request…
    assert "/narration-preferences" in script
    assert "analyze-narration" in script
    # …with every field of the preferences object (the style presets map
    # onto the complete NarrationPreferences schema).
    for field in (
        "preset",
        "acting_intensity",
        "whisper_policy",
        "whisper_paragraphs",
        "tag_density",
        "custom_direction",
    ):
        assert field in script, field
