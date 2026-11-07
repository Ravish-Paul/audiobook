"""Delivery-style enum compatibility tests — MiMo and ElevenLabs always mocked.

Reproduces and guards the reported failure: MiMo returned an out-of-enum
`paragraphs[3].delivery_style` and the whole plan was rejected after 3
attempts. Covers:

* Natural, Restrained Cinematic and Expressive presets end to end;
* every valid `delivery_style` accepted, invalid ones rejected;
* MiMo returning "cinematic" (a preset name) as `delivery_style` — never
  normalized into a style, but corrected through targeted retry feedback;
* unknown values rejected after bounded retries with a user-friendly,
  structured error (no stack traces);
* the defensive normalization layer (trim / casing / safe aliases only);
* preferences preserved unchanged in the final plan;
* no ElevenLabs generation when plan validation fails;
* every successful plan contains only allowed `delivery_style` values;
* the original raw MiMo response preserved for debugging.

No real credits are spent.
"""

from __future__ import annotations

import json
from pathlib import Path

import dataclasses
import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import DEFAULT_ELEVENLABS_BASE_URL, get_settings
from app.main import app
from app.routes import audio as audio_routes
from app.routes import narration as narration_routes
from app.schemas.narration import DELIVERY_STYLES, NarrationPlan
from app.services import AudioService, ElevenLabsClient, NarrationPlanner
from app.services.narration_planner import (
    _format_validation_error,
    normalize_delivery_style,
    normalize_plan_payload,
    split_paragraphs,
)
from fakes import FAKE_MP3, FakeComposer, FakeMiMo

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"

# Four paragraphs so `paragraphs.3.delivery_style` exists — the exact field
# from the reported error.
SCRIPT = (
    "The lane was empty at midnight.\n\n"
    "Somewhere a dog barked twice.\n\n"
    "She counted her steps to the iron gate.\n\n"
    "By dawn the whole street would know."
)
PARAGRAPHS = split_paragraphs(SCRIPT)
assert len(PARAGRAPHS) == 4

# Styles proven compliant with every preset's policy budget (see
# test_narration_preferences: "dramatic" counts against the dramatic budget).
SAFE_STYLES = ("calm", "conversational", "soft", "reflective", "slow", "mysterious")

PRESETS = {
    "natural": {
        "preset": "natural",
        "acting_intensity": 2,
        "whisper_policy": "never",
        "whisper_paragraphs": [],
        "tag_density": "minimal",
        "custom_direction": "",
    },
    "cinematic": {
        "preset": "restrained_cinematic",
        "acting_intensity": 2,
        "whisper_policy": "user_selected",
        "whisper_paragraphs": [],
        "tag_density": "minimal",
        "custom_direction": "",
    },
    "expressive": {
        "preset": "expressive",
        "acting_intensity": 3,
        "whisper_policy": "user_selected",
        "whisper_paragraphs": [],
        "tag_density": "standard",
        "custom_direction": "",
    },
}


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def project_with_script(client: TestClient) -> str:
    response = client.post("/projects", json={"name": "Delivery Style Test"})
    assert response.status_code == 201, response.text
    project_id = response.json()["project_id"]
    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", SCRIPT.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201
    return project_id


def plan_with_styles(
    styles: list[str],
    *,
    texts: list[str] | None = None,
    intensity: int = 2,
    language: str = "English",
) -> dict:
    """A policy-compliant plan whose `delivery_style`s are exactly `styles`."""
    texts = texts if texts is not None else PARAGRAPHS
    paragraphs = []
    for index, text in enumerate(texts):
        paragraphs.append(
            {
                "paragraph_index": index,
                "original_text": text,
                "tts_text": text,
                "emotion": "neutral",
                "delivery_style": styles[index],
                "intensity": intensity,
                "narration_direction": (
                    f"Read paragraph {index} as natural, controlled storytelling."
                ),
                "voice_settings": {
                    "speed": 1.0,
                    "stability": 0.6,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
                "pause_after_ms": 300 + 100 * index,
            }
        )
    return {
        "overall_story_tone": "quiet suspense",
        "language": language,
        "paragraphs": paragraphs,
    }


def _use_fake_mimo(monkeypatch: pytest.MonkeyPatch, fake: FakeMiMo) -> FakeMiMo:
    monkeypatch.setattr(
        narration_routes,
        "_planner",
        lambda: NarrationPlanner(get_settings(), mimo_client=fake),
    )
    return fake


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(
            200, content=FAKE_MP3, headers={"content-type": "audio/mpeg"},
            request=request,
        )


def _use_fake_audio(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    settings = dataclasses.replace(
        get_settings(),
        elevenlabs_api_key="sk-style-test",
        elevenlabs_voice_id="voice-style",
        elevenlabs_base_url=DEFAULT_ELEVENLABS_BASE_URL,
    )
    recorder = _Recorder()
    service = AudioService(
        settings,
        tts_client=ElevenLabsClient(settings, transport=httpx.MockTransport(recorder)),
        composer=FakeComposer(),
    )
    monkeypatch.setattr(audio_routes, "_audio_service", lambda: service)
    return recorder


def _analyze(client: TestClient, project_id: str, preferences: dict):
    return client.post(
        f"/projects/{project_id}/analyze-narration", json={"preferences": preferences}
    )


def _debug_payloads(project_id: str) -> list[dict]:
    debug_dir = TEST_STORAGE / project_id / "mimo_debug"
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(debug_dir.glob("*.json"))
    ] if debug_dir.is_dir() else []


# --------------------------------------------------------------------------- #
# Enum: valid and invalid values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("style", DELIVERY_STYLES)
def test_every_allowed_delivery_style_is_valid(style: str) -> None:
    data = plan_with_styles([style], texts=["Only one paragraph."])
    plan = NarrationPlan.model_validate(data)
    assert plan.paragraphs[0].delivery_style == style


@pytest.mark.parametrize(
    "bad_value",
    ["cinematic", "natural", "expressive", "whispered", "warm", "storytelling", ""],
)
def test_invalid_delivery_styles_are_rejected(bad_value: str) -> None:
    data = plan_with_styles([bad_value], texts=["Only one paragraph."])
    with pytest.raises(ValidationError):
        NarrationPlan.model_validate(data)


# --------------------------------------------------------------------------- #
# Defensive normalization (trim / casing / safe aliases only)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Calm ", "calm"),
        ("CONVERSATIONAL", "conversational"),
        ("  Soft  ", "soft"),
        ("Slow", "slow"),
        ("conversation", "conversational"),  # safe morphological alias
        ("Softly", "soft"),
    ],
)
def test_normalization_fixes_unambiguous_variants(raw: str, expected: str) -> None:
    value, changed = normalize_delivery_style(raw)
    assert value == expected
    assert changed is True


@pytest.mark.parametrize(
    "raw", ["cinematic", "natural", "expressive", "warm", "whispered", "soft-spoken"]
)
def test_normalization_never_guesses_unknown_styles(raw: str) -> None:
    # Preset names and semantic guesses stay exactly as returned — they must
    # be rejected loudly, never mapped onto an arbitrary allowed value.
    value, changed = normalize_delivery_style(raw)
    assert value == raw
    assert changed is False


def test_normalize_plan_payload_preserves_the_original_value() -> None:
    data = plan_with_styles([" CALM ", "cinematic", "conversation", "soft"])
    payload, notes = normalize_plan_payload(data)

    assert payload["paragraphs"][0]["delivery_style"] == "calm"
    assert payload["paragraphs"][1]["delivery_style"] == "cinematic"  # untouched
    assert payload["paragraphs"][2]["delivery_style"] == "conversational"
    assert payload["paragraphs"][3]["delivery_style"] == "soft"

    # Notes preserve the raw value for debugging.
    assert any("' CALM '" in note and "'calm'" in note for note in notes)
    assert any("'conversation'" in note for note in notes)
    assert not any("cinematic" in note for note in notes)
    # The global preferences (when echoed) are never touched by normalization.
    data_with_prefs = {
        **plan_with_styles(["calm"], texts=["Only one paragraph."]),
        "preferences": {"preset": "cinematic"},
    }
    payload2, _ = normalize_plan_payload(data_with_prefs)
    assert payload2["preferences"] == {"preset": "cinematic"}


# --------------------------------------------------------------------------- #
# Targeted validation feedback
# --------------------------------------------------------------------------- #
def test_validation_feedback_names_index_field_value_and_enum() -> None:
    styles = ["calm", "conversational", "soft", "cinematic"]  # bad at index 3
    data = plan_with_styles(styles)
    with pytest.raises(ValidationError) as exc:
        NarrationPlan.model_validate(data)

    problems = _format_validation_error(exc.value)
    feedback = " ".join(problems)
    assert "paragraphs.3.delivery_style" in feedback  # exact field
    assert "paragraph 3" in feedback  # exact paragraph index
    assert "'cinematic'" in feedback  # the ACTUAL invalid value
    assert "Allowed values:" in feedback
    for allowed in DELIVERY_STYLES:
        assert allowed in feedback  # complete accepted enum list
    assert "Regenerate ONLY" in feedback  # regenerate just this field
    assert "keep every other plan value unchanged" in feedback


# --------------------------------------------------------------------------- #
# All presets end to end (fake MiMo)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("preset", ["natural", "cinematic", "expressive"])
def test_all_presets_produce_valid_plans(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
    preset: str,
) -> None:
    preferences = PRESETS[preset]
    intensity = preferences["acting_intensity"]
    styles = [SAFE_STYLES[i % len(SAFE_STYLES)] for i in range(len(PARAGRAPHS))]
    fake = _use_fake_mimo(
        monkeypatch,
        FakeMiMo([json.dumps(plan_with_styles(styles, intensity=intensity))]),
    )

    response = _analyze(client, project_with_script, preferences)

    assert response.status_code == 200, response.text
    body = response.json()
    produced = [p["delivery_style"] for p in body["paragraphs"]]
    assert set(produced) <= set(DELIVERY_STYLES)  # only allowed values
    # The global preferences are preserved unchanged in the final plan.
    assert body["preferences"] == preferences
    assert fake.call_count == 1


def test_every_successful_plan_uses_only_allowed_styles(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Across presets and styles, a successful plan never leaves the enum."""
    seen: set[str] = set()
    for round_index, preset in enumerate(("natural", "cinematic", "expressive")):
        project = client.post("/projects", json={"name": f"styles-{round_index}"})
        project_id = project.json()["project_id"]
        client.post(
            f"/projects/{project_id}/upload-script",
            files={"file": ("script.txt", SCRIPT.encode("utf-8"), "text/plain")},
        )
        styles = list(SAFE_STYLES[round_index:]) + list(SAFE_STYLES[:round_index])
        styles = [styles[i] for i in range(len(PARAGRAPHS))]
        preferences = PRESETS[preset]
        _use_fake_mimo(
            monkeypatch,
            FakeMiMo(
                [json.dumps(plan_with_styles(styles, intensity=preferences["acting_intensity"]))]
            ),
        )

        response = _analyze(client, project_id, preferences)

        assert response.status_code == 200, response.text
        for paragraph in response.json()["paragraphs"]:
            assert paragraph["delivery_style"] in DELIVERY_STYLES
            seen.add(paragraph["delivery_style"])
    assert seen  # the loop actually exercised plans


# --------------------------------------------------------------------------- #
# The reported failure: "cinematic" as delivery_style
# --------------------------------------------------------------------------- #
def test_preset_name_as_delivery_style_is_corrected_on_retry(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reproduces the field from the bug report exactly: paragraphs.3 with a
    # global preset name where a per-paragraph style belongs.
    styles = ["calm", "conversational", "soft", "cinematic"]
    rejected = plan_with_styles(styles)
    fixed = plan_with_styles(["calm", "conversational", "soft", "mysterious"])
    fake = _use_fake_mimo(
        monkeypatch,
        FakeMiMo(
            [json.dumps(rejected), json.dumps(fixed)]
        ),
    )

    response = _analyze(client, project_with_script, PRESETS["cinematic"])

    assert response.status_code == 200, response.text
    assert fake.call_count == 2  # invalid value corrected during retry

    retry_feedback = fake.calls[1]["user"]
    assert "REJECTED" in retry_feedback
    assert "paragraphs.3.delivery_style" in retry_feedback
    assert "'cinematic'" in retry_feedback  # the actual invalid value
    assert "Allowed values:" in retry_feedback
    assert "Regenerate ONLY" in retry_feedback

    body = response.json()
    assert body["paragraphs"][3]["delivery_style"] == "mysterious"
    assert {p["delivery_style"] for p in body["paragraphs"]} <= set(DELIVERY_STYLES)
    # Preferences survive the retry unchanged.
    assert body["preferences"] == PRESETS["cinematic"]

    # The original raw MiMo response is preserved for debugging.
    preserved = " ".join(
        raw for payload in _debug_payloads(project_with_script)
        for raw in payload.get("raw_attempts", [])
    )
    assert "cinematic" in preserved


def test_sloppy_casing_is_normalized_before_validation(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    styles = [" CALM ", "Conversational", "Softly", "slow"]
    fake = _use_fake_mimo(
        monkeypatch, FakeMiMo([json.dumps(plan_with_styles(styles))])
    )

    response = _analyze(client, project_with_script, PRESETS["natural"])

    assert response.status_code == 200, response.text
    produced = [p["delivery_style"] for p in response.json()["paragraphs"]]
    assert produced == ["calm", "conversational", "soft", "slow"]
    assert fake.call_count == 1  # no retry needed for unambiguous variants


# --------------------------------------------------------------------------- #
# Unknown values are rejected after bounded retries
# --------------------------------------------------------------------------- #
def test_unknown_style_is_rejected_after_bounded_retries(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = plan_with_styles(["calm", "conversational", "soft", "whispered"])
    fake = _use_fake_mimo(monkeypatch, FakeMiMo([json.dumps(bad)]))  # never fixes it

    response = _analyze(client, project_with_script, PRESETS["cinematic"])

    assert response.status_code == 502
    body = response.json()
    # User-friendly, structured error — no stack traces.
    assert "Unable to prepare the narration plan" in body["detail"]
    assert "no audio was generated" in body["detail"]
    assert "Traceback" not in body["detail"]
    assert body["attempts"] == get_settings().mimo_max_attempts  # bounded
    feedback = " ".join(body["problems"])
    assert "'whispered'" in feedback  # the actual invalid value
    assert "paragraphs.3.delivery_style" in feedback
    for allowed in DELIVERY_STYLES:
        assert allowed in feedback

    assert fake.call_count == get_settings().mimo_max_attempts  # never silent
    assert "REJECTED" in fake.calls[1]["user"]
    # Raw responses preserved for debugging even on failure.
    preserved = " ".join(
        raw for payload in _debug_payloads(project_with_script)
        for raw in payload.get("raw_attempts", [])
    )
    assert "whispered" in preserved


def test_no_elevenlabs_generation_when_plan_validation_fails(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _use_fake_audio(monkeypatch)
    bad = plan_with_styles(["calm", "conversational", "soft", "cinematic"])
    _use_fake_mimo(monkeypatch, FakeMiMo([json.dumps(bad)]))

    response = _analyze(client, project_with_script, PRESETS["cinematic"])

    assert response.status_code == 502
    assert recorder.requests == []  # ElevenLabs was never called
    assert not (TEST_STORAGE / project_with_script / "narration_plan.json").exists()
    assert not (
        TEST_STORAGE / project_with_script / "audio" / "audiobook.mp3"
    ).exists()
