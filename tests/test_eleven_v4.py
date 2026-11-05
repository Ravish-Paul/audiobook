"""Advanced ElevenLabs v4 narration engine tests — zero credits spent.

Everything is mocked: ElevenLabs via `httpx.MockTransport`, FFmpeg via
`FakeComposer`, MiMo via directly saved narration plans / `FakeMiMo`.

Covered here:
* model capability verification (speed/style/clone_strength filtering),
* inline audio tags on Hindi scripts (whisper / suspense transitions),
* paragraph continuity (neighbour text first, request IDs on regeneration),
* pronunciation dictionary configuration,
* text normalization defaults + the Hindi language-normalization ban,
* audio effects + the official effects catalog,
* configurable output formats (MP3 44.1 kHz + PCM/WAV intermediates),
* validation of every MiMo TTS instruction before API submission.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import (
    DEFAULT_ELEVENLABS_BASE_URL,
    get_settings,
    parse_audio_effects,
    parse_pronunciation_dictionaries,
)
from app.main import app
from app.routes import audio as audio_routes
from app.routes import elevenlabs as elevenlabs_routes
from app.schemas.narration import NarrationPlan
from app.services import AudioService, ElevenLabsClient, ProjectService
from app.services.audio_tags import (
    spoken_words_equal,
    strip_tags,
    validate_tts_text,
)
from app.services.eleven_models import get_capabilities
from app.services.elevenlabs_client import SynthesisContext
from app.services.errors import ElevenLabsConfigError
from fakes import FAKE_MP3, FakeComposer, FakeMiMo, make_plan_dict

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"

# Default GET /v1/audio-effects/catalog response used by V4Recorder.
DEFAULT_CATALOG: dict[str, Any] = {
    "filter_presets": [
        {"filter_preset_id": "old_radio", "name": "Old Radio", "description": None}
    ],
    "environments": [
        {"environment_id": "hall", "name": "Hall", "description": None}
    ],
    "background_noises": [
        {"background_noise_id": "cafe", "name": "Cafe", "description": None}
    ],
}

HINDI_SCRIPT = (
    "रात बहुत अंधेरी थी और चारों तरफ सन्नाटा पसरा हुआ था।\n\n"
    "अचानक दरवाजे पर किसी ने दस्तक दी।\n\n"
    "वह धीरे-धीरे दरवाजे की ओर बढ़ा।"
)
HINDI_PARAGRAPHS = [
    "रात बहुत अंधेरी थी और चारों तरफ सन्नाटा पसरा हुआ था।",
    "अचानक दरवाजे पर किसी ने दस्तक दी।",
    "वह धीरे-धीरे दरवाजे की ओर बढ़ा।",
]
# MiMo's inline audio tags for the three paragraphs (whisper + suspense).
TAGGED_TTS_TEXTS = [
    f"[quiet, tense delivery] {HINDI_PARAGRAPHS[0]}",
    f"[whispers] {HINDI_PARAGRAPHS[1]} [sighs]",
    f"[low, ominous narrator voice] {HINDI_PARAGRAPHS[2]}",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _v4_settings(**overrides: Any):
    """Deterministic v4 settings with fake credentials (no real traffic)."""
    base = dict(
        elevenlabs_api_key="sk-v4-test",
        elevenlabs_voice_id="voice-v4-test",
        elevenlabs_base_url=DEFAULT_ELEVENLABS_BASE_URL,
        elevenlabs_model_id="eleven_v4",
        elevenlabs_output_format="mp3_44100_128",
        elevenlabs_intermediate_output_format="",
        elevenlabs_language_code="",
        elevenlabs_apply_text_normalization="auto",
        elevenlabs_apply_language_text_normalization=False,
        elevenlabs_pronunciation_dictionaries=(),
        elevenlabs_audio_effects=None,
        elevenlabs_voice_is_pvc=False,
        elevenlabs_clone_strength=None,
    )
    base.update(overrides)
    return dataclasses.replace(get_settings(), **base)


class V4Recorder:
    """Fake ElevenLabs endpoint recording body, URL and `request-id` header."""

    def __init__(self, fail_text: Optional[str] = None, status: int = 500) -> None:
        self.requests: list[dict[str, Any]] = []
        self.fail_text = fail_text
        self.status = status
        self._counter = 0
        self.catalog_fetches = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        # The client validates configured audio effects against the official
        # catalog before the first synthesis request.
        if request.method == "GET" and request.url.path.endswith(
            "/audio-effects/catalog"
        ):
            self.catalog_fetches += 1
            return httpx.Response(200, json=DEFAULT_CATALOG, request=request)

        body = json.loads(request.content)
        self._counter += 1
        request_id = f"req-{self._counter:04d}-mock"
        self.requests.append(
            {
                "body": body,
                "url": str(request.url),
                "path": request.url.path,
                "params": dict(request.url.params),
                "request_id": request_id,
            }
        )
        if self.fail_text is not None and body.get("text") == self.fail_text:
            return httpx.Response(
                self.status,
                json={"detail": "simulated upstream failure"},
                request=request,
            )
        return httpx.Response(
            200,
            content=FAKE_MP3,
            headers={"content-type": "audio/mpeg", "request-id": request_id},
            request=request,
        )


def _env_settings():
    """Current env settings + fake credentials (for env-driven tests)."""
    return dataclasses.replace(
        get_settings(),
        elevenlabs_api_key="sk-env-test",
        elevenlabs_voice_id="voice-env-test",
        elevenlabs_base_url=DEFAULT_ELEVENLABS_BASE_URL,
    )


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def project_with_script(client: TestClient) -> str:
    project_id = client.post("/projects", json={"name": "V4 Test"}).json()[
        "project_id"
    ]
    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", HINDI_SCRIPT.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201
    return project_id


def _wire_service(
    monkeypatch: pytest.MonkeyPatch,
    recorder: V4Recorder,
    *,
    settings=None,
    composer: Optional[FakeComposer] = None,
) -> FakeComposer:
    """Route generate-audio through a mocked ElevenLabs + composer."""
    settings = settings or _v4_settings()
    composer = composer or FakeComposer()
    service = AudioService(
        settings,
        tts_client=ElevenLabsClient(settings, transport=httpx.MockTransport(recorder)),
        composer=composer,
    )
    monkeypatch.setattr(audio_routes, "_audio_service", lambda: service)
    return composer


def _save_plan(
    project_id: str,
    texts=None,
    tts_texts: Optional[list[str]] = None,
    language: str = "Hindi",
) -> NarrationPlan:
    texts = texts if texts is not None else HINDI_PARAGRAPHS
    plan = NarrationPlan.model_validate(
        make_plan_dict(texts, tts_texts=tts_texts, language=language)
    )
    ProjectService(get_settings()).save_narration_plan(project_id, plan)
    return plan


def _payload(**overrides: Any) -> ElevenLabsClient:
    settings = _v4_settings(**overrides)
    return ElevenLabsClient(settings)


# --------------------------------------------------------------------------- #
# 1. Model configuration + capability verification
# --------------------------------------------------------------------------- #
def test_default_model_is_eleven_v4() -> None:
    assert get_settings().elevenlabs_model_id == "eleven_v4"
    assert get_capabilities("eleven_v4").supports_audio_tags is True


def test_voice_id_and_model_id_are_configurable() -> None:
    settings = _v4_settings(elevenlabs_voice_id="voice-XYZ", elevenlabs_model_id="eleven_v4_turbo")
    client = ElevenLabsClient(settings)
    assert client.endpoint.endswith("/text-to-speech/voice-XYZ")
    assert client.capabilities.model_id == "eleven_v4_turbo"


def test_v4_voice_settings_exclude_speed_and_style() -> None:
    client = _payload()
    merged = client._merge_voice_settings(
        {"speed": 1.2, "stability": 0.4, "similarity_boost": 0.8, "style": 0.9,
         "use_speaker_boost": True, "emotion": "suspense"}  # emotion must vanish
    )
    assert "speed" not in merged       # eleven_v4 has no speed slider
    assert "style" not in merged       # eleven_v4 has no style slider
    assert "emotion" not in merged     # narration metadata is never an API field
    assert merged["stability"] == 0.4
    assert merged["similarity_boost"] == 0.8
    assert merged["use_speaker_boost"] is True


def test_speed_and_style_survive_for_models_that_support_them() -> None:
    client = _payload(elevenlabs_model_id="eleven_multilingual_v2")
    merged = client._merge_voice_settings(
        {"speed": 1.1, "stability": 0.3, "style": 0.5, "similarity_boost": 0.7,
         "use_speaker_boost": False}
    )
    assert merged["speed"] == 1.1
    assert merged["style"] == 0.5
    assert merged["use_speaker_boost"] is False


def test_clone_strength_requires_pvc_flag_and_v4_model() -> None:
    plan_settings = {"stability": 0.5, "similarity_boost": 0.75, "clone_strength": 0.4}

    # flag off -> never sent (voice is not an eligible v4 PVC)
    assert "clone_strength" not in _payload()._merge_voice_settings(plan_settings)

    # flag on + v4 -> sent
    pvc = _payload(elevenlabs_voice_is_pvc=True)
    assert pvc._merge_voice_settings(plan_settings)["clone_strength"] == 0.4

    # flag on + non-v4 model -> still dropped (capability first)
    legacy = _payload(
        elevenlabs_voice_is_pvc=True, elevenlabs_model_id="eleven_multilingual_v2"
    )
    assert "clone_strength" not in legacy._merge_voice_settings(plan_settings)

    # config-level default applies when the plan does not set one
    default = _payload(elevenlabs_voice_is_pvc=True, elevenlabs_clone_strength=0.25)
    assert default._merge_voice_settings({"stability": 0.5})["clone_strength"] == 0.25


def test_invalid_output_format_is_rejected_locally() -> None:
    client = _payload(elevenlabs_output_format="mp3_99999_999")
    with pytest.raises(ElevenLabsConfigError) as exc:
        client._require_configuration()
    assert "ELEVENLABS_OUTPUT_FORMAT" in str(exc.value)
    assert "mp3_44100_128" in str(exc.value)  # helpful allowed-values message


def test_invalid_text_normalization_is_rejected_locally() -> None:
    client = _payload(elevenlabs_apply_text_normalization="sometimes")
    with pytest.raises(ElevenLabsConfigError) as exc:
        client._require_configuration()
    assert "APPLY_TEXT_NORMALIZATION" in str(exc.value)


# --------------------------------------------------------------------------- #
# 2. Audio tag validation (every TTS instruction before submission)
# --------------------------------------------------------------------------- #
def test_hindi_tts_text_with_tags_is_valid() -> None:
    assert validate_tts_text(HINDI_PARAGRAPHS[0], TAGGED_TTS_TEXTS[0]) == []


def test_tts_text_may_be_absent() -> None:
    assert validate_tts_text(HINDI_PARAGRAPHS[0], None) == []


def test_tts_text_that_changes_spoken_words_is_rejected() -> None:
    problems = validate_tts_text(HINDI_PARAGRAPHS[0], "[whispers] रात बहुत अंधेरी थी।")
    assert problems and "EXACTLY" in problems[0]


def test_tts_text_with_spoken_instructions_outside_brackets_is_rejected() -> None:
    # A director note written as prose would be SPOKEN by ElevenLabs.
    problems = validate_tts_text(
        HINDI_PARAGRAPHS[1], f"whisper this line: {HINDI_PARAGRAPHS[1]}"
    )
    assert problems and "spoken words" in problems[0]


def test_ssml_break_tags_are_rejected() -> None:
    problems = validate_tts_text(
        HINDI_PARAGRAPHS[1],
        f'{HINDI_PARAGRAPHS[1]}<break time="1.5s" /> [whispers]',
    )
    assert problems and any("SSML" in p for p in problems)


def test_too_many_tags_are_rejected() -> None:
    text = " ".join(["[whispers]"] * 9) + HINDI_PARAGRAPHS[2]
    problems = validate_tts_text(HINDI_PARAGRAPHS[2], text)
    assert problems and any("at most" in p for p in problems)


def test_tags_with_code_like_content_are_rejected() -> None:
    problems = validate_tts_text(
        HINDI_PARAGRAPHS[2], f"[pause 3s] {HINDI_PARAGRAPHS[2]}"
    )
    assert problems and any("delivery cue" in p for p in problems)


def test_strip_tags_keeps_brackets_that_belong_to_the_script() -> None:
    original = "He read the sign: [EXIT]"
    assert strip_tags(original, preserve=original) == original
    assert spoken_words_equal(original, f"[whispers] {original}")
    # ...while a real directive tag is removed from the spoken words.
    assert strip_tags(f"[whispers] {original}", preserve=original).strip() == original


def test_pydantic_rejects_invalid_tts_text() -> None:
    bad = make_plan_dict(HINDI_PARAGRAPHS, tts_texts=TAGGED_TTS_TEXTS)
    bad["paragraphs"][0]["tts_text"] = "[whispers] totally different words"
    with pytest.raises(Exception) as exc:
        NarrationPlan.model_validate(bad)
    assert "tts_text" in str(exc.value)


# --------------------------------------------------------------------------- #
# 3. MiMo planner: generates tts_text, validates it, retries on bad output
# --------------------------------------------------------------------------- #
def _use_fake_mimo(monkeypatch: pytest.MonkeyPatch, fake: FakeMiMo) -> None:
    from app.routes import narration as narration_routes
    from app.services import NarrationPlanner

    monkeypatch.setattr(
        narration_routes,
        "_planner",
        lambda: NarrationPlanner(get_settings(), mimo_client=fake),
    )


def test_planner_accepts_plan_with_inline_audio_tags(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = make_plan_dict(HINDI_PARAGRAPHS, tts_texts=TAGGED_TTS_TEXTS)
    _use_fake_mimo(monkeypatch, FakeMiMo([json.dumps(good, ensure_ascii=False)]))

    response = client.post(f"/projects/{project_with_script}/analyze-narration")
    assert response.status_code == 200, response.text
    body = response.json()

    # original_text untouched, tts_text carries the tags.
    assert [p["original_text"] for p in body["paragraphs"]] == HINDI_PARAGRAPHS
    assert body["paragraphs"][0]["tts_text"] == TAGGED_TTS_TEXTS[0]
    assert "[whispers]" in body["paragraphs"][1]["tts_text"]


def test_planner_rejects_and_retries_when_tts_text_changes_words(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = make_plan_dict(HINDI_PARAGRAPHS, tts_texts=TAGGED_TTS_TEXTS)
    bad["paragraphs"][1]["tts_text"] = "[whispers] अचानक दरवाजे पर दस्तक दी।"  # words dropped
    good = make_plan_dict(HINDI_PARAGRAPHS, tts_texts=TAGGED_TTS_TEXTS)
    fake = FakeMiMo([json.dumps(bad, ensure_ascii=False), json.dumps(good, ensure_ascii=False)])
    _use_fake_mimo(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/analyze-narration")
    assert response.status_code == 200, response.text

    assert fake.call_count == 2
    assert "tts_text" in fake.calls[1]["user"]  # feedback names the problem
    assert response.json()["paragraphs"][1]["tts_text"] == TAGGED_TTS_TEXTS[1]


def test_planner_rejects_ssml_in_tts_text(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = make_plan_dict(HINDI_PARAGRAPHS, tts_texts=TAGGED_TTS_TEXTS)
    bad["paragraphs"][0]["tts_text"] = (
        f'{HINDI_PARAGRAPHS[0]}<break time="1s" />'
    )
    good = make_plan_dict(HINDI_PARAGRAPHS, tts_texts=TAGGED_TTS_TEXTS)
    fake = FakeMiMo([json.dumps(bad, ensure_ascii=False), json.dumps(good, ensure_ascii=False)])
    _use_fake_mimo(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/analyze-narration")
    assert response.status_code == 200, response.text
    assert fake.call_count == 2
    assert "SSML" in fake.calls[1]["user"]


# --------------------------------------------------------------------------- #
# 4. Inline tags reach ElevenLabs (whisper / suspense transitions)
# --------------------------------------------------------------------------- #
def test_hindi_audio_tags_are_sent_to_elevenlabs(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_plan(project_with_script, tts_texts=TAGGED_TTS_TEXTS)
    recorder = V4Recorder()
    _wire_service(monkeypatch, recorder)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text
    assert len(recorder.requests) == 3

    texts = [r["body"]["text"] for r in recorder.requests]
    assert texts[0].startswith("[quiet, tense delivery]")
    assert texts[1].startswith("[whispers]")
    assert texts[1].endswith("[sighs]")
    assert texts[2].startswith("[low, ominous narrator voice]")
    # The spoken Hindi words are all still there, byte-for-byte.
    for index, original in enumerate(HINDI_PARAGRAPHS):
        assert original in texts[index]

    # v4 model + normalization defaults on every request.
    for request in recorder.requests:
        assert request["body"]["model_id"] == "eleven_v4"
        assert request["body"]["apply_text_normalization"] == "auto"
        assert "speed" not in request["body"]["voice_settings"]
        assert "style" not in request["body"]["voice_settings"]


def test_audio_tags_are_stripped_for_models_without_tag_support() -> None:
    client = _payload(elevenlabs_model_id="eleven_multilingual_v2")
    payload = client._build_payload(
        HINDI_PARAGRAPHS[1],
        {"stability": 0.5},
        tts_text=TAGGED_TTS_TEXTS[1],
    )
    # Tags would be SPOKEN as words by this model -> removed.
    assert payload["text"] == HINDI_PARAGRAPHS[1]
    assert "[whispers]" not in payload["text"]


# --------------------------------------------------------------------------- #
# 5. Paragraph continuity (text first-time, request IDs on regeneration)
# --------------------------------------------------------------------------- #
def test_first_time_generation_uses_neighbour_text_not_request_ids(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_plan(project_with_script)
    recorder = V4Recorder()
    _wire_service(monkeypatch, recorder)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text
    bodies = [r["body"] for r in recorder.requests]

    # First paragraph: only a NEXT neighbour.
    assert "previous_text" not in bodies[0]
    assert "previous_request_ids" not in bodies[0]
    assert bodies[0].get("next_text") == HINDI_PARAGRAPHS[1]
    assert "next_request_ids" not in bodies[0]

    # Middle paragraph: both neighbours as plain text (ordinary first run).
    assert bodies[1].get("previous_text") == HINDI_PARAGRAPHS[0]
    assert bodies[1].get("next_text") == HINDI_PARAGRAPHS[2]
    assert "previous_request_ids" not in bodies[1]
    assert "next_request_ids" not in bodies[1]

    # Last paragraph: only a PREVIOUS neighbour.
    assert bodies[2].get("previous_text") == HINDI_PARAGRAPHS[1]
    assert "previous_request_ids" not in bodies[2]
    assert "next_text" not in bodies[2]
    assert "next_request_ids" not in bodies[2]

    # Every response carried a request-id which was stored for later runs.
    metadata = json.loads(
        (TEST_STORAGE / project_with_script / "audio" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    stored_ids = [seg["request_id"] for seg in metadata["segments"]]
    assert stored_ids == [r["request_id"] for r in recorder.requests]


def test_regeneration_of_middle_segment_uses_request_ids_only(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_plan(project_with_script)
    first_run = V4Recorder()
    _wire_service(monkeypatch, first_run)
    assert (
        client.post(f"/projects/{project_with_script}/generate-audio").status_code
        == 200
    )

    # Delete ONLY the middle segment -> the next run must regenerate it.
    middle = TEST_STORAGE / project_with_script / "audio" / "segments" / "paragraph_0002.mp3"
    middle.unlink()

    second_run = V4Recorder()
    _wire_service(monkeypatch, second_run)
    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text

    # Only paragraph 2 was re-saved; segments 1 and 3 were not re-billed.
    assert len(second_run.requests) == 1
    body = second_run.requests[0]["body"]
    assert body["text"] == HINDI_PARAGRAPHS[1]

    # Continuity via stored request IDs — and NOT via text (the XOR rule).
    assert body["previous_request_ids"] == [first_run.requests[0]["request_id"]]
    assert body["next_request_ids"] == [first_run.requests[2]["request_id"]]
    assert "previous_text" not in body
    assert "next_text" not in body


def test_stale_request_ids_fall_back_to_neighbour_text() -> None:
    """IDs older than ~2 hours must not be sent (ElevenLabs expires them)."""
    service = AudioService(_v4_settings())
    ordered = NarrationPlan.model_validate(
        make_plan_dict(HINDI_PARAGRAPHS)
    ).paragraphs

    stale = {
        0: {
            "index": 0,
            "status": "generated",
            "request_id": "old-id",
            "model_id": "eleven_v4",
            "generated_at": "2020-01-01T00:00:00+00:00",
        }
    }
    context = service._build_context(
        position=1,
        ordered=ordered,
        prior_records=stale,
        using_prior_ids=True,
    )
    assert context is not None
    assert context.previous_request_ids == ()
    assert context.previous_text == HINDI_PARAGRAPHS[0]  # text fallback


def test_context_never_sends_text_and_ids_for_the_same_side() -> None:
    client = _payload()
    payload = client._build_payload(
        HINDI_PARAGRAPHS[1],
        {"stability": 0.5},
        context=SynthesisContext(
            previous_text="old text",
            next_text="new text",
            previous_request_ids=("id-a", "id-b"),
            next_request_ids=(),
        ),
    )
    # IDs win for the previous side; text must not travel alongside them.
    assert payload["previous_request_ids"] == ["id-a", "id-b"]
    assert "previous_text" not in payload
    # The next side has no IDs, so its text is used.
    assert payload["next_text"] == "new text"
    assert "next_request_ids" not in payload


def test_request_id_lists_are_capped_at_three() -> None:
    client = _payload()
    payload = client._build_payload(
        HINDI_PARAGRAPHS[1],
        {"stability": 0.5},
        context=SynthesisContext(
            previous_request_ids=("1", "2", "3", "4", "5"),
            next_request_ids=("9", "8"),
        ),
    )
    assert payload["previous_request_ids"] == ["3", "4", "5"]
    assert payload["next_request_ids"] == ["9", "8"]


def test_stitching_is_skipped_for_models_without_support() -> None:
    client = _payload(elevenlabs_model_id="eleven_v3")
    payload = client._build_payload(
        HINDI_PARAGRAPHS[1],
        {"stability": 0.5},
        context=SynthesisContext(previous_text="x", next_text="y"),
    )
    # Request Stitching is not available for eleven_v3 (docs).
    assert "previous_text" not in payload
    assert "next_text" not in payload
    # ...but audio tags still work on v3.
    assert client.capabilities.supports_audio_tags is True


# --------------------------------------------------------------------------- #
# 6. Pronunciation dictionaries
# --------------------------------------------------------------------------- #
def test_parse_pronunciation_dictionaries_from_json_env() -> None:
    locators = parse_pronunciation_dictionaries(
        ["dict-1", {"id": "dict-2", "version_id": "v9"}]
    )
    assert locators == (
        {"pronunciation_dictionary_id": "dict-1"},
        {"pronunciation_dictionary_id": "dict-2", "version_id": "v9"},
    )


def test_parse_pronunciation_dictionaries_drops_placeholders_and_caps_at_three() -> None:
    locators = parse_pronunciation_dictionaries(
        ["your_dictionary_id_here", "", "d1", "d2", "d3", "d4"]
    )
    # placeholder/empty IDs are never invented or sent; max 3 locators.
    assert len(locators) == 3
    assert all(not v.startswith("your_") for loc in locators for v in loc.values())


def test_pronunciation_locators_reach_the_payload() -> None:
    client = _payload(
        elevenlabs_pronunciation_dictionaries=(
            {"pronunciation_dictionary_id": "hindi-names"},
        )
    )
    payload = client._build_payload(HINDI_PARAGRAPHS[0], {"stability": 0.5})
    assert payload["pronunciation_dictionary_locators"] == [
        {"pronunciation_dictionary_id": "hindi-names"}
    ]


def test_pronunciation_locators_absent_when_unconfigured() -> None:
    payload = _payload()._build_payload(HINDI_PARAGRAPHS[0], {"stability": 0.5})
    assert "pronunciation_dictionary_locators" not in payload


# --------------------------------------------------------------------------- #
# 7. Text normalization (default auto; never language-normalize Hindi)
# --------------------------------------------------------------------------- #
def test_text_normalization_defaults_to_auto() -> None:
    payload = _payload()._build_payload(HINDI_PARAGRAPHS[0], {"stability": 0.5})
    assert payload["apply_text_normalization"] == "auto"


def test_text_normalization_is_configurable() -> None:
    payload = _payload(
        elevenlabs_apply_text_normalization="off"
    )._build_payload(HINDI_PARAGRAPHS[0], {"stability": 0.5})
    assert payload["apply_text_normalization"] == "off"


def test_language_text_normalization_is_never_enabled_for_hindi() -> None:
    client = _payload(
        elevenlabs_apply_language_text_normalization=True,
        elevenlabs_language_code="hi",
    )
    payload = client._build_payload(HINDI_PARAGRAPHS[0], {"stability": 0.5})
    assert payload["language_code"] == "hi"
    # Hindi is not Japanese -> the (Japanese-only) knob stays off/absent.
    assert "apply_language_text_normalization" not in payload


def test_language_text_normalization_allowed_only_for_japanese() -> None:
    client = _payload(
        elevenlabs_apply_language_text_normalization=True,
        elevenlabs_language_code="ja",
    )
    payload = client._build_payload("こんにちは", {"stability": 0.5})
    assert payload["apply_language_text_normalization"] is True


def test_language_code_not_sent_to_models_that_do_not_support_it() -> None:
    client = _payload(
        elevenlabs_model_id="eleven_multilingual_v2",
        elevenlabs_language_code="hi",
    )
    payload = client._build_payload("नमस्ते", {"stability": 0.5})
    assert "language_code" not in payload


def test_language_code_derived_from_plan_language_when_not_configured() -> None:
    # The service maps MiMo's `language` ("Hindi") to ISO 639-1 ("hi").
    service = AudioService(_v4_settings())
    plan = NarrationPlan.model_validate(make_plan_dict(HINDI_PARAGRAPHS, language="Hindi"))
    assert service._language_code(plan) == "hi"

    # ...and an unknown language is never guessed.
    unknown = NarrationPlan.model_validate(
        make_plan_dict(HINDI_PARAGRAPHS, language="Klingon")
    )
    assert service._language_code(unknown) is None

    # Configured ISO code wins over the plan-derived one.
    configured = AudioService(_v4_settings(elevenlabs_language_code="HI"))
    assert configured._language_code(unknown) == "hi"

    payload = _payload()._build_payload(
        HINDI_PARAGRAPHS[0], {"stability": 0.5}, language_code="hi"
    )
    assert payload["language_code"] == "hi"


# --------------------------------------------------------------------------- #
# 8. Audio effects + official catalog
# --------------------------------------------------------------------------- #
def test_parse_audio_effects_drops_unknown_keys_and_clamps_ranges() -> None:
    effects = parse_audio_effects(
        {
            "filter_preset_id": "phone",
            "mystery_key": "nope",
            "distance": 3.0,       # clamped to 1.0
            "pan": -4.0,           # clamped to -1.0
            "send_level": "0.5",
        }
    )
    assert effects == {"filter_preset_id": "phone", "distance": 1.0, "pan": -1.0, "send_level": 0.5}
    assert parse_audio_effects({}) is None
    assert parse_audio_effects("not-a-dict") is None


def test_configured_effects_are_sent_and_unknown_catalog_ids_dropped() -> None:
    client = _payload(
        elevenlabs_audio_effects={
            "filter_preset_id": "old_radio",
            "environment_id": "not_in_catalog",
            "background_noise_id": "cafe",
            "distance": 0.2,
        }
    )
    # Simulate a fetched catalog (GET /v1/audio-effects/catalog response).
    client._effects_catalog = {
        "filter_presets": [
            {"filter_preset_id": "old_radio", "name": "Old Radio", "description": None}
        ],
        "environments": [
            {"environment_id": "hall", "name": "Hall", "description": None}
        ],
        "background_noises": [
            {"background_noise_id": "cafe", "name": "Cafe", "description": None}
        ],
    }
    payload = client._build_payload(HINDI_PARAGRAPHS[0], {"stability": 0.5})
    effects = payload["audio_effects"]
    assert effects["filter_preset_id"] == "old_radio"
    assert effects["background_noise_id"] == "cafe"
    assert "environment_id" not in effects  # unknown ID -> never sent (404)


def test_no_effects_configured_means_no_effects_parameter() -> None:
    payload = _payload()._build_payload(HINDI_PARAGRAPHS[0], {"stability": 0.5})
    assert "audio_effects" not in payload


def test_reverb_and_pan_are_dropped_for_raw_pcm_outputs() -> None:
    client = _payload(
        elevenlabs_intermediate_output_format="pcm_44100",
        elevenlabs_audio_effects={"environment_id": "hall", "pan": -0.5, "distance": 0.3},
    )
    payload = client._build_payload(HINDI_PARAGRAPHS[0], {"stability": 0.5})
    # pcm_* output + stereo effects = API 400 -> only `distance` survives.
    assert payload["audio_effects"] == {"distance": 0.3}


def test_audio_effects_catalog_endpoint(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned = {
        "filter_presets": [
            {"filter_preset_id": "phone", "name": "Phone", "description": "tinny"}
        ],
        "environments": [
            {"environment_id": "small_room", "name": "Small Room", "description": None}
        ],
        "background_noises": [
            {"background_noise_id": "keyboard", "name": "Keyboard", "description": None}
        ],
    }

    class StubClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def fetch_audio_effects_catalog(self) -> dict:
            return canned

    monkeypatch.setattr(elevenlabs_routes, "ElevenLabsClient", StubClient)
    response = client.get("/elevenlabs/audio-effects/catalog")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["filter_presets"][0]["filter_preset_id"] == "phone"
    assert body["environments"][0]["environment_id"] == "small_room"
    assert body["background_noises"][0]["background_noise_id"] == "keyboard"


# --------------------------------------------------------------------------- #
# 9. Output quality: MP3 44.1 kHz + PCM/WAV intermediates
# --------------------------------------------------------------------------- #
def test_segment_requests_use_the_configured_intermediate_format(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _v4_settings(elevenlabs_intermediate_output_format="wav_44100")
    _save_plan(project_with_script)
    recorder = V4Recorder()
    _wire_service(monkeypatch, recorder, settings=settings)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text

    # Segments fetched as 44.1 kHz WAV (editable master)...
    for request in recorder.requests:
        assert request["params"]["output_format"] == "wav_44100"

    # ...saved with honest .wav names and merged ONCE into MP3 by FFmpeg.
    segments_dir = TEST_STORAGE / project_with_script / "audio" / "segments"
    assert (segments_dir / "paragraph_0001.wav").is_file()
    assert not (segments_dir / "paragraph_0001.mp3").exists()

    metadata = json.loads(
        (TEST_STORAGE / project_with_script / "audio" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["segment_output_format"] == "wav_44100"
    assert metadata["audiobook"]["filename"] == "audiobook.mp3"

    # The API response still reports the final MP3 format.
    assert response.json()["output_format"] == "mp3_44100_128"


def test_default_requests_mp3_44100_128(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_plan(project_with_script)
    recorder = V4Recorder()
    _wire_service(monkeypatch, recorder)
    client.post(f"/projects/{project_with_script}/generate-audio")

    for request in recorder.requests:
        assert request["params"]["output_format"] == "mp3_44100_128"


def test_single_pass_still_uses_the_final_format(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy mode writes the file directly -> no intermediate involved."""
    settings = _v4_settings(elevenlabs_intermediate_output_format="wav_44100")
    recorder = V4Recorder()
    _wire_service(monkeypatch, recorder, settings=settings)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text
    assert response.json()["generation_mode"] == "single_pass"
    assert recorder.requests[0]["params"]["output_format"] == "mp3_44100_128"


# --------------------------------------------------------------------------- #
# 10. Config plumbing through the API
# --------------------------------------------------------------------------- #
def test_pronunciation_dictionaries_from_env_reach_the_request(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ELEVENLABS_PRONUNCIATION_DICTIONARIES",
        json.dumps(["hindi-names-dict", {"id": "hard-words", "version_id": "v2"}]),
    )
    _save_plan(project_with_script)
    recorder = V4Recorder()
    _wire_service(monkeypatch, recorder, settings=_env_settings())

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text
    locators = recorder.requests[0]["body"]["pronunciation_dictionary_locators"]
    assert locators == [
        {"pronunciation_dictionary_id": "hindi-names-dict"},
        {"pronunciation_dictionary_id": "hard-words", "version_id": "v2"},
    ]


def test_audio_effects_from_env_reach_the_request(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ELEVENLABS_AUDIO_EFFECTS",
        json.dumps({"filter_preset_id": "old_radio", "distance": 0.25}),
    )
    _save_plan(project_with_script)
    recorder = V4Recorder()
    _wire_service(monkeypatch, recorder, settings=_env_settings())

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text
    # The configured effects are validated against the (mocked) official
    # catalog — `old_radio` is a known ID, so it is sent.
    assert recorder.requests[0]["body"]["audio_effects"] == {
        "filter_preset_id": "old_radio",
        "distance": 0.25,
    }
    # The catalog was fetched exactly once (cached for later segments).
    assert recorder.catalog_fetches == 1


# --------------------------------------------------------------------------- #
# 11. Audio merging with intermediates (real FFmpeg, skipped if unavailable)
# --------------------------------------------------------------------------- #
def _ffmpeg_available() -> bool:
    from app.services.audio_composer import resolve_ffmpeg
    from app.services.errors import FFmpegNotFoundError

    try:
        resolve_ffmpeg(get_settings())
        return True
    except FFmpegNotFoundError:
        return False


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg is not installed")
def test_audio_merging_accepts_wav_intermediate_segments(tmp_path: Path) -> None:
    """FFmpeg merges WAV intermediates into the final audiobook once."""
    import subprocess

    from app.services.audio_composer import FFmpegAudioComposer, Segment, resolve_ffmpeg

    ffmpeg = resolve_ffmpeg(get_settings())
    paths = []
    for index in range(2):
        tone = tmp_path / f"paragraph_{index + 1:04d}.wav"
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi",
                "-i", f"sine=frequency={330 + index * 110}:duration=0.3",
                str(tone),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        paths.append(tone)

    settings = _v4_settings(elevenlabs_intermediate_output_format="wav_44100")
    output = tmp_path / "audiobook.mp3"
    FFmpegAudioComposer(settings).compose(
        [Segment(path=p, pause_after_ms=200) for p in paths], output
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert output.read_bytes()[:3] == b"ID3"  # final encode happens ONCE here
