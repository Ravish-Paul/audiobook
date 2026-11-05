"""Regression tests for `tts_text` (Eleven v4 audio tags) — zero credits.

Requirement checklist covered here:
1. `tts_text` is generated for expressive paragraphs (planner enforces it);
2. whisper tags reach ElevenLabs verbatim;
3. narrator + character delivery changes inside one paragraph are preserved;
4. stripping all tags reproduces `original_text` EXACTLY;
5. original single-pass and older-model workflows stay compatible.

Everything is mocked: ElevenLabs via `httpx.MockTransport`, MiMo via
`FakeMiMo`. No credits are ever spent.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import DEFAULT_ELEVENLABS_BASE_URL, get_settings
from app.main import app
from app.routes import audio as audio_routes
from app.routes import narration as narration_routes
from app.schemas.narration import NarrationPlan
from app.services import (
    AudioService,
    ElevenLabsClient,
    NarrationPlanner,
    ProjectService,
    require_tts_text,
)
from app.services.audio_tags import strip_tags_exact, validate_tts_text
from fakes import FAKE_MP3, FakeComposer, FakeMiMo, make_plan_dict

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"

HINDI_SCRIPT = (
    "सुमन मुस्कुराई।\n\n"
    "अचानक दरवाजे पर किसी ने दस्तक दी।\n\n"
    "वह धीरे-धीरे दरवाजे की ओर बढ़ा।"
)
HINDI_PARAGRAPHS = [
    "सुमन मुस्कुराई।",
    "अचानक दरवाजे पर किसी ने दस्तक दी।",
    "वह धीरे-धीरे दरवाजे की ओर बढ़ा।",
]

# Narration -> dialogue -> narration inside ONE paragraph: each part gets
# its own delivery instruction (requirements 3 + 5).
NARRATOR_ORIGINAL = (
    "सुमन मुस्कुराई। उसने कहा, मुझे डर लग रहा है। फिर वह चुप हो गई।"
)
NARRATOR_TTS = (
    "[quiet, tense delivery] सुमन मुस्कुराई। "
    "[sad] उसने कहा, मुझे डर लग रहा है। "
    "[sighs] फिर वह चुप हो गई।"
)
WHISPER_TTS = f"[whispers] {HINDI_PARAGRAPHS[0]}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class Recorder:
    """Fake ElevenLabs endpoint recording bodies, params and request IDs."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._counter = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith(
            "/audio-effects/catalog"
        ):
            return httpx.Response(200, json={}, request=request)
        body = json.loads(request.content)
        self._counter += 1
        request_id = f"req-{self._counter:04d}-mock"
        self.requests.append(
            {"body": body, "params": dict(request.url.params), "request_id": request_id}
        )
        return httpx.Response(
            200,
            content=FAKE_MP3,
            headers={"content-type": "audio/mpeg", "request-id": request_id},
            request=request,
        )


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def project_with_script(client: TestClient) -> str:
    project_id = client.post("/projects", json={"name": "tts_text regression"}).json()[
        "project_id"
    ]
    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", HINDI_SCRIPT.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201
    return project_id


def _settings(**overrides: Any):
    base = dict(
        elevenlabs_api_key="sk-tts-test",
        elevenlabs_voice_id="voice-tts-test",
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


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    recorder: Recorder,
    *,
    settings=None,
) -> None:
    settings = settings or _settings()
    service = AudioService(
        settings,
        tts_client=ElevenLabsClient(settings, transport=httpx.MockTransport(recorder)),
        composer=FakeComposer(),
    )
    monkeypatch.setattr(audio_routes, "_audio_service", lambda: service)


def _save_plan(project_id: str, texts=None, tts_texts=None) -> NarrationPlan:
    texts = texts if texts is not None else HINDI_PARAGRAPHS
    plan = NarrationPlan.model_validate(
        make_plan_dict(texts, tts_texts=tts_texts)
    )
    ProjectService(get_settings()).save_narration_plan(project_id, plan)
    return plan


def _use_fake_mimo(monkeypatch: pytest.MonkeyPatch, fake: FakeMiMo) -> None:
    monkeypatch.setattr(
        narration_routes,
        "_planner",
        lambda: NarrationPlanner(get_settings(), mimo_client=fake),
    )


# --------------------------------------------------------------------------- #
# 1. tts_text is generated for expressive paragraphs
# --------------------------------------------------------------------------- #
def test_planner_rejects_plan_without_tts_text_and_retries(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MiMo omitting `tts_text` must NOT silently produce a plan without it."""
    omitted = make_plan_dict(HINDI_PARAGRAPHS)
    for paragraph in omitted["paragraphs"]:
        paragraph.pop("tts_text", None)
    complete = make_plan_dict(
        HINDI_PARAGRAPHS,
        tts_texts=[
            f"[quiet, tense delivery] {HINDI_PARAGRAPHS[0]}",
            f"[whispers] {HINDI_PARAGRAPHS[1]}",
            HINDI_PARAGRAPHS[2],
        ],
    )
    fake = FakeMiMo(
        [json.dumps(omitted, ensure_ascii=False), json.dumps(complete, ensure_ascii=False)]
    )
    _use_fake_mimo(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/analyze-narration")
    assert response.status_code == 200, response.text

    # First attempt was rejected and the feedback named the missing field.
    assert fake.call_count == 2
    assert "tts_text is required" in fake.calls[1]["user"]

    # Every paragraph of the SAVED plan now carries tts_text.
    body = response.json()
    assert len(body["paragraphs"]) == 3
    for paragraph in body["paragraphs"]:
        assert paragraph["tts_text"]
    assert body["paragraphs"][0]["tts_text"].startswith("[quiet, tense delivery]")
    assert body["paragraphs"][1]["tts_text"].startswith("[whispers]")

    saved = json.loads(
        (TEST_STORAGE / project_with_script / "narration_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert all("tts_text" in p for p in saved["paragraphs"])


def test_require_tts_text_reports_each_missing_paragraph() -> None:
    plan = NarrationPlan.model_validate(make_plan_dict(HINDI_PARAGRAPHS))
    data = plan.model_dump()
    data["paragraphs"][1].pop("tts_text")
    broken = NarrationPlan.model_validate(data)  # schema allows absence…

    problems = require_tts_text(broken)  # …but the planner does not
    assert len(problems) == 1
    assert "paragraph 1" in problems[0]
    assert "tts_text is required" in problems[0]

    intact = NarrationPlan.model_validate(make_plan_dict(HINDI_PARAGRAPHS))
    assert require_tts_text(intact) == []


def test_tts_text_equal_to_original_is_valid_when_no_tags_needed() -> None:
    plan = NarrationPlan.model_validate(
        make_plan_dict(HINDI_PARAGRAPHS)  # make_plan_dict: tts_text == text
    )
    assert require_tts_text(plan) == []
    assert validate_tts_text(HINDI_PARAGRAPHS[0], HINDI_PARAGRAPHS[0]) == []


def test_plan_generation_never_silently_falls_back_when_tts_text_missing(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If MiMo NEVER returns tts_text, analysis fails loudly — no fallback.

    A new narration-plan generation must never silently proceed with
    `original_text` in place of `tts_text`: the plan is rejected after all
    attempts (502) and NOTHING is saved.
    """
    omitted = make_plan_dict(HINDI_PARAGRAPHS)
    for paragraph in omitted["paragraphs"]:
        paragraph.pop("tts_text", None)
    fake = FakeMiMo([json.dumps(omitted, ensure_ascii=False)])  # repeats forever
    _use_fake_mimo(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/analyze-narration")

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "tts_text" in detail
    assert "no audio was generated" in detail
    # Every allowed attempt was spent, with feedback each time.
    assert fake.call_count == get_settings().mimo_max_attempts
    assert all("tts_text is required" in call["user"] for call in fake.calls[1:])
    # No plan file was written — the old plan (if any) is never overwritten
    # by a fallback result.
    assert not (TEST_STORAGE / project_with_script / "narration_plan.json").exists()


# --------------------------------------------------------------------------- #
# 2. whisper tags reach ElevenLabs (verbatim — tts_text, not original_text)
# --------------------------------------------------------------------------- #
def test_whisper_tag_reaches_elevenlabs_verbatim(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _save_plan(
        project_with_script,
        tts_texts=[WHISPER_TTS, HINDI_PARAGRAPHS[1], HINDI_PARAGRAPHS[2]],
    )
    recorder = Recorder()
    _wire(monkeypatch, recorder)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text

    # The request text IS tts_text — byte-identical, tags included.
    sent = recorder.requests[0]["body"]["text"]
    assert sent == plan.paragraphs[0].tts_text
    assert sent.startswith("[whispers]")
    assert sent == WHISPER_TTS
    # ...and the untagged paragraphs go through unchanged.
    assert recorder.requests[1]["body"]["text"] == HINDI_PARAGRAPHS[1]


# --------------------------------------------------------------------------- #
# 3. narrator + character delivery changes are preserved
# --------------------------------------------------------------------------- #
def test_narrator_and_dialogue_delivery_are_preserved(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The uploaded script itself must contain the narrator+dialogue paragraph
    # (plans are always cross-checked against the stored script).
    custom_script = (
        f"{NARRATOR_ORIGINAL}\n\n{HINDI_PARAGRAPHS[1]}\n\n{HINDI_PARAGRAPHS[2]}"
    )
    upload = client.post(
        f"/projects/{project_with_script}/upload-script",
        files={"file": ("script.txt", custom_script.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201

    texts = [NARRATOR_ORIGINAL, HINDI_PARAGRAPHS[1], HINDI_PARAGRAPHS[2]]
    tts_texts = [NARRATOR_TTS, HINDI_PARAGRAPHS[1], HINDI_PARAGRAPHS[2]]
    plan = _save_plan(project_with_script, texts=texts, tts_texts=tts_texts)

    # The stored plan keeps every part of the triple-tagged paragraph.
    stored_tts = plan.paragraphs[0].tts_text
    assert stored_tts == NARRATOR_TTS
    assert stored_tts.index("[quiet, tense delivery]") < stored_tts.index("[sad]")
    assert stored_tts.index("[sad]") < stored_tts.index("[sighs]")

    recorder = Recorder()
    _wire(monkeypatch, recorder)
    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text

    # All three delivery instructions reach eleven_v4, in order …
    sent = recorder.requests[0]["body"]["text"]
    assert sent == NARRATOR_TTS
    assert sent.index("[quiet, tense delivery]") < sent.index("[sad]") < sent.index(
        "[sighs]"
    )
    # … while every spoken word stays exactly where the script put it.
    for sentence in (
        "सुमन मुस्कुराई।",
        "उसने कहा, मुझे डर लग रहा है।",
        "फिर वह चुप हो गई।",
    ):
        assert sentence in sent
    assert strip_tags_exact(sent, preserve=NARRATOR_ORIGINAL) == NARRATOR_ORIGINAL


# --------------------------------------------------------------------------- #
# 4. spoken words remain EXACTLY unchanged (tag stripping == original_text)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("tts_text", "original"),
    [
        ("[whispers] hello world", "hello world"),
        ("hello world [sighs]", "hello world"),
        ("one [laughs] two", "one two"),
        ("[sad] a [curious] b", "a b"),
        ("[quiet, tense delivery] narrator line. [shouts] Character line!", "narrator line. Character line!"),
        ("नमस्ते [whispers] दोस्तों", "नमस्ते दोस्तों"),
        ("no tags at all", "no tags at all"),
        (NARRATOR_TTS, NARRATOR_ORIGINAL),
    ],
    ids=[
        "leading", "trailing", "middle", "two-tags",
        "narrator-dialogue", "hindi", "untagged", "triple",
    ],
)
def test_stripping_tags_reproduces_original_text_exactly(
    tts_text: str, original: str
) -> None:
    assert strip_tags_exact(tts_text, preserve=original) == original
    assert validate_tts_text(original, tts_text) == []


@pytest.mark.parametrize(
    ("tts_text", "original"),
    [
        ("[whispers]  hello world", "hello world"),  # double space after tag
        ("[whispers] hello world", "Hello world"),   # changed a word
        ("hello world [laughs] extra", "hello world"),  # added spoken words
        ("hello  world [sad]", "hello world"),       # spacing changed
        ("whispered: hello world", "hello world"),   # instruction outside tags
    ],
    ids=["extra-space", "changed-word", "added-words", "spacing", "prose"],
)
def test_tts_text_that_does_not_reproduce_original_is_rejected(
    tts_text: str, original: str
) -> None:
    assert validate_tts_text(original, tts_text) != []


def test_pydantic_plan_rejects_non_exact_tts_text() -> None:
    data = make_plan_dict(HINDI_PARAGRAPHS, tts_texts=[WHISPER_TTS] * 3)
    data["paragraphs"][0]["tts_text"] = f"[whispers] {HINDI_PARAGRAPHS[0]} extra words"
    with pytest.raises(Exception) as exc:
        NarrationPlan.model_validate(data)
    assert "tts_text" in str(exc.value)


# --------------------------------------------------------------------------- #
# 5. original single-pass and older workflows remain compatible
# --------------------------------------------------------------------------- #
def test_single_pass_without_plan_still_sends_the_original_script(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No plan -> legacy whole-script request, text untouched, no tags."""
    recorder = Recorder()
    _wire(monkeypatch, recorder)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text
    assert response.json()["generation_mode"] == "single_pass"
    assert len(recorder.requests) == 1
    assert recorder.requests[0]["body"]["text"] == HINDI_SCRIPT


def test_pre_upgrade_plan_without_tts_text_still_generates_audio(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plans saved BEFORE the tts_text upgrade must keep working (409-free)."""
    old_data = make_plan_dict(HINDI_PARAGRAPHS)
    for paragraph in old_data["paragraphs"]:
        paragraph.pop("tts_text", None)
    ProjectService(get_settings()).save_narration_plan(
        project_with_script, NarrationPlan.model_validate(old_data)
    )

    recorder = Recorder()
    _wire(monkeypatch, recorder)
    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text
    assert response.json()["generation_mode"] == "narration_plan"

    # Without tts_text the original paragraph text is spoken, unchanged.
    for index, paragraph in enumerate(recorder.requests):
        assert paragraph["body"]["text"] == HINDI_PARAGRAPHS[index]

    metadata = json.loads(
        (TEST_STORAGE / project_with_script / "audio" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(seg["tts_text"] is None for seg in metadata["segments"])
    assert all(seg["has_audio_tags"] is False for seg in metadata["segments"])


def test_older_model_keeps_speed_and_style_in_the_request(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """eleven_multilingual_v2 (older workflow) still receives speed/style."""
    settings = _settings(elevenlabs_model_id="eleven_multilingual_v2")
    plan = _save_plan(
        project_with_script,
        tts_texts=[WHISPER_TTS, HINDI_PARAGRAPHS[1], HINDI_PARAGRAPHS[2]],
    )
    recorder = Recorder()
    _wire(monkeypatch, recorder, settings=settings)

    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text

    voice_settings = recorder.requests[0]["body"]["voice_settings"]
    assert voice_settings["speed"] == plan.paragraphs[0].voice_settings.speed
    assert voice_settings["style"] == plan.paragraphs[0].voice_settings.style
    assert voice_settings["stability"] == plan.paragraphs[0].voice_settings.stability

    # This model has no audio-tag support: tags are stripped so a `[whispers]`
    # cue is never SPOKEN as literal words — the words themselves are intact.
    sent = recorder.requests[0]["body"]["text"]
    assert sent == HINDI_PARAGRAPHS[0]  # the whisper-tagged paragraph, untagged
    assert "[whispers]" not in sent


def test_v4_request_omits_speed_and_style_while_plan_keeps_them(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 9: speed/style survive as planning metadata ONLY."""
    plan = _save_plan(project_with_script)  # plan voice_settings include speed/style
    assert all("speed" in p.voice_settings.model_dump() for p in plan.paragraphs)
    assert all("style" in p.voice_settings.model_dump() for p in plan.paragraphs)

    recorder = Recorder()
    _wire(monkeypatch, recorder)
    response = client.post(f"/projects/{project_with_script}/generate-audio")
    assert response.status_code == 200, response.text

    # Effective eleven_v4 request: no speed, no style.
    for request in recorder.requests:
        effective = request["body"]["voice_settings"]
        assert "speed" not in effective
        assert "style" not in effective
        assert "stability" in effective
        assert "similarity_boost" in effective

    # …and the stored metadata shows both views side by side.
    metadata = json.loads(
        (TEST_STORAGE / project_with_script / "audio" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    for segment in metadata["segments"]:
        assert "speed" in segment["voice_settings"]       # planning metadata
        assert "style" in segment["voice_settings"]
        assert "speed" not in segment["effective_voice_settings"]
        assert "style" not in segment["effective_voice_settings"]
        assert segment["effective_voice_settings"]["stability"] == (
            segment["voice_settings"]["stability"]
        )


def test_original_text_is_kept_separately_for_verification(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """original_text (untouched) and tts_text are both stored for review."""
    plan = _save_plan(
        project_with_script,
        tts_texts=[WHISPER_TTS, HINDI_PARAGRAPHS[1], HINDI_PARAGRAPHS[2]],
    )

    saved = json.loads(
        (TEST_STORAGE / project_with_script / "narration_plan.json").read_text(
            encoding="utf-8"
        )
    )
    for paragraph, expected in zip(saved["paragraphs"], HINDI_PARAGRAPHS):
        assert paragraph["original_text"] == expected          # untouched script
        assert "tts_text" in paragraph                          # delivery version
        assert strip_tags_exact(
            paragraph["tts_text"], preserve=paragraph["original_text"]
        ) == paragraph["original_text"]                         # verifiable

    # The GET endpoint exposes both fields too.
    fetched = client.get(
        f"/projects/{project_with_script}/narration-plan"
    ).json()
    assert fetched["paragraphs"][0]["original_text"] == HINDI_PARAGRAPHS[0]
    assert fetched["paragraphs"][0]["tts_text"] == plan.paragraphs[0].tts_text
