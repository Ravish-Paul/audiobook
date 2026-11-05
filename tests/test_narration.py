"""Narration analysis tests — MiMo is always mocked (no credits spent).

Covers: paragraph splitting, Hindi analysis, emotion classification,
per-paragraph voice settings, Pydantic validation, retries with feedback,
missing API key, MiMo timeouts/failures and the plan storage endpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.routes import narration as narration_routes
from app.schemas.narration import DELIVERY_STYLES, EMOTIONS, NarrationPlan
from app.services import NarrationPlanner
from app.services.errors import MiMoAPIError, MiMoTimeoutError
from app.services.narration_planner import split_paragraphs, validate_against_script
from fakes import FakeMiMo, make_plan_dict, plan_json

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"

HINDI_SCRIPT = (
    "रात बहुत अंधेरी थी और चारों तरफ सन्नाटा पसरा हुआ था।\n\n"
    "अचानक दरवाजे पर किसी ने दस्तक दी।\n\n"
    "वह धीरे-धीरे दरवाजे की ओर बढ़ा और दरवाजा खोला।"
)
HINDI_PARAGRAPHS = split_paragraphs(HINDI_SCRIPT)


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def project_with_script(client: TestClient) -> str:
    project_id = client.post("/projects", json={"name": "Narration Test"}).json()[
        "project_id"
    ]
    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", HINDI_SCRIPT.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201
    return project_id


def _use_fake_mimo(monkeypatch: pytest.MonkeyPatch, fake: FakeMiMo) -> None:
    """Route the analyze endpoint through a planner with a fake MiMo."""
    monkeypatch.setattr(
        narration_routes,
        "_planner",
        lambda: NarrationPlanner(get_settings(), mimo_client=fake),
    )


def _analyze(client: TestClient, project_id: str) -> dict:
    response = client.post(f"/projects/{project_id}/analyze-narration")
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------- #
# Paragraph splitting
# --------------------------------------------------------------------------- #
def test_split_paragraphs_on_blank_lines() -> None:
    script = "पैराग्राफ़ एक।\n\nदो।\n\n\nतीन।\n"
    assert split_paragraphs(script) == ["पैराग्राफ़ एक।", "दो।", "तीन।"]


def test_split_paragraphs_falls_back_to_lines() -> None:
    script = "line one\nline two\nline three"
    assert split_paragraphs(script) == ["line one", "line two", "line three"]


def test_split_paragraphs_ignores_blank_only_input() -> None:
    assert split_paragraphs("   \n\n  ") == []


# --------------------------------------------------------------------------- #
# Happy path — Hindi analysis
# --------------------------------------------------------------------------- #
def test_analyze_narration_success(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert len(HINDI_PARAGRAPHS) == 3
    _use_fake_mimo(monkeypatch, FakeMiMo([plan_json(HINDI_PARAGRAPHS)]))

    body = _analyze(client, project_with_script)

    assert body["project_id"] == project_with_script
    assert body["saved"] is True
    assert body["saved_at"] is not None
    assert body["language"] == "Hindi"
    assert body["overall_story_tone"] == "suspenseful"
    assert body["paragraph_count"] == 3

    # Every original paragraph survives untouched, in order.
    texts = [p["original_text"] for p in body["paragraphs"]]
    assert texts == HINDI_PARAGRAPHS

    # Emotion classification uses the controlled vocabulary.
    for paragraph in body["paragraphs"]:
        assert paragraph["emotion"] in EMOTIONS
        assert paragraph["delivery_style"] in DELIVERY_STYLES
        assert 1 <= paragraph["intensity"] <= 5
        assert 0 <= paragraph["pause_after_ms"] <= 3000
        assert paragraph["narration_direction"]
        settings = paragraph["voice_settings"]
        assert 0.7 <= settings["speed"] <= 1.2
        assert 0.0 <= settings["stability"] <= 1.0
        assert 0.0 <= settings["similarity_boost"] <= 1.0
        assert 0.0 <= settings["style"] <= 1.0
        assert isinstance(settings["use_speaker_boost"], bool)

    # Different paragraphs get different decisions (not one blanket setting).
    assert len({p["emotion"] for p in body["paragraphs"]}) > 1
    assert len({p["voice_settings"]["stability"] for p in body["paragraphs"]}) > 1
    assert len({p["pause_after_ms"] for p in body["paragraphs"]}) > 1


def test_analyze_never_modifies_the_uploaded_script(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_mimo(monkeypatch, FakeMiMo([plan_json(HINDI_PARAGRAPHS)]))
    _analyze(client, project_with_script)

    stored = (TEST_STORAGE / project_with_script / "script.txt").read_text(
        encoding="utf-8"
    )
    assert stored == HINDI_SCRIPT


def test_narration_plan_is_saved_to_disk(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_mimo(monkeypatch, FakeMiMo([plan_json(HINDI_PARAGRAPHS)]))
    _analyze(client, project_with_script)

    path = TEST_STORAGE / project_with_script / "narration_plan.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["project_id"] == project_with_script
    assert payload["saved_at"]
    assert len(payload["paragraphs"]) == 3
    assert payload["paragraphs"][0]["original_text"] == HINDI_PARAGRAPHS[0]


# --------------------------------------------------------------------------- #
# GET /projects/{id}/narration-plan
# --------------------------------------------------------------------------- #
def test_get_plan_returns_saved_plan(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_mimo(monkeypatch, FakeMiMo([plan_json(HINDI_PARAGRAPHS)]))
    analyzed = _analyze(client, project_with_script)

    response = client.get(f"/projects/{project_with_script}/narration-plan")
    assert response.status_code == 200
    body = response.json()
    assert body["paragraph_count"] == analyzed["paragraph_count"]
    assert body["paragraphs"] == analyzed["paragraphs"]
    assert body["saved_at"] == analyzed["saved_at"]


def test_get_plan_is_404_before_analysis(
    client: TestClient, project_with_script: str
) -> None:
    response = client.get(f"/projects/{project_with_script}/narration-plan")
    assert response.status_code == 404
    assert "analyze-narration" in response.json()["detail"]


def test_analyze_unknown_project_is_404(client: TestClient) -> None:
    response = client.post(
        "/projects/00000000-0000-0000-0000-000000000000/analyze-narration"
    )
    assert response.status_code == 404


def test_analyze_without_script_is_404(client: TestClient) -> None:
    project_id = client.post("/projects", json={"name": "No script"}).json()[
        "project_id"
    ]
    response = client.post(f"/projects/{project_id}/analyze-narration")
    assert response.status_code == 404
    assert "No script has been uploaded" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Configuration + MiMo failures
# --------------------------------------------------------------------------- #
def test_missing_mimo_api_key_is_500(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No fake injected: the real client must fail fast, before any network I/O.
    monkeypatch.setenv("MIMO_API_KEY", "")

    response = client.post(f"/projects/{project_with_script}/analyze-narration")
    assert response.status_code == 500
    assert "MIMO_API_KEY" in response.json()["detail"]


def test_mimo_timeout_is_504(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_mimo(
        monkeypatch, FakeMiMo([MiMoTimeoutError("MiMo did not respond within 120 seconds")])
    )

    response = client.post(f"/projects/{project_with_script}/analyze-narration")
    assert response.status_code == 504
    assert "120 seconds" in response.json()["detail"]


def test_mimo_failure_is_502(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_fake_mimo(
        monkeypatch, FakeMiMo([MiMoAPIError("Could not reach the Xiaomi MiMo API")])
    )

    response = client.post(f"/projects/{project_with_script}/analyze-narration")
    assert response.status_code == 502
    assert "Could not reach" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Validation + safe retries (never trust the LLM)
# --------------------------------------------------------------------------- #
def test_invalid_json_is_retried_then_rejected(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = get_settings().mimo_max_attempts
    fake = FakeMiMo(["this is not json at all"])
    _use_fake_mimo(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/analyze-narration")

    assert response.status_code == 502
    assert fake.call_count == attempts
    assert "not valid JSON" in response.json()["detail"]
    assert "no audio was generated" in response.json()["detail"]
    # The retry prompt carries the failure back to the model.
    assert "REJECTED" in fake.calls[1]["user"]
    assert "not valid JSON" in fake.calls[1]["user"]


def test_out_of_range_value_is_rejected_and_retried(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = make_plan_dict(HINDI_PARAGRAPHS)
    bad["paragraphs"][1]["voice_settings"]["stability"] = 1.5  # > 1.0
    fake = FakeMiMo(
        [json.dumps(bad, ensure_ascii=False), plan_json(HINDI_PARAGRAPHS)]
    )
    _use_fake_mimo(monkeypatch, fake)

    body = _analyze(client, project_with_script)  # second attempt succeeds

    assert fake.call_count == 2
    assert "stability" in fake.calls[1]["user"]
    assert body["paragraph_count"] == 3
    assert all(
        p["voice_settings"]["stability"] <= 1.0 for p in body["paragraphs"]
    )


def test_modified_original_text_is_rejected(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    tampered = make_plan_dict(HINDI_PARAGRAPHS)
    tampered["paragraphs"][0]["original_text"] = "रात बहुत अंधेरी थी।"  # paraphrased
    fake = FakeMiMo(
        [json.dumps(tampered, ensure_ascii=False), plan_json(HINDI_PARAGRAPHS)]
    )
    _use_fake_mimo(monkeypatch, fake)

    body = _analyze(client, project_with_script)

    assert fake.call_count == 2
    assert "original_text" in fake.calls[1]["user"]
    assert body["paragraphs"][0]["original_text"] == HINDI_PARAGRAPHS[0]


def test_repeatedly_invalid_plan_returns_useful_error(
    client: TestClient, project_with_script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = make_plan_dict(HINDI_PARAGRAPHS)
    bad["paragraphs"] = bad["paragraphs"][:2]  # one paragraph missing
    fake = FakeMiMo([json.dumps(bad, ensure_ascii=False)])
    _use_fake_mimo(monkeypatch, fake)

    response = client.post(f"/projects/{project_with_script}/analyze-narration")

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "missing paragraph_index" in detail
    assert str(get_settings().mimo_max_attempts) in detail
    # No plan was saved and no audio was generated.
    assert not (TEST_STORAGE / project_with_script / "narration_plan.json").exists()


def test_invalid_emotion_is_rejected_by_schema() -> None:
    bad = make_plan_dict(["only one paragraph"])
    bad["paragraphs"][0]["emotion"] = "excited"  # not in the vocabulary
    with pytest.raises(Exception) as exc:
        NarrationPlan.model_validate(bad)
    assert "emotion" in str(exc.value)


def test_intensity_and_pause_bounds_are_enforced() -> None:
    bad = make_plan_dict(["only one paragraph"])
    bad["paragraphs"][0]["intensity"] = 9
    bad["paragraphs"][0]["pause_after_ms"] = 99999
    with pytest.raises(Exception) as exc:
        NarrationPlan.model_validate(bad)
    message = str(exc.value)
    assert "intensity" in message
    assert "pause_after_ms" in message


def test_duplicate_and_unknown_indices_are_reported() -> None:
    plan = NarrationPlan.model_validate(make_plan_dict(["a", "b"]))
    plan.paragraphs[1].paragraph_index = 0  # duplicate, index 1 missing

    problems = validate_against_script(plan, ["a", "b"])
    joined = " ".join(problems)
    assert "duplicate" in joined
    assert "missing" in joined


# --------------------------------------------------------------------------- #
# Optional live test — disabled unless explicitly enabled. Costs MiMo credits.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.getenv("RUN_MIMO_LIVE_TEST") != "1",
    reason="set RUN_MIMO_LIVE_TEST=1 with valid .env credentials to run",
)
def test_live_mimo_narration_analysis(
    client: TestClient, project_with_script: str
) -> None:
    settings = get_settings()
    assert settings.mimo_configured, (
        "RUN_MIMO_LIVE_TEST=1 requires MIMO_API_KEY in .env"
    )

    response = client.post(f"/projects/{project_with_script}/analyze-narration")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["paragraph_count"] == len(HINDI_PARAGRAPHS)
    assert body["language"]
    assert body["overall_story_tone"]

    # The real model must never alter the Hindi script.
    assert [p["original_text"] for p in body["paragraphs"]] == HINDI_PARAGRAPHS
    for paragraph in body["paragraphs"]:
        assert paragraph["emotion"] in EMOTIONS
        assert paragraph["delivery_style"] in DELIVERY_STYLES

    # ...and the plan must be retrievable afterwards.
    saved = client.get(f"/projects/{project_with_script}/narration-plan")
    assert saved.status_code == 200
    assert saved.json()["paragraphs"] == body["paragraphs"]
