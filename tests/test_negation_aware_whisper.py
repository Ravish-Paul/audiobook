"""Negation-aware whisper detection tests (MiMo/ElevenLabs always mocked).

Guards the false positive found by the real-MiMo smoke test: the direction

    "Keep the voice low and still, never breathy or whispered."

explicitly PROHIBITS whispering, but the old substring matcher saw "whispered"
and rejected the plan. Covers:

* negations that must NOT count as whisper requests (never/do not/avoid/no/
  not/without + the exact smoke-test sentence + "unwhispered");
* affirmative whisper requests that MUST still count (whisper the sentence,
  deliver in a whisper, [whispers], quiet whispered delivery, speak
  breathlessly);
* negation scope (no leaking across sentences, clause commas or `but`),
  including mixed negated + affirmative instructions;
* audio tags in `tts_text`, and prose in `tts_text` that merely quotes the
  script ("She whispered.");
* emotion / delivery_style cannot smuggle whisper wording (closed enums);
* the existing whisper policy rules (never / user-selected / rare) unchanged.

No real credits are spent.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import get_settings
from app.main import app
from app.routes import narration as narration_routes
from app.schemas.narration import DELIVERY_STYLES, EMOTIONS, NarrationPlan
from app.schemas.narration_preferences import NarrationPreferences
from app.services.eleven_models import get_capabilities
from app.services.narration_planner import NarrationPlanner
from app.services.narration_policy import (
    WHISPER_RE,
    NarrationPolicyValidator,
    has_whisper_instruction,
    whisper_budget,
    whisper_snippets,
    whisper_terms,
)
from fakes import FakeMiMo

TEXT = "The lane was empty at midnight."

# --------------------------------------------------------------------------- #
# Requirement 2 — prohibitions must never count as whisper requests
# --------------------------------------------------------------------------- #
NEGATIVE_CASES = [
    "never whisper",
    "do not whisper",
    "don't whisper",
    "avoid whispering",
    "no whispering",
    "not breathy",
    "without whispering",
    # the exact sentence from the real-MiMo smoke test
    "Keep the voice low and still, never breathy or whispered.",
    # the sentence that triggered the same false positive on the 2nd attempt
    "keep the voice plain and unwhispered, just measured and still.",
    "no breathy or whispered delivery anywhere",
    "Neither whispered nor hushed.",
    "deliver it rather plain; never hushed",
]

# --------------------------------------------------------------------------- #
# Requirement 3 — affirmative instructions must still count
# --------------------------------------------------------------------------- #
POSITIVE_CASES = [
    "whisper the sentence",
    "deliver in a whisper",
    "quiet whispered delivery",
    "speak breathlessly",
    "read the line in a hushed voice",
    "soft breathy narration",
    "[whispers]",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def one_paragraph_plan(
    direction: str, tts_text: Optional[str] = None, *, text: str = TEXT
) -> NarrationPlan:
    data = {
        "overall_story_tone": "quiet suspense",
        "language": "English",
        "paragraphs": [
            {
                "paragraph_index": 0,
                "original_text": text,
                "tts_text": text if tts_text is None else tts_text,
                "emotion": "neutral",
                "delivery_style": "calm",
                "intensity": 2,
                "narration_direction": direction,
                "voice_settings": {
                    "speed": 1.0,
                    "stability": 0.6,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
                "pause_after_ms": 300,
            }
        ],
    }
    return NarrationPlan.model_validate(data)


def preferences(
    policy: str = "never", whisper_paragraphs: Optional[Sequence[int]] = None
) -> NarrationPreferences:
    return NarrationPreferences(
        preset="natural",
        acting_intensity=2,
        whisper_policy=policy,
        whisper_paragraphs=list(whisper_paragraphs or []),
        tag_density="minimal",
        custom_direction="",
    )


def whisper_problems(
    direction: str,
    *,
    policy: str = "never",
    whisper_paragraphs: Optional[Sequence[int]] = None,
    tts_text: Optional[str] = None,
    text: str = TEXT,
) -> list[str]:
    """Only the whisper-related problems the policy raises for one plan.

    Matches both message shapes: "found whispering (…)" and the rare-budget
    "whispering appears in N paragraphs (…)".
    """
    validator = NarrationPolicyValidator(
        preferences(policy, whisper_paragraphs),
        capabilities=get_capabilities("eleven_v4"),
    )
    problems = validator.validate(one_paragraph_plan(direction, tts_text, text=text))
    return [problem for problem in problems if "whisper" in problem]


# --------------------------------------------------------------------------- #
# The detector itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", NEGATIVE_CASES)
def test_negated_whisper_wording_is_never_a_request(text: str) -> None:
    assert whisper_terms(text) == []
    assert has_whisper_instruction(text) is False


@pytest.mark.parametrize("text", POSITIVE_CASES)
def test_affirmative_whisper_wording_is_a_request(text: str) -> None:
    assert whisper_terms(text)
    assert has_whisper_instruction(text) is True


def test_unprefixed_whisper_word_is_still_detected() -> None:
    """Word boundaries must not hide real instructions ("whispered")."""
    assert whisper_terms("whispered") == ["whispered"]
    assert whisper_terms("unwhispered") == []  # the prefix negates it


# --------------------------------------------------------------------------- #
# Requirement 2/3 — at policy level (whisper_policy="never")
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", NEGATIVE_CASES)
def test_prohibitions_survive_the_whisper_policy(text: str) -> None:
    assert whisper_problems(text) == []


@pytest.mark.parametrize("text", POSITIVE_CASES)
def test_affirmative_instructions_are_still_rejected(text: str) -> None:
    problems = whisper_problems(text)
    assert problems, f"'{text}' must be a whisper violation"
    assert "whisper_policy is 'Never'" in problems[0]


def test_positive_tag_in_tts_text_is_rejected() -> None:
    problems = whisper_problems(
        "Keep it even and plain.", tts_text=f"[whispers] {TEXT}"
    )
    assert problems
    assert "[whispers]" in problems[0]


def test_negated_tag_is_not_rejected() -> None:
    assert whisper_problems("Keep it even and plain.", tts_text=f"[no whispers] {TEXT}") == []


def test_direction_negation_does_not_silence_a_positive_tag() -> None:
    """Mixed instruction: the tag still requests whispering."""
    problems = whisper_problems(
        "never breathy or whispered", tts_text=f"[whispers] {TEXT}"
    )
    assert problems
    assert "[whispers]" in problems[0]


# --------------------------------------------------------------------------- #
# Negation scope
# --------------------------------------------------------------------------- #
def test_negation_does_not_leak_into_the_next_sentence() -> None:
    text = "Keep the voice low and still, never breathy or whispered. Whisper the ending."
    assert whisper_terms(text) == ["Whisper"]
    assert has_whisper_instruction(text) is True


def test_negation_does_not_cross_a_clause_comma() -> None:
    text = "never raise the voice, whisper the sentence"
    assert whisper_terms(text) == ["whisper"]


def test_contrastive_but_ends_the_negation_scope() -> None:
    text = "never whispered, but whisper the final line"
    assert whisper_terms(text) == ["whisper"]


def test_negation_scope_is_bounded() -> None:
    assert whisper_terms("never do any of that whisper") == []  # inside the window
    assert whisper_terms("never " + "plain " * 6 + "whisper") == []  # 7 words
    # …but the scope is a window, not the whole paragraph:
    assert whisper_terms("never " + "plain " * 9 + "whisper") == ["whisper"]


# --------------------------------------------------------------------------- #
# Mixed instructions
# --------------------------------------------------------------------------- #
def test_narrow_negator_does_not_cover_the_next_clause() -> None:
    # "avoid whispering" is negated, "speak breathlessly" is still requested.
    text = "avoid whispering and speak breathlessly"
    assert whisper_terms(text) == ["breathlessly"]


def test_wide_negator_covers_coordinated_phrases() -> None:
    assert whisper_terms("never whisper or speak breathlessly") == []


def test_narrow_negator_covers_its_own_coordinated_list() -> None:
    assert whisper_terms("avoid breathy or whispered delivery") == []


# --------------------------------------------------------------------------- #
# Requirement 1 — the fields whisper can actually come from
# --------------------------------------------------------------------------- #
def test_emotion_and_delivery_style_cannot_carry_whisper_wording() -> None:
    """Both are closed enums: no free text can reach the policy through them."""
    assert [value for value in EMOTIONS if WHISPER_RE.search(value)] == []
    assert [value for value in DELIVERY_STYLES if WHISPER_RE.search(value)] == []


def test_a_whispered_delivery_style_never_reaches_the_policy() -> None:
    data = json.loads(one_paragraph_plan("keep it plain").model_dump_json())
    data["paragraphs"][0]["delivery_style"] = "whispered"
    with pytest.raises(ValidationError):
        NarrationPlan.model_validate(data)  # schema rejects it first


def test_script_prose_in_tts_text_is_not_a_whisper_instruction() -> None:
    """`tts_text` prose is the spoken script — "She whispered." is content.

    The schema only allows delivery instructions INSIDE [audio tags], so prose
    that happens to contain a whisper word is script text, never a request.
    """
    spoken = "She whispered the secret."
    assert whisper_problems("keep it plain", tts_text=spoken, text=spoken) == []


def test_whisper_instructions_outside_tags_are_rejected_by_the_schema() -> None:
    with pytest.raises(ValidationError):
        one_paragraph_plan("keep it plain", tts_text="whispered: hello world")


def test_whisper_snippets_only_flag_requests() -> None:
    plan = one_paragraph_plan("never breathy or whispered")
    assert whisper_snippets(plan.paragraphs[0]) == []
    plan = one_paragraph_plan("whisper the sentence")
    snippets = whisper_snippets(plan.paragraphs[0])
    assert snippets and snippets[0].startswith("narration_direction")


# --------------------------------------------------------------------------- #
# Requirement 6 — the existing whisper policy rules stay intact
# --------------------------------------------------------------------------- #
def test_selected_paragraph_is_still_allowed_to_whisper() -> None:
    assert whisper_problems("whisper the line", policy="user_selected", whisper_paragraphs=[0]) == []


def test_unselected_paragraph_is_still_rejected() -> None:
    problems = whisper_problems("whisper the line", policy="user_selected")
    assert problems
    assert "Only user-selected lines" in problems[0]
    assert "selected paragraphs: none" in problems[0]


def test_rare_budget_counts_requests_but_not_prohibitions() -> None:
    # Four paragraphs of "never whispered" whisper nothing -> within budget.
    assert whisper_budget(4) == 1
    validator = NarrationPolicyValidator(preferences("rare"), capabilities=get_capabilities("eleven_v4"))
    data = json.loads(one_paragraph_plan("never breathy or whispered").model_dump_json())
    data["paragraphs"] = [dict(data["paragraphs"][0], paragraph_index=i) for i in range(4)]
    assert [p for p in validator.validate(NarrationPlan.model_validate(data)) if "whisper" in p] == []

    data["paragraphs"][0]["narration_direction"] = "whisper the sentence"
    data["paragraphs"][1]["narration_direction"] = "deliver in a whisper"
    problems = [
        p
        for p in validator.validate(NarrationPlan.model_validate(data))
        if "whisper" in p
    ]
    assert problems  # two whispered paragraphs exceed the budget of 1


# --------------------------------------------------------------------------- #
# End-to-end regression through the API (fake MiMo)
# --------------------------------------------------------------------------- #
SCRIPT = (
    "The lane was empty at midnight.\n\n"
    "Somewhere a dog barked twice.\n\n"
    "By dawn the whole street would know."
)


def _plan_json(directions: Sequence[str]) -> str:
    paragraphs = []
    for index, text in enumerate(SCRIPT.split("\n\n")):
        paragraphs.append(
            {
                "paragraph_index": index,
                "original_text": text,
                "tts_text": text,
                "emotion": "neutral",
                "delivery_style": "calm",
                "intensity": 2,
                "narration_direction": directions[index],
                "voice_settings": {
                    "speed": 1.0,
                    "stability": 0.6,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
                "pause_after_ms": 300,
            }
        )
    return json.dumps(
        {"overall_story_tone": "quiet suspense", "language": "English", "paragraphs": paragraphs},
        ensure_ascii=False,
    )


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def project_with_script(client: TestClient) -> str:
    project_id = client.post("/projects", json={"name": "Negation Whisper Test"}).json()[
        "project_id"
    ]
    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", SCRIPT.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201
    return project_id


def _use_fake_mimo(monkeypatch: pytest.MonkeyPatch, fake: FakeMiMo) -> FakeMiMo:
    monkeypatch.setattr(
        narration_routes,
        "_planner",
        lambda: NarrationPlanner(get_settings(), mimo_client=fake),
    )
    return fake


def _analyze(client: TestClient, project_id: str, prefs: dict[str, Any]):
    return client.post(
        f"/projects/{project_id}/analyze-narration", json={"preferences": prefs}
    )


def test_smoke_test_sentence_is_accepted_end_to_end(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact false positive from the real-MiMo smoke test must pass."""
    directions = [
        "Read the title cleanly and evenly.",
        "Keep the voice low and still, never breathy or whispered.",
        "Deliver the final line with quiet, even composure.",
    ]
    fake = _use_fake_mimo(monkeypatch, FakeMiMo([_plan_json(directions)]))

    response = _analyze(
        client,
        project_with_script,
        {
            "preset": "restrained_cinematic",
            "acting_intensity": 2,
            "whisper_policy": "user_selected",
            "whisper_paragraphs": [],
            "tag_density": "minimal",
            "custom_direction": "",
        },
    )

    assert response.status_code == 200, response.text
    assert fake.call_count == 1  # no wasted retry


def test_affirmative_whisper_direction_is_still_rejected_end_to_end(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directions = [
        "Read the title cleanly and evenly.",
        "whisper the sentence",
        "Deliver the final line with quiet, even composure.",
    ]
    fake = _use_fake_mimo(monkeypatch, FakeMiMo([_plan_json(directions)]))

    response = _analyze(
        client,
        project_with_script,
        {
            "preset": "natural",
            "acting_intensity": 2,
            "whisper_policy": "never",
            "whisper_paragraphs": [],
            "tag_density": "minimal",
            "custom_direction": "",
        },
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "found whispering" in detail
    assert "whisper_policy is 'Never'" in detail
    assert fake.call_count == get_settings().mimo_max_attempts
