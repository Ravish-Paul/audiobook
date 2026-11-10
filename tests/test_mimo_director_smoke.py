"""Opt-in, PLAN-ONLY smoke tests against the REAL MiMo endpoint.

Skipped by default. Enable BOTH flags (and a configured `.env`):

    RUN_MIMO_LIVE_TEST=1 RUN_MIMO_DIRECTOR_SMOKE=1 \\
        .\\.venv\\Scripts\\python.exe -m pytest tests/test_mimo_director_smoke.py -q

Six representative stories (Hindi horror, English romantic drama, adult
NON-explicit sensual romance, emotional/sad, action/adventure, educational)
are each run through BOTH modes:

* auto   — `POST /analyze-story` (real whole-story analysis) followed by
           `POST /analyze-narration` with `preset=auto`: the plan must embed
           the analysis and resolve `effective_profile` to the recommendation;
* manual — a second `POST /analyze-narration` under a user-chosen profile:
           the manual profile must WIN while the saved analysis stays as
           context (it is reused, no third MiMo analysis call).

Hard safety rules for this file (nothing else in it may change):

* real MiMo ONLY — no ElevenLabs synthesis, no pyannote, no FFmpeg, no audio
  generation of any kind (the conftest blanks those credentials);
* no quota / allowance interaction — beta mode is OFF for the suite;
* no production project changes — every run creates a FRESH project inside
  the isolated `tests/_test_storage` directory;
* plan-only assertions: exact original text, valid emotion/delivery enums,
  policy-clean plan (whisper/tag/variety budgets), sensual boundaries.

Passing these tests verifies REAL narration planning with the real model.
It does NOT verify anything about generated audio — no audio is produced.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.schemas.narration import DELIVERY_STYLES, EMOTIONS, NarrationPlan
from app.schemas.narration_preferences import (
    NARRATION_PROFILES,
    NarrationPreferences,
)
from app.schemas.story_analysis import StoryDirectorAnalysis
from app.services.narration_planner import preferences_for_profile, split_paragraphs
from app.services.narration_policy import NarrationPolicyValidator

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"

_SKIP = (
    os.getenv("RUN_MIMO_DIRECTOR_SMOKE") != "1"
    or os.getenv("RUN_MIMO_LIVE_TEST") != "1"
)
_SKIP_REASON = (
    "set RUN_MIMO_LIVE_TEST=1 RUN_MIMO_DIRECTOR_SMOKE=1 with valid .env "
    "credentials to run (plan-only: real MiMo, never ElevenLabs/pyannote/"
    "FFmpeg/audio)"
)

# --------------------------------------------------------------------------- #
# Six representative stories (original test copy, 5 short paragraphs each so
# the cross-plan variety rules actually apply, ≥4 paragraphs).
# --------------------------------------------------------------------------- #
SCRIPTS: dict[str, str] = {
    "hindi_horror": (
        "दोपहर की धूप में हम पुरानी हवेली के दरवाज़े पर पहुँचे।\n\n"
        "अंदर धूल से ढका फर्नीचर और चुप दीवारें शांत लग रही थीं।\n\n"
        "शाम होते-होते सीढ़ियों से किसी के चलने की आवाज़ आई।\n\n"
        "अंधेरे कोने में एक दीया अपने आप जल उठा और हवा ठंडी हो गई।\n\n"
        "दरवाज़ा अपने आप बंद हुआ और दीवार पर एक परछाई दौड़ गई।\n\n"
        "भोर होते ही सन्नाटा टूटा और हमने राहत की साँस ली।\n\n"
        "लौटते हुए धूप में हवेली पीछे रह गई और दिन सामान्य रहा।"
    ),
    "english_romantic_drama": (
        "The rain stopped just as Meera reached the old bookstore.\n\n"
        "Inside, the smell of paper and tea made her forget the city.\n\n"
        "Aarav looked up from the counter and smiled without a word.\n\n"
        "They spoke about ordinary things until the lamps came on outside.\n\n"
        "When she left, the evening felt lighter than it had in months."
    ),
    "adult_sensual_romance": (
        "After the guests left, the apartment fell into a warm quiet.\n\n"
        "They sat close on the balcony, shoulders touching, words unhurried.\n\n"
        "Her laughter came softly when he told the story of the broken lamp.\n\n"
        "He touched her hand and the rest of the city seemed to fade.\n\n"
        "They stayed like that until the night turned cool and ordinary."
    ),
    "emotional_sad": (
        "The hospital corridor stayed quiet after the last visitor left.\n\n"
        "He read the old message again and put the phone in his pocket.\n\n"
        "On the walk home he counted the years they had lost to silence.\n\n"
        "He bought hot tea at the market and texted his mother that he was fine.\n\n"
        "His sister called, and they laughed about their father's bad cooking.\n\n"
        "At the grave he left the flowers and said the old words aloud.\n\n"
        "That night he slept, and the corridor was only a corridor."
    ),
    "action_adventure": (
        "The bridge exploded a second after Nikhil cleared it.\n\n"
        "He ran along the cliff path while the dust rose behind him.\n\n"
        "The boat engine coughed, then roared as the tide pulled them out.\n\n"
        "Below, the searchlights swept the water in hard white lines.\n\n"
        "By dawn the sea was calm and the danger had passed."
    ),
    "educational": (
        "Photosynthesis happens inside tiny structures called chloroplasts.\n\n"
        "Chlorophyll captures light energy and turns it into chemical energy.\n\n"
        "The plant then uses that energy to build sugars from carbon dioxide.\n\n"
        "Oxygen leaves the leaf through small pores called stomata.\n\n"
        "That is why forests feel cool and fresh on a bright afternoon."
    ),
}

# The manual profile each story is re-planned under (the analysis still
# recommends its own profile — the test proves the manual override wins).
MANUAL_PROFILE = {
    "hindi_horror": "horror",
    "english_romantic_drama": "romantic",
    "adult_sensual_romance": "sensual",
    "emotional_sad": "emotional",
    "action_adventure": "action",
    "educational": "professional",
}


def _auto_prefs() -> dict[str, Any]:
    """The 'Let AI choose the mood' panel payload (frontend AUTO_MOOD)."""
    return {
        "preset": "auto",
        "acting_intensity": 3,
        "whisper_policy": "user_selected",
        "whisper_paragraphs": [],
        "tag_density": "standard",
        "custom_direction": "",
    }


def _manual_prefs(profile: str) -> dict[str, Any]:
    payload = _auto_prefs()
    payload["preset"] = profile
    return payload


# --------------------------------------------------------------------------- #
# Assertions shared by both modes
# --------------------------------------------------------------------------- #
def _assert_plan_is_clean(
    body: dict[str, Any],
    paragraphs: list[str],
    planning_prefs: NarrationPreferences,
    *,
    expect_analysis: bool,
) -> None:
    """The saved plan must honour the script, the enums AND the policy."""
    assert [p["original_text"] for p in body["paragraphs"]] == paragraphs
    assert body["paragraph_count"] == len(paragraphs)

    for paragraph in body["paragraphs"]:
        assert paragraph["emotion"] in EMOTIONS
        assert paragraph["delivery_style"] in DELIVERY_STYLES
        assert 1 <= paragraph["intensity"] <= planning_prefs.acting_intensity
        assert paragraph.get("tts_text")  # a plan without tts_text never speaks

    if expect_analysis:
        assert isinstance(body.get("story_analysis"), dict)
        assert body.get("effective_profile") == planning_prefs.preset

    # The FULL policy (whisper, tags, intensity, markers, variety) must pass
    # for the preferences that were actually used for planning.
    plan = NarrationPlan.model_validate(body)
    problems = NarrationPolicyValidator(planning_prefs).validate(plan)
    assert problems == [], problems

    # Variety in plain terms: no one non-neutral emotion on the whole story.
    emotions = [p["emotion"] for p in body["paragraphs"]]
    if len(emotions) >= 4:
        non_neutral = [e for e in emotions if e != "neutral"]
        if non_neutral:
            top = Counter(non_neutral).most_common(1)[0][1]
            assert top < math.ceil(len(emotions) * 0.75), (
                f"one emotion dominates {top}/{len(emotions)} paragraphs: "
                f"{Counter(non_neutral)}"
            )


def _assert_sensual_boundaries(bodies: list[dict[str, Any]]) -> None:
    """Sensual & Intimate: no unsolicited whispering, no intimacy everywhere."""
    family = {"intimate", "romance", "tenderness", "vulnerable"}
    for body in bodies:
        paragraphs = body["paragraphs"]
        for paragraph in paragraphs:
            assert "[whisper" not in (paragraph.get("tts_text") or "").casefold()
            assert "breath" not in (paragraph.get("tts_text") or "").casefold()
        hits = sum(1 for p in paragraphs if p["emotion"] in family)
        assert hits < math.ceil(len(paragraphs) * 0.6), (
            f"intimate emotions in {hits}/{len(paragraphs)} paragraphs"
        )


# --------------------------------------------------------------------------- #
# The smoke test (real MiMo, plan only)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
@pytest.mark.parametrize(
    ("case", "manual_profile"),
    [(case, MANUAL_PROFILE[case]) for case in SCRIPTS],
    ids=list(SCRIPTS),
)
def test_real_mimo_director_plan_only(case: str, manual_profile: str) -> None:
    settings = get_settings()
    assert settings.mimo_configured, (
        "RUN_MIMO_LIVE_TEST=1 RUN_MIMO_DIRECTOR_SMOKE=1 requires MIMO_API_KEY in .env"
    )

    script = SCRIPTS[case]
    paragraphs = split_paragraphs(script)
    assert len(paragraphs) >= 4  # the variety rules must apply to every case

    with TestClient(app) as client:
        project_id = client.post(
            "/projects", json={"name": f"director-smoke-{case}"}
        ).json()["project_id"]
        upload = client.post(
            f"/projects/{project_id}/upload-script",
            files={"file": ("script.txt", script.encode("utf-8"), "text/plain")},
        )
        assert upload.status_code == 201, upload.text

        # 1) The whole-story analysis (real MiMo, auxiliary call).
        analysis_response = client.post(
            f"/projects/{project_id}/analyze-story", json={"force": True}
        )
        assert analysis_response.status_code == 200, analysis_response.text
        analysis = analysis_response.json()["analysis"]
        assert analysis["recommended_profile"] in NARRATION_PROFILES
        assert analysis["scene_progression"], "the analysis must describe scenes"
        assert analysis["recommended_direction"], "the analysis must direct something"
        analysis_obj = StoryDirectorAnalysis.model_validate(analysis)

        # 2) AUTO mode: the plan must be directed by that analysis.
        auto_response = client.post(
            f"/projects/{project_id}/analyze-narration",
            json={"preferences": _auto_prefs()},
        )
        assert auto_response.status_code == 200, auto_response.text
        auto_plan = auto_response.json()
        assert auto_plan["preferences"]["preset"] == "auto"
        assert auto_plan["effective_profile"] == analysis["recommended_profile"]
        assert (
            auto_plan["story_analysis"]["recommended_profile"]
            == analysis["recommended_profile"]
        )
        auto_planning = preferences_for_profile(
            NarrationPreferences(**_auto_prefs()), analysis_obj
        )
        assert auto_planning is not None
        assert auto_planning.preset == analysis["recommended_profile"]
        _assert_plan_is_clean(auto_plan, paragraphs, auto_planning, expect_analysis=True)

        # 3) MANUAL mode: the user's profile wins, the analysis stays context.
        manual_response = client.post(
            f"/projects/{project_id}/analyze-narration",
            json={"preferences": _manual_prefs(manual_profile)},
        )
        assert manual_response.status_code == 200, manual_response.text
        manual_plan = manual_response.json()
        assert manual_plan["preferences"]["preset"] == manual_profile
        assert manual_plan["effective_profile"] == manual_profile
        # The recommendation was NOT adopted, but the analysis is still there.
        assert manual_plan["story_analysis"]["recommended_profile"] == analysis[
            "recommended_profile"
        ]
        _assert_plan_is_clean(
            manual_plan,
            paragraphs,
            NarrationPreferences(**_manual_prefs(manual_profile)),
            expect_analysis=True,
        )

        # 4) Sensual-specific boundaries on BOTH modes for the sensual story.
        if case == "adult_sensual_romance":
            _assert_sensual_boundaries([auto_plan, manual_plan])

        # 5) PLAN-ONLY: this smoke test must never produce an audiobook.
        directory = TEST_STORAGE / project_id
        audio_dir = directory / "audio"
        assert not audio_dir.exists() or not any(audio_dir.iterdir())
        assert not (directory / "audiobook.mp3").exists()

        # Both plans are retrievable for post-mortem inspection.
        saved = client.get(f"/projects/{project_id}/narration-plan")
        assert saved.status_code == 200
        assert saved.json()["paragraphs"] == manual_plan["paragraphs"]

        # The analysis is reused (not re-billed) for any further call.
        reused = client.post(
            f"/projects/{project_id}/analyze-story", json={"force": False}
        )
        assert reused.status_code == 200 and reused.json()["reused"] is True


def test_smoke_cases_cover_all_six_representative_stories() -> None:
    """Never skipped: guards the smoke matrix itself."""
    assert set(SCRIPTS) == {
        "hindi_horror",
        "english_romantic_drama",
        "adult_sensual_romance",
        "emotional_sad",
        "action_adventure",
        "educational",
    }
    assert set(MANUAL_PROFILE) == set(SCRIPTS)
    assert all(MANUAL_PROFILE.values())
    for script in SCRIPTS.values():
        assert len(split_paragraphs(script)) >= 4
