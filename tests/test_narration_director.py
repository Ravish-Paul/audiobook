"""AI Narration Director tests — MiMo and ElevenLabs are ALWAYS mocked.

Covers (no real provider calls, no credits, no audiobook generation):

* the twelve global profiles: registry, labels/descriptions/guidance, prompt
  propagation (each profile's own guidance — and only its own — reaches the
  MiMo prompt) and distinct policy budgets;
* auto mode: the whole-story analysis runs first, the plan embeds it and the
  RESOLVED profile (not the label) directs the prompt; manual mode: the user's
  chosen profile wins while the analysis stays as context;
* the story-analysis endpoints: reuse / force / 404 / staleness after a script
  change, plus auxiliary-path classification (never an allowance path);
* scene/paragraph variety (`_check_variety`): repeated direction, one emotion
  everywhere, the sensual family stamped on every paragraph, repeated tags —
  and the <4-paragraph exemption plus a compliant mixed plan;
* sensual boundaries: prompt rules (no forced whispering, no added explicit
  content) and the hard whisper policy for unsolicited whispering;
* narrator/character separation and stable voice identity in the multi-voice
  prompt, including the analysis block in auto mode;
* eleven_v4: speed/style stay planning metadata, tag stripping for models
  without tag support;
* the narration-preview AUDIO endpoint: one MP3 on explicit action, nothing
  written to the project, auxiliary classification;
* plan identity: fingerprint v2 and plan hash change with profile/analysis,
  and generate-audio is rejected (409) when the analysis changed.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx
import pytest
from fastapi.testclient import TestClient

from app import plan_identity, security
from app.config import DEFAULT_ELEVENLABS_BASE_URL, get_settings
from app.main import app
from app.routes import audio as audio_routes
from app.routes import narration as narration_routes
from app.schemas.narration import (
    DELIVERY_STYLES,
    EMOTIONS,
    NarrationPlan,
    tts_projection,
)
from app.schemas.narration_preferences import (
    NARRATION_PROFILES,
    PRESET_LABELS,
    PROFILE_DESCRIPTIONS,
    PROFILE_GUIDANCE,
    NarrationPreferences,
    normalize_profile_id,
)
from app.schemas.story_analysis import StoryDirectorAnalysis
from app.services import AudioService, ElevenLabsClient, NarrationPlanner, StoryAnalysisService
from app.services.audio_service import plan_hash
from app.services.audio_tags import sanitize_for_model
from app.services.multi_voice_planner import (
    SYSTEM_PROMPT as MULTI_VOICE_SYSTEM_PROMPT,
    build_user_prompt as multi_voice_user_prompt,
)
from app.services.narration_planner import (
    build_system_prompt,
    build_user_prompt,
    split_paragraphs,
)
from app.services.narration_policy import (
    PRESET_MARKER_FRACTION,
    NarrationPolicyValidator,
)
from fakes import (
    FAKE_MP3,
    FakeComposer,
    FakeMiMo,
    director_analysis_json,
    make_director_analysis_dict,
    make_plan_dict,
)

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"

SCRIPT = (
    "रात बहुत अंधेरी थी और चारों तरफ सन्नाटा पसरा हुआ था।\n\n"
    "अचानक दरवाजे पर किसी ने दस्तक दी।\n\n"
    "वह धीरे-धीरे दरवाजे की ओर बढ़ा और दरवाजा खोला।"
)
PARAGRAPHS = split_paragraphs(SCRIPT)  # exactly 3 (fits the director fixture)

# Six plain paragraphs for the policy-level variety tests (the variety check
# only runs for plans with at least 4 paragraphs).
SIX = [f"Scene {index}: the story continues with ordinary narration." for index in range(6)]

EXPECTED_PROFILES = (
    "natural",
    "cinematic",
    "horror",
    "romantic",
    "sensual",
    "emotional",
    "action",
    "mystery",
    "fantasy",
    "comedy",
    "professional",
    "children",
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


def _use_fake_mimo(monkeypatch: pytest.MonkeyPatch, fake: FakeMiMo) -> FakeMiMo:
    """Route every narration endpoint through a planner with a fake MiMo."""
    monkeypatch.setattr(
        narration_routes,
        "_planner",
        lambda: NarrationPlanner(get_settings(), mimo_client=fake),
    )
    return fake


def _use_fake_story(monkeypatch: pytest.MonkeyPatch, fake: FakeMiMo) -> FakeMiMo:
    """Route the story-analysis endpoints through a fake MiMo."""
    monkeypatch.setattr(
        narration_routes,
        "_story_service",
        lambda: StoryAnalysisService(get_settings(), mimo_client=fake),
    )
    return fake


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


def _use_fake_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire a fully mocked AudioService into the generate-audio route."""
    settings = dataclasses.replace(
        get_settings(),
        elevenlabs_api_key="sk-director-test",
        elevenlabs_voice_id="voice-director",
        elevenlabs_base_url=DEFAULT_ELEVENLABS_BASE_URL,
    )
    service = AudioService(
        settings,
        tts_client=ElevenLabsClient(settings, transport=httpx.MockTransport(_Recorder())),
        composer=FakeComposer(),
    )
    monkeypatch.setattr(audio_routes, "_audio_service", lambda: service)


def _v4_settings(**overrides: Any):
    """Deterministic eleven_v4 settings with fake credentials (no traffic)."""
    base = dict(
        elevenlabs_api_key="sk-director-test",
        elevenlabs_voice_id="voice-director",
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


def policy_plan(
    texts: Sequence[str],
    *,
    emotions: Optional[Sequence[str]] = None,
    directions: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[Optional[str]]] = None,
    styles: Optional[Sequence[str]] = None,
) -> NarrationPlan:
    """A Pydantic-valid plan for policy tests (inert v4 speed/style, intensity 2)."""
    default_emotions = (
        "neutral",
        "happiness",
        "curiosity",
        "sadness",
        "anger",
        "confidence",
    )
    default_styles = ("calm", "conversational", "soft", "reflective", "slow", "mysterious")
    paragraphs: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        emotion = (
            emotions[index]
            if emotions is not None
            else default_emotions[index % len(default_emotions)]
        )
        style = styles[index] if styles is not None else default_styles[index % len(default_styles)]
        direction = (
            directions[index]
            if directions is not None
            else f"Read paragraph {index} as natural, controlled storytelling."
        )
        tag = tags[index] if tags is not None else None
        tts_text = f"[{tag}] {text}" if tag else text
        paragraphs.append(
            {
                "paragraph_index": index,
                "original_text": text,
                "tts_text": tts_text,
                "emotion": emotion,
                "delivery_style": style,
                "intensity": 2,
                "narration_direction": direction,
                "voice_settings": {
                    "speed": 1.0,  # not supported by eleven_v4 -> inert
                    "stability": 0.6,
                    "similarity_boost": 0.75,
                    "style": 0.0,  # not supported by eleven_v4 -> inert
                    "use_speaker_boost": True,
                },
                "pause_after_ms": 300 + 50 * index,
            }
        )
    return NarrationPlan.model_validate(
        {
            "overall_story_tone": "varied",
            "language": "English",
            "paragraphs": paragraphs,
        }
    )


def problems_for(
    texts: Sequence[str], preset: str = "natural", **plan_kwargs: Any
) -> list[str]:
    """Every policy violation for `texts` under `preset`."""
    validator = NarrationPolicyValidator(NarrationPreferences(**prefs(preset=preset)))
    return validator.validate(policy_plan(texts, **plan_kwargs))


def compliant_json(texts: Sequence[str]) -> str:
    """JSON of `policy_plan` — a policy-clean response from a well-behaved MiMo."""
    return json.dumps(policy_plan(texts).model_dump(mode="json"), ensure_ascii=False)


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def project_with_script(client: TestClient) -> str:
    project_id = client.post("/projects", json={"name": "Director Test"}).json()[
        "project_id"
    ]
    upload = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", SCRIPT.encode("utf-8"), "text/plain")},
    )
    assert upload.status_code == 201
    return project_id


# --------------------------------------------------------------------------- #
# §1 — the twelve profiles (registry + prompt propagation)
# --------------------------------------------------------------------------- #
def test_exactly_twelve_profiles_with_labels_descriptions_and_guidance() -> None:
    assert NARRATION_PROFILES == EXPECTED_PROFILES  # the exposed set, exactly
    labels = [PRESET_LABELS[p] for p in NARRATION_PROFILES]
    descriptions = [PROFILE_DESCRIPTIONS[p] for p in NARRATION_PROFILES]
    assert len(set(labels)) == 12
    assert len(set(descriptions)) == 12
    assert all(descriptions) and all(len(text) <= 100 for text in descriptions)
    # Every profile carries real planning guidance (not just a label).
    assert all(PROFILE_GUIDANCE[p].strip() for p in NARRATION_PROFILES)
    # No legacy preset leaks into the exposed profile set …
    for legacy in ("restrained_cinematic", "expressive", "dramatic", "soft", "smooth", "auto"):
        assert legacy not in NARRATION_PROFILES
    # … but the legacy ids stay accepted for old plans/localStorage.
    for legacy in ("restrained_cinematic", "expressive", "dramatic", "soft", "smooth"):
        assert normalize_profile_id(legacy) == legacy


def test_profile_ids_are_neither_emotions_nor_delivery_styles() -> None:
    assert not set(NARRATION_PROFILES) & set(EMOTIONS)
    assert not set(NARRATION_PROFILES) & set(DELIVERY_STYLES)
    # The user prompt repeats this rule: profiles are GLOBAL, never a value.
    user = build_user_prompt(
        SIX[:2], preferences=NarrationPreferences(preset="cinematic")
    )
    assert "GLOBAL direction only" in user
    for profile in NARRATION_PROFILES:
        assert profile not in DELIVERY_STYLES


def test_every_profile_prompt_contains_only_its_own_guidance() -> None:
    prompts = {
        profile: build_system_prompt(NarrationPreferences(preset=profile))
        for profile in NARRATION_PROFILES
    }
    assert len(set(prompts.values())) == 12  # twelve DISTINCT planning prompts

    for profile, prompt in prompts.items():
        assert PRESET_LABELS[profile] in prompt
        assert PROFILE_GUIDANCE[profile] in prompt  # this profile's rules …
        for other in NARRATION_PROFILES:
            if other != profile:
                # … and no other profile's rules steer this plan.
                assert PROFILE_GUIDANCE[other] not in prompt


def test_profiles_use_distinct_policy_marker_budgets() -> None:
    budgets = [PRESET_MARKER_FRACTION[p] for p in NARRATION_PROFILES]
    assert PRESET_MARKER_FRACTION["natural"] == 0.0
    assert PRESET_MARKER_FRACTION["professional"] == 0.0
    assert PRESET_MARKER_FRACTION["action"] > PRESET_MARKER_FRACTION["romantic"]
    assert PRESET_MARKER_FRACTION["horror"] > PRESET_MARKER_FRACTION["natural"]
    # The twelve profiles must not collapse into one shared budget.
    assert len(set(budgets)) >= 6


def test_normalize_profile_id_accepts_ids_and_human_labels_only() -> None:
    assert normalize_profile_id("horror") == "horror"
    assert normalize_profile_id("Horror & Suspense") == "horror"
    assert normalize_profile_id("Children's Story") == "children"
    assert normalize_profile_id("AI Automatic") == "auto"
    assert normalize_profile_id("definitely not a profile") is None
    assert normalize_profile_id("") is None


# --------------------------------------------------------------------------- #
# §2 — auto mode (whole-story analysis drives the plan)
# --------------------------------------------------------------------------- #
def test_auto_mode_analyses_the_story_and_resolves_the_effective_profile(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _use_fake_mimo(
        monkeypatch,
        FakeMiMo([director_analysis_json(), compliant_json(PARAGRAPHS)]),
    )

    response = client.post(
        f"/projects/{project_with_script}/analyze-narration",
        json={"preferences": prefs(preset="auto", acting_intensity=3, tag_density="standard")},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    # The user's choice stays `auto`; the RESOLVED profile directs the plan.
    assert body["preferences"]["preset"] == "auto"
    assert body["effective_profile"] == "horror"
    assert body["story_analysis"]["recommended_profile"] == "horror"
    # The analysis is persisted for reuse and plan identity.
    assert (TEST_STORAGE / project_with_script / "story_analysis.json").is_file()

    # Two MiMo calls: (1) the whole-story analysis, (2) the paragraph plan.
    assert fake.call_count == 2
    analysis_system = fake.calls[0]["system"]
    assert "whole-story" in analysis_system.casefold() or "story analyst" in analysis_system.casefold()

    system = fake.calls[1]["system"]
    assert "AI-automatic mood" in system  # auto guidance block …
    assert (
        "- recommended global profile (chosen by AI Automatic analysis): "
        "Horror & Suspense"
    ) in system
    # … the RESOLVED profile's boundary reaches the prompt as real guidance …
    assert "- global profile boundary (apply this as the direction):" in system
    assert PROFILE_GUIDANCE["horror"] in system
    assert PROFILE_GUIDANCE["romantic"] not in system  # only the resolved profile
    # … and the analysis itself is present as context, never as one hard mood.
    assert "- primary genre: horror; secondary genre: mystery" in system
    assert "- scene progression: 0-1: calm (intensity 2); 2-2: tense (intensity 4)" in system
    assert "- character tendencies: सुमन: calm and protective" in system
    assert "Never apply one mood everywhere" in system


# --------------------------------------------------------------------------- #
# §3 — manual mode (the user's profile wins, analysis stays context)
# --------------------------------------------------------------------------- #
def test_manual_profile_overrides_the_recommendation_but_keeps_context(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_fake_story(monkeypatch, FakeMiMo([director_analysis_json()]))
    analysed = client.post(
        f"/projects/{project_with_script}/analyze-story", json={"force": False}
    )
    assert analysed.status_code == 200, analysed.text

    fake = _use_fake_mimo(monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS)]))
    response = client.post(
        f"/projects/{project_with_script}/analyze-narration",
        json={"preferences": prefs(preset="romantic")},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["preferences"]["preset"] == "romantic"
    assert body["effective_profile"] == "romantic"  # manual wins …
    assert body["story_analysis"]["recommended_profile"] == "horror"  # … over 'horror'

    system = fake.calls[0]["system"]
    assert "Manual mode: the user's chosen global profile WINS" in system
    assert (
        "- AI recommendation (the user OVERRODE it with Romantic & Intimate): "
        "Horror & Suspense"
    ) in system
    assert PROFILE_GUIDANCE["romantic"] in system  # the chosen profile directs
    assert PROFILE_GUIDANCE["horror"] not in system  # the recommendation does not
    assert "global profile boundary" not in system  # (auto-only line)


# --------------------------------------------------------------------------- #
# §4 — story-analysis endpoints (reuse / force / 404 / staleness / aux)
# --------------------------------------------------------------------------- #
def test_analyze_story_reuses_the_saved_analysis_unless_forced(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _use_fake_story(monkeypatch, FakeMiMo([director_analysis_json()]))
    url = f"/projects/{project_with_script}/analyze-story"

    first = client.post(url, json={"force": False})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["reused"] is False and body["saved"] is True
    assert body["analysis"]["recommended_profile"] == "horror"
    assert fake.call_count == 1

    # Same script → the saved analysis is reused (no second MiMo call).
    again = client.post(url, json={"force": False})
    assert again.status_code == 200, again.text
    assert again.json()["reused"] is True
    assert again.json()["saved_at"] == body["saved_at"]
    assert fake.call_count == 1

    fetched = client.get(f"/projects/{project_with_script}/story-analysis")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["analysis"]["recommended_profile"] == "horror"

    # `force` explicitly asks for a fresh analysis.
    forced = client.post(url, json={"force": True})
    assert forced.status_code == 200, forced.text
    assert forced.json()["reused"] is False
    assert fake.call_count == 2


def test_story_analysis_is_404_before_analysis_and_stale_after_a_script_change(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = f"/projects/{project_with_script}/story-analysis"
    # No analysis yet → the GET endpoint is a clean 404 (no MiMo call needed).
    _use_fake_story(monkeypatch, FakeMiMo([director_analysis_json()]))
    assert client.get(url).status_code == 404

    assert client.post(
        f"/projects/{project_with_script}/analyze-story", json={}
    ).status_code == 200

    # Replacing the script invalidates the saved analysis (stale hash) …
    (TEST_STORAGE / project_with_script / "script.txt").write_text(
        "नई शुरुआत हुई।\n\nबीच का दृश्य बदल गया।\n\nअब कहानी का अंत आ गया।",
        encoding="utf-8",
    )
    assert client.get(url).status_code == 404
    # … and a fresh analysis can be produced for the new script.
    fresh = client.post(
        f"/projects/{project_with_script}/analyze-story", json={"force": False}
    )
    assert fresh.status_code == 200, fresh.text
    assert fresh.json()["reused"] is False


def test_story_and_preview_audio_endpoints_are_auxiliary_paths() -> None:
    """Never an allowance path — the aux list wins over generation suffixes."""
    assert security._generation_kind("/projects/x/analyze-story") == "aux"
    assert security._generation_kind("/projects/x/narration-preview/audio") == "aux"
    assert security._generation_kind("/projects/x/analyze-narration/preview") == "aux"
    # … while the billable workflow endpoints keep their classification.
    assert security._generation_kind("/projects/x/analyze-narration") == "plan"
    assert security._generation_kind("/projects/x/create-multi-voice-plan") == "plan"
    assert security._generation_kind("/projects/x/generate-audio") == "generate"
    assert "/analyze-story" in security._AUX_SUFFIXES
    assert "/narration-preview/audio" in security._AUX_SUFFIXES


# --------------------------------------------------------------------------- #
# §5 — scene/paragraph variety (`_check_variety`)
# --------------------------------------------------------------------------- #
def test_variety_rejects_one_direction_stamped_on_every_paragraph() -> None:
    repeated = ["Keep the tension steady and hold this exact tone."] * 6
    problems = problems_for(SIX, directions=repeated)

    assert any("repeated almost identically" in problem for problem in problems)


def test_variety_rejects_one_non_neutral_emotion_everywhere() -> None:
    problems = problems_for(
        SIX, emotions=["fear"] * 5 + ["neutral"]
    )

    assert any("do NOT apply one mood to the whole story" in problem for problem in problems)


def test_variety_rejects_the_sensual_profile_applied_to_every_paragraph() -> None:
    problems = problems_for(
        SIX,
        preset="sensual",
        emotions=["intimate", "romance", "tenderness", "vulnerable", "neutral", "neutral"],
    )

    assert any(
        "the 'sensual' profile is applied to" in problem for problem in problems
    ), problems


def test_variety_rejects_a_repeated_audio_tag() -> None:
    problems = problems_for(SIX, tags=["softly", "softly", "softly", "softly", None, None])

    assert any("is repeated in" in problem and "[softly]" in problem for problem in problems)


def test_variety_tolerates_short_plans_and_passes_a_directed_mix() -> None:
    # Preview windows / tiny scripts (<4 paragraphs) are exempt.
    short = problems_for(
        PARAGRAPHS,
        emotions=["fear", "fear", "fear"],
        directions=["Hold the tension."] * 3,
    )
    assert short == []

    # A properly directed plan: neutral narration survives, intimacy only
    # where the scene earns it, no copied directions or tags.
    mixed = problems_for(
        SIX,
        preset="sensual",
        emotions=["intimate", "neutral", "romance", "neutral", "happiness", "neutral"],
        directions=[f"Direct scene {index} specifically." for index in range(6)],
        tags=["gentle", None, None, None, None, None],
    )
    assert mixed == [], mixed


# --------------------------------------------------------------------------- #
# §6 — sensual boundaries
# --------------------------------------------------------------------------- #
def test_sensual_prompt_forbids_forced_whispering_and_added_content() -> None:
    prompt = build_system_prompt(NarrationPreferences(preset="sensual"))

    assert "NEVER force whispering, breathiness or seductive delivery" in prompt
    assert "never add explicit content that is absent from the source" in prompt
    assert "ordinary romance, flirting, arguments and" in prompt
    assert "only where the text is genuinely intimate" in prompt


def test_sensual_unsolicited_whisper_is_rejected_by_policy() -> None:
    texts = SIX[:3]
    problems = problems_for(
        texts,
        preset="sensual",
        emotions=["intimate", "neutral", "neutral"],
        tags=[None, "whispers", None],
    )

    assert any("was not selected" in problem for problem in problems), problems


# --------------------------------------------------------------------------- #
# §7 — narrator vs. characters (multi-voice planning)
# --------------------------------------------------------------------------- #
def test_multi_voice_prompt_keeps_narrator_and_characters_separate() -> None:
    assert "NARRATOR STAYS SEPARATE" in MULTI_VOICE_SYSTEM_PROMPT
    assert "never copy a character's emotional direction onto the narrator" in (
        MULTI_VOICE_SYSTEM_PROMPT
    )
    assert "CHARACTER IDENTITY IS STABLE" in MULTI_VOICE_SYSTEM_PROMPT
    assert "the locked voice stay fixed" in MULTI_VOICE_SYSTEM_PROMPT
    assert "the system assigns locked voices" in MULTI_VOICE_SYSTEM_PROMPT


def test_multi_voice_user_prompt_carries_analysis_and_profile_boundary() -> None:
    analysis = StoryDirectorAnalysis.model_validate(make_director_analysis_dict())

    prompt = multi_voice_user_prompt(
        ["alpha paragraph", "beta paragraph", "gamma paragraph"],
        [
            {"speaker_id": "narrator", "speaker_type": "narrator"},
            {"speaker_id": "suman", "speaker_type": "character"},
        ],
        preferences=NarrationPreferences(preset="auto"),
        story_analysis=analysis,
    )

    assert "WHOLE-STORY ANALYSIS" in prompt
    assert (
        "- recommended global profile (chosen by AI Automatic analysis): "
        "Horror & Suspense"
    ) in prompt
    assert "- global profile boundary (apply this as the direction):" in prompt
    assert PROFILE_GUIDANCE["horror"] in prompt
    assert "Never apply one mood everywhere" in prompt


# --------------------------------------------------------------------------- #
# §8 — eleven_v4 parameters and tags (planning metadata ≠ request)
# --------------------------------------------------------------------------- #
def test_v4_effective_voice_settings_never_contain_speed_or_style() -> None:
    client = ElevenLabsClient(_v4_settings())
    effective = client.effective_voice_settings(
        {
            "speed": 1.15,
            "style": 0.9,
            "stability": 0.35,
            "similarity_boost": 0.6,
            "use_speaker_boost": False,
        }
    )

    assert "speed" not in effective  # unsupported for eleven_v4 → never sent
    assert "style" not in effective
    assert effective["stability"] == 0.35
    assert effective["similarity_boost"] == 0.6
    assert effective["use_speaker_boost"] is False


def test_tts_projection_lists_speed_and_style_as_planning_only() -> None:
    plan = NarrationPlan.model_validate(make_plan_dict(PARAGRAPHS))

    info, per_paragraph = tts_projection(plan, _v4_settings())

    assert info.model_id == "eleven_v4"
    assert info.supports_speed is False
    assert info.supports_style is False
    assert "speed" in info.planning_only_parameters
    assert "style" in info.planning_only_parameters
    assert per_paragraph  # every paragraph still has effective settings
    for item in per_paragraph:
        assert "speed" not in item.settings
        assert "style" not in item.settings
        assert "stability" in item.settings


def test_audio_tags_are_stripped_for_models_without_tag_support() -> None:
    original = "She laughed and closed the door."
    tts_text = f"[laughs] {original}"

    assert sanitize_for_model(tts_text, original, supports_tags=True) == tts_text
    assert sanitize_for_model(tts_text, original, supports_tags=False) == original


# --------------------------------------------------------------------------- #
# §9 — narration preview audio (explicit action, nothing written)
# --------------------------------------------------------------------------- #
def test_narration_preview_audio_returns_one_mp3_and_writes_nothing(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.elevenlabs_client import SynthesisResult

    _use_fake_mimo(monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS)]))
    seen: dict[str, Any] = {}

    async def fake_synthesize(
        self: ElevenLabsClient,
        text: str,
        voice_settings: Optional[dict[str, object]] = None,
        *,
        tts_text: Optional[str] = None,
        **kwargs: Any,
    ) -> SynthesisResult:
        seen["text"] = text
        seen["tts_text"] = tts_text
        seen["voice_settings"] = voice_settings
        return SynthesisResult(audio=FAKE_MP3, model_id="eleven_v4")

    monkeypatch.setattr(ElevenLabsClient, "synthesize_result", fake_synthesize)

    response = client.post(
        f"/projects/{project_with_script}/narration-preview/audio",
        json={"paragraph_index": 1, "preferences": prefs()},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.headers["content-disposition"].find("narration-preview.mp3") != -1
    assert response.content == FAKE_MP3
    # The EXACT original text is what gets spoken.
    assert seen["text"] == PARAGRAPHS[1]
    assert seen["tts_text"] == PARAGRAPHS[1]

    directory = TEST_STORAGE / project_with_script
    assert not (directory / "narration_plan.json").exists()  # preview never saves a plan
    audio_dir = directory / "audio"
    assert not audio_dir.exists() or not any(audio_dir.iterdir())  # no audiobook artefacts
    # No generation workflow was opened (aux must never touch plan identity).
    assert plan_identity.read_state(get_settings(), project_with_script) is None


# --------------------------------------------------------------------------- #
# §10 — plan identity (fingerprint v2, plan hash, staleness on generate)
# --------------------------------------------------------------------------- #
def test_fingerprint_v2_changes_with_profile_and_story_analysis(
    client: TestClient,
    project_with_script: str,
) -> None:
    settings = get_settings()
    directory = TEST_STORAGE / project_with_script
    # compute_fingerprint needs script + cast; preferences/analysis join in.
    (directory / "character_cast.json").write_text(
        json.dumps(
            {
                "story": {"language": "hi"},
                "approved": True,
                "casting": {
                    "narrator": {
                        "status": "approved",
                        "selected": {"voice_id": "voice-narrator-01"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    base = plan_identity.compute_fingerprint(settings, project_with_script)
    assert base is not None and base.startswith("v2:")
    # Unchanged inputs → unchanged fingerprint.
    assert plan_identity.compute_fingerprint(settings, project_with_script) == base

    # Changing the generation-relevant profile …
    saved = client.put(
        f"/projects/{project_with_script}/narration-preferences",
        json=prefs(preset="horror"),
    )
    assert saved.status_code == 200, saved.text
    with_profile = plan_identity.compute_fingerprint(settings, project_with_script)
    assert with_profile is not None and with_profile != base

    # … and adding the whole-story analysis …
    StoryAnalysisService(
        settings, mimo_client=FakeMiMo([director_analysis_json()])
    ).analyze(project_with_script)
    with_analysis = plan_identity.compute_fingerprint(settings, project_with_script)
    assert with_analysis is not None and with_analysis != with_profile

    # … and changing that analysis all produce DIFFERENT plan identities.
    StoryAnalysisService(
        settings,
        mimo_client=FakeMiMo([director_analysis_json(recommended_profile="emotional")]),
    ).analyze(project_with_script, force=True)
    changed = plan_identity.compute_fingerprint(settings, project_with_script)
    assert changed is not None and changed != with_analysis


def test_plan_hash_covers_story_analysis_and_effective_profile() -> None:
    base = make_plan_dict(PARAGRAPHS)  # legacy shape: none of the new fields
    legacy = NarrationPlan.model_validate(base)
    with_analysis = NarrationPlan.model_validate(
        {**base, "story_analysis": make_director_analysis_dict()}
    )
    with_profile = NarrationPlan.model_validate(
        {**base, "effective_profile": "horror"}
    )

    hashes = {plan_hash(legacy), plan_hash(with_analysis), plan_hash(with_profile)}
    assert len(hashes) == 3  # each new generation-relevant input changes it


def test_generate_audio_is_rejected_when_the_story_analysis_changed(
    client: TestClient,
    project_with_script: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_fake_audio(monkeypatch)
    _use_fake_story(monkeypatch, FakeMiMo([director_analysis_json()]))
    assert client.post(
        f"/projects/{project_with_script}/analyze-story", json={"force": False}
    ).status_code == 200

    _use_fake_mimo(monkeypatch, FakeMiMo([compliant_json(PARAGRAPHS)]))
    planned = client.post(
        f"/projects/{project_with_script}/analyze-narration",
        json={"preferences": prefs(preset="auto")},
    )
    assert planned.status_code == 200, planned.text
    assert planned.json()["story_analysis"]["recommended_profile"] == "horror"

    # A FRESH analysis now describes the same script → the old plan is stale.
    _use_fake_story(
        monkeypatch,
        FakeMiMo([director_analysis_json(recommended_profile="emotional")]),
    )
    forced = client.post(
        f"/projects/{project_with_script}/analyze-story", json={"force": True}
    )
    assert forced.status_code == 200, forced.text
    assert forced.json()["analysis"]["recommended_profile"] == "emotional"

    response = client.post(f"/projects/{project_with_script}/generate-audio")

    assert response.status_code == 409
    assert "whole-story analysis changed" in response.text
