"""ElevenLabs Voice Discovery + matching tests (spec §3/§4) — zero credits.

Every request goes through `httpx.MockTransport`; the official query
parameter names, pagination behaviour and payload normalisation are asserted
against the current API documentation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any, Optional

import httpx
import pytest

from app.config import DEFAULT_ELEVENLABS_BASE_URL, get_settings
from app.schemas.casting import CharacterProfile
from app.services.errors import ElevenLabsAPIError, ElevenLabsVoiceError
from app.services.voice_discovery import (
    ElevenLabsVoiceDiscovery,
    VoiceDetailsNotFoundError,
    VoiceSearchFilters,
)
from app.services.voice_matcher import (
    MIN_ACCEPT_SCORE,
    normalize_account_voice,
    normalize_shared_voice,
    rank_candidates,
    score_voice,
)


# --------------------------------------------------------------------------- #
# Payload builders (shapes from the official docs)
# --------------------------------------------------------------------------- #
def account_voice(
    voice_id: str,
    *,
    name: str = "Voice",
    languages: tuple[str, ...] = ("hi",),
    accent: str = "indian",
    gender: Optional[str] = None,
    age: Optional[str] = None,
    use_case: str = "audiobook",
    description: str = "Deep, calm, measured narrator voice for audiobooks",
    preview_url: Optional[str] = "https://example.com/preview.mp3",
    safety_control: Optional[str] = None,
    category: str = "premade",
    fiat_rate: Optional[float] = None,
    notice_period: Optional[int] = None,
) -> dict[str, Any]:
    labels: dict[str, str] = {"accent": accent}
    if gender:
        labels["gender"] = gender
    if age:
        labels["age"] = age
    if use_case:
        labels["use_case"] = use_case
    item: dict[str, Any] = {
        "voice_id": voice_id,
        "name": name,
        "category": category,
        "labels": labels,
        "description": description,
        "preview_url": preview_url,
        "verified_languages": [
            {"language": lang, "model_id": "eleven_v4", "accent": accent}
            for lang in languages
        ],
    }
    if safety_control:
        item["safety_control"] = safety_control
    if fiat_rate is not None:
        item["sharing"] = {"fiat_rate": fiat_rate, "rate": 1.0,
                           "notice_period": notice_period or 0}
    return item


def shared_voice(
    voice_id: str,
    *,
    owner: str = "owner-abc",
    name: str = "Shared Voice",
    language: Optional[str] = "hi",
    locale: Optional[str] = "hi-IN",
    accent: str = "indian",
    gender: str = "female",
    age: str = "young",
    descriptive: str = "calm",
    use_case: str = "narrative_story",
    category: str = "professional",
    description: Optional[str] = "Soft warm Hindi narration voice",
    preview_url: Optional[str] = "https://example.com/shared.mp3",
    fiat_rate: Optional[float] = None,
    notice_period: Optional[int] = None,
    free_users_allowed: bool = True,
) -> dict[str, Any]:
    return {
        "public_owner_id": owner,
        "voice_id": voice_id,
        "date_unix": 1714423232,
        "name": name,
        "accent": accent,
        "gender": gender,
        "age": age,
        "descriptive": descriptive,
        "use_case": use_case,
        "category": category,
        "usage_character_count_1y": 1000,
        "usage_character_count_7d": 10,
        "play_api_usage_character_count_1y": 1000,
        "cloned_by_count": 3,
        "free_users_allowed": free_users_allowed,
        "live_moderation_enabled": False,
        "featured": False,
        "language": language,
        "locale": locale,
        "description": description,
        "preview_url": preview_url,
        "fiat_rate": fiat_rate,
        "notice_period": notice_period,
    }


class DiscoveryMock:
    """Mock ElevenLabs voice endpoints; records every request."""

    def __init__(
        self,
        *,
        account_by_token: Optional[dict[str, dict]] = None,
        account_default: Optional[dict] = None,
        shared_by_page: Optional[dict[str, dict]] = None,
        shared_default: Optional[dict] = None,
        details: Optional[dict[str, dict]] = None,
    ) -> None:
        self.account_by_token = account_by_token or {}
        self.account_default = account_default or {"voices": [], "has_more": False}
        self.shared_by_page = shared_by_page or {}
        self.shared_default = shared_default or {"voices": [], "has_more": False}
        self.details = details or {}
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = dict(request.url.params)
        self.requests.append(
            {"method": request.method, "path": path, "params": params,
             "body": request.content}
        )

        if path == "/v2/voices":
            token = params.get("next_page_token") or ""
            page = self.account_by_token.get(token, self.account_default)
            return httpx.Response(200, json=page, request=request)

        if path == "/v1/shared-voices":
            page_key = params.get("page", "0")
            page = self.shared_by_page.get(page_key, self.shared_default)
            return httpx.Response(200, json=page, request=request)

        if path.startswith("/v1/voices/add/"):
            voice_id = path.rstrip("/").split("/")[-1]
            return httpx.Response(200, json={"voice_id": voice_id},
                                  request=request)

        if path.startswith("/v1/voices/"):
            voice_id = path.rstrip("/").split("/")[-1]
            detail = self.details.get(voice_id)
            if detail is None:
                return httpx.Response(
                    404, json={"detail": "Voice not found"}, request=request
                )
            return httpx.Response(200, json=detail, request=request)

        return httpx.Response(404, json={"detail": "unknown path"}, request=request)

    @property
    def paths(self) -> list[str]:
        return [r["path"] for r in self.requests]

    def params_for(self, path: str, index: int = 0) -> dict[str, str]:
        matches = [r["params"] for r in self.requests if r["path"] == path]
        return matches[index]


def _discovery(mock: DiscoveryMock) -> ElevenLabsVoiceDiscovery:
    settings = dataclasses.replace(
        get_settings(),
        elevenlabs_api_key="sk-voice-test",
        elevenlabs_base_url=DEFAULT_ELEVENLABS_BASE_URL,
    )
    return ElevenLabsVoiceDiscovery(
        settings, transport=httpx.MockTransport(mock)
    )


# --------------------------------------------------------------------------- #
# Account voices — GET /v2/voices
# --------------------------------------------------------------------------- #
def test_account_voice_search_uses_documented_parameters() -> None:
    mock = DiscoveryMock(
        account_default={"voices": [], "has_more": False}
    )
    discovery = _discovery(mock)

    asyncio.run(discovery.list_account_voices(
        VoiceSearchFilters(
            language="hi",
            gender="female",
            age="young",
            accent="indian",
            use_case="audiobook",
            category="professional",
            search="narrator",
        )
    ))

    params = mock.params_for("/v2/voices")
    assert params["page_size"] == "100"
    assert params["language"] == json.dumps(["hi"])  # JSON list, per the docs
    assert params["gender"] == "female"
    assert params["age"] == "young"
    assert params["accent"] == "indian"
    assert params["use_cases"] == json.dumps(["audiobook"])
    assert params["category"] == "professional"
    assert params["search"] == "narrator"
    assert params["include_total_count"] == "false"
    assert "next_page_token" not in params  # first page


def test_account_voices_paginate_with_next_page_token() -> None:
    mock = DiscoveryMock(
        account_by_token={
            "": {
                "voices": [account_voice("a1")],
                "has_more": True,
                "next_page_token": "TOKEN-1",
            },
            "TOKEN-1": {
                "voices": [account_voice("a2")],
                "has_more": False,
                "next_page_token": None,
            },
        }
    )
    discovery = _discovery(mock)

    voices = asyncio.run(discovery.list_account_voices())

    assert [v["voice_id"] for v in voices] == ["a1", "a2"]
    assert mock.paths == ["/v2/voices", "/v2/voices"]
    assert "next_page_token" not in mock.params_for("/v2/voices", 0)
    assert mock.params_for("/v2/voices", 1)["next_page_token"] == "TOKEN-1"


def test_account_pagination_stops_without_token() -> None:
    """has_more without a token must not loop forever."""
    mock = DiscoveryMock(
        account_default={
            "voices": [account_voice("a1")],
            "has_more": True,
            "next_page_token": None,
        }
    )
    discovery = _discovery(mock)

    voices = asyncio.run(discovery.list_account_voices())

    assert len(voices) == 1
    assert len(mock.requests) == 1


# --------------------------------------------------------------------------- #
# Shared voices — GET /v1/shared-voices
# --------------------------------------------------------------------------- #
def test_shared_voice_search_uses_documented_parameters() -> None:
    mock = DiscoveryMock(shared_default={"voices": [], "has_more": False})
    discovery = _discovery(mock)

    asyncio.run(discovery.list_shared_voices(
        VoiceSearchFilters(
            language="hi",
            locale="hi-IN",
            gender="female",
            age="young",
            accent="indian",
            use_case="audiobook",
            category="professional",
            search="warm",
        )
    ))

    params = mock.params_for("/v1/shared-voices")
    assert params["page"] == "0"
    assert params["page_size"] == "100"
    assert params["language"] == "hi"
    assert params["locale"] == "hi-IN"
    assert params["gender"] == "female"
    assert params["age"] == "young"
    assert params["accent"] == "indian"
    assert params["use_cases"] == json.dumps(["audiobook"])
    assert params["category"] == "professional"
    assert params["search"] == "warm"
    assert "sort" in params  # proven voices first


def test_shared_voices_paginate_by_page_number() -> None:
    mock = DiscoveryMock(
        shared_by_page={
            "0": {
                "voices": [shared_voice("s1", owner="owner-1")],
                "has_more": True,
            },
            "1": {
                "voices": [shared_voice("s2", owner="owner-2")],
                "has_more": False,
            },
        }
    )
    discovery = _discovery(mock)

    voices = asyncio.run(discovery.list_shared_voices())

    assert [v["voice_id"] for v in voices] == ["s1", "s2"]
    assert mock.params_for("/v1/shared-voices", 0)["page"] == "0"
    assert mock.params_for("/v1/shared-voices", 1)["page"] == "1"
    assert len(mock.requests) == 2


# ------------------------------------------------------------------ #
# Voice details + import
# ------------------------------------------------------------------ #
def test_get_voice_returns_details() -> None:
    detail = account_voice("v-detail", name="Details Voice")
    mock = DiscoveryMock(details={"v-detail": detail})
    discovery = _discovery(mock)

    result = asyncio.run(discovery.get_voice("v-detail"))
    assert result["voice_id"] == "v-detail"
    assert mock.paths == ["/v1/voices/v-detail"]


def test_get_voice_not_found_raises_typed_error() -> None:
    mock = DiscoveryMock(details={})
    discovery = _discovery(mock)

    with pytest.raises(VoiceDetailsNotFoundError):
        asyncio.run(discovery.get_voice("missing-voice"))


def test_add_shared_voice_posts_to_official_path() -> None:
    mock = DiscoveryMock()
    discovery = _discovery(mock)

    added = asyncio.run(
        discovery.add_shared_voice("owner-xyz", "shared-1", "My Cast")
    )

    assert added == "shared-1"
    request = mock.requests[-1]
    assert request["method"] == "POST"
    assert request["path"] == "/v1/voices/add/owner-xyz/shared-1"
    assert json.loads(request["body"]) == {"new_name": "My Cast"}


def test_add_shared_voice_requires_owner_id() -> None:
    discovery = _discovery(DiscoveryMock())
    with pytest.raises(ElevenLabsAPIError):
        asyncio.run(discovery.add_shared_voice("", "shared-1", "Name"))


# --------------------------------------------------------------------------- #
# Normalisation of both payload shapes
# --------------------------------------------------------------------------- #
def test_normalize_account_voice_reads_labels_and_verified_languages() -> None:
    raw = normalize_account_voice(
        account_voice(
            "acc-1",
            languages=("hi", "en"),
            gender="female",
            age="middle-aged",
            fiat_rate=2.0,
            notice_period=45,
        )
    )
    assert raw is not None
    assert raw.origin == "account"
    assert raw.languages == ["hi", "en"]
    assert raw.gender == "female"
    assert raw.age == "middle-aged"
    assert raw.accent == "indian"
    assert raw.use_case == "audiobook"
    assert raw.custom_rate is True           # fiat_rate present -> custom rate
    assert raw.notice_period_days == 45
    assert raw.restricted_reason is None


def test_normalize_shared_voice_reads_library_fields() -> None:
    raw = normalize_shared_voice(
        shared_voice("sh-1", gender="male", age="mature", free_users_allowed=False)
    )
    assert raw is not None
    assert raw.origin == "shared"
    assert raw.public_owner_id == "owner-abc"
    assert raw.languages == ["hi"]
    assert raw.locale == "hi-IN"
    assert raw.gender == "male"
    assert raw.age == "mature"
    assert raw.free_users_allowed is False


def test_restricted_voice_is_marked_restricted() -> None:
    raw = normalize_account_voice(account_voice("banned", safety_control="BAN"))
    assert raw is not None
    assert raw.restricted_reason is not None


# --------------------------------------------------------------------------- #
# Ranking (spec §4)
# --------------------------------------------------------------------------- #
def _profile(**overrides: Any) -> CharacterProfile:
    base = dict(
        character_id="narrator",
        character_name="Narrator",
        role="narrator",
        language="hi",
        voice_description="Deep, calm, measured Hindi storytelling voice",
        speaking_style="Natural, restrained audiobook narration",
        has_spoken_dialogue=True,
    )
    base.update(overrides)
    return CharacterProfile(**base)


def test_rank_excludes_language_incompatible_voices() -> None:
    hi_good = normalize_account_voice(account_voice("hi-1"))          # verified hi
    en_only = normalize_account_voice(
        account_voice("en-1", languages=("en",), accent="american",
                      description="American english voice for podcasts")
    )
    assert hi_good and en_only

    ranked = rank_candidates(_profile(), [en_only, hi_good],
                             required_language="hi")
    assert [c.voice_id for c in ranked] == ["hi-1"]  # en voice never proposed


def test_rank_excludes_restricted_voices() -> None:
    banned = normalize_account_voice(account_voice("banned-1", safety_control="BAN"))
    good = normalize_account_voice(account_voice("ok-1"))
    assert banned and good

    ranked = rank_candidates(_profile(), [banned, good], required_language="hi")
    assert [c.voice_id for c in ranked] == ["ok-1"]


def test_rank_drops_weak_candidates_instead_of_filling_the_cast() -> None:
    """A technically-compatible but poor match must not be auto-selected."""
    weak = normalize_account_voice(
        account_voice(
            "weak-1",
            languages=(),            # language unverified -> 0.4
            description="",          # no personality signal
            preview_url=None,        # no preview
            use_case="social_media", # not audiobook
            category="generated",
        )
    )
    assert weak is not None
    scores, _ = score_voice(_profile(), weak, required_language="hi")
    assert scores is not None
    from app.services.voice_matcher import total_score

    assert total_score(scores) < MIN_ACCEPT_SCORE

    assert rank_candidates(_profile(), [weak], required_language="hi") == []


def test_rank_returns_best_three_with_score_breakdown() -> None:
    pool = [
        normalize_account_voice(
            account_voice(f"v{i}", name=f"Voice {i}",
                          description=desc, preview_url=prev)
        )
        for i, (desc, prev) in enumerate([
            ("Deep calm measured Hindi storytelling voice for audiobooks",
             "https://example.com/p1.mp3"),
            ("Warm narrator voice, calm and measured for stories",
             "https://example.com/p2.mp3"),
            ("Calm measured narrator voice for audiobook stories",
             "https://example.com/p3.mp3"),
            ("Random chatter voice", "https://example.com/p4.mp3"),
        ])
    ]
    pool = [v for v in pool if v]

    ranked = rank_candidates(_profile(), pool, required_language="hi")

    assert len(ranked) == 3                      # top-3 only
    assert [c.rank for c in ranked] == [1, 2, 3]
    assert ranked[0].total_score >= ranked[1].total_score >= ranked[2].total_score
    for candidate in ranked:
        assert set(candidate.scores) == {
            "language", "accent", "personality", "demographics",
            "audiobook_fit", "quality", "cost_and_limits",
        }
        assert candidate.total_score >= MIN_ACCEPT_SCORE
        assert candidate.preview_url


def test_unestablished_demographics_are_never_penalised() -> None:
    """age/gender unknown -> neutral score; mismatch only when established."""
    profile = _profile(gender=None, age=None, uncertain_attributes=[])
    voice = normalize_account_voice(
        account_voice("v1", gender="female", age="young")
    )
    assert voice is not None
    scores, _ = score_voice(profile, voice, required_language="hi")
    assert scores is not None
    assert scores["demographics"] == 0.7  # neutral, not a mismatch

    # Established male profile vs female voice -> penalised.
    established = _profile(gender="male", age=None, uncertain_attributes=[])
    scores2, notes = score_voice(established, voice, required_language="hi")
    assert scores2 is not None
    assert scores2["demographics"] == 0.0
    assert any("gender mismatch" in n for n in notes)


def test_custom_rate_and_notice_period_are_surfaced_in_notes() -> None:
    voice = normalize_account_voice(
        account_voice("v1", fiat_rate=3.0, notice_period=60)
    )
    assert voice is not None
    scores, notes = score_voice(_profile(), voice, required_language="hi")
    assert scores is not None
    assert scores["cost_and_limits"] < 1.0
    assert any("custom usage rate" in n for n in notes)
    assert any("notice period" in n for n in notes)


def test_locale_match_scores_higher_than_other_region() -> None:
    """Story locale (hi-IN) prefers voices verified for the same region."""
    from app.services.voice_matcher import total_score

    exact = normalize_account_voice(account_voice("reg-1"))          # no locale
    other = normalize_account_voice(account_voice("reg-2"))
    assert exact and other
    exact.locale = "hi-IN"
    other.locale = "en-US"

    exact_scores, _ = score_voice(
        _profile(), exact, required_language="hi", required_locale="hi-IN"
    )
    other_scores, other_notes = score_voice(
        _profile(), other, required_language="hi", required_locale="hi-IN"
    )
    assert exact_scores and other_scores
    assert total_score(exact_scores) > total_score(other_scores)
    assert any("locale differs" in n for n in other_notes)
