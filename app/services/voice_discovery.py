"""ElevenLabs Voice Discovery service (spec §3).

Wraps the four official endpoints — nothing is hardcoded and every call is
seam-injectable so automated tests never touch the network:

    GET  /v2/voices                         account voices (next_page_token/has_more)
    GET  /v1/shared-voices                  library discovery (page/has_more)
    GET  /v1/voices/{voice_id}              selected voice details
    POST /v1/voices/add/{public_user_id}/{voice_id}   add a shared voice

Supported search parameters (from the current official docs):

    account: page_size, next_page_token, search, sort, sort_direction,
             voice_type, category, fine_tuning_state, collection_id, gender,
             age, language (list), accent, use_cases (list),
             min_notice_period_days, include_custom_rates,
             include_live_moderated, high_quality, include_total_count,
             voice_ids (list)
    shared:  page_size, page, category, gender, age, accent, language, locale,
             search, use_cases (list), descriptives (list), featured,
             min_notice_period_days, include_custom_rates,
             include_live_moderated, reader_app_enabled, owner_id, sort

List-valued query params are JSON-encoded exactly like the official SDK
examples (`language=["hi"]`). Pagination is capped (`MAX_PAGES`) so a broad
search can never loop forever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from app.config import Settings, get_settings
from app.services.errors import (
    ElevenLabsAPIError,
    ElevenLabsAuthError,
    ElevenLabsConfigError,
    ElevenLabsRateLimitError,
    ElevenLabsTimeoutError,
)

# Pagination limits (page_size 100 = official maximum per request).
ACCOUNT_PAGE_SIZE = 100
SHARED_PAGE_SIZE = 100
MAX_PAGES = 5

_MAX_ERROR_DETAIL = 300


class VoiceDetailsNotFoundError(ElevenLabsAPIError):
    """`GET /v1/voices/{voice_id}` returned 404 — voice not in this account."""


@dataclass
class VoiceSearchFilters:
    """Normalised filters shared by both discovery endpoints."""

    language: Optional[str] = None          # ISO 639-1, e.g. "hi"
    locale: Optional[str] = None            # e.g. "hi-IN" (shared endpoint)
    gender: Optional[str] = None            # matched case-insensitively
    age: Optional[str] = None
    accent: Optional[str] = None
    use_case: Optional[str] = None          # e.g. "audiobook" / "narrative_story"
    category: Optional[str] = None          # professional | famous | high_quality | …
    search: Optional[str] = None            # free-text (name/description/labels)
    extra: dict[str, Any] = field(default_factory=dict)


def _safe_detail(response: httpx.Response) -> str:
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            raw = payload.get("detail") or payload.get("error") or payload
            if isinstance(raw, dict):
                raw = raw.get("message") or raw.get("status_message") or raw
            detail = str(raw)
        else:
            detail = str(payload)
    except Exception:
        try:
            detail = response.text[:_MAX_ERROR_DETAIL]
        except Exception:
            detail = ""
    detail = " ".join(detail.split())
    return f" — {detail[:_MAX_ERROR_DETAIL]}" if detail else ""


class ElevenLabsVoiceDiscovery:
    """Read-only voice discovery + shared-voice import (per-request injectable)."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.settings = settings or get_settings()
        # Test seam: tests inject httpx.MockTransport — no credits, no network.
        self._transport = transport

    # ------------------------------------------------------------------ #
    # HTTP plumbing
    # ------------------------------------------------------------------ #
    def _require_configuration(self) -> None:
        if not self.settings.elevenlabs_api_key:
            raise ElevenLabsConfigError(
                "ELEVENLABS_API_KEY is not configured. Copy .env.example to .env "
                "and add your ElevenLabs API key."
            )

    def _headers(self) -> dict[str, str]:
        return {
            "xi-api-key": self.settings.elevenlabs_api_key,
            "accept": "application/json",
        }

    @property
    def _v1_base(self) -> str:
        """`ELEVENLABS_BASE_URL` — the API root INCLUDING `/v1` (convention)."""
        return self.settings.elevenlabs_base_url.rstrip("/")

    @property
    def _root(self) -> str:
        """Host root without `/v1` — `/v2/voices` lives OUTSIDE `/v1`."""
        base = self._v1_base
        if base.endswith("/v1"):
            return base[: -len("/v1")]
        return base

    async def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        self._require_configuration()
        client_kwargs: dict[str, object] = {
            "timeout": self.settings.elevenlabs_timeout_seconds,
            "follow_redirects": False,
        }
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        clean_params = {k: v for k, v in params.items() if v not in (None, "")}
        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.get(
                    url, headers=self._headers(), params=clean_params
                )
        except httpx.TimeoutException as exc:
            raise ElevenLabsTimeoutError(
                "ElevenLabs voice discovery did not respond within "
                f"{self.settings.elevenlabs_timeout_seconds:.0f} seconds"
            ) from exc
        except httpx.HTTPError as exc:
            raise ElevenLabsAPIError(
                f"Could not reach the ElevenLabs API ({exc.__class__.__name__})"
            ) from exc

        return self._to_json(response, context=f"GET {url}")

    async def _post(
        self, url: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        self._require_configuration()
        client_kwargs: dict[str, object] = {
            "timeout": self.settings.elevenlabs_timeout_seconds,
            "follow_redirects": False,
        }
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        headers = dict(self._headers())
        headers["content-type"] = "application/json"
        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(
                    url, headers=headers, json=body
                )
        except httpx.TimeoutException as exc:
            raise ElevenLabsTimeoutError(
                "ElevenLabs did not respond within "
                f"{self.settings.elevenlabs_timeout_seconds:.0f} seconds"
            ) from exc
        except httpx.HTTPError as exc:
            raise ElevenLabsAPIError(
                f"Could not reach the ElevenLabs API ({exc.__class__.__name__})"
            ) from exc

        return self._to_json(response, context=f"POST {url}")

    def _to_json(self, response: httpx.Response, *, context: str) -> dict[str, Any]:
        status = response.status_code
        detail = _safe_detail(response)

        if status in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            raise ElevenLabsAuthError(
                "ElevenLabs rejected the API key — check ELEVENLABS_API_KEY"
                + detail
            )
        if status == httpx.codes.TOO_MANY_REQUESTS:
            retry_after = response.headers.get("retry-after")
            hint = f"; retry after {retry_after}s" if retry_after else ""
            raise ElevenLabsRateLimitError(
                "ElevenLabs rate limit reached" + hint + detail
            )
        if status == httpx.codes.NOT_FOUND:
            raise ElevenLabsAPIError(f"{context} — not found (404)" + detail)
        if status >= 400:
            raise ElevenLabsAPIError(
                f"{context} failed with HTTP {status}" + detail
            )

        try:
            data = response.json()
        except Exception as exc:
            raise ElevenLabsAPIError(
                f"{context} returned invalid JSON"
            ) from exc
        if not isinstance(data, dict):
            raise ElevenLabsAPIError(
                f"{context} returned an unexpected payload"
            )
        return data

    # ------------------------------------------------------------------ #
    # Account voices — GET /v2/voices
    # ------------------------------------------------------------------ #
    async def list_account_voices(
        self, filters: Optional[VoiceSearchFilters] = None
    ) -> list[dict[str, Any]]:
        """All matching account voices (paginated via next_page_token)."""
        filters = filters or VoiceSearchFilters()
        voices: list[dict[str, Any]] = []
        token: Optional[str] = None

        for _page in range(MAX_PAGES):
            params: dict[str, Any] = {
                "page_size": ACCOUNT_PAGE_SIZE,
                "include_total_count": "false",
            }
            if filters.language:
                params["language"] = json.dumps([filters.language])
            if filters.gender:
                params["gender"] = filters.gender
            if filters.age:
                params["age"] = filters.age
            if filters.accent:
                params["accent"] = filters.accent
            if filters.use_case:
                params["use_cases"] = json.dumps([filters.use_case])
            if filters.category:
                params["category"] = filters.category
            if filters.search:
                params["search"] = filters.search
            if token:
                params["next_page_token"] = token
            params.update(filters.extra)

            data = await self._get(f"{self._root}/v2/voices", params)
            page_items = data.get("voices")
            if isinstance(page_items, list):
                voices.extend(page_items)

            if not data.get("has_more"):
                break
            token = data.get("next_page_token")
            if not token:
                break
        return voices

    # ------------------------------------------------------------------ #
    # Shared voices — GET /v1/shared-voices
    # ------------------------------------------------------------------ #
    async def list_shared_voices(
        self, filters: Optional[VoiceSearchFilters] = None
    ) -> list[dict[str, Any]]:
        """Library voices matching the filters (paginated via `page`)."""
        filters = filters or VoiceSearchFilters()
        voices: list[dict[str, Any]] = []

        for page in range(MAX_PAGES):
            params: dict[str, Any] = {
                "page_size": SHARED_PAGE_SIZE,
                "page": page,
                "sort": "usage_character_count_1y",  # proven voices first
            }
            if filters.language:
                params["language"] = filters.language
            if filters.locale:
                params["locale"] = filters.locale
            if filters.gender:
                params["gender"] = filters.gender
            if filters.age:
                params["age"] = filters.age
            if filters.accent:
                params["accent"] = filters.accent
            if filters.use_case:
                params["use_cases"] = json.dumps([filters.use_case])
            if filters.category:
                params["category"] = filters.category
            if filters.search:
                params["search"] = filters.search
            params.update(filters.extra)

            data = await self._get(f"{self._v1_base}/shared-voices", params)
            page_items = data.get("voices")
            if isinstance(page_items, list):
                voices.extend(page_items)

            if not data.get("has_more"):
                break
        return voices

    # ------------------------------------------------------------------ #
    # Voice details — GET /v1/voices/{voice_id}
    # ------------------------------------------------------------------ #
    async def get_voice(self, voice_id: str) -> dict[str, Any]:
        try:
            return await self._get(f"{self._v1_base}/voices/{voice_id}", {})
        except ElevenLabsAPIError as exc:
            # The API answers 404 — and sometimes 400 "not found" — for a
            # voice that is not in this account (e.g. a library voice that
            # was never imported). Both mean the same to the caller.
            if "not found" in str(exc).lower():
                raise VoiceDetailsNotFoundError(
                    f"voice '{voice_id}' is not available in this account"
                ) from exc
            raise

    # ------------------------------------------------------------------ #
    # Import a shared voice — POST /v1/voices/add/{public_user_id}/{voice_id}
    # ------------------------------------------------------------------ #
    async def add_shared_voice(
        self, public_owner_id: str, voice_id: str, new_name: str
    ) -> str:
        """Add a library voice to the account; returns the (possibly new) id."""
        if not public_owner_id.strip():
            raise ElevenLabsAPIError(
                "public_owner_id is required to add a shared voice"
            )
        data = await self._post(
            f"{self._v1_base}/voices/add/{public_owner_id}/{voice_id}",
            {"new_name": new_name or f"Shared voice {voice_id[:8]}"},
        )
        added = data.get("voice_id")
        return str(added) if added else voice_id
