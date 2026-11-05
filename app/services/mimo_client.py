"""Xiaomi MiMo client — the AI narration director.

MiMo is reached through Xiaomi's OpenAI-compatible endpoint using the
official `openai` SDK:

    OpenAI(api_key=MIMO_API_KEY, base_url="https://api.xiaomimimo.com/v1")

MiMo decides **how** each paragraph is narrated; it never receives ElevenLabs
credentials and never rewrites the script.

Security notes:
* the key comes from `.env` only, is excluded from `repr()` and is never
  logged or returned in API responses;
* `max_retries=0` on the SDK client — retry policy belongs to the planner,
  which must first validate the JSON plan.
"""

from __future__ import annotations

from typing import Any, Optional

import openai
from openai import OpenAI

from app.config import Settings, get_settings
from app.services.errors import (
    MiMoAPIError,
    MiMoAuthError,
    MiMoConfigError,
    MiMoRateLimitError,
    MiMoTimeoutError,
)

# The narration director system prompt lives with the planner (it needs the
# paragraph schema); this module only owns transport + error mapping.


class MiMoClient:
    """Thin, testable wrapper around `chat.completions.create`."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.settings = settings or get_settings()
        # DI seam: tests inject a fake object with `.chat.completions.create`.
        self._client = client

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    @property
    def client(self) -> OpenAI:
        """Lazily build the SDK client (raises when .env is not set up)."""
        if self._client is None:
            if not self.settings.mimo_api_key:
                raise MiMoConfigError(
                    "MIMO_API_KEY is not configured. Copy .env.example to .env "
                    "and add your Xiaomi MiMo API key."
                )
            self._client = OpenAI(
                api_key=self.settings.mimo_api_key,
                base_url=self.settings.mimo_base_url,
                timeout=self.settings.mimo_timeout_seconds,
                max_retries=0,  # the planner owns the retry policy
            )
        return self._client

    # ------------------------------------------------------------------ #
    # Completion
    # ------------------------------------------------------------------ #
    def complete_json(self, *, system: str, user: str) -> str:
        """Ask MiMo for a JSON object and return the raw content string.

        Raises a `ProjectError` subclass so routes can map it to an HTTP
        status: 500 (no key) / 401 / 429 / 504 (timeout) / 502.
        """
        settings = self.settings
        request_kwargs: dict[str, Any] = {
            "model": settings.mimo_model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": settings.mimo_max_completion_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
            # MiMo-specific: disable the reasoning/thinking channel.
            "extra_body": {"thinking": {"type": "disabled"}},
        }

        try:
            completion = self.client.chat.completions.create(**request_kwargs)
        except openai.APITimeoutError as exc:
            raise MiMoTimeoutError(
                f"MiMo did not respond within "
                f"{settings.mimo_timeout_seconds:.0f} seconds"
            ) from exc
        except openai.AuthenticationError as exc:
            raise MiMoAuthError(
                "MiMo rejected the API key — check MIMO_API_KEY"
            ) from exc
        except openai.RateLimitError as exc:
            raise MiMoRateLimitError(
                "MiMo rate limit reached — retry in a moment"
            ) from exc
        except openai.APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status in (401, 403):
                raise MiMoAuthError(
                    "MiMo rejected the API key — check MIMO_API_KEY"
                ) from exc
            if status == 429:
                raise MiMoRateLimitError(
                    "MiMo rate limit reached — retry in a moment"
                ) from exc
            raise MiMoAPIError(
                f"MiMo API failed with HTTP {status or 'unknown'}"
            ) from exc
        except openai.APIConnectionError as exc:
            raise MiMoAPIError(
                f"Could not reach the Xiaomi MiMo API ({exc.__class__.__name__})"
            ) from exc
        except openai.OpenAIError as exc:
            raise MiMoAPIError(
                f"MiMo request failed ({exc.__class__.__name__})"
            ) from exc

        content = self._extract_content(completion)
        if not content:
            raise MiMoAPIError("MiMo returned an empty response")
        return content

    # ------------------------------------------------------------------ #
    # Response handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_content(completion: Any) -> str:
        """Pull `choices[0].message.content` out of any SDK-ish object."""
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        return content.strip() if isinstance(content, str) else ""
