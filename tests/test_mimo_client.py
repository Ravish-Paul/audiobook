"""Tests for the Xiaomi MiMo transport client (fully mocked, no credits)."""

from __future__ import annotations

import httpx
import pytest

from app.services import MiMoClient
from app.services.errors import (
    MiMoAPIError,
    MiMoAuthError,
    MiMoConfigError,
    MiMoRateLimitError,
    MiMoTimeoutError,
)
from fakes import FakeCompletion, FakeOpenAIClient

SYSTEM = "You are an expert audiobook narration director."
USER = "Analyze the complete audiobook script."


@pytest.fixture()
def with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic MiMo credentials/settings, independent of any .env."""
    monkeypatch.setenv("MIMO_API_KEY", "sk-mimo-test-key")
    monkeypatch.setenv("MIMO_MODEL_ID", "mimo-v2.6-flash")
    monkeypatch.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    monkeypatch.setenv("MIMO_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("MIMO_MAX_COMPLETION_TOKENS", "8192")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_missing_mimo_api_key_is_reported_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MIMO_API_KEY", "")

    client = MiMoClient()
    with pytest.raises(MiMoConfigError) as exc:
        client.complete_json(system=SYSTEM, user=USER)

    assert exc.value.status_code == 500
    assert "MIMO_API_KEY" in str(exc.value)


def test_api_key_is_excluded_from_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIMO_API_KEY", "sk-SECRET-DO-NOT-LEAK")
    from app.config import get_settings

    assert "sk-SECRET-DO-NOT-LEAK" not in repr(get_settings())


# --------------------------------------------------------------------------- #
# Request contract (official OpenAI SDK shape + Xiaomi endpoint)
# --------------------------------------------------------------------------- #
def test_request_contract_uses_xiaomi_endpoint(
    with_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.services.mimo_client as module

    constructor_kwargs: dict[str, object] = {}
    sdk_client = FakeOpenAIClient(result=FakeCompletion('{"ok": true}'))

    def fake_openai(**kwargs: object) -> object:
        constructor_kwargs.update(kwargs)
        return sdk_client

    monkeypatch.setattr(module, "OpenAI", fake_openai)

    client = MiMoClient()
    content = client.complete_json(system=SYSTEM, user=USER)

    # --- SDK construction ------------------------------------------------- #
    assert constructor_kwargs["api_key"] == "sk-mimo-test-key"
    assert constructor_kwargs["base_url"] == "https://api.xiaomimimo.com/v1"
    assert constructor_kwargs["max_retries"] == 0  # planner owns retries
    assert constructor_kwargs["timeout"] == 120.0

    # --- chat.completions.create ------------------------------------------ #
    request = sdk_client.calls[0]
    assert request["model"] == "mimo-v2.6-flash"
    assert request["max_completion_tokens"] == 8192
    assert request["response_format"] == {"type": "json_object"}
    assert request["stream"] is False
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert request["messages"][0]["content"] == SYSTEM
    assert request["messages"][1]["content"] == USER

    assert content == '{"ok": true}'


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #
def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.xiaomimimo.com/v1/chat/completions")


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=_request())


def test_timeout_maps_to_504(
    with_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.services.mimo_client as module

    error = openai_timeout_error()
    monkeypatch.setattr(
        module, "OpenAI", lambda **_: FakeOpenAIClient(error=error)
    )

    with pytest.raises(MiMoTimeoutError) as exc:
        MiMoClient().complete_json(system=SYSTEM, user=USER)
    assert exc.value.status_code == 504
    assert "seconds" in str(exc.value)


def test_invalid_api_key_maps_to_401(
    with_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.services.mimo_client as module

    error = openai_auth_error()
    monkeypatch.setattr(
        module, "OpenAI", lambda **_: FakeOpenAIClient(error=error)
    )

    with pytest.raises(MiMoAuthError) as exc:
        MiMoClient().complete_json(system=SYSTEM, user=USER)
    assert exc.value.status_code == 401
    assert "MIMO_API_KEY" in str(exc.value)


def test_rate_limit_maps_to_429(
    with_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.services.mimo_client as module

    error = openai_rate_limit_error()
    monkeypatch.setattr(
        module, "OpenAI", lambda **_: FakeOpenAIClient(error=error)
    )

    with pytest.raises(MiMoRateLimitError) as exc:
        MiMoClient().complete_json(system=SYSTEM, user=USER)
    assert exc.value.status_code == 429


def test_connection_error_maps_to_502(
    with_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.services.mimo_client as module

    error = openai_connection_error()
    monkeypatch.setattr(
        module, "OpenAI", lambda **_: FakeOpenAIClient(error=error)
    )

    with pytest.raises(MiMoAPIError) as exc:
        MiMoClient().complete_json(system=SYSTEM, user=USER)
    assert exc.value.status_code == 502
    assert "Could not reach" in str(exc.value)


def test_server_error_maps_to_502(
    with_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.services.mimo_client as module

    error = openai_status_error(500)
    monkeypatch.setattr(
        module, "OpenAI", lambda **_: FakeOpenAIClient(error=error)
    )

    with pytest.raises(MiMoAPIError) as exc:
        MiMoClient().complete_json(system=SYSTEM, user=USER)
    assert exc.value.status_code == 502
    assert "500" in str(exc.value)


def test_empty_response_maps_to_502(
    with_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.services.mimo_client as module

    monkeypatch.setattr(
        module, "OpenAI", lambda **_: FakeOpenAIClient(result=FakeCompletion(None))
    )

    with pytest.raises(MiMoAPIError) as exc:
        MiMoClient().complete_json(system=SYSTEM, user=USER)
    assert "empty" in str(exc.value).lower()


# --------------------------------------------------------------------------- #
# openai exception factories
# --------------------------------------------------------------------------- #
def openai_timeout_error():
    import openai

    return openai.APITimeoutError(request=_request())


def openai_connection_error():
    import openai

    return openai.APIConnectionError(message="conn refused", request=_request())


def openai_auth_error():
    import openai

    return openai.AuthenticationError(
        "invalid api key", response=_response(401), body=None
    )


def openai_rate_limit_error():
    import openai

    return openai.RateLimitError("rate limited", response=_response(429), body=None)


def openai_status_error(status: int):
    import openai

    return openai.APIStatusError(
        f"upstream {status}", response=_response(status), body=None
    )
