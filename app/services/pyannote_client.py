"""Async client for the official pyannoteAI REST API.

Reference: https://docs.pyannote.ai (OpenAPI spec validated 2026-11)

    POST {base}/v1/media/input      -> {"url": "<pre-signed PUT URL>"}   (201)
    PUT  <pre-signed URL>           -> upload the local audio bytes
    POST {base}/v1/diarize          -> {"jobId": ..., "status": ...}
         body: {"url": "media://OBJECT_KEY", "model": "precision-3",
                "transcription": true,
                "transcriptionConfig": {"model": "faster-whisper-large-v3-turbo"}}
    GET  {base}/v1/jobs/{jobId}     -> {"jobId", "status", "output": {...}}
         status: created | pending | running | succeeded | failed | canceled
         `output` exists ONLY while the job is succeeded and is deleted
         24h after completion — we persist it immediately (see service).

Billing note (docs.pyannote.ai/administration/billing): every SUCCESSFUL
/v1/diarize job has a **20-second minimum charge**. Nothing in this module
submits a job by itself — submission only happens through the analysis
service, and the automated test suite never reaches a real endpoint.

Security notes:
* the API key is sent as an `Authorization: Bearer …` header only, never
  logged, never returned to clients, excluded from `repr()`;
* pre-signed upload URLs are secrets: they are never logged and never
  included in error messages or API responses (bodies are redacted);
* `media://` object keys are unique per analysis (project + analysis UUID),
  never a fixed shared key.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import httpx

from app.config import Settings, get_settings
from app.services.errors import (
    PyannoteAPIError,
    PyannoteAuthError,
    PyannoteConfigError,
    PyannoteRateLimitError,
    PyannoteRequestError,
    PyannoteTimeoutError,
)

# Upstream error messages are useful but must stay short and safe to show.
_MAX_ERROR_DETAIL = 300

# The provider documents `media://object-key` with this pattern
# (maxLength 255 for the whole `media://…` URL).
_MEDIA_KEY_RE = re.compile(r"^[a-zA-Z0-9\-_./]{1,240}$")

# Any URL with a query string is treated as a potential pre-signed secret.
_URL_WITH_QUERY_RE = re.compile(r"https?://[^\s\"']+\?[^\s\"']+")


def _redact(text: str) -> str:
    """Mask URLs carrying query strings (pre-signed credentials)."""
    return _URL_WITH_QUERY_RE.sub("[redacted-url]", text)


def redact_url_secrets(text: str) -> str:
    """Public helper: strip pre-signed URL credentials from any text."""
    return _redact(text)


def _safe_detail(response: httpx.Response) -> str:
    """Extract a short, human readable message from a pyannoteAI error body.

    The request URL itself is never included (the PUT target is a
    pre-signed URL); response bodies are redacted defensively as well.
    """
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            raw = payload.get("message") or payload.get("error") or payload
            if isinstance(raw, dict):
                raw = raw.get("message") or raw
            detail = str(raw)
        else:
            detail = str(payload)
    except Exception:  # non-JSON body (HTML error page, empty body, ...)
        try:
            detail = response.text[:_MAX_ERROR_DETAIL]
        except Exception:
            detail = ""

    detail = " ".join(_redact(detail).split())
    if not detail:
        return ""
    return f" — {detail[:_MAX_ERROR_DETAIL]}"


class PyannoteClient:
    """Small wrapper around the three pyannoteAI endpoints we use."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.settings = settings or get_settings()
        # Test seam: tests inject `httpx.MockTransport(...)` so no real
        # request is ever made and no pyannote credits are consumed.
        self._transport = transport

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    @property
    def media_endpoint(self) -> str:
        return f"{self.settings.pyannote_base_url}/v1/media/input"

    @property
    def diarize_endpoint(self) -> str:
        return f"{self.settings.pyannote_base_url}/v1/diarize"

    def job_endpoint(self, job_id: str) -> str:
        return f"{self.settings.pyannote_base_url}/v1/jobs/{job_id}"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.pyannote_api_key}",
            "accept": "application/json",
        }

    def _require_configuration(self) -> None:
        """Fail fast with a clear message when .env is not set up yet."""
        if not self.settings.pyannote_api_key:
            raise PyannoteConfigError(
                "PYANNOTE_API_KEY is not configured. Copy .env.example to "
                ".env and add your pyannoteAI API key "
                "(https://dashboard.pyannote.ai)."
            )

    # ------------------------------------------------------------------ #
    # Step 1: temporary media upload (local files are not public)
    # ------------------------------------------------------------------ #
    async def create_media_upload_url(self, object_key: str) -> str:
        """POST /v1/media/input -> pre-signed PUT URL for `object_key`.

        The returned URL is a secret: callers must only PUT the audio
        bytes to it and must never log or expose it.
        """
        self._require_configuration()
        key = object_key.strip().lstrip("/")
        if not key or not _MEDIA_KEY_RE.match(key):
            raise PyannoteRequestError(
                "Invalid media object key — only letters, digits, '-', '_', "
                "'.' and '/' are allowed (max 240 characters)."
            )

        data = await self._request(
            "POST",
            self.media_endpoint,
            json_body={"url": f"media://{key}"},
            # Idempotent: the same object key simply yields a fresh URL.
            retries=self.settings.pyannote_max_attempts - 1,
        )
        url = data.get("url") if isinstance(data, dict) else None
        if not isinstance(url, str) or not url.strip():
            raise PyannoteAPIError(
                "pyannoteAI media upload did not return a pre-signed URL"
            )
        return url.strip()

    async def upload_media(self, presigned_url: str, data: bytes) -> None:
        """PUT the local audiobook bytes to the pre-signed URL."""
        self._require_configuration()
        if not data:
            raise PyannoteRequestError(
                "Refusing to upload an empty audio file to pyannoteAI"
            )

        # Pre-signed URLs carry credentials in the query string: the error
        # messages below deliberately never contain the URL itself.
        await self._request(
            "PUT",
            presigned_url,
            content=data,
            headers={"content-type": "application/octet-stream"},
            retries=self.settings.pyannote_max_attempts - 1,
            upload=True,
        )

    # ------------------------------------------------------------------ #
    # Step 2: submit the asynchronous diarization job
    # ------------------------------------------------------------------ #
    def build_diarize_payload(self, object_key: str) -> dict[str, Any]:
        """The documented POST /v1/diarize body (validated against the spec)."""
        return {
            "url": f"media://{object_key.strip().lstrip('/')}",
            "model": self.settings.pyannote_model_id,
            "transcription": True,
            "transcriptionConfig": {
                "model": self.settings.pyannote_transcription_model
            },
        }

    async def submit_diarization(self, object_key: str) -> dict[str, Any]:
        """POST /v1/diarize -> {"jobId", "status", "warning"?}.

        NEVER retried automatically: a duplicate submission would run (and
        bill) the same audio twice. Timeouts surface as a 504 and the
        caller decides whether to re-check for an existing job.
        """
        self._require_configuration()
        data = await self._request(
            "POST",
            self.diarize_endpoint,
            json_body=self.build_diarize_payload(object_key),
            retries=0,
        )
        if not isinstance(data, dict) or not data.get("jobId"):
            raise PyannoteAPIError(
                "pyannoteAI did not return a job id for the diarization job"
            )
        return data

    # ------------------------------------------------------------------ #
    # Step 3: job status / result
    # ------------------------------------------------------------------ #
    async def get_job(self, job_id: str) -> dict[str, Any]:
        """GET /v1/jobs/{job_id} -> job dict (with `output` once succeeded).

        Retried on transport errors/5xx (idempotent GET, never billed).
        """
        self._require_configuration()
        job_id = (job_id or "").strip()
        if not job_id:
            raise PyannoteRequestError("No pyannoteAI job id was stored")

        data = await self._request(
            "GET",
            self.job_endpoint(job_id),
            retries=self.settings.pyannote_max_attempts - 1,
        )
        if not isinstance(data, dict):
            raise PyannoteAPIError(
                "pyannoteAI returned an unexpected job payload"
            )
        status = data.get("status")
        if not isinstance(status, str) or not status.strip():
            raise PyannoteAPIError(
                "pyannoteAI job response is missing its status"
            )
        return data

    # ------------------------------------------------------------------ #
    # Transport + error mapping
    # ------------------------------------------------------------------ #
    async def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        content: Optional[bytes] = None,
        headers: Optional[dict[str, str]] = None,
        retries: int = 0,
        upload: bool = False,
    ) -> Any:
        """Perform one HTTP call with bounded retries for idempotent calls.

        `upload=True` targets a pre-signed URL (different host, no API key
        header — the signature in the URL is the credential).
        """
        self._require_configuration()
        timeout = self.settings.pyannote_timeout_seconds
        attempts = max(0, retries) + 1
        merged_headers = dict(headers or {})
        if not upload:
            merged_headers.update(self._headers)

        last_transport_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            client_kwargs: dict[str, object] = {
                "timeout": timeout,
                # Pre-signed storage URLs may redirect (307/308) on PUT.
                "follow_redirects": upload,
            }
            if self._transport is not None:
                client_kwargs["transport"] = self._transport

            try:
                async with httpx.AsyncClient(**client_kwargs) as client:
                    request = client.build_request(
                        method,
                        url,
                        headers=merged_headers,
                        json=json_body,
                        content=content,
                    )
                    response = await client.send(request)
            except httpx.TimeoutException as exc:
                last_transport_error = exc
                if attempt < attempts:
                    continue
                raise PyannoteTimeoutError(
                    f"pyannoteAI did not respond within "
                    f"{timeout:.0f} seconds"
                ) from exc
            except httpx.HTTPError as exc:
                last_transport_error = exc
                if attempt < attempts:
                    continue
                target = "the pyannoteAI API" if not upload else "the media upload"
                raise PyannoteAPIError(
                    f"Could not reach {target} ({exc.__class__.__name__})"
                ) from exc

            status = response.status_code
            if 200 <= status < 300:
                if method.upper() == "PUT":
                    return None
                try:
                    return response.json()
                except Exception as exc:
                    raise PyannoteAPIError(
                        "pyannoteAI returned a non-JSON response"
                    ) from exc

            # Server-side hiccup on an idempotent call -> bounded retry.
            if status >= 500 and attempt < attempts:
                continue

            raise self._map_error(response, upload=upload)

        # Defensive: the loop always returns or raises.
        raise PyannoteAPIError(
            f"pyannoteAI request failed ({last_transport_error!r})"
        )

    @staticmethod
    def _map_error(response: httpx.Response, *, upload: bool) -> Exception:
        """Translate an upstream HTTP status into a domain error."""
        status = response.status_code
        detail = _safe_detail(response)
        where = "the media upload" if upload else "pyannoteAI"

        if status in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            return PyannoteAuthError(
                f"{where} rejected the API key — check PYANNOTE_API_KEY" + detail
            )
        if status in (httpx.codes.BAD_REQUEST, httpx.codes.UNPROCESSABLE_ENTITY):
            return PyannoteRequestError(
                f"{where} rejected the request payload" + detail
            )
        if status == httpx.codes.PAYMENT_REQUIRED:
            return PyannoteAPIError(
                "pyannoteAI reports that a subscription/payment is required "
                "(HTTP 402)" + detail
            )
        if status == httpx.codes.TOO_MANY_REQUESTS:
            retry_after = response.headers.get("retry-after")
            hint = f"; retry after {retry_after}s" if retry_after else ""
            return PyannoteRateLimitError(
                "pyannoteAI rate limit reached" + hint + detail
            )
        if status == httpx.codes.NOT_FOUND:
            return PyannoteAPIError(
                f"{where} target was not found (HTTP 404) — a media:// key or "
                "job id may have expired" + detail
            )
        return PyannoteAPIError(
            f"{where} failed with HTTP {status}" + detail
        )
