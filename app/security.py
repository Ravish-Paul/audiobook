"""Minimal private-beta security layer (no user-management system).

Activated ONLY when tester tokens are configured in the environment
(`BETA_ACCESS_TOKENS="name:token,…"` and/or `BETA_ACCESS_TOKEN`). With no
tokens configured the app behaves exactly as before (development/tests).

What it adds when beta mode is on:

* **Access control** — everything requires a token: the frontend, every API
  route, project routes and media downloads. Only the login page, its
  stylesheet and the logout endpoint are public. A token arrives either as
  `Authorization: Bearer <token>` / `X-Beta-Token: <token>` (API clients)
  or as an HMAC-signed HttpOnly session cookie set by `POST /auth/login`.
* **Project isolation** — every project records its owning tester; requests
  for another tester's project (including `/media/projects/{id}/…`) get the
  same 404 as a missing project, so existence never leaks.
* **Limits** — per-tester generation allowances (persistent, per-tester
  configurable via `BETA_TESTER_GENERATION_LIMITS`), per-minute rate
  limiting (stricter on login) and duplicate-submission protection for the
  credit-spending endpoints.
* **API docs** — `/docs`, `/openapi.json` and `/redoc` are disabled.

Provider API keys never leave the backend: they are read from the
environment here and are never rendered anywhere.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import re
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from app import plan_identity
from app.config import Settings, get_settings

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SESSION_COOKIE = "beta_session"

# Public in beta mode: the login/logout endpoints, the login page stylesheet
# and the browser favicon request. EVERYTHING else requires a token
# (including /static/app.js and the frontend at /).
PUBLIC_PATHS = frozenset({"/auth/login", "/auth/logout", "/favicon.ico", "/static/style.css"})

# API documentation is disabled for external beta testers.
DOCS_PATHS = frozenset({"/docs", "/redoc", "/openapi.json"})

# Endpoints that spend provider credits (MiMo / ElevenLabs / pyannoteAI).
GENERATION_SUFFIXES = (
    "/analyze-narration",
    "/analyze-narration/preview",
    "/generate-audio",
    "/analyze-characters",
    "/create-multi-voice-plan",
    "/audio-analysis",
    "/voice-casting/audition",
)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


# --------------------------------------------------------------------------- #
# Helpers: identity
# --------------------------------------------------------------------------- #
def current_tester(request: Request) -> str:
    """FastAPI dependency: the tester name for the current request.

    Returns "local" when beta mode is off (development / tests), so project
    ownership still works deterministically there.
    """
    return getattr(request.state, "tester_id", None) or "local"


def _lookup_token(settings: Settings, token: str) -> Optional[str]:
    """Constant-time token lookup -> tester name (None when invalid)."""
    for name, configured in settings.beta_access_tokens:
        if hmac.compare_digest(token, configured):
            return name
    return None


def _authenticate(request: Request, settings: Settings) -> Optional[str]:
    """Resolve the requester's tester name, or None when unauthenticated."""
    authorization = request.headers.get("authorization", "")
    if authorization:
        if not authorization.lower().startswith("bearer "):
            return None
        return _lookup_token(settings, authorization[7:].strip())

    header_token = request.headers.get("x-beta-token", "")
    if header_token:
        return _lookup_token(settings, header_token.strip())

    cookie = request.cookies.get(SESSION_COOKIE, "")
    if cookie:
        return _verify_session(cookie, settings)
    return None


# --------------------------------------------------------------------------- #
# Helpers: session cookies (signed with the environment-configured key)
# --------------------------------------------------------------------------- #
def _sign_session(name: str, settings: Settings) -> tuple[str, int]:
    expires_at = int(time.time()) + settings.beta_session_ttl_hours * 3600
    payload = f"{name}.{expires_at}"
    signature = hmac.new(
        settings.beta_session_key, payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{payload}.{signature}", expires_at


def _verify_session(value: str, settings: Settings) -> Optional[str]:
    parts = value.split(".")
    if len(parts) != 3:
        return None
    name, expires_raw, signature = parts
    payload = f"{name}.{expires_raw}"
    expected = hmac.new(
        settings.beta_session_key, payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return None
    try:
        expires_at = int(expires_raw)
    except ValueError:
        return None
    if expires_at < time.time():
        return None
    if name not in dict(settings.beta_access_tokens):
        return None  # token rotated away -> session invalid
    return name


# --------------------------------------------------------------------------- #
# Helpers: rate limiting + duplicate-submission protection (in-memory)
# --------------------------------------------------------------------------- #
_RATE_LOCK = threading.Lock()
_RATE_WINDOWS: dict[str, tuple[float, int]] = {}


def _rate_ok(key: str, limit_per_minute: int) -> bool:
    """Fixed one-minute window counter. `limit <= 0` disables the check."""
    if limit_per_minute <= 0:
        return True
    with _RATE_LOCK:
        now = time.time()
        start, count = _RATE_WINDOWS.get(key, (now, 0))
        if now - start >= 60.0:
            start, count = now, 0
        _RATE_WINDOWS[key] = (start, count + 1)
        if len(_RATE_WINDOWS) > 10_000:  # cheap pruning
            cutoff = now - 120.0
            for stale in [k for k, (s, _) in _RATE_WINDOWS.items() if s < cutoff]:
                _RATE_WINDOWS.pop(stale, None)
        return count + 1 <= limit_per_minute


_DEDUP_LOCK = threading.Lock()
_IN_FLIGHT: dict[str, bool] = {}
_LAST_COMPLETED: dict[str, float] = {}


def _duplicate_begin(key: str, window_seconds: int) -> Optional[str]:
    """Reserve a generation slot, or return the duplicate error message."""
    with _DEDUP_LOCK:
        if _IN_FLIGHT.get(key):
            return "This generation is already being processed — please wait."
        last = _LAST_COMPLETED.get(key, 0.0)
        if window_seconds > 0 and time.time() - last < window_seconds:
            return (
                "Duplicate submission — this generation request was just "
                "completed. Please wait a moment before retrying."
            )
        _IN_FLIGHT[key] = True
        return None


def _duplicate_end(key: str, succeeded: bool) -> None:
    with _DEDUP_LOCK:
        _IN_FLIGHT.pop(key, None)
        if succeeded:
            _LAST_COMPLETED[key] = time.time()


# --------------------------------------------------------------------------- #
# Helpers: per-tester generation ledger (persisted on disk)
# --------------------------------------------------------------------------- #
# One COMPLETE audiobook workflow — narration planning (MiMo + plan
# validation) followed by audio generation (ElevenLabs segments, audio
# processing, FFmpeg assembly, storage) — shares ONE `generation_id` and
# consumes EXACTLY ONE generation allowance: the planning call opens the
# workflow, the generate-audio call joins it. MiMo retries and internal
# pipeline steps never charge separately. Auxiliary provider calls
# (narration preview, character analysis, casting audition, diarization)
# are NOT audiobook generations: they never reserve or consume the
# allowance and are never blocked by it — the allowance counts complete
# audiobook workflows only. They keep their rate limiting and
# duplicate-submission protection.
#
# Usage per tester is persistent and lifetime: the stored `used` counter
# never resets on restart or configuration change, and the effective limit
# is re-read from the environment on every request, so raising a limit makes
# the extra allowance available immediately while lowering it below the
# current usage blocks generation again. `remaining = max(0, limit - used)`.
#
# Conservative provider-failure policy: an allowance is reserved when the
# request is accepted and is NEVER refunded afterwards — a provider call may
# already have been billed.
#
# DEPLOYMENT CONSTRAINT — SINGLE UVICORN WORKER ONLY. `_QUOTA_LOCK` is a
# `threading.Lock`: it makes the check-and-increment atomic only inside ONE
# OS process. The beta deployment must therefore run a single worker
# (`uvicorn app.main:app --workers 1`, the default). Multi-worker quota
# safety is NOT provided and must not be claimed: with `--workers N > 1`
# each process would hold its own lock, read-modify-write the same
# `quota.json` concurrently and could lose updates (two requests spending
# the same last allowance). Before enabling multiple workers the ledger
# needs cross-process coordination — e.g. an OS file lock around the
# read-modify-write of `quota.json` / `generations.json`, or moving the
# counters into a transactional shared store (SQLite/Redis). The
# persistence format itself is unchanged for the single-worker beta.
_QUOTA_LOCK = threading.Lock()

_GENERATION_KIND_PLAN = "plan"
_GENERATION_KIND_GENERATE = "generate"
_GENERATION_KIND_AUX = "aux"
_GENERATION_KIND_OTHER = "other"

# Paths whose requests open the shared workflow of a complete audiobook.
_PLAN_SUFFIXES = ("/analyze-narration", "/create-multi-voice-plan")

# Auxiliary provider calls that are NOT audiobook generations. They never
# reserve or consume the tester's complete-audiobook allowance and are never
# blocked by it — the allowance counts complete audiobook workflows only.
# They keep their rate limiting and duplicate-submission protection.
_AUX_SUFFIXES = (
    "/analyze-narration/preview",
    "/analyze-characters",
    "/voice-casting/audition",
    "/audio-analysis",
)


def _quota_path(settings: Settings) -> Path:
    # `_beta` is not a UUID, so it can never collide with a project and is
    # unreachable through /media/projects/{uuid}/…
    return settings.storage_dir / "_beta" / "quota.json"


def _generations_path(settings: Settings) -> Path:
    """Ledger of generation transactions (open workflows + history)."""
    return settings.storage_dir / "_beta" / "generations.json"


def _generation_kind(path: str) -> str:
    """Classify a credit-spending path for the generation ledger.

    `aux` paths are auxiliary provider calls (preview, character analysis,
    audition, diarization) — never audiobook generations. Unknown billable
    paths fall back to `other` (one allowance, conservative default).
    """
    if path.endswith("/generate-audio"):
        return _GENERATION_KIND_GENERATE
    if any(path.endswith(suffix) for suffix in _AUX_SUFFIXES):
        return _GENERATION_KIND_AUX
    if any(path.endswith(suffix) for suffix in _PLAN_SUFFIXES):
        return _GENERATION_KIND_PLAN
    return _GENERATION_KIND_OTHER


def effective_generation_limit(settings: Settings, tester: str) -> int:
    """The tester's generation allowance (individual override wins).

    Central resolver — every generation route accounts against it.
    """
    overrides = settings.beta_tester_generation_limits
    if tester in overrides:
        return overrides[tester]
    return settings.beta_generation_limit


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass  # a missing counter file must not break generation


def _used_of(entry: Any) -> int:
    """Lifetime generation count (legacy entries stored it as `count`)."""
    if not isinstance(entry, dict):
        return 0
    try:
        return int(entry.get("used", entry.get("count", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def generation_usage(settings: Settings, tester: str) -> dict[str, int]:
    """The authenticated tester's OWN usage — never anyone else's."""
    with _QUOTA_LOCK:
        used = _used_of(_read_json(_quota_path(settings)).get(tester))
    limit = effective_generation_limit(settings, tester)
    return {
        "generation_limit": limit,
        "generations_used": used,
        "generations_remaining": max(0, limit - used),
        "max_script_characters": settings.max_script_characters,
    }


# The generation_id of the request currently being handled (set by the
# middleware). Every internal operation of the workflow reads the SAME id:
# narration planning, MiMo retries, plan validation, ElevenLabs segments,
# audio processing, FFmpeg assembly and the stored audiobook.
_GENERATION_ID: ContextVar[str] = ContextVar("generation_id", default="")


def current_generation_id() -> Optional[str]:
    """The `generation_id` of the ongoing generation transaction, if any."""
    return _GENERATION_ID.get() or None


# True when THIS `create-multi-voice-plan` request joined a plan workflow
# whose allowance was already spent (set by the middleware, like
# `_GENERATION_ID`). The route forwards it to the planner so a recovered
# plan is never rebuilt or charged twice.
_MV_PLAN_JOINED: ContextVar[bool] = ContextVar("mv_plan_joined", default=False)


def mv_plan_joined() -> bool:
    """Whether the current plan request reused an already-charged workflow."""
    return bool(_MV_PLAN_JOINED.get())


def _ledger_charge(
    settings: Settings,
    tester: str,
    project_id: Optional[str],
    kind: str,
    *,
    mv_plan_request: bool = False,
) -> tuple[Optional[str], Optional[str], bool]:
    """Reserve one generation allowance for this request.

    Returns `(error_message, generation_id, charged)`. A `generate-audio`
    call joins the workflow opened by the project's planning call (same
    `generation_id`, no second charge); every other call starts its own
    generation and consumes exactly one allowance. The check-and-increment
    is atomic under `_QUOTA_LOCK`, so two requests can never both spend the
    same remaining allowance. Reservations are never refunded.

    Auxiliary provider calls (`kind == "aux"` — narration preview, character
    analysis, voice audition, diarization) are NOT audiobook generations:
    they never reserve or consume the allowance and are never blocked by it.
    They still receive a `generation_id` for artifact traceability and are
    recorded in the history with `charged: false`.

    `mv_plan_request` marks `POST …/create-multi-voice-plan`. It JOINS the
    project's existing plan workflow — same server-computed fingerprint,
    joinable lifecycle status, open ledger workflow with the same id — so a
    repeated plan request (e.g. after the browser lost `localStorage`)
    reuses the allowance already spent instead of charging again. A fresh
    charge atomically records `begin_creation` (status `creating`) in the
    same critical section, so charge and plan identity can never diverge.
    """
    if kind == _GENERATION_KIND_AUX:
        return None, str(uuid.uuid4()), False

    limit = effective_generation_limit(settings, tester)
    with _QUOTA_LOCK:
        usage = _read_json(_quota_path(settings))
        ledger = _read_json(_generations_path(settings))
        workflows = ledger.setdefault("workflows", {})
        history = ledger.setdefault("history", [])
        key = f"{tester}:{project_id}" if project_id else None
        now = datetime.now(timezone.utc).isoformat()

        fingerprint: Optional[str] = None
        join_state: Optional[dict[str, Any]] = None
        if mv_plan_request and project_id:
            fingerprint = plan_identity.compute_fingerprint(settings, project_id)
            join_state = plan_identity.joinable_state(
                settings, project_id, tester, fingerprint
            )

        charged = False
        if kind == _GENERATION_KIND_GENERATE and key and key in workflows:
            # Same complete workflow (planning already consumed its allowance).
            generation_id = workflows[key].get("generation_id") or str(uuid.uuid4())
        elif (
            join_state is not None
            and key in workflows
            and workflows[key].get("generation_id") == join_state.get("generation_id")
        ):
            # Durable plan recovery: this exact plan workflow is already paid
            # for and still open — join it at ZERO extra allowance.
            generation_id = join_state.get("generation_id") or str(uuid.uuid4())
        else:
            used = _used_of(usage.get(tester))
            if used >= limit:
                return (
                    f"Generation limit reached ({used} of {limit} audiobook "
                    "generations used for this tester account) — contact the "
                    "administrator for an increased allowance.",
                    None,
                    False,
                )
            usage[tester] = {
                "used": used + 1,
                "last_generation_at": now,
            }
            generation_id = str(uuid.uuid4())
            charged = True
            if kind == _GENERATION_KIND_PLAN and key:
                # Open the shared workflow; generate-audio will join it.
                # (A newer planning call supersedes an older open workflow.)
                workflows[key] = {
                    "generation_id": generation_id,
                    "tester": tester,
                    "project_id": project_id,
                    "opened_at": now,
                }
            if mv_plan_request and project_id:
                # Atomically with the charge: this plan build is now in
                # flight under this generation_id (the planner flips it to
                # `ready`, the failure path to `failed`).
                plan_identity.begin_creation(
                    settings, project_id, tester, fingerprint, generation_id
                )

        _write_json(_quota_path(settings), usage)
        _write_json(_generations_path(settings), ledger)
        return None, generation_id, charged


def _ledger_complete(
    settings: Settings,
    tester: str,
    project_id: Optional[str],
    kind: str,
    generation_id: Optional[str],
    charged: bool,
    succeeded: bool,
    path: str = "",
) -> None:
    """Record the finished request; close the workflow when audio is done.

    The allowance is never released here (conservative provider-failure
    policy) — completion only keeps the transaction history accurate.

    A FAILED `create-multi-voice-plan` (``path``) also closes its plan
    record (`creating → failed`): the reservation stays spent and the next
    attempt opens a fresh workflow.
    """
    with _QUOTA_LOCK:
        ledger = _read_json(_generations_path(settings))
        workflows = ledger.setdefault("workflows", {})
        history = ledger.setdefault("history", [])
        key = f"{tester}:{project_id}" if project_id else None
        history.append(
            {
                "generation_id": generation_id,
                "tester": tester,
                "project_id": project_id,
                "kind": kind,
                "charged": charged,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "outcome": "ok" if succeeded else "failed",
            }
        )
        if (
            kind == _GENERATION_KIND_GENERATE
            and key
            and workflows.get(key, {}).get("generation_id") == generation_id
        ):
            workflows.pop(key, None)  # the audiobook is generated (or failed)
        if not succeeded and project_id and path.endswith(
            "/create-multi-voice-plan"
        ):
            plan_identity.mark_failed(settings, project_id, generation_id)
        _write_json(_generations_path(settings), ledger)


# --------------------------------------------------------------------------- #
# Helpers: paths
# --------------------------------------------------------------------------- #
def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _wants_html(request: Request) -> bool:
    return "text/html" in request.headers.get("accept", "")


def _project_id_in_path(path: str) -> Optional[str]:
    """The UUID project id in /projects/{id}… or /media/projects/{id}/… URLs."""
    for prefix in ("/projects/", "/media/projects/"):
        if path.startswith(prefix):
            segment = path[len(prefix) :].split("/", 1)[0]
            return segment if _UUID_RE.match(segment) else None
    return None


def _project_owned_by(settings: Settings, project_id: str, tester: str) -> bool:
    meta_path = settings.storage_dir / project_id / "project.json"
    try:
        record = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("owner") == tester


def _is_generation_path(path: str) -> bool:
    return path.startswith("/projects/") and any(
        path.endswith(suffix) for suffix in GENERATION_SUFFIXES
    )


# --------------------------------------------------------------------------- #
# ASGI middleware
# --------------------------------------------------------------------------- #
def _json(detail: str, status_code: int, headers: Optional[dict] = None) -> Response:
    return JSONResponse(
        {"detail": detail}, status_code=status_code, headers=headers or {}
    )


class BetaSecurityMiddleware:
    """Pure-ASGI middleware enforcing the private-beta security layer.

    With no tokens configured (`settings.beta_enabled == False`) it is a
    transparent pass-through, so development and the existing test suite are
    unaffected.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        settings = get_settings()
        if not settings.beta_enabled:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        path = request.url.path
        method = request.method.upper()

        # 1) API documentation is disabled for the beta.
        if path in DOCS_PATHS or path.startswith("/docs/"):
            await _json("Not Found", 404)(scope, receive, send)
            return

        # 2) Public endpoints (login page + friends).
        if path in PUBLIC_PATHS:
            if path == "/auth/login" and method == "POST":
                if not _rate_ok(
                    f"login:{_client_ip(request)}",
                    settings.beta_login_rate_limit_per_minute,
                ):
                    await _json(
                        "Too many login attempts — please wait a minute.", 429,
                        {"Retry-After": "60"},
                    )(scope, receive, send)
                    return
            await self.app(scope, receive, send)
            return

        # 3) Authentication (bearer/header token or session cookie).
        tester = _authenticate(request, settings)
        if tester is None:
            if _wants_html(request) and method == "GET":
                target = f"/auth/login?next={path}"
                await RedirectResponse(target, status_code=302)(scope, receive, send)
            else:
                await _json("Authentication required", 401)(scope, receive, send)
            return

        # Downstream handlers (and `current_tester`) read this.
        scope.setdefault("state", {})["tester_id"] = tester

        # 4) Basic rate limiting per tester.
        if not _rate_ok(f"req:{tester}", settings.beta_rate_limit_per_minute):
            await _json(
                "Rate limit exceeded — please slow down.", 429, {"Retry-After": "60"}
            )(scope, receive, send)
            return

        # 5) Project isolation: another tester's project is a 404, never a 403.
        project_id = _project_id_in_path(path)
        if project_id is not None and not _project_owned_by(settings, project_id, tester):
            await _json(f"Project '{project_id}' not found", 404)(scope, receive, send)
            return

        # 6) Generation endpoints: duplicate guard + per-tester allowance.
        #    One complete audiobook workflow (planning + generate-audio)
        #    shares one generation_id and consumes exactly one allowance.
        #    Auxiliary calls (preview / character analysis / audition /
        #    diarization) keep their duplicate guard but never touch the
        #    complete-audiobook allowance.
        if method == "POST" and _is_generation_path(path):
            key = f"{tester}:{path}"
            duplicate = _duplicate_begin(key, settings.beta_duplicate_window_seconds)
            if duplicate:
                await _json(duplicate, 409)(scope, receive, send)
                return

            kind = _generation_kind(path)
            mv_plan_request = (
                kind == _GENERATION_KIND_PLAN
                and path.endswith("/create-multi-voice-plan")
            )
            quota_error, generation_id, charged = _ledger_charge(
                settings, tester, project_id, kind, mv_plan_request=mv_plan_request
            )
            if quota_error:
                _duplicate_end(key, succeeded=False)
                await _json(quota_error, 429)(scope, receive, send)
                return

            # Downstream services attach this id to the artifacts they store.
            joined = mv_plan_request and not charged
            scope.setdefault("state", {})["generation_id"] = generation_id
            scope["state"]["mv_plan_joined"] = joined
            generation_token = _GENERATION_ID.set(generation_id or "")
            joined_token = _MV_PLAN_JOINED.set(joined)
            status_box: dict[str, int] = {}

            async def send_wrapper(message) -> None:  # noqa: ANN001
                if message["type"] == "http.response.start":
                    status_box["status"] = message["status"]
                await send(message)

            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                succeeded = status_box.get("status", 500) < 400
                _GENERATION_ID.reset(generation_token)
                _MV_PLAN_JOINED.reset(joined_token)
                _duplicate_end(key, succeeded=succeeded)
                _ledger_complete(
                    settings, tester, project_id, kind,
                    generation_id, charged, succeeded, path,
                )
            return

        await self.app(scope, receive, send)


# --------------------------------------------------------------------------- #
# Login / logout endpoints
# --------------------------------------------------------------------------- #
auth_router = APIRouter(prefix="/auth", tags=["Auth"])

_PRIVACY_NOTICE = (
    "<p class=\"privacy-note\">This is a <strong>private beta</strong>. "
    "Uploaded scripts and generated audio are stored on the beta server for "
    "testing purposes only — do not upload confidential manuscripts. "
    "You can delete your projects and their files at any time.</p>"
)


def _login_page(next_url: str, error: str = "") -> str:
    error_html = (
        f'<p class="status error">{html.escape(error)}</p>' if error else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Private beta — AI Audiobook</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <header class="page-header">
    <h1>Create your audiobook</h1>
    <p class="subtitle">Private beta access — enter your tester token.</p>
  </header>
  <main class="page">
    <section class="card">
      {error_html}
      <form method="post" action="/auth/login">
        <input type="hidden" name="next" value="{html.escape(next_url)}" />
        <label for="token">Beta access token</label>
        <input id="token" name="token" type="password"
               autocomplete="current-password" required
               class="title-input" style="font-size:1rem" />
        <div class="result-actions" style="margin-top:1rem">
          <button class="btn btn-primary" type="submit">Enter the beta</button>
        </div>
      </form>
      {_PRIVACY_NOTICE}
    </section>
  </main>
</body>
</html>"""


def _safe_next(url: str) -> str:
    """Only same-site absolute paths — never an open redirect."""
    if url.startswith("/") and not url.startswith("//") and "\\" not in url:
        return url
    return "/"


@auth_router.get("/login", response_class=HTMLResponse)
def login_form(request: Request) -> Response:
    """Login page (public)."""
    settings = get_settings()
    if not settings.beta_enabled:
        return RedirectResponse("/", status_code=302)
    return HTMLResponse(_login_page(_safe_next(request.query_params.get("next", "/"))))


@auth_router.post("/login")
async def login_submit(request: Request) -> Response:
    """Validate the tester token and set the signed session cookie."""
    settings = get_settings()
    if not settings.beta_enabled:
        return RedirectResponse("/", status_code=303)

    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        body = await request.json()
        token = str((body or {}).get("token", ""))
        next_url = _safe_next(str((body or {}).get("next", "/")))
    else:
        form = await request.form()
        token = str(form.get("token", ""))
        next_url = _safe_next(str(form.get("next", "/")))

    name = _lookup_token(settings, token.strip())
    if name is None:
        if content_type.startswith("application/json"):
            return _json("Invalid access token", 401)
        return HTMLResponse(
            _login_page(next_url, error="That access token is not valid."),
            status_code=401,
        )

    cookie_value, _ = _sign_session(name, settings)
    response = RedirectResponse(next_url, status_code=303)
    response.set_cookie(
        SESSION_COOKIE,
        cookie_value,
        max_age=settings.beta_session_ttl_hours * 3600,
        httponly=True,
        samesite="lax",
        secure=settings.beta_cookie_secure,
        path="/",
    )
    return response


@auth_router.post("/logout")
def logout() -> Response:
    """Clear the session cookie (public — logging out must always work)."""
    response = RedirectResponse("/auth/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response
