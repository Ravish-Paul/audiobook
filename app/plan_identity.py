"""Server-side identity + lifecycle of a project's multi-voice plan.

The browser is NEVER trusted for plan existence or validity: a plan is
"the same logical workflow" only when the SERVER recomputes the same
fingerprint from the stored project data (script, character cast and
generation-relevant narration preferences) and the persisted plan state
still matches it.

Storage layout (one file per project, next to the other project artefacts):

    storage/projects/{project_id}/multi_voice_plan_state.json
        {
          "version": 1,
          "project_id": "...",
          "owner": "tester-name",          # authenticated project owner
          "plan_id": "uuid",               # identity of this plan record
          "fingerprint": "v1:<sha256>",    # canonical input fingerprint
          "generation_id": "uuid|null",    # the workflow that paid for it
          "status": "creating|ready|started|completed|failed",
          "created_at": "...", "updated_at": "...",
          "history": [ {…superseded records…} ]
        }

Lifecycle:

    creating  -> the allowance was reserved, the plan is being built
    ready     -> multi_voice_plan.json exists and is reusable
    started   -> generate-audio is producing the audiobook
    completed -> the audiobook finished (a NEW audiobook needs a new charge)
    failed    -> the workflow ended in an error (allowance is never refunded)

Only `ready`/`started` (an UNFINISHED workflow) may be recovered without a
second allowance; `completed`/`failed` follow the existing regeneration and
conservative provider-failure policies.

Import rule: this module must stay importable from `app.security` without
pulling in `app.services` (which imports `app.security`), so it only uses
the standard library plus `app.config`.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import Settings

PLAN_STATE_FILENAME = "multi_voice_plan_state.json"

# Bump when the hashed inputs change shape — old fingerprints then simply
# stop matching instead of being compared against a different algorithm.
FINGERPRINT_VERSION = "v2"

# Statuses whose workflow is still unfinished and therefore recoverable.
REUSABLE_STATUSES = ("ready", "started")

# Statuses whose OPEN ledger workflow may be joined by another plan request
# without spending a second allowance. `creating` is included so that a
# concurrent/retried request during a long build joins the build that is
# already paid for instead of charging again.
JOINABLE_STATUSES = ("ready", "started", "creating")

# Previous statuses whose already-built plan artefact may be reused for
# unchanged inputs (no second MiMo call): an unfinished workflow, an
# audiobook that already completed, or a workflow that failed.
REUSABLE_PREVIOUS_STATUSES = ("ready", "started", "completed", "failed")

_STATE_LOCK = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _project_dir(settings: Settings, project_id: str) -> Path:
    return settings.storage_dir / project_id


def state_path(settings: Settings, project_id: str) -> Path:
    """storage/projects/{project_id}/multi_voice_plan_state.json"""
    return _project_dir(settings, project_id) / PLAN_STATE_FILENAME


def plan_path(settings: Settings, project_id: str) -> Path:
    """storage/projects/{project_id}/multi_voice_plan.json (the artefact)."""
    return _project_dir(settings, project_id) / settings.multi_voice_plan_filename


def _canonical_json(value: Any) -> str:
    """Deterministic serialisation: key order, spacing and unicode never
    change the bytes, so equivalent inputs always hash identically."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def cast_identity(cast: dict[str, Any]) -> dict[str, Any]:
    """Generation-relevant slice of `character_cast.json`.

    The story analysis (character identities, roles, descriptions), the
    approval flag and the locked speaker -> voice mapping decide what a
    multi-voice plan is built from. Candidate *rankings* are deliberately
    excluded: re-ranking voices without touching a lock must not invalidate
    a paid plan.
    """
    casting = cast.get("casting")
    locks: dict[str, Any] = {}
    if isinstance(casting, dict):
        for speaker_id in sorted(casting):
            entry = casting.get(speaker_id) or {}
            selected = entry.get("selected") if isinstance(entry, dict) else None
            locks[speaker_id] = {
                "voice_id": (selected or {}).get("voice_id")
                if isinstance(selected, dict)
                else None,
                "status": entry.get("status") if isinstance(entry, dict) else None,
            }
    return {
        "story": cast.get("story"),
        "approved": bool(cast.get("approved")),
        "locks": locks,
    }


def compute_fingerprint(settings: Settings, project_id: str) -> Optional[str]:
    """Canonical fingerprint of the CURRENT script + cast + preferences.

    Returns `None` when an input is missing or unreadable — an unknown
    fingerprint must never be treated as "the same plan".
    """
    if not project_id:
        return None
    directory = _project_dir(settings, project_id)

    script_path = directory / settings.stored_script_filename
    if not script_path.is_file():
        return None
    try:
        script_text = script_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None

    cast_path = directory / settings.character_cast_filename
    if not cast_path.is_file():
        return None
    cast = _read_json(cast_path)
    if not isinstance(cast, dict):
        return None

    prefs_path = directory / settings.narration_preferences_filename
    preferences: Any = None
    if prefs_path.is_file():
        payload = _read_json(prefs_path)
        if isinstance(payload, dict):
            preferences = payload.get("preferences", payload)
        elif payload is not None:
            preferences = payload

    analysis_path = directory / settings.story_analysis_filename
    story_analysis: Any = None
    if analysis_path.is_file():
        stored = _read_json(analysis_path)
        if isinstance(stored, dict):
            story_analysis = stored.get("analysis", stored)

    payload = {
        "version": FINGERPRINT_VERSION,
        "script": _sha256(script_text),
        "cast": cast_identity(cast),
        "preferences": preferences,
        "story_analysis": story_analysis,
    }
    return f"{FINGERPRINT_VERSION}:{_sha256(_canonical_json(payload))}"


# --------------------------------------------------------------------------- #
# Plan state record
# --------------------------------------------------------------------------- #
def read_state(settings: Settings, project_id: str) -> Optional[dict[str, Any]]:
    """The persisted plan state, or `None` when there is none yet."""
    if not project_id:
        return None
    data = _read_json(state_path(settings, project_id))
    return data if isinstance(data, dict) and data else None


def _write_state(settings: Settings, project_id: str, state: dict[str, Any]) -> None:
    path = state_path(settings, project_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)  # readers never see a half-written record
    except OSError:
        pass  # a missing state file only means "no plan to reuse"


def begin_creation(
    settings: Settings,
    project_id: str,
    owner: Optional[str],
    fingerprint: Optional[str],
    generation_id: Optional[str],
) -> dict[str, Any]:
    """Record that a NEW billable plan workflow was opened for this project.

    Called from the quota ledger while `_QUOTA_LOCK` is held, so the charge
    and the "a plan is being created" marker land together: a concurrent
    matching request reuses this workflow instead of paying for a second one.
    """
    now = _utcnow().isoformat()
    with _STATE_LOCK:
        previous = read_state(settings, project_id)
        history = list(previous.get("history") or []) if previous else []
        if previous:
            history.append(
                {
                    key: previous.get(key)
                    for key in (
                        "plan_id",
                        "generation_id",
                        "fingerprint",
                        "status",
                        "created_at",
                        "updated_at",
                    )
                }
            )
        state: dict[str, Any] = {
            "version": 1,
            "project_id": project_id,
            "owner": owner,
            "plan_id": str(uuid.uuid4()),
            "fingerprint": fingerprint,
            "generation_id": generation_id,
            "status": "creating",
            # The artefact on disk (if any) still belongs to the SUPERSEDED
            # record — `create()` must validate it against `previous_*`, not
            # against the new workflow's fingerprint.
            "previous_status": previous.get("status") if previous else None,
            "previous_fingerprint": previous.get("fingerprint") if previous else None,
            "created_at": now,
            "updated_at": now,
            "history": history[-20:],  # keep the record bounded
        }
        _write_state(settings, project_id, state)
        return state


def _update(
    settings: Settings,
    project_id: str,
    generation_id: Optional[str],
    *,
    statuses: tuple[str, ...],
    new_status: str,
    adopt_generation: bool = False,
) -> Optional[dict[str, Any]]:
    """Transition the record when it belongs to this workflow.

    Status writes only ever touch the record of the generation they were
    called for, so a stale response can never rewrite a newer workflow.
    `adopt_generation` lets `mark_started` take over a record whose previous
    workflow is definitively over (`ready` / `completed` / `failed`) — e.g.
    the first `generate-audio` after a completed plan, which always opens a
    new `generation_id`.
    """
    with _STATE_LOCK:
        state = read_state(settings, project_id)
        if not state:
            return None
        if state.get("status") not in statuses:
            return None
        record_id = state.get("generation_id")
        if generation_id and record_id not in (None, generation_id):
            if not (
                adopt_generation
                and state.get("status") in ("ready", "completed", "failed")
            ):
                return None  # another workflow owns this record
            state["generation_id"] = generation_id
        elif adopt_generation and generation_id:
            state["generation_id"] = generation_id
        state["status"] = new_status
        state["updated_at"] = _utcnow().isoformat()
        _write_state(settings, project_id, state)
        return state


def mark_ready(
    settings: Settings,
    project_id: str,
    generation_id: Optional[str],
    fingerprint: Optional[str],
) -> Optional[dict[str, Any]]:
    """The plan artefact is saved and reusable (`creating` -> `ready`)."""
    with _STATE_LOCK:
        state = read_state(settings, project_id)
        if not state:
            return None
        if generation_id:
            state["generation_id"] = generation_id
        if fingerprint:
            state["fingerprint"] = fingerprint
        state["status"] = "ready"
        state["updated_at"] = _utcnow().isoformat()
        _write_state(settings, project_id, state)
        return state


def mark_started(
    settings: Settings, project_id: str, generation_id: Optional[str]
) -> Optional[dict[str, Any]]:
    return _update(
        settings,
        project_id,
        generation_id,
        statuses=("ready", "started", "creating", "completed", "failed"),
        new_status="started",
        adopt_generation=True,
    )


def mark_completed(
    settings: Settings, project_id: str, generation_id: Optional[str]
) -> Optional[dict[str, Any]]:
    return _update(
        settings,
        project_id,
        generation_id,
        statuses=("started", "ready", "creating"),
        new_status="completed",
    )


def mark_failed(
    settings: Settings, project_id: str, generation_id: Optional[str]
) -> Optional[dict[str, Any]]:
    """Terminal failure — the reservation is never refunded (existing policy)."""
    return _update(
        settings,
        project_id,
        generation_id,
        statuses=("creating", "started", "ready"),
        new_status="failed",
    )


def joinable_state(
    settings: Settings,
    project_id: str,
    owner: Optional[str],
    fingerprint: Optional[str],
) -> Optional[dict[str, Any]]:
    """The record when an OPEN workflow for these exact inputs may be joined.

    Used by the quota ledger so a repeated/`localStorage`-less plan request
    reuses the allowance that was already spent instead of charging again.
    The plan artefact itself is validated by the planner (it may still be
    under construction while `status == "creating"`).
    """
    if not fingerprint:
        return None
    state = read_state(settings, project_id)
    if not state:
        return None
    if state.get("status") not in JOINABLE_STATUSES:
        return None
    if state.get("fingerprint") != fingerprint:
        return None  # script / cast / preferences changed (Case C)
    if not state.get("generation_id"):
        return None
    if owner is not None and state.get("owner") not in (None, owner):
        return None  # never hand another tester's plan to this session
    return state


def artefact_is_reusable(
    state: Optional[dict[str, Any]], fingerprint: Optional[str]
) -> bool:
    """May the `multi_voice_plan.json` on disk be reused for these inputs?

    * `ready` / `started` — the artefact belongs to this very record;
    * `creating` — a fresh charge superseded the previous record, so the
      artefact belongs to `previous_*` (it must match the current inputs
      and come from a finished or unfinished build);
    * `completed` / `failed` — no fresh charge ran (direct planner use);
      the artefact still describes the current inputs.
    """
    if not state or not fingerprint:
        return False
    status = state.get("status")
    if status in REUSABLE_STATUSES:
        return state.get("fingerprint") == fingerprint
    if status == "creating":
        return (
            state.get("previous_status") in REUSABLE_PREVIOUS_STATUSES
            and state.get("previous_fingerprint") == fingerprint
        )
    if status in ("completed", "failed"):
        return state.get("fingerprint") == fingerprint
    return False


def plan_status(
    settings: Settings, project_id: str, owner: Optional[str] = None
) -> dict[str, Any]:
    """Server-authoritative status payload for `GET .../multi-voice-plan/status`."""
    fingerprint = compute_fingerprint(settings, project_id)
    state = read_state(settings, project_id)
    if owner is not None and state and state.get("owner") not in (None, owner):
        state = None  # cross-tester records are never exposed
    exists = state is not None
    status = state.get("status") if state else "none"
    valid = bool(
        exists
        and fingerprint
        and plan_path(settings, project_id).is_file()
        and artefact_is_reusable(state, fingerprint)
    )
    ready = bool(valid and status in REUSABLE_STATUSES)
    return {
        "project_id": project_id,
        "exists": exists,
        "status": status if exists else "none",
        "valid": valid,
        "ready_for_generate": ready,
        "plan_id": state.get("plan_id") if state else None,
        "plan_fingerprint": state.get("fingerprint") if state else None,
        "generation_id": state.get("generation_id") if state else None,
        "created_at": state.get("created_at") if state else None,
        "updated_at": state.get("updated_at") if state else None,
    }
