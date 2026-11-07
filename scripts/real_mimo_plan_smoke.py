#!/usr/bin/env python3
"""REAL MiMo narration-plan smoke test for the "Room 207" project.

Safety contract (enforced by guards in this process, not by convention):

* a REAL MiMo API call is made through the existing `MiMoClient` +
  `NarrationPlanner` (no fakes, no mocks);
* ElevenLabs is NEVER called  -> its three network methods and every async
  HTTP request are booby-trapped (the local capability read used by the
  response projection is allowed, it never touches the network);
* FFmpeg is NEVER run         -> `subprocess.Popen` is booby-trapped;
* NO audiobook audio is generated and NO audiobook generation quota is spent:
  the planner is called as a plain service (no HTTP request), so the beta
  middleware that counts quota/rate/duplicate submissions never runs;
* the project's existing files (project.json, script.txt, any audio) must be
  byte-identical afterwards — only `narration_preferences.json`,
  `narration_plan.json` and `mimo_debug/*` may appear.

Preferences: Cinematic preset (restrained_cinematic), acting intensity 2,
Minimal audio tags (tag_density=minimal).

Run from the repo root:

    .\\.venv\\Scripts\\python.exe scripts\\real_mimo_plan_smoke.py

Exit code 0 = plan built and every check passed; 1 = something failed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROJECT_ID = "857167a6-c734-4ac9-a62e-9caf8d48b260"  # "कमरा नंबर 207"
REPORT_DIR_NAME = "mimo_debug"

# Files this smoke test is allowed to create inside the project folder.
ALLOWED_NEW = ("narration_preferences.json", "narration_plan.json", REPORT_DIR_NAME)

# Words that indicate over-acting in the instruction fields.
OVERACTING_RE = re.compile(
    r"\b(whisper\w*|hushed|hush|trembling|tremble|breathless|breathlessly"
    r"|shout\w*|yell\w*|scream\w*|sob\w*|cry\w*|weep\w*"
    r"|dramatically|dramatic|overact\w*)\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Guards — installed BEFORE any service that could spend money is constructed.
# --------------------------------------------------------------------------- #
GUARDS = {"subprocess_calls": [], "elevenlabs_calls": 0, "async_http_calls": []}


def install_guards() -> None:
    """Make forbidden side effects fail loudly instead of silently happening."""
    import subprocess

    # Must stay a CLASS: asyncio.windows_events subclasses subprocess.Popen.
    real_popen = subprocess.Popen

    class ForbiddenPopen(real_popen):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            GUARDS["subprocess_calls"].append(str(args[0]) if args else repr(kwargs))
            raise RuntimeError(
                "SMOKE TEST GUARD: spawning a process (FFmpeg) is forbidden here"
            )

    subprocess.Popen = ForbiddenPopen  # type: ignore[assignment]

    from app.services import audio_service, elevenlabs_client

    def forbidden_elevenlabs(*args, **kwargs):  # type: ignore[no-untyped-def]
        GUARDS["elevenlabs_calls"] += 1
        raise RuntimeError(
            "SMOKE TEST GUARD: an ElevenLabs network call is forbidden here"
        )

    # Constructing the client is harmless (tts_projection only reads local
    # capability data) — the three methods that touch the network are not.
    for method in ("synthesize", "synthesize_result", "fetch_audio_effects_catalog"):
        setattr(elevenlabs_client.ElevenLabsClient, method, forbidden_elevenlabs)
    # Belt and braces: NO async HTTP at all. The openai/MiMo client is
    # synchronous, so this cannot interfere with the real MiMo call.
    import httpx

    real_async_send = httpx.AsyncClient.send

    async def forbidden_async_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
        GUARDS["async_http_calls"].append(str(request.url))
        raise RuntimeError(
            f"SMOKE TEST GUARD: async HTTP request forbidden: {request.url}"
        )

    httpx.AsyncClient.send = forbidden_async_send  # type: ignore[method-assign]
    GUARDS["_real_async_send"] = real_async_send

    audio_service.AudioService.__init__ = forbidden_elevenlabs  # type: ignore[method-assign]

    # Keep a reference so linters cannot claim `real_popen` is unused.
    GUARDS["_real_popen"] = real_popen


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_project(project_dir: Path) -> dict[str, str]:
    return {
        str(p.relative_to(project_dir)).replace("\\", "/"): sha256(p)
        for p in sorted(project_dir.rglob("*"))
        if p.is_file()
    }


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    install_guards()

    from app.config import get_settings
    from app.schemas.narration import DELIVERY_STYLES, NarrationPlan
    from app.schemas.narration_preferences import NarrationPreferences
    from app.services.errors import ProjectError
    from app.services.narration_planner import NarrationPlanner

    settings = get_settings()
    project_dir = settings.storage_dir / PROJECT_ID
    quota_path = settings.storage_dir / "_beta" / "quota.json"

    # ------------------------------------------------------------------ #
    # Preconditions
    # ------------------------------------------------------------------ #
    if not settings.mimo_configured:
        print("FAIL: MIMO_API_KEY is not configured — cannot make a real call.")
        return 1
    if not (project_dir / "script.txt").exists():
        print(f"FAIL: no script.txt for project {PROJECT_ID}")
        return 1

    before = snapshot_project(project_dir)
    before_quota = {
        "exists": quota_path.exists(),
        "sha256": sha256(quota_path) if quota_path.exists() else None,
        "mtime": quota_path.stat().st_mtime if quota_path.exists() else None,
    }
    audio_before = sorted(
        str(p.relative_to(project_dir)) for p in project_dir.rglob("*")
        if p.is_file() and "audio" in p.parts
    )

    preferences = NarrationPreferences(
        preset="restrained_cinematic",
        acting_intensity=2,
        whisper_policy="user_selected",
        whisper_paragraphs=[],
        tag_density="minimal",
        custom_direction="",
    )

    print("=" * 78)
    print("REAL MiMo narration-plan smoke test — Room 207")
    print(f"  project     : {PROJECT_ID}")
    print(f"  model       : {settings.mimo_model_id} @ {settings.mimo_base_url}")
    print(f"  attempts    : {settings.mimo_max_attempts} (timeout "
          f"{settings.mimo_timeout_seconds:.0f}s each)")
    print(f"  preferences : preset=restrained_cinematic intensity=2 "
          f"tags=minimal whisper=user_selected")
    print(f"  guards      : ElevenLabs=FORBIDDEN subprocess(FFmpeg)=FORBIDDEN")
    print("=" * 78)

    # ------------------------------------------------------------------ #
    # The real MiMo round trip (Pydantic validation + policy checks inside)
    # ------------------------------------------------------------------ #
    planner = NarrationPlanner(settings)
    started = time.time()
    try:
        response = planner.analyze(PROJECT_ID, preferences=preferences)
    except ProjectError as exc:
        elapsed = time.time() - started
        print(f"\nFAIL after {elapsed:.1f}s: {exc.__class__.__name__}")
        print(f"  {exc}")
        failures = getattr(exc, "problems", None)
        if failures:
            print("  validation problems:")
            for problem in failures:
                print(f"    - {problem}")
        _write_report(
            project_dir, settings, ok=False, error=str(exc),
            problems=list(failures or []), elapsed=elapsed,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - guard trips / unexpected errors
        elapsed = time.time() - started
        import traceback

        print(f"\nFAIL after {elapsed:.1f}s: {exc.__class__.__name__}: {exc}")
        traceback.print_exc()
        _write_report(
            project_dir, settings, ok=False,
            error=f"{exc.__class__.__name__}: {exc}", elapsed=elapsed,
        )
        return 1
    elapsed = time.time() - started

    # ------------------------------------------------------------------ #
    # Checks
    # ------------------------------------------------------------------ #
    plan_path = project_dir / "narration_plan.json"
    checks: dict[str, object] = {}

    # 1) Pydantic validation of the persisted plan.
    try:
        persisted = NarrationPlan.model_validate(read_json(plan_path))
        pydantic_ok, pydantic_detail = True, "persisted narration_plan.json re-validates"
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        persisted, pydantic_ok, pydantic_detail = None, False, f"{exc}"
    checks["pydantic_validation"] = {
        "passed": pydantic_ok and response.saved_at is not None,
        "detail": pydantic_detail,
        "saved_at": str(response.saved_at),
    }

    # 2) delivery_style values.
    styles = [p.delivery_style for p in response.paragraphs]
    invalid = [s for s in styles if s not in DELIVERY_STYLES]
    checks["delivery_styles"] = {
        "all_allowed": not invalid,
        "used": {s: styles.count(s) for s in sorted(set(styles))},
        "invalid": invalid,
        "allowed": list(DELIVERY_STYLES),
    }

    # 3) preferences preserved in the final plan.
    got_prefs = response.preferences.model_dump() if response.preferences else None
    want_prefs = preferences.model_dump()
    checks["preferences_preserved"] = {
        "preserved": got_prefs == want_prefs,
        "expected": want_prefs,
        "actual": got_prefs,
    }

    # 4) whispering / dramatic instructions in the instruction fields.
    hits = []
    fields = ("narration_direction", "tts_text", "emotion", "delivery_style")
    for paragraph in response.paragraphs:
        for field in fields:
            value = getattr(paragraph, field, None)
            if isinstance(value, str):
                for match in OVERACTING_RE.finditer(value):
                    hits.append(
                        {
                            "paragraph_index": paragraph.paragraph_index,
                            "field": field,
                            "term": match.group(0),
                            "context": value[max(0, match.start() - 40): match.end() + 40],
                        }
                    )
    tone_hits = [m.group(0) for m in OVERACTING_RE.finditer(response.overall_story_tone)]
    checks["overacting_instructions"] = {
        "whisper_or_dramatic_hits": hits,
        "overall_story_tone_hits": tone_hits,
        "clean": not hits and not tone_hits,
        "note": (
            "Scanned narration_direction, tts_text, emotion, delivery_style and "
            "overall_story_tone for whisper/hushed/trembling/breathless/shout/"
            "yell/scream/sob/crie/weep/dramatic/overact."
        ),
    }

    # 5) locations.
    debug_dir = project_dir / REPORT_DIR_NAME
    raw_files = sorted(str(p.relative_to(project_dir)).replace("\\", "/")
                       for p in debug_dir.glob("*.json")) if debug_dir.is_dir() else []
    checks["artifacts"] = {
        "plan": str(plan_path),
        "raw_mimo_attempts_dir": str(debug_dir),
        "raw_mimo_attempts_files": raw_files,
        "preferences": str(project_dir / "narration_preferences.json"),
    }

    # ------------------------------------------------------------------ #
    # Safety verification (nothing else may have changed)
    # ------------------------------------------------------------------ #
    after = snapshot_project(project_dir)
    new_files = sorted(set(after) - set(before))
    changed_files = sorted(k for k in set(after) & set(before) if after[k] != before[k])
    # The smoke test's OWN artifacts may change on a re-run; everything else
    # (project.json, script.txt, any audio, …) must stay byte-identical.
    is_artifact = lambda name: name.startswith(ALLOWED_NEW)  # noqa: E731
    unexpected_new = [f for f in new_files if not is_artifact(f)]
    protected_changed = [f for f in changed_files if not is_artifact(f)]
    safety = {
        "protected_files_unchanged": not protected_changed,
        "changed_files": changed_files,
        "protected_changed_files": protected_changed,
        "new_files": new_files,
        "unexpected_new_files": unexpected_new,
        "audio_files_before": audio_before,
        "audio_files_after": sorted(
            str(p.relative_to(project_dir)) for p in project_dir.rglob("*")
            if p.is_file() and "audio" in p.parts
        ),
        "elevenlabs_network_calls": GUARDS["elevenlabs_calls"],
        "async_http_calls": GUARDS["async_http_calls"],
        "subprocess_ffmpeg_calls": GUARDS["subprocess_calls"],
        "quota_before": before_quota,
        "quota_after": {
            "exists": quota_path.exists(),
            "sha256": sha256(quota_path) if quota_path.exists() else None,
            "mtime": quota_path.stat().st_mtime if quota_path.exists() else None,
        },
    }
    safety["quota_untouched"] = safety["quota_before"] == safety["quota_after"]
    safety["no_audio_created"] = safety["audio_files_after"] == audio_before
    safety["ok"] = all(
        [
            safety["protected_files_unchanged"],
            not unexpected_new,
            safety["elevenlabs_network_calls"] == 0,
            not safety["async_http_calls"],
            not safety["subprocess_ffmpeg_calls"],
            safety["quota_untouched"],
            safety["no_audio_created"],
        ]
    )

    checks["safety"] = safety
    checks["elapsed_seconds"] = round(elapsed, 1)
    checks["paragraph_count"] = len(response.paragraphs)
    checks["language"] = response.language
    checks["overall_story_tone"] = response.overall_story_tone

    all_ok = bool(
        checks["pydantic_validation"]["passed"]
        and checks["delivery_styles"]["all_allowed"]
        and checks["preferences_preserved"]["preserved"]
        and checks["overacting_instructions"]["clean"]
        and safety["ok"]
    )
    checks["all_checks_passed"] = all_ok

    report_path = _write_report(project_dir, settings, ok=all_ok, checks=checks,
                                elapsed=elapsed)
    _print_report(checks, report_path)
    return 0 if all_ok else 1


def _write_report(
    project_dir: Path,
    settings,  # noqa: ANN001 - Settings
    *,
    ok: bool,
    elapsed: float = 0.0,
    checks: dict | None = None,
    error: str | None = None,
    problems: list | None = None,
) -> Path:
    debug_dir = project_dir / REPORT_DIR_NAME
    debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = debug_dir / f"smoke_report_{stamp}.json"
    dump(
        path,
        {
            "test": "real MiMo narration-plan smoke test",
            "project_id": PROJECT_ID,
            "model": settings.mimo_model_id,
            "ok": ok,
            "error": error,
            "problems": problems or [],
            "elapsed_seconds": round(elapsed, 1),
            "checks": checks or {},
        },
    )
    return path


def _print_report(checks: dict, report_path: Path) -> None:
    def mark(value: bool) -> str:
        return "PASS" if value else "FAIL"

    pv = checks["pydantic_validation"]
    ds = checks["delivery_styles"]
    pf = checks["preferences_preserved"]
    oa = checks["overacting_instructions"]
    sf = checks["safety"]

    print()
    print("=" * 78)
    print(f"RESULT: {'ALL CHECKS PASSED' if checks['all_checks_passed'] else 'CHECKS FAILED'}"
          f"  ({checks['elapsed_seconds']}s, {checks['paragraph_count']} paragraphs)")
    print("=" * 78)
    print(f"1. Pydantic validation ............ {mark(bool(pv['passed']))} — {pv['detail']}")
    print(f"   saved_at ...................... {pv['saved_at']}")
    print(f"2. delivery_style values ......... {mark(bool(ds['all_allowed']))}")
    print(f"   used .......................... {ds['used']}")
    if ds["invalid"]:
        print(f"   INVALID ....................... {ds['invalid']}")
    print(f"3. preferences preserved ......... {mark(bool(pf['preserved']))}")
    print(f"   expected ...................... {pf['expected']}")
    print(f"   actual ........................ {pf['actual']}")
    print(f"4. no whisper/dramatic directions  {mark(bool(oa['clean']))}")
    print(f"   hits .......................... {len(oa['whisper_or_dramatic_hits'])}"
          f" (tone hits: {len(oa['overall_story_tone_hits'])})")
    for hit in oa["whisper_or_dramatic_hits"]:
        print(f"     - paragraph {hit['paragraph_index']} [{hit['field']}] "
              f"'{hit['term']}' :: {hit['context']}")
    print(f"5. artifacts")
    print(f"   plan ........................... {checks['artifacts']['plan']}")
    print(f"   raw MiMo attempts ............. {checks['artifacts']['raw_mimo_attempts_dir']}")
    for name in checks["artifacts"]["raw_mimo_attempts_files"]:
        print(f"       - {name}")
    print(f"   preferences ................... {checks['artifacts']['preferences']}")
    print()
    print(f"safety: {'PASS' if sf['ok'] else 'FAIL'}")
    print(f"   protected files unchanged ..... {sf['protected_files_unchanged']}"
          f" (changed: {sf['protected_changed_files']})")
    print(f"   new files ..................... {sf['new_files']}")
    print(f"   ElevenLabs network calls ...... {sf['elevenlabs_network_calls']}")
    print(f"   async HTTP calls .............. {len(sf['async_http_calls'])}"
          f" {sf['async_http_calls']}")
    print(f"   subprocess/FFmpeg calls ....... {len(sf['subprocess_ffmpeg_calls'])}")
    print(f"   generation quota untouched .... {sf['quota_untouched']}")
    print(f"   audio files untouched ......... {sf['no_audio_created']}"
          f" (before: {sf['audio_files_before']})")
    print(f"\nreport written to: {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
