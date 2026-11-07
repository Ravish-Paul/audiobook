"""Multi-character voice casting UI — markup, script-limit counter, pipeline.

Everything is mocked: NO real MiMo / ElevenLabs / pyannote / FFmpeg calls,
no provider credits, no real tester tokens (isolated `tests/_test_storage`
+ test identities only).

Coverage:

* the `Use Multiple Character Voices` option, its description and the
  casting controls exist in the served markup (spec §3) and are styled for
  narrow screens;
* the SERVED application (§13): the casting section sits between "Your
  Script" and "Narration style", carries the "Multi-Character Voice
  Casting" heading and step badges, and every served asset revalidates
  (`Cache-Control: no-cache`) so a stale browser cache can never hide it —
  the behavioural checks themselves are driven through the real `app.js`
  by `tests/js/casting.test.js`;
* the script maximum shown in the counter comes from `GET /usage` — the
  configured server value, never a hardcoded JavaScript literal (§9);
* one complete multi-character workflow (analysis -> casting -> select ->
  approve -> audition -> create-multi-voice-plan -> generate-audio) costs
  EXACTLY ONE allowance and produces `generation_mode == "multi_voice"`
  audio that really used the locked voices, while `analyze-narration` is
  never called (§5, §7).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.routes import narration as narration_routes
from app.services import ProjectService
from fakes import MV_SCRIPT, FakeMiMo, multi_voice_plan_json, story_json
from test_multi_voice import (
    _default_mock,
    _use_analyzer,
    _use_planner,
    _wire_audio,
    _wire_casting,
)
from test_quota_audit import _create_project, _generations, _usage

TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"
JS_DIR = Path(__file__).resolve().parent / "js"
REPO_ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

ALICE = {"Authorization": "Bearer alice-secret-1"}


# --------------------------------------------------------------------------- #
# Fixtures (same conventions as test_generation_limits / test_quota_audit)
# --------------------------------------------------------------------------- #
def _reset_runtime(clear_storage: bool = True) -> None:
    """Clear in-memory middleware state (and optionally the ledger files)."""
    from app import security as security_mod

    with security_mod._RATE_LOCK:
        security_mod._RATE_WINDOWS.clear()
    with security_mod._DEDUP_LOCK:
        security_mod._IN_FLIGHT.clear()
        security_mod._LAST_COMPLETED.clear()
    if clear_storage:
        shutil.rmtree(TEST_STORAGE / "_beta", ignore_errors=True)


@pytest.fixture()
def beta_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Private beta mode with one tester and a deterministic allowance."""
    monkeypatch.setenv("BETA_ACCESS_TOKENS", "alice:alice-secret-1,bob:bob-secret-2")
    monkeypatch.setenv("BETA_TESTER_GENERATION_LIMITS", "")
    monkeypatch.setenv("BETA_GENERATION_LIMIT", "2")
    monkeypatch.setenv("BETA_RATE_LIMIT_PER_MINUTE", "200")
    monkeypatch.setenv("BETA_LOGIN_RATE_LIMIT_PER_MINUTE", "10")
    monkeypatch.setenv("BETA_DUPLICATE_WINDOW_SECONDS", "10")
    _reset_runtime()
    yield
    _reset_runtime()


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# 1. The 16 focused frontend check — run the real app.js under node
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_frontend_casting_behavior() -> None:
    """20 behavioural checks through the real app.js + mocked FastAPI."""
    result = subprocess.run(
        [NODE, str(JS_DIR / "casting.test.js")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all casting frontend checks passed" in result.stdout
    assert result.stdout.count("ok - ") == 20, result.stdout


# --------------------------------------------------------------------------- #
# 2. Markup: toggle, description, casting modes (spec §3)
# --------------------------------------------------------------------------- #
def test_casting_option_markup_and_description(client: TestClient) -> None:
    html = client.get("/", headers={"Accept": "text/html"}).text
    flat = " ".join(html.split())

    assert 'id="casting-card"' in html
    assert 'id="mv-toggle"' in html
    assert "Use Multiple Character Voices" in flat
    assert (
        "Let AI identify the narrator and characters, recommend suitable voices, "
        "and create a multi-voice audiobook."
    ) in flat

    # Casting body: progress, roles table, mode switch, cards, continue.
    for marker in (
        'id="casting-body"',
        'id="cast-progress"',
        'id="cast-roles-wrap"',
        'id="cast-roles-body"',
        'id="cast-note"',
        'id="cast-modes"',
        'id="cast-mode-auto"',
        'id="cast-mode-manual"',
        'id="cast-cards"',
        'id="btn-casting-continue"',
    ):
        assert marker in html, marker

    # The two casting modes are user-facing, not technical labels.
    assert "Let AI choose the voices" in flat
    assert "I&rsquo;ll choose the voices" in flat
    # The action button names the NEXT step; its later label ("Continue to
    # Audiobook Generation") is pinned on the served app.js below.
    assert "Character Analyser" in flat

    # The option is OFF by default and its panel starts hidden.
    assert 'class="card hidden" id="casting-card"' in html
    assert '<input type="checkbox" id="mv-toggle" />' in html

    # Never expose internal identifiers in the markup (existing UI contract).
    for banned in ("project_id", "btn-preview", "preview-paragraph"):
        assert banned not in html, banned


def test_casting_styles_include_narrow_screen_rules(client: TestClient) -> None:
    css = client.get("/static/style.css").text

    for marker in (
        ".toggle-row",
        ".roles-table",
        ".mode-switch",
        ".mode-option",
        ".cast-card",
        ".cast-controls",
        ".cast-audition",
        ".casting-actions",
    ):
        assert marker in css, marker

    # Casting controls stack on phones (single responsive block).
    assert css.count("@media (max-width: 720px)") == 1
    media = css[css.index("@media (max-width: 720px)") :]
    assert ".cast-controls { flex-direction: column; align-items: stretch; }" in media
    assert ".mode-switch { flex-direction: column; }" in media


# --------------------------------------------------------------------------- #
# 3. §9 — the script maximum comes from the configured server value
# --------------------------------------------------------------------------- #
def test_usage_reports_the_configured_script_character_limit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`GET /usage` is authoritative for `max_script_characters`."""
    monkeypatch.setenv("MAX_SCRIPT_CHARACTERS", "3000")

    body = client.get("/usage").json()
    assert body["max_script_characters"] == 3000
    assert set(body) == {
        "generation_limit",
        "generations_used",
        "generations_remaining",
        "max_script_characters",
    }


def test_static_javascript_never_hardcodes_the_script_limit() -> None:
    """No `200000`/`3000` literals in the shipped JavaScript (§9)."""
    scripts = sorted((REPO_ROOT / "app" / "static").glob("*.js"))
    assert scripts, "static JavaScript files must exist"
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        assert re.search(r"(?<!\d)200000(?!\d)", source) is None, script.name
        assert re.search(r"(?<!\d)3000(?!\d)", source) is None, script.name
        # The counter reads the value from the /usage payload.
        assert "max_script_characters" in source, script.name


@pytest.mark.skipif(
    not (REPO_ROOT / ".env").exists(), reason="repository .env is not present"
)
def test_configured_server_script_limit_is_three_thousand() -> None:
    """The real server value (from .env) that GET /usage serves."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "MAX_SCRIPT_CHARACTERS"  # conftest pins 200000 for the suite
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.config import get_settings; "
            "print(get_settings().max_script_characters)",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3000", result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# 4. §5/§7 — one complete multi-character workflow costs exactly ONE allowance
# --------------------------------------------------------------------------- #
def test_multi_voice_casting_workflow_costs_exactly_one_allowance(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Analysis/casting/audition are free; plan+generate = ONE allowance.

    The generated audio must genuinely use the locked character voices
    (`generation_mode == "multi_voice"` + one TTS request per locked voice),
    and the single-voice narration planner must never run in this flow.
    """
    analyzer = FakeMiMo([story_json()])
    _use_analyzer(monkeypatch, analyzer)
    planner = FakeMiMo([multi_voice_plan_json()])
    _use_planner(monkeypatch, planner)
    # ONE shared mock: casting/audition discovery + generation TTS (all fake).
    discovery = _default_mock()
    _wire_casting(monkeypatch, discovery)
    composer = _wire_audio(monkeypatch, discovery)

    def _single_voice_planning_must_not_run() -> Any:
        raise AssertionError(
            "analyze-narration must not run in the multi-voice casting workflow"
        )

    monkeypatch.setattr(narration_routes, "_planner", _single_voice_planning_must_not_run)

    with TestClient(app) as client:
        project = _create_project(client, ALICE, content=MV_SCRIPT)

        # 1) Character analysis — auxiliary, costs nothing.
        analysis = client.post(
            f"/projects/{project}/analyze-characters", headers=ALICE
        )
        assert analysis.status_code == 200, analysis.text
        speaker_ids = analysis.json()["speaker_ids"]
        assert speaker_ids[0] == "narrator"
        assert len(speaker_ids) > 1, "the fixture script has speaking characters"
        assert _usage(client, ALICE)["generations_used"] == 0

        # 2) Ranked candidates per speaker — costs nothing.
        built = client.post(f"/projects/{project}/voice-casting", headers=ALICE)
        assert built.status_code == 200, built.text
        cast = built.json()
        assert _usage(client, ALICE)["generations_used"] == 0

        # 3) Automatic casting: lock the rank-1 recommendation per speaker.
        selected: dict[str, str] = {}
        for speaker_id in speaker_ids:
            entry = (
                cast["narrator"]
                if speaker_id == "narrator"
                else cast["characters"][speaker_id]
            )
            top = sorted(entry["candidates"], key=lambda c: c["rank"])[0]
            picked = client.post(
                f"/projects/{project}/voice-casting/select",
                json={"speaker_id": speaker_id, "voice_id": top["voice_id"]},
                headers=ALICE,
            )
            assert picked.status_code == 200, picked.text
            selected[speaker_id] = top["voice_id"]
        assert _usage(client, ALICE)["generations_used"] == 0

        # 4) Approve the whole cast — costs nothing.
        approved = client.post(
            f"/projects/{project}/voice-casting/approve", headers=ALICE
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["approved"] is True

        # The frontend re-reads this authoritative state after a reload.
        state = client.get(f"/projects/{project}/voice-casting", headers=ALICE)
        assert state.status_code == 200, state.text
        assert state.json()["approved"] is True
        assert _usage(client, ALICE)["generations_used"] == 0

        # 5) A voice preview (audition) — auxiliary, costs nothing.
        preview_speaker = next(
            (sid for sid in speaker_ids if sid != "narrator"), speaker_ids[0]
        )
        audition = client.post(
            f"/projects/{project}/voice-casting/audition",
            json={
                "speaker_id": preview_speaker,
                "text": "नमस्ते, यह मेरी आवाज़ का छोटा सा नमूना है।",
                "voice_id": selected[preview_speaker],
            },
            headers=ALICE,
        )
        assert audition.status_code == 200, audition.text
        assert audition.headers["content-type"].startswith("audio/mpeg")
        assert _usage(client, ALICE)["generations_used"] == 0

        # 6) The multi-voice plan is the ONE allowance (opens the workflow).
        plan = client.post(
            f"/projects/{project}/create-multi-voice-plan", headers=ALICE
        )
        assert plan.status_code == 200, plan.text
        assert _usage(client, ALICE)["generations_used"] == 1

        # 7) generate-audio JOINS that workflow — still exactly one allowance.
        audio = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
        assert audio.status_code == 200, audio.text
        body = audio.json()
        assert body["generation_mode"] == "multi_voice"
        assert body["segment_count"] == 6

        usage = _usage(client, ALICE)
        assert usage["generations_used"] == 1  # EXACTLY one allowance
        assert usage["generations_remaining"] == 1

        # §5 proof: every locked voice really produced TTS audio.
        paths = [request["path"] for request in discovery.tts_requests]
        for voice_id in selected.values():
            assert any(path.endswith(f"/{voice_id}") for path in paths), (
                voice_id,
                paths,
            )
        assert composer.segment_counts == [6]

        # The single-voice narration plan was never built (no 2nd allowance).
        assert analyzer.call_count == 1
        assert planner.call_count == 1

        # 8) Ledger: one reservation, shared generation_id, aux never charged.
        ledger = _generations()
        entries = [
            entry
            for entry in ledger["history"]
            if entry.get("tester") == "alice"
        ]
        charged = [entry for entry in entries if entry["charged"]]
        assert len(charged) == 1

        plan_ids = {e["generation_id"] for e in entries if e["kind"] == "plan"}
        generate_ids = {e["generation_id"] for e in entries if e["kind"] == "generate"}
        assert len(plan_ids) == 1
        assert plan_ids == generate_ids

        aux_entries = [e for e in entries if e["kind"] == "aux"]
        assert aux_entries, "analysis/audition were still recorded"
        assert all(not e["charged"] for e in aux_entries)
        aux_ids = {e["generation_id"] for e in aux_entries}
        assert aux_ids.isdisjoint(plan_ids)  # aux never joins the workflow

        # The workflow was closed by generate-audio.
        assert ledger.get("workflows") == {}

        # The stored audiobook carries the workflow's generation_id.
        metadata = ProjectService(get_settings()).get_audio_metadata(project)
        assert metadata is not None
        assert metadata["generation_id"] in plan_ids


def test_multi_voice_regeneration_costs_one_more_allowance(
    beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second complete generation is a second allowance (never a free ride)."""
    _use_analyzer(monkeypatch, FakeMiMo([story_json()]))
    _use_planner(monkeypatch, FakeMiMo([multi_voice_plan_json()]))
    discovery = _default_mock()
    _wire_casting(monkeypatch, discovery)
    _wire_audio(monkeypatch, discovery)

    with TestClient(app) as client:
        project = _create_project(client, ALICE, content=MV_SCRIPT)
        assert (
            client.post(
                f"/projects/{project}/analyze-characters", headers=ALICE
            ).status_code
            == 200
        )
        cast = client.post(f"/projects/{project}/voice-casting", headers=ALICE).json()
        for speaker_id in ["narrator", *cast["characters"].keys()]:
            entry = (
                cast["narrator"]
                if speaker_id == "narrator"
                else cast["characters"][speaker_id]
            )
            top = sorted(entry["candidates"], key=lambda c: c["rank"])[0]
            assert (
                client.post(
                    f"/projects/{project}/voice-casting/select",
                    json={"speaker_id": speaker_id, "voice_id": top["voice_id"]},
                    headers=ALICE,
                ).status_code
                == 200
            )
        assert (
            client.post(
                f"/projects/{project}/voice-casting/approve", headers=ALICE
            ).status_code
            == 200
        )
        assert (
            client.post(
                f"/projects/{project}/create-multi-voice-plan", headers=ALICE
            ).status_code
            == 200
        )
        first = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
        assert first.status_code == 200, first.text
        assert first.json()["generation_mode"] == "multi_voice"
        assert _usage(client, ALICE)["generations_used"] == 1

        # Regenerating inside the duplicate window is refused (never a free
        # double-spend)…
        second = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
        assert second.status_code == 409
        assert "Duplicate submission" in second.json()["detail"]
        assert _usage(client, ALICE)["generations_used"] == 1

        # …and after the guard window a new generation is a NEW allowance.
        from app import security as security_mod

        with security_mod._DEDUP_LOCK:
            security_mod._LAST_COMPLETED.clear()
        third = client.post(f"/projects/{project}/generate-audio", headers=ALICE)
        assert third.status_code == 200, third.text
        assert third.json()["generation_mode"] == "multi_voice"
        assert _usage(client, ALICE)["generations_used"] == 2

        charged = [
            entry
            for entry in _generations()["history"]
            if entry.get("tester") == "alice" and entry["charged"]
        ]
        assert len(charged) == 2


# --------------------------------------------------------------------------- #
# 6. §13 — the SERVED application: order, badges, freshness, boundary
# --------------------------------------------------------------------------- #
def test_served_html_orders_casting_between_script_and_style(
    client: TestClient,
) -> None:
    """Placement: 1 Your Script · 2 Multi-Character Voice Casting · 3 style · 4 Generate."""
    response = client.get("/", headers={"Accept": "text/html"})
    assert response.status_code == 200
    html = response.text

    script_at = html.index('id="upload-heading"')
    casting_at = html.index('id="casting-card"')
    style_at = html.index('id="style-heading"')
    generate_at = html.index('id="generate-heading"')
    assert script_at < casting_at < style_at < generate_at, (
        "the casting section must sit between the script and the style sections"
    )

    flat = " ".join(html.split())
    assert "Multi-Character Voice Casting" in flat

    # Step badges follow the same numbering.
    assert (
        '<span class="step-badge" id="step-badge-casting" aria-hidden="true">2</span>'
        in html
    )
    assert (
        '<span class="step-badge" id="step-badge-style" aria-hidden="true">3</span>'
        in html
    )
    assert (
        '<span class="step-badge" id="step-badge-generate" aria-hidden="true">4</span>'
        in html
    )

    # The toggle is plain, unchecked markup — never pre-enabled.
    assert '<input type="checkbox" id="mv-toggle" />' in html
    assert '<input type="checkbox" id="mv-toggle" checked' not in html


def test_served_assets_revalidate_and_carry_the_current_code(
    client: TestClient,
) -> None:
    """A stale browser cache can never hide the casting UI (§4/§13)."""
    page = client.get("/", headers={"Accept": "text/html"})
    assert page.headers.get("cache-control") == "no-cache"

    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert js.headers.get("cache-control") == "no-cache"
    css = client.get("/static/style.css")
    assert css.status_code == 200
    assert css.headers.get("cache-control") == "no-cache"

    # The server serves exactly the files on disk (no stale build artefacts).
    disk_js = (REPO_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert js.text == disk_js

    # …and those files really contain the current casting implementation.
    for marker in (
        "renderCasting",  # the section render loop (incl. after upload)
        "analyze-characters",  # existing analysis endpoint
        "voice-casting",  # existing casting endpoints
        "create-multi-voice-plan",
        "multi-voice-plan/status",  # server-side plan recovery (§9)
        "fetchPlanStatus",
        "Character Analyser",  # explicit button trigger — never auto analysis
        "onCastingAction",
        "Save & Approve Casting",  # the button's second phase (never implies generation)
    ):
        assert marker in js.text, marker
    assert ".cast-card" in css.text
    assert ".toggle-row" in css.text


def test_script_limit_boundary_3000_accepted_3001_rejected(
    client: TestClient, beta_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The REAL configured limit: 3000 code points in, 3001 rejected (§3)."""
    monkeypatch.setenv("MAX_SCRIPT_CHARACTERS", "3000")

    assert client.get("/usage", headers=ALICE).json()["max_script_characters"] == 3000

    accepted = client.post(
        "/projects/from-script",
        files={"file": ("s.txt", ("a" * 3000).encode("utf-8"), "text/plain")},
        headers=ALICE,
    )
    assert accepted.status_code == 201, accepted.text
    assert accepted.json()["character_count"] == 3000  # Unicode code points

    rejected = client.post(
        "/projects/from-script",
        files={"file": ("s.txt", ("a" * 3001).encode("utf-8"), "text/plain")},
        headers=ALICE,
    )
    assert rejected.status_code == 400
    assert "limit: 3000 characters" in rejected.json()["detail"]
