"""Shared test configuration.

Goals:
* keep every test inside an isolated `tests/_test_storage` folder;
* keep the automated suite **hermetic** — a developer's real `.env`
  credentials must never be used and no MiMo / ElevenLabs credits may be
  spent.

Explicit opt-ins for the live tests:

    RUN_ELEVENLABS_LIVE_TEST=1 pytest -q    # real ElevenLabs roundtrip
    RUN_MIMO_LIVE_TEST=1       pytest -q    # real MiMo narration analysis
    RUN_PYANNOTE_LIVE_TEST=1   pytest -q    # real pyannoteAI job (BILLS!)
"""

from __future__ import annotations

import os
from pathlib import Path

# Isolated storage for every test (independent from real project data).
os.environ["STORAGE_DIR"] = str(Path(__file__).resolve().parent / "_test_storage")

if os.getenv("RUN_ELEVENLABS_LIVE_TEST") != "1":
    # Blank the credentials and point at an unroutable port so that even a
    # forgotten mock cannot reach the real ElevenLabs API.
    os.environ["ELEVENLABS_API_KEY"] = ""
    os.environ["ELEVENLABS_VOICE_ID"] = "test-voice-id"
    os.environ["ELEVENLABS_BASE_URL"] = "http://127.0.0.1:9"
    # Pin the Eleven v4 contract deterministically — capability filtering
    # (no speed/style, audio tags kept, language_code allowed) must not
    # depend on whatever a developer's .env happens to configure.
    os.environ["ELEVENLABS_MODEL_ID"] = "eleven_v4"
    os.environ["ELEVENLABS_OUTPUT_FORMAT"] = "mp3_44100_128"
    os.environ["ELEVENLABS_INTERMEDIATE_OUTPUT_FORMAT"] = ""
    os.environ["ELEVENLABS_LANGUAGE_CODE"] = ""
    os.environ["ELEVENLABS_APPLY_TEXT_NORMALIZATION"] = "auto"
    os.environ["ELEVENLABS_APPLY_LANGUAGE_TEXT_NORMALIZATION"] = "false"
    os.environ["ELEVENLABS_PRONUNCIATION_DICTIONARIES"] = ""
    os.environ["ELEVENLABS_AUDIO_EFFECTS"] = ""
    os.environ["ELEVENLABS_VOICE_IS_PROFESSIONAL_CLONE"] = "false"
    os.environ["ELEVENLABS_CLONE_STRENGTH"] = ""

if os.getenv("RUN_MIMO_LIVE_TEST") != "1":
    # Same protection for the Xiaomi MiMo narration director.
    os.environ["MIMO_API_KEY"] = ""
    os.environ["MIMO_BASE_URL"] = "http://127.0.0.1:9"

if os.getenv("RUN_PYANNOTE_LIVE_TEST") != "1":
    # Same protection for pyannoteAI — a real diarization job BILLS (20 s
    # minimum per successful job), so the automated suite must never reach
    # the real API. Unit tests inject httpx.MockTransport instead.
    os.environ["PYANNOTE_API_KEY"] = ""
    os.environ["PYANNOTE_BASE_URL"] = "http://127.0.0.1:9"
    # Pin the contract deterministically regardless of a local .env.
    os.environ["PYANNOTE_MODEL_ID"] = "precision-3"
    os.environ["PYANNOTE_TRANSCRIPTION_MODEL"] = "faster-whisper-large-v3-turbo"
    os.environ["PYANNOTE_TIMEOUT_SECONDS"] = "30"
    os.environ["PYANNOTE_MAX_ATTEMPTS"] = "2"
    os.environ["PYANNOTE_DURATION_TOLERANCE_MS"] = "500"

# Private-beta access control must never leak in from a developer's .env:
# the suite runs with beta mode OFF (open app, no login), just like the
# provider keys above. Tests that cover the beta layer enable it explicitly
# with monkeypatch.setenv(...) and get deterministic limits from here.
os.environ["BETA_ACCESS_TOKENS"] = ""
os.environ["BETA_ACCESS_TOKEN"] = ""
os.environ["BETA_SESSION_SECRET"] = ""
os.environ["BETA_COOKIE_SECURE"] = "false"
os.environ["BETA_SESSION_TTL_HOURS"] = "72"
os.environ["BETA_GENERATION_LIMIT"] = "10"
os.environ["BETA_TESTER_GENERATION_LIMITS"] = ""
os.environ["BETA_GENERATION_WINDOW_HOURS"] = "24"
os.environ["BETA_RATE_LIMIT_PER_MINUTE"] = "120"
os.environ["BETA_LOGIN_RATE_LIMIT_PER_MINUTE"] = "10"
os.environ["BETA_DUPLICATE_WINDOW_SECONDS"] = "10"
# Pin the script/length limits so a local .env cannot change validation.
os.environ["MAX_SCRIPT_SIZE_MB"] = "5"
os.environ["MAX_SCRIPT_CHARACTERS"] = "200000"
os.environ["MAX_AUDIO_DURATION_MINUTES"] = "120"
os.environ["SPEECH_CHARS_PER_MINUTE"] = "900"
