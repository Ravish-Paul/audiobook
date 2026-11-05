"""Shared test configuration.

Goals:
* keep every test inside an isolated `tests/_test_storage` folder;
* keep the automated suite **hermetic** — a developer's real `.env`
  credentials must never be used and no MiMo / ElevenLabs credits may be
  spent.

Explicit opt-ins for the live tests:

    RUN_ELEVENLABS_LIVE_TEST=1 pytest -q    # real ElevenLabs roundtrip
    RUN_MIMO_LIVE_TEST=1       pytest -q    # real MiMo narration analysis
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
