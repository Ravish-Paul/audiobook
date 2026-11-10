# AI Audiobook Platform — MiMo Narration Director + ElevenLabs TTS

A FastAPI backend where an **AI narration director** decides how every
paragraph of your script should sound, ElevenLabs speaks it, and FFmpeg
assembles the final audiobook.

```
Upload script
  → simple creator UI at `/` (upload → narration style → Generate Audiobook):
    the project + title + language are detected automatically
  → mood mode: "Let AI choose" (whole-story analysis picks the profile) or
    "I'll choose the mood" (one of 12 narration profiles) → persisted as the
    Narration Preferences (preset, intensity, whisper policy, tag density)
  → Xiaomi MiMo reads the whole story (context + emotional progression)
  → MiMo decides emotion / delivery / voice settings / pauses per paragraph
    AND writes `tts_text` with inline Eleven v4 audio tags ([whispers], …)
  → FastAPI validates the JSON plan (Pydantic + tag rules) AND enforces the
    preferences (NarrationPolicyValidator → targeted retry, never accepted
    silently) → narration_plan.json (preferences snapshot in the plan hash)
  → ElevenLabs (eleven_v4) generates one segment per paragraph:
    capability-filtered voice settings, neighbour text/request-ID continuity,
    pronunciation dictionaries, text normalization, optional audio effects
  → FFmpeg merges the segments with MiMo's pauses (once, into audiobook.mp3)
  → playable audiobook.mp3
```

**Multi-voice pipeline (AI character casting):**

```
POST analyze-characters      → MiMo extracts tone/genre/language + cast
                               (narrator + characters, no invented attributes)
                               → character_cast.json
POST voice-casting           → ElevenLabs voice discovery (account first,
                               shared library fallback) → top-3 per speaker
POST voice-casting/select    → permanent voice lock per speaker
POST voice-casting/approve   → cast approved (required before generation)
POST create-multi-voice-plan → MiMo splits paragraphs into speaker-attributed
                               utterances (exact words preserved)
POST generate-audio          → per-utterance voices → FFmpeg → audiobook.mp3
                               (generation_mode: "multi_voice")
```

**Narration Director (12 profiles · auto/manual mood · whole-story analysis):**

```
POST analyze-story           → MiMo reads the WHOLE script → genre, language,
                               scene-by-scene mood progression + recommended
                               profile → story_analysis.json (AUX: no allowance;
                               a matching saved analysis is reused, {"force": true}
                               re-runs it)
  mood mode "auto"           → the recommended profile becomes the global
                               direction ("Use recommended style" in the UI)
  mood mode "manual"         → the user's chosen profile WINS — the analysis
                               stays context only, never a mood stamp
  three layers reach the plan:
    1. global profile boundary  (PROFILE_GUIDANCE rendered into the prompt)
    2. whole-story analysis     (auto mode: scene progression → mood changes)
    3. per-paragraph direction  (emotion, delivery, tts_text tags, pauses)
  → NarrationPolicyValidator enforces VARIETY — no one-mood-everywhere
POST narration-preview/audio → ONE paragraph rendered as narration-preview.mp3
                               (explicit action only, AUX: no allowance, never
                               touches the audiobook's files)
```

**Stage 1 — Audio Analysis Engine (Pyannote diarization + transcription):**

```
POST /projects/{id}/audio-analysis    → SHA-256 + ffprobe (real sample rate)
                                        → duplicate check → upload audiobook to
                                        pyannoteAI (/v1/media/input) → submit job
                                        (/v1/diarize, transcription on) → returns
                                        immediately (200 + analysis_id)
GET  /projects/{id}/audio-analysis/status → polls provider job (bounded backoff),
                                        saves raw → normalized → metadata JSON
GET  /projects/{id}/audio-analysis    → normalized word/speaker timeline
```

**Separation of responsibilities (important):**

| Component | Responsibility |
| --------- | -------------- |
| **Xiaomi MiMo** (`mimo-v2.6-flash`) | Decides *how* to narrate — emotion, intensity, inline audio tags, voice transitions, continuity — **within the user's narration preferences**. Never rewrites the script, never touches ElevenLabs. |
| **ElevenLabs** (`eleven_v4`) | Only converts text → speech with the numeric settings it is given. It makes **no** narration decisions. |
| **FFmpeg** | Concatenates the segments, inserts the pauses, and encodes the final MP3 (once) — background music/SFX belong here. |
| **Narration Preferences** (set via the simple creator at `/`) | The user's contract for how the narrator sounds: the **narration profile** (one of 12, or `auto` = AI picks after the whole-story analysis), `acting_intensity`, `whisper_policy`, `tag_density`, `custom_direction`. Persisted per project, sent to MiMo, enforced after validation by `NarrationPolicyValidator`, embedded in the plan (→ plan hash). Never a frontend-only setting. |

## Narration profiles — the 12 moods

The user-facing global direction (`narration_preferences.preset`). Each one
renders its own `PROFILE_GUIDANCE` into the MiMo system prompt, so the twelve
**actually plan differently** (asserted by `tests/test_narration_director.py`):

| `preset` id | Label | Character |
| ----------- | ----- | --------- |
| `natural` | Natural & Conversational | Clear, conversational and realistic storytelling. |
| `cinematic` | Cinematic & Immersive | Atmospheric, immersive and dramatically controlled. |
| `horror` | Horror & Suspense | Gradual tension, unease and suspense. |
| `romantic` | Romantic & Intimate | Warm, affectionate and emotionally close. |
| `sensual` | Sensual & Intimate | Close, subtle intimacy only where the scene allows it. |
| `emotional` | Emotional & Reflective | Sincere, reflective and gently emotional. |
| `action` | Action & Adventure | Energetic, urgent and sharply focused. |
| `mystery` | Mystery & Thriller | Measured, curious and quietly tense. |
| `fantasy` | Fantasy & Epic | Atmospheric, wondrous and epic in scale. |
| `comedy` | Comedy & Lighthearted | Natural timing with playful lightness. |
| `professional` | Calm & Professional | Clear, steady and composed. |
| `children` | Children's Story | Warm, clear and friendly storytelling. |
| `auto` | AI Automatic | The whole-story analysis picks the profile (mood mode *Let AI choose*). |

Legacy ids `restrained_cinematic`, `expressive`, `dramatic`, `soft`, `smooth`
stay accepted for older saved preferences/plans but are hidden from the UI
(`LEGACY_STYLE_ALIASES` maps old localStorage values onto the nearest current
profile). The profile is a **boundary, not a mood stamp**: neutral narration
stays neutral and the narrator never inherits a character's emotion or voice.

## Tech stack

* Python 3.11+ · FastAPI · Uvicorn · Pydantic v2
* `openai` SDK → Xiaomi MiMo (OpenAI-compatible endpoint)
* `httpx` → ElevenLabs REST API (async)
* FFmpeg → segment composition
* `python-dotenv` → secrets from `.env`

## Endpoints

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| GET | `/` | **Audiobook creator UI** for browsers (`Accept: text/html`) · welcome JSON for API clients |
| GET | `/auth/login` · POST `/auth/login` · POST `/auth/logout` | Private-beta login (session cookie) / logout |
| GET | `/static/*` | Panel assets (`style.css`, `app.js`) served from `app/static/` |
| GET | `/health` | Health check (storage, MiMo, ElevenLabs, FFmpeg status) |
| POST | `/projects` | Create a project → UUID |
| POST | `/projects/from-script` | **One-step import for the creator UI**: upload a `.txt` script → auto-created project with detected title + language, paragraph count, preview |
| DELETE | `/projects` | **Privacy**: delete every project (and file) of the current tester |
| GET | `/projects/{project_id}` | Project details, `404` if not found |
| GET | `/projects` | The authenticated tester's own projects (newest first) — restores the project after browser localStorage loss |
| DELETE | `/projects/{project_id}` | **Privacy**: delete one project and all its files |
| POST | `/projects/{project_id}/upload-script` | Upload a `.txt` script (max 5 MB, UTF-8) |
| GET | `/projects/{project_id}/narration-preferences` | Saved narration preferences, or the defaults (`saved: false`) |
| PUT | `/projects/{project_id}/narration-preferences` | Save narration preferences **without** generating a plan |
| POST | `/projects/{project_id}/analyze-story` | **Whole-story analysis** (genre, language, scene-by-scene mood progression, recommended profile) → `story_analysis.json` · optional body `{"force": true}` · **aux — never consumes an allowance** |
| GET | `/projects/{project_id}/story-analysis` | The saved analysis (`404` when none exists for the current script) |
| POST | `/projects/{project_id}/analyze-narration` | **MiMo directs the script → narration_plan.json** (optional body `{"preferences": {...}}`; incl. `tts_text` audio tags) |
| POST | `/projects/{project_id}/analyze-narration/preview` | Preview ONE paragraph (±1 neighbour as context) — never overwrites the plan |
| POST | `/projects/{project_id}/narration-preview/audio` | **Render ONE paragraph as `narration-preview.mp3`** (explicit action only; aux — no allowance, writes nothing to the audiobook) |
| GET | `/projects/{project_id}/narration-plan` | Review the saved plan before generating audio |
| POST | `/projects/{project_id}/generate-audio` | Generate the audiobook (plan-driven, or legacy single-pass) |
| GET | `/projects/{project_id}/audio` | Generated audio info + playable URL |
| POST | `/projects/{project_id}/analyze-characters` | MiMo story + character cast → `character_cast.json` |
| POST | `/projects/{project_id}/voice-casting` | Discover + rank top-3 ElevenLabs voices per speaker |
| GET | `/projects/{project_id}/voice-casting` | Candidates, selections, approval state |
| POST | `/projects/{project_id}/voice-casting/select` | Permanently lock a voice for one speaker |
| POST | `/projects/{project_id}/voice-casting/audition` | Short audition line (≤400 chars, optional) |
| POST | `/projects/{project_id}/voice-casting/approve` | Approve the locked cast |
| POST | `/projects/{project_id}/create-multi-voice-plan` | MiMo utterance plan (speaker-attributed) |
| GET | `/projects/{project_id}/multi-voice-plan/status` | Server-authoritative plan status (never charged) — durable recovery after reload / localStorage loss |
| GET | `/elevenlabs/audio-effects/catalog` | Valid IDs for optional `ELEVENLABS_AUDIO_EFFECTS` |
| POST | `/projects/{project_id}/audio-analysis` | Start pyannoteAI diarization + word transcription (async) |
| GET | `/projects/{project_id}/audio-analysis/status` | Poll local + provider job status (saves results on success) |
| GET | `/projects/{project_id}/audio-analysis` | Normalized word-level timeline (speakers, words, turns) |
| GET | `/usage` | The caller's **own** limits: `generation_limit`, `generations_used`, `generations_remaining`, `max_script_characters` |
| GET | `/media/projects/{project_id}/{path}` | Media files (playback/download of `audiobook.mp3`, previews) |

## Project structure

```
audiobook_web/
├── app/
│   ├── main.py                    # app, /media + /static mounts, error handlers, startup log
│   ├── config.py                  # Settings (env + .env driven)
│   ├── static/                    # Simple audiobook creator UI (no build step, served at /static)
│   │   ├── index.html             #   upload card, mood-mode switch (AI/manual), 12 profile cards,
│   │   │                          #   story analysis + narration preview, one Generate action
│   │   ├── app.js                 #   auto project import, profile→preferences mapping, story analysis,
│   │   │                          #   narration preview audio, generation pipeline
│   │   └── style.css
│   ├── routes/
│   │   ├── health.py              # GET / (creator UI for browsers, welcome JSON for APIs), GET /health
│   │   ├── projects.py            # projects + script upload
│   │   ├── narration.py           # analyze-story, story-analysis, analyze-narration (+ preview),
│   │   │                          # narration-preview/audio, narration-preferences, narration-plan
│   │   ├── casting.py             # characters, voice casting, utterance plan
│   │   ├── audio.py               # generate-audio, audio info
│   │   ├── audio_analysis.py       # audio-analysis start/status/normalized
│   │   ├── media.py               # /media/projects/… file serving
│   │   ├── usage.py               # GET /usage (the caller's own limits)
│   │   └── elevenlabs.py          # audio-effects catalog proxy
│   ├── schemas/
│   │   ├── project.py             # project / script models
│   │   ├── narration.py           # NarrationPlan, ParagraphPlan (tts_text), VoiceSettings, response models
│   │   ├── narration_preferences.py # NarrationPreferences: 12 profiles + auto + legacy ids, guidance
│   │   ├── story_analysis.py       # whole-story analysis (genre, scenes, recommended profile)
│   │   ├── casting.py             # characters, voice profiles, candidates, locks
│   │   ├── multi_voice.py         # Utterance, MultiVoicePlan
│   │   ├── audio.py               # audio response model
│   │   ├── audio_analysis.py       # normalized timeline + start/status models
│   │   └── elevenlabs.py          # audio effects catalog models
│   └── services/
│       ├── mimo_client.py         # Xiaomi MiMo (openai SDK) + error mapping
│       ├── narration_planner.py   # paragraphs → MiMo → validated plan (+ preferences, preview, preview audio)
│       ├── narration_policy.py    # NarrationPolicyValidator: preferences + direction/mood VARIETY checks
│       ├── story_analysis_service.py # whole-story analysis → story_analysis.json (recommended profile)
│       ├── script_import.py       # one-step import (title/language detection)
│       ├── script_limits.py       # script length / estimated-duration limits
│       ├── character_analyzer.py  # story + character discovery (cast)
│       ├── voice_discovery.py     # /v2/voices, /v1/shared-voices, add voice
│       ├── voice_matcher.py       # voice scoring/ranking (top-3, threshold)
│       ├── voice_casting_service.py # discovery → select/lock → approve → audition
│       ├── multi_voice_planner.py # utterance plan (exact words preserved)
│       ├── audio_tags.py          # Eleven v4 audio tag parse/validate/strip
│       ├── eleven_models.py       # per-model capability registry (v4, v3, …)
│       ├── elevenlabs_client.py   # ElevenLabs v4 TTS (async httpx)
│       ├── audio_composer.py      # FFmpeg merge + pauses (+ PCM/WAV inputs)
│       ├── audio_service.py       # multi-voice / plan / single-pass modes
│       ├── audio_probe.py         # ffprobe/ffmpeg → real sample rate + duration
│       ├── pyannote_client.py     # pyannoteAI media/diarize/jobs (async httpx)
│       ├── audio_analysis_normalizer.py # Decimal ROUND_HALF_UP → ms/samples
│       ├── audio_analysis_service.py    # dedupe, submit, poll, persist (Stage 1)
│       ├── project_service.py     # storage layout & metadata
│       └── errors.py              # domain errors → HTTP status codes
├── storage/projects/{project_id}/
│   ├── project.json
│   ├── script.txt
│   ├── narration_plan.json          # paragraph plan (single-voice flow)
│   ├── narration_preferences.json   # the user's narration preferences (per project)
│   ├── story_analysis.json          # whole-story analysis (auto mode + UI panel, script-hash bound)
│   ├── character_cast.json          # cast + voice profiles + permanent locks
│   ├── multi_voice_plan.json        # utterance plan (multi-voice flow)
│   └── audio/
│       ├── segments/paragraph_0001.mp3|wav|pcm …     # paragraph mode
│       ├── segments/utterance_p0000_u0000.mp3|wav …  # multi-voice mode
│       ├── audiobook.mp3
│       ├── metadata.json            # per-utterance speaker/voice records
│       └── analysis/
│           ├── audio_analysis_metadata.json          # job bookkeeping
│           └── {analysis_id}/
│               ├── audio_analysis_raw.json           # verbatim provider output
│               └── audio_analysis_normalized.json    # Decimal-normalized timeline
├── tests/
│   ├── conftest.py                # hermetic env (no real credentials)
│   ├── fakes.py                   # FakeMiMo / FakeComposer / plan builders
│   ├── test_api.py                # projects + upload
│   ├── test_audio.py              # legacy single-pass audio
│   ├── test_audio_analysis.py     # pyannote Stage 1 (all mocked, 37 tests)
│   ├── test_beta_security.py      # tokens, sessions, 401-reservation, project isolation
│   ├── test_casting_ui.py         # casting markup/style + mocked end-to-end workflow
│   ├── test_composer.py           # real FFmpeg composition
│   ├── test_delivery_style_compat.py # legacy delivery_style plans still load
│   ├── test_eleven_v4.py          # v4 tags, continuity, dictionaries, formats
│   ├── test_frontend.py           # served creator UI contract
│   ├── test_generation_limits.py  # allowance reservation/join rules
│   ├── test_mimo_client.py        # MiMo transport + error mapping
│   ├── test_mimo_director_smoke.py # OPT-IN real-MiMo story analysis (6 cases, plan only)
│   ├── test_multi_voice.py        # casting, locks, utterances, generation
│   ├── test_multi_voice_recovery.py # durable plan recovery / fingerprint
│   ├── test_narration.py          # planning, validation, retries, Hindi
│   ├── test_narration_director.py # 12 profiles, auto/manual modes, variety policy (26 tests)
│   ├── test_narration_preferences.py # preferences, policy validator, preview, hashing (28 tests)
│   ├── test_negation_aware_whisper.py # whisper selection edge cases
│   ├── test_plan_audio.py         # segments, recovery, metadata, playback
│   ├── test_quota_audit.py        # cross-process quota ledger audit
│   ├── test_script_import.py      # from-script import (title/language)
│   ├── test_tts_text_regression.py # tts_text mandatory + exact-word rules
│   ├── test_voice_discovery.py    # voice search, pagination, ranking
│   └── js/                        # node behaviour suites vs the real app.js
│       ├── app_behavior.test.js   #   upload → plan → generate → download
│       ├── casting.test.js        #   multi-character casting (20 checks)
│       └── limits.test.js         #   character counting / usage limits
├── scripts/
│   ├── real_mimo_plan_smoke.py    # opt-in live smoke runner (real MiMo, plan only)
│   ├── start_server.bat           # start uvicorn (exactly ONE worker — beta rule)
│   ├── stop_server.bat            # stop the server
│   └── show_tokens.bat            # show name:token list from .env (PRIVATE — never share)
├── .env.example                   # template (no real credentials)
├── requirements.txt
└── README.md
```

## Getting started

```powershell
# 1. virtual environment (once)
python -m venv .venv

# 2. dependencies (once, or after changing requirements.txt)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. configure credentials (once)
Copy-Item .env.example .env
notepad .env

# 4. run
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/> for the **audiobook creator** (upload a script,
choose AI or manual mood mode — 12 narration profiles — then generate) and
<http://127.0.0.1:8000/docs> for Swagger UI.

### `.env`

```env
# Xiaomi MiMo — the narration director
MIMO_API_KEY=your_mimo_api_key
MIMO_MODEL_ID=mimo-v2.6-flash
MIMO_BASE_URL=https://token-plan-sgp.xiaomimimo.com/v1   # regional endpoint

# ElevenLabs — the voice (eleven_v4 = audiobook flagship model)
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_VOICE_ID=your_voice_id_here
ELEVENLABS_MODEL_ID=eleven_v4

# pyannoteAI — diarization + word-level transcription (Stage 1)
PYANNOTE_API_KEY=your_pyannote_api_key
PYANNOTE_BASE_URL=https://api.pyannote.ai
PYANNOTE_MODEL_ID=precision-3
PYANNOTE_TRANSCRIPTION_MODEL=faster-whisper-large-v3-turbo
PYANNOTE_TRANSCRIPTION=true
```

Optional: `MIMO_BASE_URL`, `MIMO_TIMEOUT_SECONDS`, `MIMO_MAX_COMPLETION_TOKENS`,
`MIMO_MAX_ATTEMPTS`, `FFMPEG_PATH`, `FFPROBE_PATH`, `ELEVENLABS_*` (language,
normalization, output/intermediate formats, pronunciation dictionaries, audio
effects, PVC clone strength), `PYANNOTE_*` (timeout, retry attempts, polling
interval/backoff/cap, duration tolerance ms), `STORAGE_DIR`,
`MAX_SCRIPT_SIZE_MB` — see `.env.example`.

Startup prints status only, never secrets:

```
[startup] elevenlabs: configured | voice=7b9m… | model=eleven_v4
[startup] mimo: configured | model=mimo-v2.6-flash | base_url=https://api.xiaomimimo.com/v1
[startup] ffmpeg: C:\Users\you\ffmpeg\ffmpeg.EXE
[startup] pyannote: configured | model=precision-3 | transcription=faster-whisper-large-v3-turbo
```

## Full workflow in Swagger UI

1. **`POST /projects`** → `{"name": "My Hindi Audiobook"}` → copy `project_id`.
2. **`POST /projects/{id}/upload-script`** → choose `sample_script.txt`.
3. **Narration preferences & direction** (the creator UI at `/` does this for you):

   * `PUT /projects/{id}/narration-preferences` → persist the complete object
     (`preset`, `acting_intensity`, `whisper_policy`, `whisper_paragraphs`,
     `tag_density`, `custom_direction`) without generating anything. `preset`
     is one of the **twelve narration profiles** (`natural`, `cinematic`,
     `horror`, `romantic`, `sensual`, `emotional`, `action`, `mystery`,
     `fantasy`, `comedy`, `professional`, `children`), `auto` (MiMo picks
     after the story analysis) or a legacy id;
   * *(optional, mood mode "auto")* `POST /projects/{id}/analyze-story`
     (body: `{}` or `{"force": true}`) → whole-story analysis saved to
     `story_analysis.json`: genre, language, dominant mood, **scene-by-scene
     mood progression** and a `recommended_profile`. A saved analysis that
     still matches the script is reused; it is **aux — no allowance**.
     `GET /projects/{id}/story-analysis` reads it back (`404` if none);
   * `POST /projects/{id}/analyze-narration/preview` → direct ONE paragraph
     (its two neighbours are sent as context) before committing to the whole
     book. A preview **never** overwrites `narration_plan.json`;
   * *(optional)* `POST /projects/{id}/narration-preview/audio`
     (body: `{"paragraph_index": 0}`) → an actual `narration-preview.mp3`
     sample of that line — explicit action only, **aux: no allowance**, and
     it never touches the audiobook's files.

4. **`POST /projects/{id}/analyze-narration`** (body: `{"preferences": {...}}`,
   optional) → MiMo reads the whole story and
   returns (and saves) the plan:

   ```json
   {
     "project_id": "…",
     "overall_story_tone": "suspenseful",
     "language": "Hindi",
     "paragraph_count": 3,
     "paragraphs": [
       {
         "paragraph_index": 0,
         "original_text": "रात बहुत अंधेरी थी …",
         "tts_text": "[quiet, tense delivery] रात बहुत अंधेरी थी …",
         "emotion": "suspense",
         "delivery_style": "slow",
         "intensity": 4,
         "narration_direction": "Use a mysterious and controlled voice with a slow delivery.",
         "voice_settings": {"speed": 0.85, "stability": 0.4, "similarity_boost": 0.75, "style": 0.25, "use_speaker_boost": true},
         "pause_after_ms": 700
       }
     ],
     "preferences": {"preset": "natural", "acting_intensity": 2, "whisper_policy": "user_selected", "whisper_paragraphs": [], "tag_density": "minimal", "custom_direction": ""},
     "voice_model": {"model_id": "eleven_v4", "supports_speed": false, "supports_style": false, "planning_only_parameters": ["speed", "style"]},
     "tts_voice_settings": [{"paragraph_index": 0, "settings": {"stability": 0.5, "similarity_boost": 0.75, "use_speaker_boost": true}}],
     "saved": true,
     "saved_at": "2026-11-05T…"
   }
   ```

   `original_text` is always the untouched script; `tts_text` adds inline
   Eleven v4 audio tags (`[whispers]`, `[laughs]`, `[low, ominous narrator
   voice]`, …) — the spoken words are validated to be identical.
   `preferences` is the snapshot the plan was built with (it is part of the
   plan hash); `tts_voice_settings` are the exact per-paragraph settings that
   reach ElevenLabs — capability-filtered, so `eleven_v4` never sees
   `speed`/`style` (they are listed in `voice_model.planning_only_parameters`).
5. **`GET /projects/{id}/narration-plan`** → review it (or re-run step 4).
6. **`POST /projects/{id}/generate-audio`** → segments + FFmpeg merge:

   ```json
   {
     "success": true,
     "message": "Audio generated successfully from 3 narration segments",
     "audio_filename": "audiobook.mp3",
     "audio_url": "/media/projects/…/audio/audiobook.mp3",
     "status": "completed",
     "generation_mode": "narration_plan",
     "segment_count": 3,
     "output_format": "mp3_44100_128"
   }
   ```

7. **Play it**: open the `audio_url`, e.g.
   <http://127.0.0.1:8000/media/projects/{id}/audio/audiobook.mp3>
8. *(optional)* **`GET /elevenlabs/audio-effects/catalog`** → copy valid
   IDs into `ELEVENLABS_AUDIO_EFFECTS` in `.env`.

> If you skip step 4, `generate-audio` falls back to the original
> whole-script single pass (`generation_mode: "single_pass"`), so existing
> integrations keep working.

**Multi-voice variant** (per-character voices — see the diagram at the top):
replace step 4 with the casting flow:

1. `POST /projects/{id}/analyze-characters` → cast saved
2. `POST /projects/{id}/voice-casting` → top-3 candidates per speaker
3. `POST /projects/{id}/voice-casting/select` → one lock per speaker
   (repeat for `narrator` + every speaking character)
4. `POST /projects/{id}/voice-casting/approve` → required before generation
5. `POST /projects/{id}/create-multi-voice-plan` → utterances
6. `POST /projects/{id}/generate-audio` → runs in `generation_mode:
   "multi_voice"` (it prefers the multi-voice plan whenever one exists)

**The same flow in the web UI** (`app/static/`, no build step): switch on
**Use Multiple Character Voices** in the *Multi-Character Voice Casting*
section (visible and off by default after a script upload, so the
single-voice workflow is untouched). Nothing is analysed automatically —
the **Character Analyser** button runs `analyze-characters` **once** for
the current script when the user clicks it (on error it becomes *Retry
Character Analyser*, showing the actual failure reason). The page then
shows the detected roles (narrator kept separate, characters without
dialogue never get a speaking voice) and offers two casting modes:

* **Let AI choose the voices** — locks the backend's rank-1 recommendation
  for every speaker, no per-character controls;
* **I'll choose the voices** — one card per speaker with the description,
  the recommended voice, a dropdown of the real discovered voices, a
  `▶ Preview Voice` button (audition endpoint, one preview at a time,
  short cooldown after a successful one) and the current selection.

**Save & Approve Casting** locks every pending choice, approves
the cast and enables *Generate Audiobook*; generation then calls
`create-multi-voice-plan` (the ONE allowance) and `generate-audio` (joins
the same workflow — never `analyze-narration`), and only reports success
once the response says `generation_mode: "multi_voice"`. Replacing or
removing the script invalidates the analysis and the assignments.
Character analysis, voice audition, casting build/select/approve and the
counter (`GET /usage` → `max_script_characters`) are free of allowance.

**The Narration Director in the web UI** (the *Narration style* section):
a **mood-mode switch** — *Let AI choose based on my story* (default) vs
*I'll choose the mood*. *Analyze my story* runs `analyze-story` once and
shows the detected genre, recommended style, story mood, language and the
**scene progression**, with *Use recommended style* to adopt the
recommendation; *Preview the narration* renders one representative line via
`narration-preview/audio` into an inline `<audio>` player (explicit action,
one at a time, never spends an allowance). In manual mode the **12 profile
cards** appear and the chosen profile wins; legacy preset ids stored by
older sessions are mapped to the closest current profile
(`LEGACY_STYLE_ALIASES`) instead of being displayed.

**Durable plan recovery** — plan existence and validity are decided by the
SERVER (`multi_voice_plan_state.json` + a canonical fingerprint of the
stored script, cast and generation-relevant narration preferences), never
by browser storage. A reload or a cleared `localStorage` asks
`GET /projects/{id}/multi-voice-plan/status` and reuses a valid unfinished
plan at **zero** extra allowance; a repeated `create-multi-voice-plan` for
unchanged inputs joins the already-paid workflow instead of charging
again. Changed inputs invalidate the old plan (a new workflow follows the
normal quota policy), completed audiobooks stay available, and failures
never refund the reservation (conservative provider-failure policy).

**Stage 1 — audio analysis in Swagger UI** (after step 7 above, i.e. once
`audiobook.mp3` exists):

1. **`POST /projects/{id}/audio-analysis`** → no body → returns

   ```json
   {
     "success": true,
     "message": "Audio analysis submitted to pyannoteAI",
     "project_id": "…",
     "analysis_id": "…",
     "status": "processing",
     "job_status": "pending",
     "model_id": "precision-3",
     "transcription_model": "faster-whisper-large-v3-turbo",
     "audio_hash": "…",
     "reused": false,
     "created_at": "2026-11-06T…"
   }
   ```

   The request returns immediately — it never holds the connection while
   pyannote transcribes. `reused: true` means an identical analysis already
   exists (same audio hash + models) and **no credits were spent**.
2. **`GET /projects/{id}/audio-analysis/status`** → repeat until
   `status: "completed"` (each call polls the provider at most once, with
   exponential backoff between calls). On success the server saves
   `audio_analysis_raw.json` + `audio_analysis_normalized.json` +
   `audio_analysis_metadata.json` immediately (provider output is deleted
   after 24 h).
3. **`GET /projects/{id}/audio-analysis`** → normalized timeline:
   `speakers`, `words` (raw seconds + `start_ms`/`end_ms` + sample indexes),
   `speaker_turns`, optional `turn_level_transcriptions`,
   `validation_issues`.

> ⚠️ **Billing**: every *successful* pyannoteAI diarization job has a
> **20-second minimum charge**. Only run step 1 on a real account when you
> intend to pay for an analysis; the whole test suite runs mocked and spends
> nothing.

## Validation & safety rules

* **Never trust the LLM** — `NarrationPlan` Pydantic schema enforces
  `stability/similarity_boost/style ∈ [0,1]`, `speed ∈ [0.7,1.2]`,
  `clone_strength ∈ [0,1]`, `intensity ∈ [1,5]`, `pause_after_ms ∈ [0,3000]`,
  the emotion/delivery vocabularies, and `extra="forbid"`.
* **Script integrity** — every `original_text` must equal the uploaded
  paragraph exactly, every index must appear exactly once. Mismatches are
  rejected and retried with feedback (up to `MIMO_MAX_ATTEMPTS`), then a
  useful `502` is returned. No audio is ever generated from a bad plan.
* **Narration preferences (anti-overacting)** — the UI's complete object
  (`preset`, `acting_intensity` 1–5, `whisper_policy`, `whisper_paragraphs`,
  `tag_density`, `custom_direction`) is sent with `analyze-narration`,
  persisted per project (`narration_preferences.json`), forwarded to MiMo
  (system prompt + user prompt) and embedded in `narration_plan.json` — it is
  never a frontend-only setting. The system prompt always carries the
  natural-delivery rules (fear ≠ whispering, suspense ≠ breathless, sadness ≠
  crying, anger ≠ shouting, no tag just because a paragraph has an emotion
  label, tags only for a genuine audible delivery change). After Pydantic
  validation, `NarrationPolicyValidator` checks whisper policy, tag density
  (minimal 1 / standard 3 / expressive 8 per paragraph), the intensity
  ceiling, amplified/dramatic instructions (incl. `delivery_style:
  dramatic`), unnecessary delivery changes and voice-setting consistency;
  every violation becomes targeted feedback for the next attempt (up to
  `MIMO_MAX_ATTEMPTS`) and a plan is **never** silently accepted — after the
  retries it fails with `502` and nothing is saved.
* **Narration Director — 12 profiles, auto/manual, three layers** — `preset`
  is the global profile: the profile's `PROFILE_GUIDANCE` is rendered into
  the MiMo system prompt as an overall **boundary** (not a per-paragraph
  stamp), so each of the twelve profiles produces genuinely different
  planning behaviour; legacy ids stay accepted for old saved preferences but
  are hidden in the UI. In **auto** mode (`preset: "auto"`) the whole-story
  analysis (`analyze-story`) supplies the scene-by-scene mood progression
  and `recommended_profile`, and that profile's guidance is injected into
  BOTH the single-voice and the multi-voice prompt; in **manual** mode the
  user's chosen profile WINS and the analysis stays context only. The third
  layer is the per-paragraph plan itself. Story analysis persists to
  `story_analysis.json` (script-hash bound, part of the multi-voice plan
  fingerprint, `FINGERPRINT_VERSION = v2`) and is reused until the script
  changes.
* **No one-mood-everywhere (direction variety)** — after Pydantic
  validation, `NarrationPolicyValidator._check_variety` (skipped for
  scripts under 4 paragraphs) rejects a plan that stamps one mood on the
  whole story: a `narration_direction` repeated almost identically in ≥60 %
  of paragraphs with too little diversity, one **non-neutral emotion**
  dominating ≥75 % of paragraphs, one profile **mood family** covering
  ≥60 % of paragraphs, or a single audio tag repeated ≥60 % of paragraphs.
  Neutral narration must survive; a horror story still gets calm passages.
  Every violation feeds the same targeted retry loop — never a silent
  accept.
* **Preferences are part of the plan hash** — changing them changes
  `plan_hash`, so finished segments built under the old preferences are never
  reused; `generate-audio` additionally refuses with `409` when the stored
  preferences differ from the snapshot inside the plan. Plans created before
  preferences existed (`preferences: null`) keep their previous hash, so no
  already-paid-for segment is re-billed after the upgrade.
* **Audio tag safety** — `tts_text` is **mandatory for every new plan**
  (when no expression is needed it simply equals `original_text`; a plan
  missing the field is rejected and retried). It may only *insert* `[tags]`:
  stripping every tag and its single separating space must reproduce
  `original_text` byte-for-byte (no added/removed words, no prose
  instructions that would be spoken aloud, no SSML `<break>` — Eleven v4
  does not support SSML, and tags are capped at 8 per paragraph). Plans
  saved before the upgrade (without `tts_text`) still load and generate.
* **Model capability verification** — `app/services/eleven_models.py` knows
  what each model accepts. `eleven_v4`/`eleven_v4_turbo` never receive
  `speed`/`style`; `clone_strength` only reaches the API for a voice flagged
  as an eligible v4 Professional Voice Clone; audio tags are stripped for
  models without tag support; `language_code` is never sent to
  `eleven_multilingual_v2`.
* **Only supported fields reach ElevenLabs** — `stability`,
  `similarity_boost`, `use_speaker_boost` (+ capability-gated `speed`,
  `style`, `clone_strength`). `emotion`, `delivery_style`,
  `narration_direction` and `pause_after_ms` are metadata. The API keeps the
  two apart explicitly: `voice_model` + `tts_voice_settings` in the
  narration response show what is *supported* and what is actually *sent*
  per paragraph, while `paragraphs[].voice_settings` stays descriptive
  planning metadata.
* **Continuity without conflicts** — first-time generation sends neighbour
  `previous_text`/`next_text`; regeneration uses stored `request-id`s
  (fresh < 2 h, same model, same plan, max 3 per side). Text and request-ID
  context are never sent together for the same side.
* **Normalization** — `apply_text_normalization` defaults to `auto`;
  `apply_language_text_normalization` is only ever enabled for Japanese and
  never for Hindi.
* **Pronunciation dictionaries / audio effects** — only IDs explicitly
  configured in `.env` are sent (placeholders like `your_…` are dropped,
  max 3 locators); effect IDs are checked against the official
  `/v1/audio-effects/catalog`. Effects colour the voice only — background
  music/SFX stay in the FFmpeg mixing pipeline.
* **Audio analysis validation (Stage 1)** — provider records with missing,
  non-numeric or negative timestamps, `end <= start`, empty text, a missing
  speaker, or spans beyond `duration_tolerance_ms` past the real duration
  (default 500 ms, `PYANNOTE_DURATION_TOLERANCE_MS`) are **excluded from the
  normalized file and listed in `validation_issues`**; the raw file always
  keeps them untouched. Missing diarization array, missing transcription
  arrays, or zero usable records fail the analysis (raw still preserved).
  Speaker **overlaps are allowed and never dropped**. Real sample rate comes
  from ffprobe — never a hardcoded 44100.
* **Pyannote stays separate** — the Stage 1 analysis never feeds the
  narration planner: `analyze-narration` only ever reads `script.txt`, the
  original script remains the single source of truth for `original_text`,
  and the diarized transcript is never used as story text. No new Sound
  Design / QA features were added.
* **Secrets** — loaded from `.env`, excluded from `repr()`, never logged or
  returned. Pre-signed pyannote upload URLs are redacted before any logging
  or error message.
* **Partial failure** — if paragraph *N* fails, finished segments stay on
  disk, `metadata.json` records `status: "partial"` and the error names the
  failed index. The next run **reuses** already-generated segments (matched
  by plan hash — and, in multi-voice mode, by the **locked voice** as well,
  so replacing a character's voice always regenerates that speaker's
  utterances in the new voice) so they are not re-billed. A final MP3 is
  only reported once FFmpeg has actually written it.

## Error handling

| Situation | Status |
| --------- | ------ |
| Unknown project / no script / no plan yet | `404` |
| No character cast / no multi-voice plan yet | `404` |
| Empty script, script > `ELEVENLABS_MAX_CHARS` | `400` |
| Narration preferences invalid for this script (whisper index) | `400` |
| Invalid plan/analysis from MiMo after retries (incl. policy + variety violations) | `502` |
| Story analysis failed after retries / none saved yet (GET) | `502` / `404` |
| Script changed after analysis (stale plan) | `409` |
| Narration preferences changed after the plan was generated | `409` |
| Cast incomplete or not approved (generation) | `409` |
| Unknown speaker / voice not found / restricted voice | `400` |
| Missing `MIMO_API_KEY` / `ELEVENLABS_API_KEY` | `500` |
| MiMo/ElevenLabs rejected the key | `502` — **never `401`**: `401` is reserved for *this app's* own session/token, so a provider key problem can't log the user out of the UI |
| Rate limited | `429` |
| Upstream failure / bad response | `502` |
| MiMo timeout | `504` |
| ElevenLabs timeout | `504` |
| pyannoteAI timeout / rejected key / rate limit | `504` / `502` / `429` (same 401 reservation) |
| pyannoteAI job failed / normalization produced nothing | `502` |
| Audio analysis started but not finished yet (GET) | `409` |
| Missing `PYANNOTE_API_KEY` (analysis start) | `500` |
| FFmpeg missing / failed | `500` |
| One paragraph/utterance failed (segments preserved) | `502` |
| Disk write failure | `500` |

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

**583 tests passing + 8 skipped (the opt-in live/smoke tests), zero credits spent by default:**

* `tests/conftest.py` blanks `MIMO_API_KEY` / `ELEVENLABS_API_KEY` /
  `PYANNOTE_API_KEY` and points the base URLs at an unroutable port unless a
  live flag is set; it also pins `eleven_v4` + `mp3_44100_128` so capability
  filtering is deterministic regardless of a local `.env`;
* MiMo is replaced by `FakeMiMo` (scripted JSON / errors) — covering JSON
  parsing, Hindi analysis, emotion classification, per-paragraph settings,
  `tts_text` audio-tag generation and validation, character discovery and
  dialogue attribution, utterance splitting, Pydantic validation, retries
  with feedback, missing key, timeout, failures;
* narration preferences are covered by
  `tests/test_narration_preferences.py` (28 tests, all mocked): GET/PUT
  persistence + defaults + `422`/`400` validation, all three whisper
  policies (allowed **and** violated, with the retry feedback asserted on
  the second MiMo call), intensity ceilings, tag-density budgets
  (minimal/standard/expressive), dramatic-instruction budgets, plan hashing
  (preferences change the hash, legacy plans keep theirs), preview of a
  single paragraph (no `narration_plan.json` written, script-index mapping
  for whisper selections, `400` out of range), `409` staleness on
  generate-audio (no synthesis, no FFmpeg run), capability-filtered
  `tts_voice_settings` for `eleven_v4`, the anti-overacting prompt rules and
  the served frontend UI;
* the Narration Director is covered by `tests/test_narration_director.py`
  (26 tests, all mocked): the twelve profiles + their distinct planning
  guidance, auto vs manual mode (auto injects the analysed
  `recommended_profile`, manual keeps the user's profile while the analysis
  stays context), legacy preset ids, story-analysis persistence/reuse
  (`story_analysis.json`, script-hash bound, `404` when absent),
  `{"force": true}`, the `narration-preview/audio` MP3 (no plan overwrite,
  no allowance), and `_check_variety` — direction repetition, ≥75 % dominant
  emotion, ≥60 % mood-family/tags, the under-4-paragraph exemption, plus
  `401`-reservation regressions in `test_beta_security.py` /
  `test_multi_voice.py` / `test_mimo_client.py` (a provider key failure
  must map to `502`, never log the user out);
* ElevenLabs is intercepted with `httpx.MockTransport` to assert the exact
  request body, query params, voice URLs and `request-id` header —
  covering voice discovery + pagination (`/v2/voices`, `/v1/shared-voices`),
  candidate ranking and restriction/language filtering, permanent voice
  locking (including voice-change regeneration — only the replaced speaker
  re-synthesises), multi-voice generation with different voice IDs per
  utterance, v4 capability filtering (no `speed`/`style`), neighbour-text vs
  request-ID continuity, pronunciation dictionaries, text normalization,
  audio effects + catalog, and MP3/WAV output formats;
* FFmpeg is replaced by `FakeComposer` in unit tests (asserting utterance
  merge ORDER and pauses), while `tests/test_composer.py` and
  `tests/test_eleven_v4.py` run the **real** ffmpeg on locally generated
  tones (MP3 and WAV intermediates) and check the resulting duration
  (skipped automatically if ffmpeg is absent);
* pyannoteAI is fully **mocked** with `httpx.MockTransport`
  (`tests/test_audio_analysis.py`, 37 tests): upload → diarize → job polling,
  duplicate reuse (no second billing), Decimal `ROUND_HALF_UP` ms/sample
  conversion from the ffprobe sample rate, raw/normalized/metadata files,
  validation exclusions + overlap preservation, timeout never retried on
  submit, and every error path (404/409/429/500/502/504) — no live pyannote
  call ever runs.

The served frontend is covered too: `tests/test_frontend.py` asserts the
markup/contract of `index.html` + `style.css`, and `node` runs the
behaviour suites in `tests/js/` against the **real** `app.js` with a fake
DOM and mocked FastAPI responses — `limits.test.js` (character counting),
`app_behavior.test.js` (upload → plan → generate → download) and
`casting.test.js` (20 checks for the multi-character casting option:
analysis-once, role display, automatic/manual casting, previews,
script-replacement invalidation, duplicate guards, zero-allowance
behaviour, multi-voice generation + honest `generation_mode` check).
`tests/test_casting_ui.py` adds the markup/style checks, the `/usage`
script-limit contract and the mocked end-to-end casting workflow.

Opt-in live tests (never run by default):

```powershell
$env:RUN_MIMO_LIVE_TEST = "1"          # real MiMo analysis (costs tokens)
$env:RUN_ELEVENLABS_LIVE_TEST = "1"    # real ElevenLabs roundtrip (costs credits)

# Narration Director smoke — 6 real-MiMo story-analysis cases (auto + manual,
# Hindi-horror + emotional-sad scripts), PLAN ONLY: no ElevenLabs, no pyannote,
# no FFmpeg, no allowance/quota changes. Needs BOTH flags:
$env:RUN_MIMO_LIVE_TEST = "1"
$env:RUN_MIMO_DIRECTOR_SMOKE = "1"
.\.venv\Scripts\python.exe -m pytest tests\test_mimo_director_smoke.py -q
```

## Configuration reference

| Env var | Default | Meaning |
| ------- | ------- | ------- |
| `MIMO_API_KEY` | — (needed for analysis) | Xiaomi MiMo key |
| `MIMO_MODEL_ID` | `mimo-v2.6-flash` | MiMo model |
| `MIMO_BASE_URL` | `https://api.xiaomimimo.com/v1` | MiMo endpoint (this project uses `https://token-plan-sgp.xiaomimimo.com/v1`) |
| `MIMO_TIMEOUT_SECONDS` | `120` | MiMo request timeout |
| `MIMO_MAX_COMPLETION_TOKENS` | `8192` | Max completion tokens |
| `MIMO_MAX_ATTEMPTS` | `3` | Retries when the plan is invalid |
| `ELEVENLABS_API_KEY` | — (needed for audio) | ElevenLabs key |
| `ELEVENLABS_VOICE_ID` | — | Voice to speak with |
| `ELEVENLABS_MODEL_ID` | `eleven_v4` | TTS model (audio tags, 90+ languages) |
| `ELEVENLABS_LANGUAGE_CODE` | *(plan-derived)* | ISO 639-1 hint (`hi`, `en`, …) |
| `ELEVENLABS_MAX_CHARS` | `5000` | Max characters per request (v4 allows 10000) |
| `ELEVENLABS_TIMEOUT_SECONDS` | `60` | TTS timeout |
| `ELEVENLABS_OUTPUT_FORMAT` | `mp3_44100_128` | Final audiobook format (MP3 44.1 kHz) |
| `ELEVENLABS_INTERMEDIATE_OUTPUT_FORMAT` | *(off)* | Per-segment master, e.g. `wav_44100` / `pcm_44100` — FFmpeg encodes the MP3 once |
| `ELEVENLABS_APPLY_TEXT_NORMALIZATION` | `auto` | `auto` \| `on` \| `off` |
| `ELEVENLABS_APPLY_LANGUAGE_TEXT_NORMALIZATION` | `false` | Japanese only — never enabled for Hindi |
| `ELEVENLABS_PRONUNCIATION_DICTIONARIES` | *(off)* | JSON list of dictionary IDs/locators (max 3, never invented) |
| `ELEVENLABS_AUDIO_EFFECTS` | *(off)* | JSON `audio_effects` body (IDs from `/elevenlabs/audio-effects/catalog`) |
| `ELEVENLABS_VOICE_IS_PROFESSIONAL_CLONE` | `false` | Enables `clone_strength` (v4 PVC only) |
| `ELEVENLABS_CLONE_STRENGTH` | *(off)* | Default clone strength 0..1 for eligible PVCs |
| `FFMPEG_PATH` | `ffmpeg` (PATH) | ffmpeg binary |
| `FFPROBE_PATH` | `ffprobe` (PATH) | ffprobe binary (real sample rate/duration) |
| `FFMPEG_TIMEOUT_SECONDS` | `600` | Composition timeout |
| `PYANNOTE_API_KEY` | — (needed for analysis) | pyannoteAI key (never logged) |
| `PYANNOTE_BASE_URL` | `https://api.pyannote.ai` | pyannoteAI endpoint |
| `PYANNOTE_MODEL_ID` | `precision-3` | Diarization model |
| `PYANNOTE_TRANSCRIPTION_MODEL` | `faster-whisper-large-v3-turbo` | Word-level transcription model |
| `PYANNOTE_TRANSCRIPTION` | `true` | Request word-level transcription |
| `PYANNOTE_TIMEOUT_SECONDS` | `30` | Per-request timeout to pyannoteAI |
| `PYANNOTE_MAX_ATTEMPTS` | `2` | Attempts for idempotent GETs/uploads (submit is NEVER retried) |
| `PYANNOTE_POLL_INTERVAL_SECONDS` | `2` | Initial backoff between job polls |
| `PYANNOTE_POLL_MAX_INTERVAL_SECONDS` | `30` | Backoff cap |
| `PYANNOTE_DURATION_TOLERANCE_MS` | `500` | Allowed timestamp overshoot past audio end |
| `STORAGE_DIR` | `<repo>/storage/projects` | Storage root |
| `MAX_SCRIPT_SIZE_MB` | `5` | Upload limit |
| `MAX_SCRIPT_CHARACTERS` | `200000` | Script length limit in Unicode code points (`0` = unlimited; exactly the limit is accepted). The value served to the UI counter comes from `GET /usage`; this repository's `.env` sets `200000`. |
| `MAX_AUDIO_DURATION_MINUTES` | `120` | Cap on the estimated spoken duration (`0` = unlimited) |
| `SPEECH_CHARS_PER_MINUTE` | `900` | Heuristic used for the duration estimate |
| `BETA_ACCESS_TOKENS` | *(off)* | `name:token,name2:token2` — per-tester tokens (beta mode) |
| `BETA_ACCESS_TOKEN` | *(off)* | Single shared token (tester name `tester`) |
| `BETA_SESSION_SECRET` | *(derived)* | HMAC key for session cookies |
| `BETA_SESSION_TTL_HOURS` | `72` | Session lifetime |
| `BETA_COOKIE_SECURE` | `false` | Set `true` behind HTTPS |
| `BETA_GENERATION_LIMIT` | `10` | Generation allowance per tester (lifetime; `0` = no generations) |
| `BETA_TESTER_GENERATION_LIMITS` | *(none)* | Per-tester overrides: `name:1,name2:3` (names must exist in `BETA_ACCESS_TOKENS`) |
| `BETA_GENERATION_WINDOW_HOURS` | `24` | Deprecated (no effect — usage is lifetime) |
| `BETA_RATE_LIMIT_PER_MINUTE` | `120` | API requests per tester per minute |
| `BETA_LOGIN_RATE_LIMIT_PER_MINUTE` | `10` | Login attempts per IP per minute |
| `BETA_DUPLICATE_WINDOW_SECONDS` | `10` | Duplicate-submission window for generation calls |

## Private beta & staging deployment

A minimal security layer protects the app while one or two external clients
test it. It activates **only when tester tokens are configured** — without
tokens the app behaves exactly as before (development/tests) and the startup
log prints a warning.

**What it enforces when enabled** (`app/security.py`, one ASGI middleware):

* every request needs a token — the frontend, all API routes, project routes
  and `/media` downloads. Only `/auth/*` and `/static/style.css` are public.
  Tokens arrive as `Authorization: Bearer …`, `X-Beta-Token: …` or an
  HMAC-signed HttpOnly session cookie from `POST /auth/login`;
* **project isolation** — projects record their owning tester; another
  tester's project returns the same `404` as a missing one (no existence
  leak), including media downloads and path-traversal attempts;
* **limits** — per-tester generation allowances (persistent, configurable
  per tester via `BETA_TESTER_GENERATION_LIMITS`), per-minute rate
  limiting (stricter on login) and duplicate-submission protection
  (`409`) for the credit-spending endpoints;
* upload limits: file size (413), script length and estimated audiobook
  duration (400) at upload **and** generation time;
* `/docs`, `/openapi.json` and `/redoc` are disabled for beta users;
* privacy notice + "Delete my projects & files" (`DELETE /projects`,
  `DELETE /projects/{id}`) in the UI;
* provider API keys stay server-side — they are read from the environment
  and never rendered to any client.

**Generation allowances & usage** (`app/security.py`, `GET /usage`):

* One **complete audiobook workflow** — narration planning (MiMo retries and
  plan validation included) plus audio generation (ElevenLabs segments, audio
  processing, FFmpeg assembly, storage) — shares ONE `generation_id` and
  consumes **exactly one** allowance. The planning call opens the workflow,
  `generate-audio` joins it. Auxiliary operations (**whole-story analysis**,
  narration paragraph preview, **narration preview audio**, character
  analysis, casting audition, diarization/audio analysis) are **not**
  audiobook generations: they never reserve or consume the allowance and
  are never blocked by it — even at `0` remaining. They keep their rate
  limiting and duplicate-submission protection (`409`).
* Usage is **persistent and lifetime** per tester: `used` never resets on
  restart or configuration change, and
  `remaining = max(0, effective_limit - used)`. Raising a limit immediately
  makes the extra allowance available; lowering it below the current usage
  blocks generation again until the limit exceeds usage.
* Limits are non-negative integers; **`0` means the tester cannot generate**
  (semantic change: this used to mean "unlimited"). Negative/decimal values
  fail startup.
* Reservations follow the conservative provider-failure policy: an allowance
  is taken when a generation request is accepted and is **never refunded** —
  a provider call may already have been billed.
* `GET /usage` (authenticated) returns the caller's own numbers only —
  `{generation_limit, generations_used, generations_remaining,
  max_script_characters}`. There is no endpoint to edit allowances.
* Login token redemption and generation allowances are fully independent:
  logging in consumes no allowance, generating never invalidates a token.
* **Persistence & concurrency**: usage lives in `storage/_beta/quota.json`
  and is written under `_QUOTA_LOCK`, a `threading.Lock` — atomic **within
  one OS process only**. The beta therefore runs **one Uvicorn worker**;
  multi-worker quota safety is not provided and must not be claimed. Before
  enabling `--workers N>1`, the ledger needs cross-process coordination
  (an OS file lock around the read-modify-write, or a transactional shared
  store such as SQLite/Redis) — otherwise concurrent workers can lose usage
  updates. Verified by an isolated test that restarts the ledger in a
  second OS process on the same storage (`tests/test_quota_audit.py`).

**Staging deployment checklist:**

1. Set one token per tester: `BETA_ACCESS_TOKENS=alice:<long-random>,bob:<long-random>`
   (generate with e.g. `python -c "import secrets; print(secrets.token_urlsafe(32))"`).
   Never commit tokens — keep them in the server's `.env` only.
2. Serve over **HTTPS** and set `BETA_COOKIE_SECURE=true` (session cookies).
3. Optionally set `BETA_SESSION_SECRET` (rotating it logs all testers out).
4. Tune the guards: `BETA_GENERATION_LIMIT`, `BETA_TESTER_GENERATION_LIMITS`,
   `BETA_RATE_LIMIT_PER_MINUTE`, `MAX_SCRIPT_SIZE_MB`,
   `MAX_SCRIPT_CHARACTERS`, `MAX_AUDIO_DURATION_MINUTES`.
5. Run with the venv Python behind a process manager — **exactly one
   worker** (the quota ledger is only atomic inside a single OS process):
   `.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000`
   (the default `--workers 1`; never pass `--workers >1` for the beta —
   bind locally and put a TLS reverse proxy — Caddy/nginx/IIS — in front).
6. Verify before sending invites: `GET /` returns 302→login, `GET /docs`
   returns 404, `GET /health` returns 401 without a token.
7. Provider keys (`MIMO_API_KEY`, `ELEVENLABS_API_KEY`, `PYANNOTE_API_KEY`)
   live only in the server-side `.env`; quotas cap the worst-case spend.

## Not implemented yet (by design)

Background music · sound effects beds in the final mix · sound-design
director · timeline resolver · FFmpeg multitrack mixing · QA
engine · SFX/music/mixing controls (kept out of the simple creator until
they are ready — a static **audiobook creator** IS served at `/`, see
`app/static/`) · Celery/Redis
job queue. (Voice *colouring* via ElevenLabs `audio_effects` IS supported —
see `ELEVENLABS_AUDIO_EFFECTS`. And *story* analysis — genre, scene-by-scene
mood progression, profile recommendation — IS implemented: `analyze-story`.)

### Stage 1 operational limits (read before production use)

* **No durable worker** — pyannote jobs are polled *on demand* when
  `GET …/audio-analysis/status` is called (exponential backoff between
  provider calls). If nobody polls, the job still finishes on pyannoteAI's
  side, but the result is only fetched the next time the status endpoint is
  hit — and pyannoteAI deletes job output **24 hours** after completion.
  After that window the local analysis stays `processing` and must be
  re-submitted (billed again).
* **Restart behavior** — job state lives in
  `storage/projects/{id}/audio/analysis/audio_analysis_metadata.json`, so a
  server restart does **not** lose a submitted job; polling simply resumes
  on the next status call. There is no background scheduler and no webhook
  receiver (no stable public HTTPS endpoint yet).
* **Duplicate protection** — an identical (audio hash + model + transcription)
  analysis is reused; `POST /v1/diarize` is never auto-retried, because a
  duplicate submit is a duplicate bill (20-second minimum per successful job).
