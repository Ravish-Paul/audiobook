# AI Audiobook Platform — MiMo Narration Director + ElevenLabs TTS

A FastAPI backend where an **AI narration director** decides how every
paragraph of your script should sound, ElevenLabs speaks it, and FFmpeg
assembles the final audiobook.

```
Upload script
  → Xiaomi MiMo reads the whole story (context + emotional progression)
  → MiMo decides emotion / delivery / voice settings / pauses per paragraph
    AND writes `tts_text` with inline Eleven v4 audio tags ([whispers], …)
  → FastAPI validates the JSON plan (Pydantic + tag rules) → narration_plan.json
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

**Separation of responsibilities (important):**

| Component | Responsibility |
| --------- | -------------- |
| **Xiaomi MiMo** (`mimo-v2.6-flash`) | Decides *how* to narrate: emotion, intensity, inline audio tags, voice transitions, continuity. Never rewrites the script, never touches ElevenLabs. |
| **ElevenLabs** (`eleven_v4`) | Only converts text → speech with the numeric settings it is given. It makes **no** narration decisions. |
| **FFmpeg** | Concatenates the segments, inserts the pauses, and encodes the final MP3 (once) — background music/SFX belong here. |

## Tech stack

* Python 3.11+ · FastAPI · Uvicorn · Pydantic v2
* `openai` SDK → Xiaomi MiMo (OpenAI-compatible endpoint)
* `httpx` → ElevenLabs REST API (async)
* FFmpeg → segment composition
* `python-dotenv` → secrets from `.env`

## Endpoints

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| GET | `/` | Welcome message |
| GET | `/health` | Health check (storage, MiMo, ElevenLabs, FFmpeg status) |
| POST | `/projects` | Create a project → UUID |
| GET | `/projects/{project_id}` | Project details, `404` if not found |
| POST | `/projects/{project_id}/upload-script` | Upload a `.txt` script (max 5 MB, UTF-8) |
| POST | `/projects/{project_id}/analyze-narration` | **MiMo directs the script → narration_plan.json** (incl. `tts_text` audio tags) |
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
| GET | `/elevenlabs/audio-effects/catalog` | Valid IDs for optional `ELEVENLABS_AUDIO_EFFECTS` |

## Project structure

```
audiobook_web/
├── app/
│   ├── main.py                    # app, /media mount, error handlers, startup log
│   ├── config.py                  # Settings (env + .env driven)
│   ├── routes/
│   │   ├── health.py              # GET / , GET /health
│   │   ├── projects.py            # projects + script upload
│   │   ├── narration.py           # analyze-narration, narration-plan
│   │   ├── casting.py             # characters, voice casting, utterance plan
│   │   ├── audio.py               # generate-audio, audio info
│   │   └── elevenlabs.py          # audio-effects catalog proxy
│   ├── schemas/
│   │   ├── project.py             # project / script models
│   │   ├── narration.py           # NarrationPlan, ParagraphPlan (tts_text), VoiceSettings
│   │   ├── casting.py             # characters, voice profiles, candidates, locks
│   │   ├── multi_voice.py         # Utterance, MultiVoicePlan
│   │   ├── audio.py               # audio response model
│   │   └── elevenlabs.py          # audio effects catalog models
│   └── services/
│       ├── mimo_client.py         # Xiaomi MiMo (openai SDK) + error mapping
│       ├── narration_planner.py   # paragraphs → MiMo → validated plan
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
│       ├── project_service.py     # storage layout & metadata
│       └── errors.py              # domain errors → HTTP status codes
├── storage/projects/{project_id}/
│   ├── project.json
│   ├── script.txt
│   ├── narration_plan.json
│   └── audio/
│       ├── segments/paragraph_0001.mp3|wav|pcm …
│       ├── audiobook.mp3
│       └── metadata.json
├── tests/
│   ├── conftest.py                # hermetic env (no real credentials)
│   ├── fakes.py                   # FakeMiMo / FakeComposer / plan builders
│   ├── test_api.py                # projects + upload
│   ├── test_audio.py              # legacy single-pass audio
│   ├── test_eleven_v4.py          # v4 tags, continuity, dictionaries, formats
│   ├── test_mimo_client.py        # MiMo transport + error mapping
│   ├── test_multi_voice.py        # casting, locks, utterances, generation
│   ├── test_narration.py          # planning, validation, retries, Hindi
│   ├── test_plan_audio.py         # segments, recovery, metadata, playback
│   ├── test_tts_text_regression.py # tts_text mandatory + exact-word rules
│   ├── test_voice_discovery.py    # voice search, pagination, ranking
│   └── test_composer.py           # real FFmpeg composition
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

Open <http://127.0.0.1:8000/docs>.

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
```

Optional: `MIMO_BASE_URL`, `MIMO_TIMEOUT_SECONDS`, `MIMO_MAX_COMPLETION_TOKENS`,
`MIMO_MAX_ATTEMPTS`, `FFMPEG_PATH`, `ELEVENLABS_*` (language, normalization,
output/intermediate formats, pronunciation dictionaries, audio effects, PVC
clone strength), `STORAGE_DIR`, `MAX_SCRIPT_SIZE_MB` — see `.env.example`.

Startup prints status only, never secrets:

```
[startup] elevenlabs: configured | voice=7b9m… | model=eleven_v4
[startup] mimo: configured | model=mimo-v2.6-flash | base_url=https://api.xiaomimimo.com/v1
[startup] ffmpeg: C:\Users\you\ffmpeg\ffmpeg.EXE
```

## Full workflow in Swagger UI

1. **`POST /projects`** → `{"name": "My Hindi Audiobook"}` → copy `project_id`.
2. **`POST /projects/{id}/upload-script`** → choose `sample_script.txt`.
3. **`POST /projects/{id}/analyze-narration`** → MiMo reads the whole story and
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
     "saved": true,
     "saved_at": "2026-11-05T…"
   }
   ```

   `original_text` is always the untouched script; `tts_text` adds inline
   Eleven v4 audio tags (`[whispers]`, `[laughs]`, `[low, ominous narrator
   voice]`, …) — the spoken words are validated to be identical.
4. **`GET /projects/{id}/narration-plan`** → review it (or re-run step 3).
5. **`POST /projects/{id}/generate-audio`** → segments + FFmpeg merge:

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

6. **Play it**: open the `audio_url`, e.g.
   <http://127.0.0.1:8000/media/projects/{id}/audio/audiobook.mp3>
7. *(optional)* **`GET /elevenlabs/audio-effects/catalog`** → copy valid
   IDs into `ELEVENLABS_AUDIO_EFFECTS` in `.env`.

> If you skip step 3, `generate-audio` falls back to the original
> whole-script single pass (`generation_mode: "single_pass"`), so existing
> integrations keep working.

## Validation & safety rules

* **Never trust the LLM** — `NarrationPlan` Pydantic schema enforces
  `stability/similarity_boost/style ∈ [0,1]`, `speed ∈ [0.7,1.2]`,
  `clone_strength ∈ [0,1]`, `intensity ∈ [1,5]`, `pause_after_ms ∈ [0,3000]`,
  the emotion/delivery vocabularies, and `extra="forbid"`.
* **Script integrity** — every `original_text` must equal the uploaded
  paragraph exactly, every index must appear exactly once. Mismatches are
  rejected and retried with feedback (up to `MIMO_MAX_ATTEMPTS`), then a
  useful `502` is returned. No audio is ever generated from a bad plan.
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
  `narration_direction` and `pause_after_ms` are metadata.
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
* **Secrets** — loaded from `.env`, excluded from `repr()`, never logged or
  returned.
* **Partial failure** — if paragraph *N* fails, finished segments stay on
  disk, `metadata.json` records `status: "partial"` and the error names the
  failed index. The next run **reuses** already-generated segments (matched
  by plan hash) so they are not re-billed. A final MP3 is only reported once
  FFmpeg has actually written it.

## Error handling

| Situation | Status |
| --------- | ------ |
| Unknown project / no script / no plan yet | `404` |
| Empty script, script > `ELEVENLABS_MAX_CHARS` | `400` |
| Invalid plan from MiMo after retries | `502` |
| Script changed after analysis (stale plan) | `409` |
| Missing `MIMO_API_KEY` / `ELEVENLABS_API_KEY` | `500` |
| MiMo/ElevenLabs rejected the key | `401` |
| Rate limited | `429` |
| Upstream failure / bad response | `502` |
| MiMo timeout | `504` |
| ElevenLabs timeout | `504` |
| FFmpeg missing / failed | `500` |
| One paragraph failed (segments preserved) | `502` |
| Disk write failure | `500` |

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

**195 tests passing + 2 opt-in live tests, zero credits spent by default:**

* `tests/conftest.py` blanks `MIMO_API_KEY` / `ELEVENLABS_API_KEY` and points
  both base URLs at an unroutable port unless a live flag is set; it also
  pins `eleven_v4` + `mp3_44100_128` so capability filtering is
  deterministic regardless of a local `.env`;
* MiMo is replaced by `FakeMiMo` (scripted JSON / errors) — covering JSON
  parsing, Hindi analysis, emotion classification, per-paragraph settings,
  `tts_text` audio-tag generation and validation, character discovery and
  dialogue attribution, utterance splitting, Pydantic validation, retries
  with feedback, missing key, timeout, failures;
* ElevenLabs is intercepted with `httpx.MockTransport` to assert the exact
  request body, query params, voice URLs and `request-id` header —
  covering voice discovery + pagination (`/v2/voices`, `/v1/shared-voices`),
  candidate ranking and restriction/language filtering, permanent voice
  locking, multi-voice generation with different voice IDs per utterance,
  v4 capability filtering (no `speed`/`style`), neighbour-text vs
  request-ID continuity, pronunciation dictionaries, text normalization,
  audio effects + catalog, and MP3/WAV output formats;
* FFmpeg is replaced by `FakeComposer` in unit tests (asserting utterance
  merge ORDER and pauses), while `tests/test_composer.py` and
  `tests/test_eleven_v4.py` run the **real** ffmpeg on locally generated
  tones (MP3 and WAV intermediates) and check the resulting duration
  (skipped automatically if ffmpeg is absent).

Opt-in live tests:

```powershell
$env:RUN_MIMO_LIVE_TEST = "1"          # real MiMo analysis (costs tokens)
$env:RUN_ELEVENLABS_LIVE_TEST = "1"    # real ElevenLabs roundtrip (costs credits)
.\.venv\Scripts\python.exe -m pytest -q
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
| `FFMPEG_TIMEOUT_SECONDS` | `600` | Composition timeout |
| `STORAGE_DIR` | `<repo>/storage/projects` | Storage root |
| `MAX_SCRIPT_SIZE_MB` | `5` | Upload limit |

## Not implemented yet (by design)

Background music · sound effects beds in the final mix · multiple-character
voices · automatic script rewriting · frontend dashboard · Celery/Redis job
queue. (Voice *colouring* via ElevenLabs `audio_effects` IS supported — see
`ELEVENLABS_AUDIO_EFFECTS`.)
