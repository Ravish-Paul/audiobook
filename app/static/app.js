/* Simple audiobook creator — vanilla JS, no build step, no dependencies.
 *
 * Flow:
 *   1. pick/drop a .txt script  -> POST /projects/from-script (auto project,
 *      detected title + language, paragraph count, preview)
 *   2. pick a narration style   -> maps to NarrationPreferences
 *   3. Generate Audiobook       -> PUT preferences -> POST analyze-narration
 *      (MiMo plan) -> POST generate-audio (ElevenLabs + FFmpeg)
 *   4. listen + download the MP3
 */
"use strict";

/* ------------------------------------------------------------------------- */
/* Narration styles -> the backend NarrationPreferences object                */
/* ------------------------------------------------------------------------- */
const STYLE_PRESETS = {
  natural: {
    label: "Natural",
    preferences: {
      preset: "natural",
      acting_intensity: 2,
      whisper_policy: "never",
      whisper_paragraphs: [],
      tag_density: "minimal",
      custom_direction: "",
    },
  },
  cinematic: {
    label: "Cinematic",
    preferences: {
      preset: "restrained_cinematic",
      acting_intensity: 2,
      whisper_policy: "user_selected",
      whisper_paragraphs: [],
      tag_density: "minimal",
      custom_direction: "",
    },
  },
  expressive: {
    label: "Expressive",
    preferences: {
      preset: "expressive",
      acting_intensity: 3,
      whisper_policy: "user_selected",
      whisper_paragraphs: [],
      tag_density: "standard",
      custom_direction: "",
    },
  },
};

const MAX_SCRIPT_BYTES = 5 * 1024 * 1024;
const STORAGE_KEY = "audiobook.state.v1";

const state = {
  busy: false,
  style: "natural",
  languageOverride: "", // "" = auto-detect
  project: null, // {id, title, language, paragraphCount, preview, filename, sizeBytes, characterCount}
  planKey: null, // preferences the saved narration plan was built with
  audio: null, // {url, filename}
  usage: null, // /usage: {generation_limit, generations_used, generations_remaining, max_script_characters}
  usageStatus: "loading", // "loading" | "ready" | "error" — never fake a limit
  characterCount: 0, // script length in Unicode code points (same count as the backend)
  // Optional multi-character casting (§"Use Multiple Character Voices").
  // Everything server-authoritative (candidates, locks, approval) lives in
  // `voiceCasting` and is re-read from GET /voice-casting — never assumed.
  casting: {
    enabled: false, // the toggle (off by default: single-voice workflow)
    mode: "auto", // "auto" = Let AI choose the voices | "manual"
    phase: "idle", // idle | analyzing steps | ready | error
    step: null, // "analyse" | "rank" | "assign" while busy
    analysis: null, // CharacterAnalysisResponse for the CURRENT script
    analysisKey: null, // `${projectId}:${characterCount}` of that analysis
    voiceCasting: null, // VoiceCastingResponse (candidates + locks + approval)
    drafts: {}, // speakerId -> voice_id chosen in the UI (not yet locked)
    busy: false, // analysis / casting / approval in flight
    runId: 0, // invalidates work started for a replaced script
    auditions: {}, // speakerId -> {status, url, audio, error}
    auditionBusy: false, // one preview at a time (server guards by path)
    lastAuditionAt: 0, // cooldown after a SUCCESSFUL preview
    mvPlanKey: null, // script + cast the saved multi-voice plan was built for
    analysisError: null,
  },
};

/* ------------------------------------------------------------------------- */
/* Small helpers                                                              */
/* ------------------------------------------------------------------------- */
const $ = (id) => document.getElementById(id);

function setStatus(message, kind = "info") {
  const node = $("status");
  if (!node) {
    console.warn(`[status:${kind}]`, message);
    return;
  }
  node.textContent = message || "";
  node.className = `status ${kind}`;
}

function detailOf(error) {
  if (error && error.detail) {
    if (typeof error.detail === "string") return error.detail;
    try {
      return JSON.stringify(error.detail);
    } catch (_) {
      return String(error.detail);
    }
  }
  return String(error && error.message ? error.message : error);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    /* empty or non-JSON body */
  }
  if (!response.ok) {
    const error = new Error(detailOf(payload || { detail: `HTTP ${response.status}` }));
    error.status = response.status;
    if (response.status === 401) {
      // Session expired / private beta login required — go to the login page.
      window.location.href = "/auth/login";
    }
    throw error;
  }
  return payload;
}

function jsonRequest(method, path, body) {
  return api(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds) {
  if (!isFinite(seconds)) return "";
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`
    : `${minutes}:${String(rest).padStart(2, "0")}`;
}

function downloadName(title, fallback) {
  const base = (title || "").replace(/[\\/:*?"<>|]+/g, "").trim();
  return `${base || fallback || "audiobook"}.mp3`;
}

function saveState() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        style: state.style,
        languageOverride: state.languageOverride,
        project: state.project,
        planKey: state.planKey,
        audio: state.audio,
        casting: {
          enabled: state.casting.enabled,
          mode: state.casting.mode,
          mvPlanKey: state.casting.mvPlanKey,
          analysis: state.casting.analysis,
          analysisKey: state.casting.analysisKey,
        },
      })
    );
  } catch (error) {
    console.warn("Could not save state:", error);
  }
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    console.warn("Could not restore state:", error);
    return null;
  }
}

/* ------------------------------------------------------------------------- */
/* Progress stages (honest: each stage is a real, known backend state)        */
/* ------------------------------------------------------------------------- */
const STAGE_IDS = ["upload", "understand", "prepare", "generate", "done"];

function setStage(id, status) {
  const item = document.querySelector(`#stage-list [data-stage="${id}"]`);
  if (!item) return;
  item.classList.toggle("active", status === "active");
  item.classList.toggle("done", status === "done");
}

function resetStages() {
  STAGE_IDS.forEach((id) => setStage(id, "pending"));
}

function showProgress(note) {
  $("progress-card").classList.remove("hidden");
  $("progress-note").textContent = note || "";
}

function hideProgress() {
  $("progress-card").classList.add("hidden");
}

/* ------------------------------------------------------------------------- */
/* Generation allowance + script character limit                              */
/* ------------------------------------------------------------------------- */
function renderCharacterCount() {
  const node = $("char-count");
  if (!node) return;
  const max = state.usage ? state.usage.max_script_characters : null;
  node.textContent = ABLimits.counterText(state.characterCount, max);
  node.classList.toggle(
    "over",
    ABLimits.exceedsCharacterLimit(state.characterCount, max)
  );
}

function renderUsage() {
  const line = $("usage-line");
  if (line) {
    if (state.usage) {
      // Authoritative value from GET /usage — never a hardcoded fallback.
      line.textContent = ABLimits.usageMessage(state.usage);
    } else if (state.usageStatus === "loading") {
      line.textContent = "Loading your generation allowance and script limit\u2026";
    } else {
      line.textContent =
        "Script limit unavailable \u2014 your generation allowance could not be loaded.";
    }
  }
  const exhausted = state.usage ? ABLimits.usageState(state.usage).exhausted : false;
  const note = $("quota-note");
  if (note) {
    note.textContent = exhausted ? ABLimits.exhaustedMessage(state.usage) : "";
    note.classList.toggle("hidden", !exhausted);
  }
  renderCharacterCount();
}

async function refreshUsage() {
  try {
    // Own usage only — derived from the session server-side.
    state.usage = await api("/usage");
    state.usageStatus = "ready";
  } catch (error) {
    console.warn("Could not load your generation usage:", error);
    state.usage = null;
    state.usageStatus = "error";
  }
  renderUsage();
  refreshButtons();
}

/* ------------------------------------------------------------------------- */
/* Buttons                                                                    */
/* ------------------------------------------------------------------------- */
function refreshButtons() {
  const ready = Boolean(state.project) && !state.busy;
  const exhausted = state.usage ? ABLimits.usageState(state.usage).exhausted : false;
  const castOn = Boolean(state.casting.enabled && state.project);
  // Multi-voice generation needs the approved cast (the backend enforces the
  // same rule with 409); the single-voice path is untouched when the toggle
  // is off.
  const castReady = !castOn || castApproved();
  $("btn-generate").disabled = !ready || exhausted || !castReady;
  $("btn-regenerate").disabled = !ready || exhausted || !castReady;
  $("btn-choose-file").disabled = state.busy;
  $("btn-replace").disabled = state.busy;
  $("btn-remove").disabled = state.busy;
  $("btn-download").classList.toggle("disabled", !state.audio);
  $("generate-hint").textContent = !state.project
    ? "Upload a script to get started."
    : castOn && state.casting.phase !== "ready"
      ? "Character casting is not finished yet."
      : castOn && !castReady
        ? "Continue the character casting to enable generation."
        : `Ready to generate · ${STYLE_PRESETS[state.style].label} style.`;
  const toggle = $("mv-toggle");
  if (toggle) {
    toggle.checked = state.casting.enabled;
    toggle.disabled = state.busy || state.casting.busy;
  }
}

async function run(action) {
  if (state.busy) {
    console.warn("Ignored click: another action is still running.");
    return null;
  }
  state.busy = true;
  refreshButtons();
  try {
    return await action();
  } catch (error) {
    console.error("Audiobook action failed:", error);
    // A failed stage must not keep spinning as if it were still running.
    document
      .querySelectorAll("#stage-list .active")
      .forEach((item) => item.classList.remove("active"));
    const note = $("progress-note");
    if (note) note.textContent = "Stopped because of an error.";
    setStatus(detailOf(error), "error");
    return null;
  } finally {
    state.busy = false;
    refreshButtons();
  }
}

/* ------------------------------------------------------------------------- */
/* Script upload (auto project creation)                                      */
/* ------------------------------------------------------------------------- */
function bindUpload() {
  const dropZone = $("drop-zone");
  const input = $("file-input");

  $("btn-choose-file").addEventListener("click", (event) => {
    event.stopPropagation();
    input.click();
  });
  $("btn-replace").addEventListener("click", () => input.click());
  $("btn-remove").addEventListener("click", () => removeScript());

  dropZone.addEventListener("click", () => input.click());
  dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  ["dragover", "dragenter"].forEach((name) =>
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      dropZone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((name) =>
    dropZone.addEventListener(name, () => dropZone.classList.remove("drag-over"))
  );
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    handleFiles(event.dataTransfer.files);
  });
  input.addEventListener("change", () => handleFiles(input.files));
}

function handleFiles(fileList) {
  const file = fileList && fileList[0];
  if (!file) return;
  if (!/\.txt$/i.test(file.name)) {
    setStatus("Only .txt files are supported.", "error");
    return;
  }
  if (file.size > MAX_SCRIPT_BYTES) {
    setStatus("That file is larger than 5 MB — please split it up.", "error");
    return;
  }
  run(() => importScript(file));
}

function renderFilePanel(file, stateText, stateClass) {
  $("file-panel").classList.remove("hidden");
  $("drop-zone").classList.add("compact");
  $("file-name").textContent = file.name;
  $("file-size").textContent = formatBytes(file.size);
  const stateNode = $("file-state");
  stateNode.textContent = stateText || "";
  stateNode.className = stateClass || "";
}

function renderSummary() {
  const project = state.project;
  if (!project) return;
  $("script-summary").classList.remove("hidden");
  $("book-title").value = project.title;
  $("language-name").textContent =
    state.languageOverride || project.language || "\u2014";
  $("paragraph-count").textContent = String(project.paragraphCount);
  $("script-preview").textContent = project.preview;
  renderCharacterCount();
}

async function importScript(file) {
  renderFilePanel(file, "Reading…", "");
  showProgress("Reading your script — the project and title are created automatically.");
  resetStages();
  setStage("upload", "active");
  setStatus("Uploading your script…", "info");

  // Live character counter: Unicode code points, exactly like the backend.
  const text = await file.text();
  const count = ABLimits.countCharacters(text);
  state.characterCount = count;
  renderCharacterCount();

  const max = state.usage ? state.usage.max_script_characters : null;
  if (ABLimits.exceedsCharacterLimit(count, max)) {
    // UX-side rejection — the backend rejects the same script authoritatively.
    renderFilePanel(file, "Too long", "is-error");
    setStage("upload", "pending");
    hideProgress();
    setStatus(`Script is too long: ${count} characters (limit: ${max} characters).`, "error");
    return;
  }

  const body = new FormData();
  body.append("file", file, file.name); // multipart field name: "file"
  const data = await api("/projects/from-script", { method: "POST", body });

  state.project = {
    id: data.project_id,
    title: data.title,
    language: data.language,
    languageCode: data.language_code,
    paragraphCount: data.paragraph_count,
    preview: data.script_preview,
    filename: data.filename,
    sizeBytes: data.size_bytes,
    characterCount: data.character_count,
  };
  state.characterCount = data.character_count;
  state.audio = null;
  state.planKey = null;
  // A replaced script is a NEW project: the previous character analysis and
  // voice assignments must never be shown for it (§"invalidate on replace").
  invalidateCasting();
  // The casting section appears as soon as the project exists — visible and
  // UNCHECKED by default, before any analysis runs (this render call is what
  // removes the `hidden` class after an upload).
  renderCasting();
  // A replaced script is NEVER auto-analysed either: the Character Analyser
  // button runs the analysis whenever the user is ready.

  renderFilePanel(file, "Ready", "is-ok");
  renderSummary();
  setStage("upload", "done");
  hideProgress();
  saveState();
  refreshButtons();
  setStatus("Your script is ready — pick a narration style and generate.", "ok");
}

function removeScript() {
  state.project = null;
  state.audio = null;
  state.planKey = null;
  state.characterCount = 0;
  invalidateCasting({ full: true }); // no script → no casting option either
  $("file-input").value = "";
  $("file-panel").classList.add("hidden");
  $("drop-zone").classList.remove("compact");
  $("script-summary").classList.add("hidden");
  $("result-card").classList.add("hidden");
  hideProgress();
  renderCasting();
  renderCharacterCount();
  saveState();
  refreshButtons();
  setStatus("Script removed. Choose a .txt file to start over.", "info");
}

/** Privacy: delete every project of this tester (and its files) server-side. */
async function deleteMyData() {
  const confirmed = window.confirm(
    "Delete all your projects and generated files? This cannot be undone."
  );
  if (!confirmed) return;
  setStatus("Deleting your projects and files…", "info");
  await api("/projects", { method: "DELETE" });
  removeScript();
  setStatus("Your projects and generated files were deleted.", "ok");
}

/* ------------------------------------------------------------------------- */
/* Narration style + language                                                 */
/* ------------------------------------------------------------------------- */
function bindStyle() {
  const cards = Array.from(document.querySelectorAll(".style-card"));
  cards.forEach((card) => {
    card.addEventListener("click", () => selectStyle(card.dataset.style));
    card.addEventListener("keydown", (event) => {
      const keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp"];
      if (!keys.includes(event.key)) return;
      event.preventDefault();
      const index = cards.indexOf(card);
      const step = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1;
      const next = cards[(index + step + cards.length) % cards.length];
      selectStyle(next.dataset.style);
      next.focus();
    });
  });

  $("btn-language-override").addEventListener("click", () => {
    const row = $("language-override-row");
    const opening = row.classList.contains("hidden");
    row.classList.toggle("hidden", !opening);
    $("btn-language-override").setAttribute("aria-expanded", String(opening));
    if (opening) $("language-select").focus();
  });

  $("language-select").addEventListener("change", (event) => {
    state.languageOverride = event.target.value;
    renderSummary();
    saveState();
    setStatus(
      state.languageOverride
        ? `Narration language set to ${state.languageOverride}.`
        : "Narration language: auto-detect.",
      "info"
    );
  });

  $("book-title").addEventListener("input", (event) => {
    if (state.project) {
      state.project.title = event.target.value;
      saveState();
      // Keep the result screen + download name in sync while typing.
      if (state.audio) {
        $("result-title").textContent = state.project.title;
        $("btn-download").setAttribute(
          "download",
          downloadName(state.project.title, state.audio.filename)
        );
      }
    }
  });
}

function selectStyle(style) {
  if (!STYLE_PRESETS[style]) return;
  state.style = style;
  document.querySelectorAll(".style-card").forEach((card) => {
    const selected = card.dataset.style === style;
    card.setAttribute("aria-checked", String(selected));
    card.tabIndex = selected ? 0 : -1;
  });
  refreshButtons();
  saveState();
  if (state.audio) {
    setStatus("Style updated — click Generate Audiobook to apply it.", "info");
  }
}

function buildPreferences() {
  const base = STYLE_PRESETS[state.style].preferences;
  const preferences = JSON.parse(JSON.stringify(base));
  if (state.languageOverride) {
    // The director model receives this as plain direction — no new API calls.
    preferences.custom_direction = `Narrate the audiobook in ${state.languageOverride}.`;
  }
  return preferences;
}

/* ------------------------------------------------------------------------- */
/* Optional multi-character casting ("Use Multiple Character Voices")          */
/* ------------------------------------------------------------------------- */
/* Reuses the existing backend endpoints only — nothing here is invented:
 *
 *   POST /projects/{id}/analyze-characters        story + cast discovery (aux)
 *   POST /projects/{id}/voice-casting             ranked candidates per role
 *   GET  /projects/{id}/voice-casting             authoritative cast state
 *   POST /projects/{id}/voice-casting/select      lock one voice (per speaker)
 *   POST /projects/{id}/voice-casting/audition    short preview line (aux)
 *   POST /projects/{id}/voice-casting/approve     lock the whole cast
 *   POST /projects/{id}/create-multi-voice-plan   utterance plan (1 allowance)
 *   POST /projects/{id}/generate-audio            joins that plan's workflow
 *
 * `generation_mode === "multi_voice"` is the backend's own proof that the
 * generated audio really used the locked character voices.
 */
const AUDITION_LINE =
  "Hello, this is a short preview of my voice for this audiobook.";
/* Mirrors the server's default duplicate window (BETA_DUPLICATE_WINDOW_SECONDS):
 * after a SUCCESSFUL preview the next click waits instead of tripping the
 * backend's duplicate-submission guard. The server stays authoritative — a
 * 409 is still surfaced with a retry action. */
const AUDITION_COOLDOWN_MS = 10 * 1000;
const DUPLICATE_REQUEST_RE = /Duplicate submission|already being processed/i;
const AUDITION_LABELS = {
  idle: "\u25B6 Preview Voice",
  loading: "Generating preview\u2026",
  playing: "\u25A0 Stop preview",
  error: "\u21BB Retry preview",
};
/* Backend `role` values -> human labels (presentation only, never data). */
const ROLE_LABELS = {
  narrator: "Narrator",
  protagonist: "Main character",
  antagonist: "Antagonist",
  supporting_character: "Supporting character",
  minor_character: "Minor character",
  ensemble: "Ensemble",
};

let castingRunId = 0; // bumped when old work must be discarded

function castingKey() {
  return state.project ? `${state.project.id}:${state.characterCount}` : null;
}

function castSpeakers() {
  const analysis = state.casting.analysis;
  if (analysis && analysis.speaker_ids) return analysis.speaker_ids;
  const casting = state.casting.voiceCasting; // degraded: no cached analysis
  if (casting) {
    return ["narrator", ...Object.keys(casting.characters || {})];
  }
  return [];
}

function castProfile(speakerId) {
  const analysis = state.casting.analysis;
  if (!analysis || !analysis.story) return null;
  const story = analysis.story;
  if (speakerId === "narrator") return story.narrator || null;
  return (story.characters || []).find((c) => c.character_id === speakerId) || null;
}

function castEntry(speakerId) {
  const casting = state.casting.voiceCasting;
  if (!casting) return null;
  if (speakerId === "narrator") return casting.narrator || null;
  return (casting.characters || {})[speakerId] || null;
}

function recommendedCandidate(entry) {
  const candidates = (entry && entry.candidates) || [];
  if (!candidates.length) return null;
  return candidates
    .slice()
    .sort((a, b) => (a.rank || 1) - (b.rank || 1))[0];
}

function candidateById(entry, voiceId) {
  const candidates = (entry && entry.candidates) || [];
  return candidates.find((c) => c.voice_id === voiceId) || null;
}

function selectedVoiceId(speakerId) {
  const entry = castEntry(speakerId);
  return entry && entry.selected ? entry.selected.voice_id : null;
}

/** What the UI currently wants for a speaker: draft -> lock -> recommendation. */
function draftVoiceId(speakerId) {
  const entry = castEntry(speakerId);
  const draft = state.casting.drafts[speakerId];
  if (draft && candidateById(entry, draft)) return draft;
  const locked = selectedVoiceId(speakerId);
  if (locked) return locked; // the server-side lock is authoritative
  const recommended = recommendedCandidate(entry);
  return recommended ? recommended.voice_id : null;
}

function hasPendingDrafts() {
  return castSpeakers().some((id) => {
    const draft = state.casting.drafts[id];
    return Boolean(draft) && selectedVoiceId(id) !== draft;
  });
}

function castApproved() {
  if (!state.casting.enabled) return true;
  const casting = state.casting.voiceCasting;
  if (!casting || !casting.approved) return false;
  return !hasPendingDrafts();
}

function voiceLabel(speakerId, voiceId) {
  const entry = castEntry(speakerId);
  const lock =
    entry && entry.selected && entry.selected.voice_id === voiceId
      ? entry.selected
      : null;
  const candidate = candidateById(entry, voiceId);
  return (
    (candidate && candidate.name) ||
    (lock && lock.voice_name) ||
    voiceId ||
    ""
  );
}

function statusText(speakerId) {
  const draft = draftVoiceId(speakerId);
  const locked = selectedVoiceId(speakerId);
  if (!draft) return "No candidate available";
  if (locked && locked === draft) return `Selected: ${voiceLabel(speakerId, draft)}`;
  if (locked) {
    return `Selected: ${voiceLabel(speakerId, locked)} \u00B7 switch to ${voiceLabel(speakerId, draft)}`;
  }
  return `Recommended: ${voiceLabel(speakerId, draft)}`;
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function progressText() {
  const casting = state.casting;
  if (casting.step === "analyse") return "Analysing your script for characters\u2026";
  if (casting.step === "rank") return "Finding the best voices for each speaking role\u2026";
  if (casting.step === "assign") return "Assigning the recommended voices\u2026";
  if (casting.phase === "error") {
    return "Character analysis failed \u2014 your script and selections are unchanged.";
  }
  if (casting.phase === "ready") return "Character analysis complete.";
  return "";
}

function renderCasting() {
  const card = $("casting-card");
  const body = $("casting-body");
  const toggle = $("mv-toggle");
  if (!card || !body || !toggle) return;
  const hasProject = Boolean(state.project);
  card.classList.toggle("hidden", !hasProject);
  // Dynamic step badges: the casting section only numbers as a step while
  // it is visible (a project exists); without it style/generate move up.
  const styleBadge = $("step-badge-style");
  const generateBadge = $("step-badge-generate");
  if (styleBadge) styleBadge.textContent = hasProject ? "3" : "2";
  if (generateBadge) generateBadge.textContent = hasProject ? "4" : "3";
  toggle.checked = state.casting.enabled;
  toggle.disabled = state.busy || state.casting.busy;
  body.classList.toggle("hidden", !state.casting.enabled);
  if (!state.casting.enabled || !state.project) return;
  const progress = $("cast-progress");
  progress.textContent = progressText();
  progress.className = state.casting.phase === "error" ? "hint warn" : "hint";
  renderRoles();
  renderModes();
  renderCastCards();
  renderContinue();
}

function renderRoles() {
  const wrap = $("cast-roles-wrap");
  const note = $("cast-note");
  const analysis = state.casting.analysis;
  const ready = state.casting.phase === "ready";
  wrap.classList.toggle("hidden", !(ready && analysis));
  if (!ready) {
    $("cast-role-count").textContent = "";
    $("cast-roles-body").innerHTML = "";
    note.textContent = "";
    return;
  }
  if (!analysis) {
    // Degraded restore: the server has the cast but this browser has no
    // cached analysis and re-running detection would reset the locks.
    note.textContent =
      "Character descriptions from this browser session are unavailable. " +
      "Re-detecting characters resets the current voice selections.";
    return;
  }
  const speakers = castSpeakers();
  const rows = speakers.map((id) => {
    const profile = castProfile(id) || {};
    const name = id === "narrator" ? "Narrator" : profile.character_name || id;
    const description =
      profile.description || profile.voice_description || "\u2014";
    return (
      `<tr data-role="${escapeHtml(id)}"><td>${escapeHtml(name)}</td>` +
      `<td>${escapeHtml(description)}</td></tr>`
    );
  });
  $("cast-roles-body").innerHTML = rows.join("");
  $("cast-role-count").textContent =
    speakers.length > 1
      ? `Detected characters \u2014 ${speakers.length} roles`
      : "Detected characters \u2014 narrator only";
  const silent = ((analysis.story && analysis.story.characters) || []).filter(
    (c) => !c.has_spoken_dialogue
  );
  if (speakers.length <= 1) {
    note.textContent =
      "No additional speaking characters were detected \u2014 the audiobook can " +
      "continue with narrator-only narration.";
  } else if (silent.length) {
    note.textContent =
      "Mentioned without dialogue: " +
      silent.map((c) => c.character_name).join(", ") +
      " \u2014 not given a speaking voice.";
  } else {
    note.textContent = "";
  }
}

function renderModes() {
  const box = $("cast-modes");
  const show = state.casting.phase === "ready" && castSpeakers().length > 0;
  box.classList.toggle("hidden", !show);
  const auto = $("cast-mode-auto");
  const manual = $("cast-mode-manual");
  auto.checked = state.casting.mode === "auto";
  manual.checked = state.casting.mode === "manual";
  auto.disabled = state.casting.busy;
  manual.disabled = state.casting.busy;
  $("cast-mode-auto-label").classList.toggle(
    "is-selected",
    state.casting.mode === "auto"
  );
  $("cast-mode-manual-label").classList.toggle(
    "is-selected",
    state.casting.mode === "manual"
  );
}

function castCardHtml(speakerId) {
  const profile = castProfile(speakerId);
  const entry = castEntry(speakerId) || { candidates: [], selected: null };
  const isNarrator = speakerId === "narrator";
  const name = isNarrator ? "Narrator" : (profile && profile.character_name) || speakerId;
  const description =
    (profile && (profile.description || profile.voice_description)) || "";
  const recommended = recommendedCandidate(entry);
  const recommendedName = recommended
    ? recommended.name || recommended.voice_id
    : "\u2014";
  const manual =
    state.casting.mode === "manual" && entry.candidates.length > 0;
  const draft = draftVoiceId(speakerId);
  const options = entry.candidates
    .map((c) => {
      const label = `${c.name || c.voice_id}${c.rank === 1 ? " (recommended)" : ""}`;
      const selected = c.voice_id === draft ? " selected" : "";
      return `<option value="${escapeHtml(c.voice_id)}"${selected}>${escapeHtml(
        label
      )}</option>`;
    })
    .join("");
  const badge = isNarrator
    ? "Narrator"
    : ROLE_LABELS[(profile && profile.role) || ""] || "";
  const audition = state.casting.auditions[speakerId] || { status: "idle" };
  const auditionDisabled =
    audition.status === "loading" || state.casting.auditionBusy || state.casting.busy;
  return [
    `<article class="cast-card" data-speaker="${escapeHtml(speakerId)}">`,
    `<div class="cast-card-head"><span class="cast-name">${escapeHtml(name)}</span>`,
    badge ? `<span class="cast-badge">${escapeHtml(badge)}</span>` : "",
    `</div>`,
    description ? `<p class="cast-desc">${escapeHtml(description)}</p>` : "",
    `<p class="cast-recommended"><span>Recommended voice:</span> ${escapeHtml(
      recommendedName
    )}</p>`,
    manual
      ? `<div class="cast-controls"><label class="cast-select-label" for="cast-select-${escapeHtml(
          speakerId
        )}">Voice</label><select id="cast-select-${escapeHtml(
          speakerId
        )}" data-action="voice" data-speaker="${escapeHtml(
          speakerId
        )}">${options}</select><span class="cast-status" id="cast-status-${escapeHtml(
          speakerId
        )}">${escapeHtml(statusText(speakerId))}</span></div>`
      : `<div class="cast-controls"><span class="cast-status" id="cast-status-${escapeHtml(
          speakerId
        )}">${escapeHtml(statusText(speakerId))}</span></div>`,
    `<div class="cast-controls"><button type="button" class="btn btn-secondary cast-audition" id="cast-audition-${escapeHtml(
      speakerId
    )}" data-action="audition" data-speaker="${escapeHtml(
      speakerId
    )}"${auditionDisabled ? " disabled" : ""}>${
      AUDITION_LABELS[audition.status] || AUDITION_LABELS.idle
    }</button></div>`,
    `</article>`,
  ].join("");
}

function renderCastCards() {
  const container = $("cast-cards");
  if (!container) return;
  const casting = state.casting;
  if (casting.phase === "error") {
    // The actual reason is shown so the tester knows what happened — the
    // Retry Character Analyser button below re-runs the analysis.
    const detail = casting.analysisError
      ? `<p class="cast-desc">${escapeHtml(casting.analysisError)}</p>`
      : "";
    container.innerHTML = [
      `<article class="cast-card"><p class="cast-desc">`,
      `Character analysis could not be completed. Your script is unchanged \u2014 press Retry Character Analyser to try again.`,
      `</p>${detail}</article>`,
    ].join("");
    return;
  }
  if (casting.phase !== "ready" || !casting.voiceCasting) {
    container.innerHTML = "";
    return;
  }
  const speakers = castSpeakers();
  if (!speakers.length) {
    container.innerHTML = "";
    return;
  }
  let html = speakers.map(castCardHtml).join("");
  if (!casting.analysis) {
    html += [
      `<article class="cast-card"><p class="cast-desc">`,
      `Re-detect characters to see their names and descriptions again.`,
      `</p><div class="cast-controls">`,
      `<button type="button" class="btn btn-secondary" data-action="redetect">`,
      `Re-detect characters</button></div></article>`,
    ].join("");
  }
  container.innerHTML = html;
  speakers.forEach((id) => {
    const select = document.getElementById(`cast-select-${id}`);
    if (select) select.value = draftVoiceId(id) || "";
  });
}

/* One primary button that always names the NEXT action (never a dead end):
 *   idle/error  → Character Analyser / Retry Character Analyser (run analysis)
 *   busy        → what is currently running (disabled)
 *   ready       → Save & Approve Casting (lock every choice + approve the cast)
 *   approved    → Casting approved (disabled) */
const CASTING_BUSY_LABELS = {
  analyse: "Analysing characters\u2026",
  rank: "Finding the best voices\u2026",
  assign: "Assigning the recommended voices\u2026",
};

function castingReadyToContinue() {
  return (
    state.casting.phase === "ready" &&
    Boolean(state.casting.voiceCasting) &&
    castSpeakers().length > 0
  );
}

function renderContinue() {
  const button = $("btn-casting-continue");
  if (!button) return;
  if (castApproved()) {
    button.textContent = "Casting approved";
    button.disabled = true;
    return;
  }
  if (state.casting.busy) {
    button.textContent =
      CASTING_BUSY_LABELS[state.casting.step] || "Saving the casting\u2026";
    button.disabled = true;
    return;
  }
  if (castingReadyToContinue()) {
    button.textContent = "Save & Approve Casting";
    button.disabled = false;
    return;
  }
  button.textContent =
    state.casting.phase === "error"
      ? "Retry Character Analyser"
      : "Character Analyser";
  button.disabled = false;
}

/** The button click: analyse when the cast is missing, continue when ready. */
function onCastingAction() {
  if (castApproved()) return;
  if (castingReadyToContinue()) {
    continueCasting();
  } else {
    startCasting();
  }
}

function updateCastStatus(speakerId) {
  const node = document.getElementById(`cast-status-${speakerId}`);
  if (node) node.textContent = statusText(speakerId);
}

function renderAuditionButtons() {
  castSpeakers().forEach((id) => {
    const button = document.getElementById(`cast-audition-${id}`);
    if (!button) return;
    const status =
      (state.casting.auditions[id] || { status: "idle" }).status || "idle";
    button.textContent = AUDITION_LABELS[status] || AUDITION_LABELS.idle;
    button.disabled =
      status === "loading" || state.casting.auditionBusy || state.casting.busy;
  });
}

/** Script replaced/removed: drop analysis, assignments and the plan key. */
function invalidateCasting(options = {}) {
  castingRunId += 1; // any in-flight analysis for the old script is discarded
  stopAllPreviews();
  state.casting.analysis = null;
  state.casting.analysisKey = null;
  state.casting.voiceCasting = null;
  state.casting.drafts = {};
  state.casting.phase = "idle";
  state.casting.step = null;
  state.casting.busy = false;
  state.casting.auditionBusy = false;
  state.casting.auditions = {};
  state.casting.lastAuditionAt = 0;
  state.casting.mvPlanKey = null;
  state.casting.analysisError = null;
  state.casting.runId = 0; // in-flight work for the old script is discarded
  if (options.full) {
    state.casting.enabled = false;
    state.casting.mode = "auto";
  }
}

function stopAllPreviews() {
  Object.keys(state.casting.auditions).forEach((id) => {
    const current = state.casting.auditions[id];
    try {
      if (current && current.audio) current.audio.pause();
    } catch (_) {
      /* already stopped */
    }
    try {
      if (current && current.url && URL.revokeObjectURL) {
        URL.revokeObjectURL(current.url);
      }
    } catch (_) {
      /* not stubbed / already revoked */
    }
  });
}

/** Analysis -> candidates -> (automatic) recommendations, once per script. */
async function startCasting() {
  if (!state.project || !state.casting.enabled) return;
  if (state.casting.busy) return; // never analyse twice in parallel
  const key = castingKey();
  const runId = ++castingRunId;
  state.casting.runId = runId;
  const cachedOk = Boolean(
    state.casting.analysis && state.casting.analysisKey === key
  );
  if (cachedOk && state.casting.voiceCasting && state.casting.phase === "ready") {
    return; // already analysed + casted for THIS script
  }
  state.casting.busy = true;
  if (cachedOk) state.casting.phase = "ready";
  renderCasting();
  refreshButtons();
  try {
    if (!cachedOk) {
      state.casting.step = "analyse";
      state.casting.analysis = null;
      state.casting.voiceCasting = null;
      state.casting.drafts = {};
      renderCasting();
      const analysis = await api(`/projects/${state.project.id}/analyze-characters`, {
        method: "POST",
      });
      if (state.casting.runId !== runId || castingKey() !== key) return; // replaced
      state.casting.analysis = analysis;
      state.casting.analysisKey = key;
      state.casting.phase = "ready";
      saveState();
    }
    state.casting.step = "rank";
    renderCasting();
    await buildCasting();
    if (state.casting.runId !== runId || castingKey() !== key) return;
    if (state.casting.mode === "auto") {
      state.casting.step = "assign";
      renderCasting();
      await assignRecommended();
      if (state.casting.runId !== runId || castingKey() !== key) return;
    }
    state.casting.step = null;
    state.casting.analysisError = null;
    setStatus("Character casting is ready — review the voices and continue.", "ok");
  } catch (error) {
    if (state.casting.runId !== runId) return;
    state.casting.step = null;
    state.casting.phase = state.casting.analysis ? "ready" : "error";
    state.casting.analysisError = detailOf(error);
    setStatus(detailOf(error), "error");
    console.error("Character casting failed:", error);
  } finally {
    if (state.casting.runId === runId) {
      state.casting.busy = false;
      renderCasting();
      refreshButtons();
    }
  }
}

async function buildCasting() {
  const response = await api(`/projects/${state.project.id}/voice-casting`, {
    method: "POST",
  });
  state.casting.voiceCasting = response;
  return response;
}

/** Lock every speaker to its rank-1 backend recommendation (mode A). */
async function assignRecommended() {
  const failed = [];
  for (const speakerId of castSpeakers()) {
    const entry = castEntry(speakerId);
    const recommended = recommendedCandidate(entry);
    if (!recommended) continue;
    if (selectedVoiceId(speakerId) === recommended.voice_id) continue;
    try {
      await selectVoice(speakerId, recommended.voice_id, recommended);
    } catch (error) {
      // One failure must not leave the other speakers unassigned.
      console.error(`Could not assign the recommended voice for ${speakerId}:`, error);
      failed.push(speakerId);
    }
  }
  if (failed.length) {
    throw new Error(
      `The recommended voice could not be assigned for: ${failed.join(", ")} — review the candidates below.`
    );
  }
}

async function selectVoice(speakerId, voiceId, candidate) {
  const body = { speaker_id: speakerId, voice_id: voiceId };
  if (candidate && candidate.origin === "shared" && candidate.public_owner_id) {
    body.public_owner_id = candidate.public_owner_id; // required by the backend
  }
  const response = await jsonRequest(
    "POST",
    `/projects/${state.project.id}/voice-casting/select`,
    body
  );
  state.casting.voiceCasting = response;
  return response;
}

/** Continue: lock every pending choice, then approve the whole cast. */
async function continueCasting() {
  if (state.casting.busy || !state.project || !state.casting.voiceCasting) return;
  const key = castingKey();
  const runId = state.casting.runId;
  state.casting.busy = true;
  renderCasting();
  refreshButtons();
  try {
    for (const speakerId of castSpeakers()) {
      const draft = draftVoiceId(speakerId);
      if (!draft) {
        throw new Error(
          "No voice is available for every speaking role — open Character casting to review the candidates."
        );
      }
      if (selectedVoiceId(speakerId) !== draft) {
        await selectVoice(
          speakerId,
          draft,
          candidateById(castEntry(speakerId), draft)
        );
      }
      if (castingKey() !== key || state.casting.runId !== runId) return; // replaced
    }
    const approved = await api(
      `/projects/${state.project.id}/voice-casting/approve`,
      { method: "POST" }
    );
    if (castingKey() !== key || state.casting.runId !== runId) return;
    state.casting.voiceCasting = approved;
    state.casting.drafts = {};
    setStatus(
      "Casting approved — choose a narration style and press Generate Audiobook to create your multi-voice audiobook.",
      "ok"
    );
    const generateButton = $("btn-generate");
    if (generateButton && !generateButton.disabled && generateButton.focus) {
      generateButton.focus();
    }
  } catch (error) {
    setStatus(detailOf(error), "error");
    console.error("Could not approve the casting:", error);
  } finally {
    if (state.casting.runId === runId) {
      state.casting.busy = false;
      renderCasting();
      refreshButtons();
    }
  }
}

/** Called by generate() so a skipped "Continue" can never produce wrong audio. */
async function ensureCastingApproved() {
  const casting = state.casting.voiceCasting;
  if (state.casting.phase !== "ready" || !casting) {
    throw new Error(
      "Character casting is not ready yet — finish the Character casting section first."
    );
  }
  for (const speakerId of castSpeakers()) {
    const draft = draftVoiceId(speakerId);
    if (!draft) {
      throw new Error(
        `No voice is available for the speaking role '${speakerId}' — choose one in Character casting.`
      );
    }
    if (selectedVoiceId(speakerId) !== draft) {
      await selectVoice(
        speakerId,
        draft,
        candidateById(castEntry(speakerId), draft)
      );
    }
  }
  if (!state.casting.voiceCasting.approved) {
    const approved = await api(
      `/projects/${state.project.id}/voice-casting/approve`,
      { method: "POST" }
    );
    state.casting.voiceCasting = approved;
    renderCasting();
    refreshButtons();
  }
}

function mvPlanKey() {
  return `${state.project.id}:${state.characterCount}:${castSpeakers().join(",")}`;
}

/* ------------------------------------------------------------------ */
/* Voice previews (audition endpoint — never charged an allowance)      */
/* ------------------------------------------------------------------ */
async function fetchAudition(speakerId, voiceId) {
  const response = await fetch(
    `/projects/${state.project.id}/voice-casting/audition`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        speaker_id: speakerId,
        voice_id: voiceId,
        text: AUDITION_LINE,
      }),
    }
  );
  if (!response.ok) {
    let payload = null;
    try {
      payload = await response.json();
    } catch (_) {
      /* non-JSON error body */
    }
    const error = new Error(
      detailOf(payload || { detail: `HTTP ${response.status}` })
    );
    error.status = response.status;
    if (response.status === 401) window.location.href = "/auth/login";
    throw error;
  }
  return response.blob();
}

async function previewVoice(speakerId) {
  if (!state.project || state.casting.busy) return;
  const auditions = state.casting.auditions;
  const current = auditions[speakerId] || { status: "idle" };
  if (state.casting.auditionBusy || current.status === "loading") {
    setStatus("A voice preview is already being generated — please wait.", "info");
    return;
  }
  if (current.status === "playing") {
    stopAudition(speakerId);
    return;
  }
  const voiceId =
    draftVoiceId(speakerId) || selectedVoiceId(speakerId) || null;
  if (!voiceId) {
    setStatus("No voice is available to preview for this role yet.", "error");
    return;
  }
  const now = Date.now();
  if (
    current.status !== "error" &&
    state.casting.lastAuditionAt &&
    now - state.casting.lastAuditionAt < AUDITION_COOLDOWN_MS
  ) {
    // A successful preview just ran — the backend would answer409 here.
    setStatus("Please wait a moment before previewing another voice.", "info");
    return;
  }
  const key = castingKey();
  state.casting.auditionBusy = true;
  auditions[speakerId] = { status: "loading" };
  renderAuditionButtons();
  try {
    const blob = await fetchAudition(speakerId, voiceId);
    if (castingKey() !== key) return; // script replaced mid-preview
    const objectUrl = URL.createObjectURL(blob);
    const audio = new Audio(objectUrl);
    audio.onended = () => {
      if (state.casting.auditions[speakerId] && state.casting.auditions[speakerId].url === objectUrl) {
        stopAudition(speakerId, { silent: true });
      }
    };
    auditions[speakerId] = { status: "playing", url: objectUrl, audio };
    renderAuditionButtons();
    const played = audio.play();
    if (played && typeof played.catch === "function") {
      played.catch(() => {
        auditions[speakerId] = {
          status: "error",
          error: "The preview could not start playing.",
        };
        renderAuditionButtons();
      });
    }
    state.casting.lastAuditionAt = Date.now();
    setStatus("Playing the voice preview.", "ok");
  } catch (error) {
    auditions[speakerId] = { status: "error", error: detailOf(error) };
    setStatus(detailOf(error), "error");
    console.error("Voice preview failed:", error);
  } finally {
    state.casting.auditionBusy = false;
    renderAuditionButtons();
  }
}

function stopAudition(speakerId, options = {}) {
  const current = state.casting.auditions[speakerId];
  if (!current) return;
  try {
    if (current.audio) current.audio.pause();
  } catch (_) {
    /* already stopped */
  }
  try {
    if (current.url && URL.revokeObjectURL) URL.revokeObjectURL(current.url);
  } catch (_) {
    /* nothing to revoke */
  }
  state.casting.auditions[speakerId] = { status: "idle" };
  renderAuditionButtons();
  if (!options.silent) setStatus("Voice preview stopped.", "info");
}

/* ------------------------------------------------------------------ */
/* Toggle / mode / restore                                              */
/* ------------------------------------------------------------------ */
function onToggleCasting(event) {
  const on = Boolean(event.target.checked);
  state.casting.enabled = on;
  saveState();
  renderCasting();
  refreshButtons();
  if (on) {
    // Enabling NEVER starts the analysis by itself — the Character Analyser
    // button below is the only trigger (analyses run when the user clicks it).
    setStatus(
      "Multiple character voices are on \u2014 press Character Analyser to detect the cast.",
      "info"
    );
  } else {
    setStatus(
      "Multiple character voices are off — the single-voice workflow is unchanged.",
      "info"
    );
  }
}

async function setCastMode(mode) {
  if (mode !== "auto" && mode !== "manual") return;
  if (state.casting.mode === mode) {
    renderModes();
    return;
  }
  state.casting.mode = mode;
  state.casting.drafts = {};
  saveState();
  renderCasting();
  refreshButtons();
  if (mode === "auto" && state.casting.voiceCasting && !state.casting.busy) {
    // Re-apply the backend recommendations (mode A never keeps stale picks).
    const runId = state.casting.runId;
    state.casting.busy = true;
    state.casting.step = "assign";
    renderCasting();
    refreshButtons();
    try {
      await assignRecommended();
      setStatus("AI recommendations assigned — review them and continue.", "ok");
    } catch (error) {
      setStatus(detailOf(error), "error");
      console.error("Could not apply the recommended voices:", error);
    } finally {
      if (state.casting.runId === runId) {
        state.casting.busy = false;
        state.casting.step = null;
        renderCasting();
        refreshButtons();
      }
    }
  }
}

function redetectCasting() {
  if (state.casting.busy) return;
  const confirmed = window.confirm(
    "Re-detect the characters? This resets the current voice selections."
  );
  if (!confirmed) return;
  state.casting.analysis = null;
  state.casting.analysisKey = null;
  state.casting.voiceCasting = null;
  state.casting.drafts = {};
  state.casting.phase = "idle";
  startCasting();
}

/* ------------------------------------------------------------------ */
/* Server-side plan state (durable recovery — localStorage is a cache)  */
/* ------------------------------------------------------------------ */
/** The server's multi-voice plan status, or null when it is unknown. */
async function fetchPlanStatus() {
  if (!state.project) return null;
  try {
    return await api(`/projects/${state.project.id}/multi-voice-plan/status`);
  } catch (error) {
    console.warn("Could not read the multi-voice plan status:", error);
    return null;
  }
}

/**
 * Sync the local "plan already created" cache with the SERVER.
 *
 * The server alone decides plan existence/validity: the key is seeded only
 * when a valid unfinished plan exists (`ready_for_generate`) and cleared
 * when the server knows better. When the status is simply unknown the local
 * value is trusted (never blocks a legitimate generation).
 */
async function syncPlanKey() {
  const status = await fetchPlanStatus();
  if (status && status.ready_for_generate) {
    state.casting.mvPlanKey = mvPlanKey();
  } else if (status) {
    state.casting.mvPlanKey = null;
  }
  saveState();
}

/** Restore the casting panel after a refresh (server state wins). */
async function restoreCasting(saved) {
  const stored = saved && saved.casting;
  if (stored) {
    state.casting.enabled = Boolean(stored.enabled);
    state.casting.mode = stored.mode === "manual" ? "manual" : "auto";
    state.casting.mvPlanKey = stored.mvPlanKey || null;
    state.casting.analysis = stored.analysis || null;
    state.casting.analysisKey = stored.analysisKey || null;
  }
  if (!state.project) {
    renderCasting();
    return;
  }
  const key = castingKey();
  if (state.casting.analysisKey !== key) {
    state.casting.analysis = null;
    state.casting.analysisKey = null;
  }
  // `probing` = localStorage was lost entirely, so the option's state is
  // unknown and the SERVER's cast state decides (a stored cast means the
  // multi-character option was on — its locks and plan must not be dropped).
  const probing = !saved;
  if (!state.casting.enabled && !probing) {
    // Explicitly off: the single-voice workflow is untouched — no casting
    // request of any kind until the user switches the option on.
    state.casting.mvPlanKey = null;
    renderCasting();
    refreshButtons();
    return;
  }
  let serverCasting = null;
  try {
    serverCasting = await api(`/projects/${state.project.id}/voice-casting`);
  } catch (error) {
    serverCasting = null; // 404: no cast stored for this project (or a real error)
  }
  if (state.casting.busy) return; // a newer action took over meanwhile
  if (!serverCasting) {
    if (probing) {
      // Nothing on the server either: stay with the default, unchecked
      // option — nothing runs before the user opts in.
      state.casting.enabled = false;
      state.casting.mvPlanKey = null;
      renderCasting();
      refreshButtons();
      return;
    }
    // Enabled but no stored cast: detection is NEVER re-run automatically —
    // the explicit "Detect characters" card takes over.
    state.casting.analysis = null;
    state.casting.analysisKey = null;
    state.casting.voiceCasting = null;
    state.casting.phase = "idle";
    await syncPlanKey();
    renderCasting();
    refreshButtons();
    return;
  }
  state.casting.enabled = true; // a stored cast means the option was on
  if (!state.casting.analysis && serverCasting.story) {
    // Descriptions come from the SERVER's stored analysis — restoring never
    // re-runs detection and never touches the approved locks.
    state.casting.analysis = {
      project_id: state.project.id,
      story: serverCasting.story,
      speaker_ids: ["narrator", ...Object.keys(serverCasting.characters || {})],
    };
    state.casting.analysisKey = key;
  }
  state.casting.voiceCasting = serverCasting;
  state.casting.phase = "ready";
  await syncPlanKey();
  renderCasting();
  refreshButtons();
}

function bindCasting() {
  $("mv-toggle").addEventListener("change", onToggleCasting);
  $("cast-mode-auto").addEventListener("change", () => setCastMode("auto"));
  $("cast-mode-manual").addEventListener("change", () => setCastMode("manual"));
  $("btn-casting-continue").addEventListener("click", onCastingAction);

  const cards = $("cast-cards");
  cards.addEventListener("click", (event) => {
    const target = event.target;
    if (!target || !target.dataset) return;
    const action = target.dataset.action;
    if (action === "audition") previewVoice(target.dataset.speaker);
    else if (action === "redetect") redetectCasting();
  });
  cards.addEventListener("change", (event) => {
    const target = event.target;
    if (!target || !target.dataset || target.dataset.action !== "voice") return;
    const speakerId = target.dataset.speaker;
    state.casting.drafts[speakerId] = target.value;
    updateCastStatus(speakerId);
    renderContinue();
    refreshButtons();
  });
}

/* ------------------------------------------------------------------------- */
/* Generation pipeline                                                        */
/* ------------------------------------------------------------------------- */
async function generate() {
  if (!state.project) throw new Error("Upload a script first.");
  const multiVoice = state.casting.enabled;

  try {
    showProgress("Real stages only — no invented percentages.");
    resetStages();
    setStage("upload", "done"); // already uploaded when the summary appeared
    $("result-card").classList.add("hidden");

    // 1) Save the selected style as preferences (always before planning —
    //    the plan hash and the audio cache both key off them).
    setStage("understand", "active");
    setStatus("Saving your narration style…", "info");
    const preferences = buildPreferences();
    await jsonRequest(
      "PUT",
      `/projects/${state.project.id}/narration-preferences`,
      preferences
    );

    if (multiVoice) {
      // Multi-voice workflow: the approved cast plus the utterance plan
      // REPLACE the single-voice narration plan, so `analyze-narration` is
      // deliberately skipped — calling it would open a SECOND workflow and
      // consume a second allowance. `create-multi-voice-plan` is the one
      // planning call; `generate-audio` then joins the same workflow.
      setStatus("Checking your character casting…", "info");
      await ensureCastingApproved();
      setStage("understand", "done");
      setStage("prepare", "done");
      setStage("generate", "active");

      const planKey = mvPlanKey();
      let needPlan = planKey !== state.casting.mvPlanKey;
      if (!needPlan) {
        // The local key says the plan exists — the SERVER decides that
        // (localStorage is only a cache). When the status is unknown the
        // local key is trusted; the backend join still prevents any double
        // charge if the plan turned out to be missing.
        const status = await fetchPlanStatus();
        if (status && !status.ready_for_generate && !status.valid) {
          needPlan = true;
        }
      }
      if (needPlan) {
        setStatus("Splitting the script into character voices…", "info");
        try {
          await api(`/projects/${state.project.id}/create-multi-voice-plan`, {
            method: "POST",
          });
          state.casting.mvPlanKey = planKey;
          saveState();
        } catch (error) {
          if (error.status === 409 && DUPLICATE_REQUEST_RE.test(String(error.message))) {
            // The duplicate guard says a plan for this exact script + cast
            // was just created — reuse it instead of paying for another.
            state.casting.mvPlanKey = planKey;
            saveState();
          } else {
            throw error;
          }
        }
      }

      setStatus(
        "Generating voices and assembling your audiobook — this can take a few minutes…",
        "info"
      );
      let audioData;
      try {
        audioData = await api(`/projects/${state.project.id}/generate-audio`, {
          method: "POST",
        });
      } catch (error) {
        const message = String(error.message || "");
        if (
          error.status === 409 &&
          (message.includes("no longer matches") || message.includes("No multi-voice plan"))
        ) {
          state.casting.mvPlanKey = null; // the next attempt rebuilds the plan
          saveState();
          throw new Error(
            `${message} Your script and casting are unchanged — press Generate Audiobook again to rebuild the plan.`
          );
        }
        throw error;
      }
      if (audioData.generation_mode !== "multi_voice") {
        // Honest failure: the backend did NOT use the character voices.
        state.casting.mvPlanKey = null;
        saveState();
        throw new Error(
          "The audiobook came back as a single-voice file instead of the multi-voice " +
            "character plan. Your script and casting are unchanged — press Generate " +
            "Audiobook again to retry."
        );
      }
      setStage("generate", "done");
      state.audio = { url: audioData.audio_url, filename: audioData.audio_filename };
      setStage("done", "done");
      saveState();
      renderResult();
      setStatus("Your audiobook is ready — press play or download it.", "ok");
      return;
    }

    // 2) Narration plan (MiMo) — skipped when the existing plan already
    //    matches the current preferences (segments stay cached).
    const planKey = JSON.stringify(preferences);
    if (planKey !== state.planKey || !(await planExists())) {
      setStatus("Understanding your story — this can take a minute…", "info");
      const plan = await jsonRequest(
        "POST",
        `/projects/${state.project.id}/analyze-narration`,
        { preferences }
      );
      state.planKey = planKey;
      // The plan's language is the backend's own detection — trust it.
      if (!state.languageOverride && plan.language) {
        state.project.language = plan.language;
        renderSummary();
      }
    }
    setStage("understand", "done");

    // 3) Plan validated and saved — narration is prepared.
    setStage("prepare", "done");

    // 4) Voices + assembly (one synchronous backend job).
    setStage("generate", "active");
    setStatus(
      "Generating voices and assembling your audiobook — this can take a few minutes…",
      "info"
    );
    const audioData = await api(`/projects/${state.project.id}/generate-audio`, {
      method: "POST",
    });
    setStage("generate", "done");

    state.audio = { url: audioData.audio_url, filename: audioData.audio_filename };
    setStage("done", "done");
    saveState();
    renderResult();
    setStatus("Your audiobook is ready — press play or download it.", "ok");
  } finally {
    // The allowance changed (or a failed attempt was charged) — reflect it.
    await refreshUsage();
  }
}

async function planExists() {
  try {
    await api(`/projects/${state.project.id}/narration-plan`);
    return true;
  } catch (error) {
    return false;
  }
}

/* ------------------------------------------------------------------------- */
/* Result screen                                                              */
/* ------------------------------------------------------------------------- */
function renderResult() {
  const project = state.project;
  const audio = state.audio;
  if (!project || !audio) return;

  $("result-card").classList.remove("hidden");
  $("result-title").textContent = project.title;
  $("result-language").textContent = state.languageOverride || project.language;

  const player = $("audio-player");
  player.onloadedmetadata = () => {
    $("result-duration").textContent = formatDuration(player.duration);
  };
  player.onerror = () => {
    $("result-duration").textContent = "";
    setStatus("The audio file could not be loaded — try regenerating.", "error");
  };
  player.src = audio.url;

  const download = $("btn-download");
  download.href = audio.url;
  download.setAttribute("download", downloadName(project.title, audio.filename));

  $("result-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ------------------------------------------------------------------------- */
/* Restore after refresh                                                      */
/* ------------------------------------------------------------------------- */
/** Paragraphs exactly like the backend `split_paragraphs` (display only). */
function splitParagraphs(text) {
  const blocks = String(text == null ? "" : text)
    .split(/\n\s*\n+/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (blocks.length <= 1) {
    const lines = String(text == null ? "" : text)
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    if (lines.length > 1) return lines;
  }
  return blocks;
}

/**
 * Recover this tester's project from the SERVER when localStorage is gone.
 *
 * The browser never invents project ids: it asks the backend for its OWN
 * projects (filtered server-side by the session's tester) and resumes the
 * most recent one that has a script. Returns the upload-flow shape or null.
 */
async function fetchLatestProject() {
  try {
    const data = await api("/projects");
    const list = Array.isArray(data) ? data : (data && data.projects) || [];
    const withScript = list.filter((p) => p && p.project_id && p.has_script);
    if (!withScript.length) return null;
    withScript.sort((a, b) =>
      String(b.created_at || "").localeCompare(String(a.created_at || ""))
    );
    const project = withScript[0];
    const script = project.script || {};
    return {
      id: project.project_id,
      title: project.name,
      language: "",
      languageCode: "",
      paragraphCount: 0,
      preview: "",
      filename: script.filename || "script.txt",
      sizeBytes: script.size_bytes || 0,
      characterCount: script.character_count || 0,
      fromServer: true,
    };
  } catch (error) {
    console.warn("Could not list your projects:", error);
    return null;
  }
}

/** Fill summary-only fields from the stored script (existing media route). */
async function hydrateScriptSummary(project) {
  try {
    const response = await fetch(`/media/projects/${project.id}/script.txt`, {
      headers: { Accept: "text/plain" },
    });
    if (!response.ok) return;
    const text = await response.text();
    project.characterCount = ABLimits.countCharacters(text);
    project.paragraphCount = splitParagraphs(text).length;
    project.preview = text.slice(0, 400);
    state.characterCount = project.characterCount;
    renderSummary();
    renderCharacterCount();
    saveState();
  } catch (error) {
    console.warn("Could not reload the stored script:", error);
  }
}

async function restore() {
  const saved = loadState();
  let project = saved && saved.project ? saved.project : null;
  if (!project) {
    // localStorage is gone: the PROJECT is recovered from the server —
    // recovery must never depend on browser storage (localStorage is only a
    // performance cache).
    project = await fetchLatestProject();
    if (!project) return;
  }

  state.style = saved && STYLE_PRESETS[saved.style] ? saved.style : "natural";
  state.languageOverride = (saved && saved.languageOverride) || "";
  state.project = project;
  state.planKey = (saved && saved.planKey) || null;
  state.audio = (saved && saved.audio) || null;
  state.characterCount = project.characterCount || 0;
  $("language-select").value = state.languageOverride;
  selectStyle(state.style);

  try {
    await api(`/projects/${state.project.id}`);
  } catch (error) {
    if (error && error.status === 404) {
      console.warn("Stored project is gone — starting fresh.");
      removeScript();
      return;
    }
    throw error;
  }

  renderFilePanel(
    { name: state.project.filename || "script.txt", size: state.project.sizeBytes || 0 },
    "Ready",
    "is-ok"
  );
  renderSummary();

  if (state.project.fromServer) {
    await hydrateScriptSummary(state.project);
  }

  // Casting state: the cast itself is server-authoritative (GET voice-casting);
  // only the analysis display comes from the local cache.
  try {
    await restoreCasting(saved);
  } catch (error) {
    console.warn("Could not restore the character casting:", error);
    renderCasting();
  }

  if (!state.audio && state.project.fromServer) {
    // localStorage cleared: a finished audiobook (if any) is still on the
    // server — recover it exactly like a normal restore would show it.
    try {
      const audioData = await api(`/projects/${state.project.id}/audio`);
      state.audio = { url: audioData.audio_url, filename: audioData.audio_filename };
    } catch (_) {
      state.audio = null; // no finished audiobook on the server
    }
  }

  if (state.audio) {
    try {
      const audioData = await api(`/projects/${state.project.id}/audio`);
      state.audio = { url: audioData.audio_url, filename: audioData.audio_filename };
      renderResult();
      setStatus("Welcome back — your audiobook is ready.", "ok");
    } catch (error) {
      state.audio = null;
      saveState();
    }
  } else if (state.planKey) {
    setStatus("Your script is ready — press Generate Audiobook to continue.", "info");
  } else {
    setStatus("Your script is ready — pick a narration style and generate.", "info");
  }
  saveState();
}

/* ------------------------------------------------------------------------- */
/* Boot                                                                       */
/* ------------------------------------------------------------------------- */
function bind() {
  bindUpload();
  bindStyle();
  bindCasting();
  $("btn-generate").addEventListener("click", () => run(generate));
  $("btn-regenerate").addEventListener("click", () => run(generate));
  const deleteButton = $("btn-delete-data");
  if (deleteButton) {
    deleteButton.addEventListener("click", () => run(deleteMyData));
  }
}

async function boot() {
  try {
    bind();
  } catch (error) {
    console.error("Could not bind the UI controls:", error);
  }

  try {
    await refreshUsage();
  } catch (error) {
    console.warn("Could not load your generation allowance:", error);
  }

  // Restore FIRST — anything that calls saveState() before restore() would
  // overwrite the saved session with the empty state.
  try {
    await restore();
  } catch (error) {
    console.error("Could not restore your previous session:", error);
    setStatus("Could not restore your previous session — please upload a script.", "error");
  }

  try {
    selectStyle(state.style);
    refreshButtons();
  } catch (error) {
    console.error("Could not initialise the page:", error);
  }

  renderCasting(); // keep the casting section + step badges in sync
  refreshButtons();
  if (!state.project) {
    setStatus("Upload your script to get started.", "info");
  }
}

document.addEventListener("DOMContentLoaded", boot);
