/* Multi-character voice casting frontend — run by pytest (tests/test_casting_ui.py).
 *
 * Drives the REAL app.js in node against a minimal fake DOM and mocked
 * FastAPI responses, then asserts the casting contract of the spec:
 *   1  single-voice mode behaves exactly as before (toggle off);
 *   2  "Use Multiple Character Voices" is disabled by default;
 *   3  enabling it does NOT analyse — the Character Analyser button calls
 *      POST /projects/{id}/analyze-characters (exactly once);
 *   4  detection results are displayed from the real analysis payload;
 *   5  narrator stays separate, non-speaking people are not speaking roles,
 *      narrator-only scripts get their own message;
 *   6  automatic casting locks the backend's rank-1 recommendations;
 *   7  manual selection updates exactly the chosen character;
 *   8  preview calls POST /projects/{id}/voice-casting/audition;
 *   9  the returned preview audio plays (and can be stopped);
 *   10 replacing the script invalidates analysis + assignments;
 *   11 duplicate analysis / audition submissions are prevented;
 *   12 analysis and audition still work at zero allowance;
 *   13 multi-voice generation skips analyze-narration, creates the plan once
 *      and verifies generation_mode === "multi_voice";
 *   14 existing projects and previously generated audio stay accessible;
 *   15 the script maximum comes from GET /usage (never hardcoded);
 *   16 script upload reveals the visible, unchecked casting section;
 *   17 reload restores the server plan without re-analysis or a second plan;
 *   18 localStorage loss recovers project, cast and plan from the server;
 *   19 restore without a stored cast analyses only via the button;
 *   20 switching back to "Let AI choose" re-selects the AI recommendation.
 *
 * No network, no browser, no providers — mocked FastAPI responses only.
 */
"use strict";

const assert = require("assert");
const path = require("path");

const APP_PATH = path.join(__dirname, "..", "..", "app", "static", "app.js");
const LIMITS_PATH = path.join(__dirname, "..", "..", "app", "static", "limits.js");

/* --------------------------------------------------------------------- */
/* Fixtures (shapes copied from the real FastAPI response models)          */
/* --------------------------------------------------------------------- */
const USAGE_OK = {
  generation_limit: 2,
  generations_used: 0,
  generations_remaining: 2,
  max_script_characters: 3000,
};
const USAGE_EXHAUSTED = {
  generation_limit: 2,
  generations_used: 2,
  generations_remaining: 0,
  max_script_characters: 3000,
};

const IMPORT_P1 = {
  project_id: "p1",
  title: "Room 207",
  language: "Hindi",
  language_code: "hi",
  paragraph_count: 3,
  script_preview: "The lane was empty…",
  filename: "script.txt",
  size_bytes: 1234,
  character_count: 757, // the spec's screenshot value, now capped by /usage
};
const IMPORT_P2 = {
  project_id: "p2",
  title: "Second Script",
  language: "Hindi",
  language_code: "hi",
  paragraph_count: 1,
  script_preview: "One quiet paragraph.",
  filename: "second.txt",
  size_bytes: 99,
  character_count: 400,
};

const AUDIO_URL = "/media/projects/p1/audio/audiobook.mp3";

function narratorProfile() {
  return {
    character_id: "narrator",
    character_name: "Narrator",
    role: "narrator",
    description: "Third-person storytelling voice",
    relationships: [],
    language: "hi",
    gender: null,
    age: null,
    accent: null,
    uncertain_attributes: [],
    voice_description: "Deep, calm, measured Hindi storytelling voice",
    speaking_style: "Natural, restrained audiobook narration",
    has_spoken_dialogue: true,
    voice_id: null,
    voice_locked: false,
  };
}

function characterProfile(overrides) {
  return Object.assign(
    {
      character_id: "suman",
      character_name: "सुमन",
      role: "protagonist",
      description: "Young female protagonist who warns the visitor.",
      relationships: [],
      language: "hi",
      gender: null,
      age: null,
      accent: null,
      uncertain_attributes: [],
      voice_description: "Soft, warm, grounded voice",
      speaking_style: "Natural, restrained",
      has_spoken_dialogue: true,
      voice_id: null,
      voice_locked: false,
    },
    overrides || {}
  );
}

const ANALYSIS_P1 = {
  project_id: "p1",
  story: {
    overall_tone: "suspenseful",
    genre: "horror",
    language: "hi",
    locale: "hi-IN",
    narrator: narratorProfile(),
    characters: [
      characterProfile({}),
      // Named in the story but NEVER speaks -> no speaking voice required.
      characterProfile({
        character_id: "rahul",
        character_name: "राहुल",
        role: "minor_character",
        description: "Knocks at the door; never speaks.",
        voice_description: "",
        speaking_style: "",
        has_spoken_dialogue: false,
      }),
    ],
  },
  speaker_ids: ["narrator", "suman"],
  saved: true,
  saved_at: null,
};

const ANALYSIS_P2 = {
  project_id: "p2",
  story: {
    overall_tone: "calm",
    genre: "",
    language: "hi",
    locale: null,
    narrator: narratorProfile(),
    characters: [],
  },
  speaker_ids: ["narrator"],
  saved: true,
  saved_at: null,
};

const NARRATOR_CANDIDATES = [
  { voice_id: "voice-narrator-01", name: "Narrator Deep", origin: "account", rank: 1 },
  { voice_id: "voice-hi-3", name: "Warm Storyteller", origin: "account", rank: 2 },
];
const SUMAN_CANDIDATES = [
  { voice_id: "voice-suman-02", name: "Warm Soft", origin: "account", rank: 1 },
  { voice_id: "voice-hi-3", name: "Warm Storyteller", origin: "account", rank: 2 },
];

function makeCasting(projectId) {
  return {
    project_id: projectId,
    approved: false,
    approved_at: null,
    narrator: {
      candidates: NARRATOR_CANDIDATES.map((c) => Object.assign({}, c)),
      selected: null,
      status: "pending",
      insufficient_candidates: false,
    },
    characters: {
      suman: {
        candidates: SUMAN_CANDIDATES.map((c) => Object.assign({}, c)),
        selected: null,
        status: "pending",
        insufficient_candidates: false,
      },
    },
    missing_speakers: ["narrator", "suman"],
    saved_at: null,
  };
}

function entryOf(casting, speakerId) {
  return speakerId === "narrator" ? casting.narrator : casting.characters[speakerId];
}

function refreshMissing(casting) {
  const ids = ["narrator"].concat(Object.keys(casting.characters));
  casting.missing_speakers = ids.filter((id) => !entryOf(casting, id).selected);
}

function selectInCasting(casting, body) {
  const entry = entryOf(casting, body.speaker_id);
  const candidate = (entry.candidates || []).find((c) => c.voice_id === body.voice_id);
  entry.selected = {
    character_id: body.speaker_id,
    voice_id: body.voice_id,
    original_voice_id: null,
    voice_name: candidate ? candidate.name : body.voice_id,
    provider: "elevenlabs",
    language: "hi",
    accent: "indian",
    voice_metadata: {},
    model_id: "eleven_v4",
    supported_voice_settings: [],
    preview_url: null,
    origin: candidate ? candidate.origin : "account",
    selected_at: "2026-11-07T00:00:00+00:00",
    locked: true,
  };
  entry.status = "selected";
  refreshMissing(casting);
  casting.approved = false; // any change invalidates approval (backend rule)
  return { status: 200, body: casting };
}

function approveCasting(casting) {
  refreshMissing(casting);
  if (casting.missing_speakers.length) {
    return {
      status: 409,
      body: {
        detail: `Select and approve voices before generating audio. Still missing a voice for: ${casting.missing_speakers.join(", ")}`,
      },
    };
  }
  casting.approved = true;
  casting.approved_at = "2026-11-07T00:00:00+00:00";
  ["narrator"].concat(Object.keys(casting.characters)).forEach((id) => {
    entryOf(casting, id).status = "approved";
  });
  return { status: 200, body: casting };
}

/* A fully approved cast WITH the server-side `story` restore field. */
function approvedCasting() {
  const casting = makeCasting("p1");
  casting.story = ANALYSIS_P1.story;
  casting.approved = true;
  casting.approved_at = "2026-11-07T00:00:00+00:00";
  casting.missing_speakers = [];
  ["narrator", "suman"].forEach((id) => {
    const entry = entryOf(casting, id);
    entry.status = "approved";
    entry.selected = {
      character_id: id,
      voice_id: id === "narrator" ? "voice-narrator-01" : "voice-suman-02",
      original_voice_id: null,
      voice_name: id === "narrator" ? "Narrator Deep" : "Warm Soft",
      provider: "elevenlabs",
      selected_at: "2026-11-07T00:00:00+00:00",
      locked: true,
    };
  });
  return casting;
}

/* GET /multi-voice-plan/status payloads (the server is the authority). */
function planReadyStatus() {
  return {
    project_id: "p1",
    exists: true,
    status: "ready",
    valid: true,
    ready_for_generate: true,
    plan_id: "plan-1",
    plan_fingerprint: "v1:abc",
    generation_id: "gen-1",
    created_at: "2026-11-07T00:00:00+00:00",
    updated_at: "2026-11-07T00:00:00+00:00",
  };
}

function planEmptyStatus() {
  return {
    project_id: "p1",
    exists: false,
    status: "none",
    valid: false,
    ready_for_generate: false,
    plan_id: null,
    plan_fingerprint: null,
    generation_id: null,
    created_at: null,
    updated_at: null,
  };
}

/* --------------------------------------------------------------------- */
/* Minimal fake DOM + fetch                                               */
/* --------------------------------------------------------------------- */
class FakeElement {
  constructor(id) {
    this.id = id;
    this.textContent = "";
    this.innerHTML = "";
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this.href = "";
    this.src = "";
    this.className = "";
    this.files = [];
    this.dataset = {};
    this.attributes = {};
    this.listeners = {};
    const classes = new Set();
    this.classList = {
      add: (name) => void classes.add(name),
      remove: (name) => void classes.delete(name),
      contains: (name) => classes.has(name),
      toggle: (name, force) => {
        const on = force === undefined ? !classes.has(name) : Boolean(force);
        if (on) classes.add(name);
        else classes.delete(name);
        return on;
      },
    };
  }
  addEventListener(name, fn) {
    (this.listeners[name] = this.listeners[name] || []).push(fn);
  }
  setAttribute(name, value) {
    this.attributes[name] = value;
  }
  getAttribute(name) {
    return this.attributes[name];
  }
  click() {}
  focus() {}
  scrollIntoView() {}
}

function fire(element, type, event) {
  (element.listeners[type] || []).forEach((fn) => fn(event || { target: element }));
}

function makeResponse(spec) {
  const status = (spec && spec.status) || 200;
  return {
    ok: status < 400,
    status,
    json: async () => (spec && spec.body !== undefined ? spec.body : null),
    text: async () =>
      spec && spec.text !== undefined
        ? spec.text
        : JSON.stringify(spec && spec.body !== undefined ? spec.body : null),
    blob: async () => (spec && spec.blob) || { size: 8, type: "audio/mpeg" },
  };
}

function flush() {
  let chain = Promise.resolve();
  for (let i = 0; i < 25; i += 1) {
    chain = chain.then(() => new Promise((r) => setImmediate(r)));
  }
  return chain;
}

function deferred() {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

class FakeAudio {
  constructor(url) {
    this.url = url;
    this.paused = true;
    this.playCalls = 0;
    this.pauseCalls = 0;
    FakeAudio.instances.push(this);
  }
  play() {
    this.paused = false;
    this.playCalls += 1;
    return Promise.resolve();
  }
  pause() {
    this.paused = true;
    this.pauseCalls += 1;
  }
}
FakeAudio.instances = [];

async function boot(scenario) {
  const elements = new Map();
  const docListeners = {};
  global.document = {
    getElementById: (id) => {
      if (!elements.has(id)) elements.set(id, new FakeElement(id));
      return elements.get(id);
    },
    querySelector: () => null,
    querySelectorAll: () => [],
    addEventListener: (event, fn) => {
      (docListeners[event] = docListeners[event] || []).push(fn);
    },
  };
  global.window = { location: { href: "/" }, confirm: () => true };
  const store = new Map(Object.entries(scenario.localStorage || {}));
  global.localStorage = {
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => void store.set(key, String(value)),
    removeItem: (key) => void store.delete(key),
  };

  const fetchCalls = [];
  global.fetch = async (url, options = {}) => {
    const method = (options.method || "GET").toUpperCase();
    let body = null;
    if (options.body) {
      try {
        body = JSON.parse(options.body);
      } catch (_) {
        body = null;
      }
    }
    const call = { url, method, body, options };
    fetchCalls.push(call);
    for (const [routeMethod, routeUrl, spec] of scenario.routes || []) {
      if (routeMethod !== method || routeUrl !== url) continue;
      const resolved = typeof spec === "function" ? spec(body, call) : spec;
      if (resolved && resolved.pending) await resolved.pending;
      return makeResponse(resolved);
    }
    return makeResponse({ status: 404, body: { detail: "not found" } });
  };

  global.FormData = class {
    append() {}
  };
  let objectUrlCounter = 0;
  const revoked = [];
  global.URL = {
    createObjectURL: () => `blob:preview-${(objectUrlCounter += 1)}`,
    revokeObjectURL: (url) => void revoked.push(url),
  };
  FakeAudio.instances = [];
  global.Audio = FakeAudio;
  global.ABLimits = require(LIMITS_PATH);

  delete require.cache[require.resolve(APP_PATH)];
  require(APP_PATH);

  const element = (id) => global.document.getElementById(id);
  for (const fn of docListeners.DOMContentLoaded || []) await fn();
  await flush();
  return {
    element,
    fetchCalls,
    store,
    revoked,
    calls: (method, url) =>
      fetchCalls.filter((c) => c.method === method && c.url === url),
    audio: () => FakeAudio.instances.slice(),
  };
}

function savedState(overrides) {
  return {
    "audiobook.state.v1": JSON.stringify(
      Object.assign(
        {
          style: "natural",
          languageOverride: "",
          project: {
            id: "p1",
            title: "Room 207",
            language: "Hindi",
            languageCode: "hi",
            paragraphCount: 3,
            preview: "The lane was empty…",
            filename: "script.txt",
            sizeBytes: 1234,
            characterCount: 757,
          },
          planKey: null,
          audio: null,
        },
        overrides || {}
      )
    ),
  };
}

function upload(element, spec) {
  const file = Object.assign(
    { name: "script.txt", size: 1234, text: async () => "Once upon a time." },
    spec || {}
  );
  element("file-input").files = [file];
  fire(element("file-input"), "change", {});
}

function toggle(element, checked) {
  element("mv-toggle").checked = checked;
  fire(element("mv-toggle"), "change", { target: { checked } });
}

/* The primary button is the ONLY analysis trigger ("Character Analyser"). */
function analyse(app) {
  fire(app.element("btn-casting-continue"), "click", {});
}

function generatedAudioBody(mode) {
  return {
    status: 200,
    body: {
      audio_url: AUDIO_URL,
      audio_filename: "audiobook.mp3",
      generation_mode: mode,
      segment_count: 3,
    },
  };
}

function generateRoutes(projectId, modeHolder) {
  return [
    [
      "PUT",
      `/projects/${projectId}/narration-preferences`,
      { status: 200, body: { saved: true } },
    ],
    [
      "POST",
      `/projects/${projectId}/generate-audio`,
      () => generatedAudioBody(modeHolder.mode),
    ],
  ];
}

let failures = 0;
async function check(name, fn) {
  try {
    await fn();
    console.log(`ok - ${name}`);
  } catch (error) {
    failures += 1;
    console.error(`NOT OK - ${name}: ${error.message}`);
  }
}

/* --------------------------------------------------------------------- */
/* Scenarios                                                              */
/* --------------------------------------------------------------------- */
(async () => {
  /* 1 — single-voice mode works exactly as before (toggle off) */
  await check("single-voice workflow is unchanged with the option off", async () => {
    const app = await boot({
      localStorage: {},
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["POST", "/projects/from-script", { status: 201, body: IMPORT_P1 }],
        ...generateRoutes("p1", { mode: "narration_plan" }),
        [
          "GET",
          "/projects/p1/narration-plan",
          { status: 404, body: { detail: "not found" } },
        ],
        [
          "POST",
          "/projects/p1/analyze-narration",
          { status: 200, body: { language: "hi", saved: true, paragraph_count: 3 } },
        ],
      ],
    });
    upload(app.element);
    await flush();
    assert.strictEqual(
      app.calls("POST", "/projects/p1/analyze-characters").length,
      0,
      "toggle off must never trigger character analysis"
    );

    fire(app.element("btn-generate"), "click", {});
    await flush();

    const urls = app.fetchCalls.map((c) => `${c.method} ${c.url}`);
    assert.ok(urls.includes("PUT /projects/p1/narration-preferences"), urls.join(","));
    assert.ok(urls.includes("POST /projects/p1/analyze-narration"), urls.join(","));
    assert.ok(urls.includes("POST /projects/p1/generate-audio"), urls.join(","));
    assert.strictEqual(
      app.calls("POST", "/projects/p1/create-multi-voice-plan").length,
      0,
      "single-voice generation must not create a multi-voice plan"
    );
    assert.strictEqual(
      app.calls("POST", "/projects/p1/analyze-characters").length,
      0
    );
    assert.ok(
      app.element("status").textContent.includes("Your audiobook is ready"),
      app.element("status").textContent
    );
    assert.strictEqual(app.element("result-card").classList.contains("hidden"), false);
  });

  /* 2 — the option is disabled by default, counter comes from /usage */
  await check("multi-character option is disabled by default", async () => {
    const app = await boot({
      localStorage: savedState(),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
      ],
    });
    assert.strictEqual(app.element("mv-toggle").checked, false);
    assert.strictEqual(app.element("casting-body").classList.contains("hidden"), true);
    assert.strictEqual(
      app.element("casting-card").classList.contains("hidden"),
      false,
      "the option is offered once a script exists"
    );
    assert.strictEqual(app.element("btn-generate").disabled, false);
    // §9: the maximum shown is the one /usage reported, not a hardcoded value.
    assert.strictEqual(
      app.element("char-count").textContent,
      "757 / 3000 characters"
    );
    assert.strictEqual(
      app.fetchCalls.filter((c) => c.url.includes("voice-casting")).length,
      0,
      "no casting request until the option is switched on"
    );
  });

  /* 3 — enabling does NOT analyse; the Character Analyser button runs it once */
  await check("enabling the option does not analyse; the Character Analyser button runs it once", async () => {
    const gate = deferred();
    const app = await boot({
      localStorage: savedState(),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        [
          "POST",
          "/projects/p1/analyze-characters",
          () => ({ status: 200, body: ANALYSIS_P1, pending: gate.promise }),
        ],
        ["POST", "/projects/p1/voice-casting", { status: 200, body: makeCasting("p1") }],
        ["POST", "/projects/p1/voice-casting/select", (body) => selectInCasting(makeCasting("p1"), body)],
      ],
    });

    toggle(app.element, true);
    await flush();
    assert.strictEqual(
      app.calls("POST", "/projects/p1/analyze-characters").length,
      0,
      "checking the option must NOT start the analysis"
    );
    assert.strictEqual(
      app.element("btn-casting-continue").textContent,
      "Character Analyser"
    );
    assert.strictEqual(app.element("btn-casting-continue").disabled, false);
    assert.strictEqual(app.element("cast-progress").textContent, "");

    // The button starts the analysis — with progress and locked controls.
    analyse(app);
    await flush();
    assert.strictEqual(
      app.calls("POST", "/projects/p1/analyze-characters").length,
      1
    );
    assert.strictEqual(
      app.element("cast-progress").textContent.includes("Analysing your script"),
      true,
      app.element("cast-progress").textContent
    );
    assert.strictEqual(app.element("mv-toggle").disabled, true, "controls lock while running");
    assert.strictEqual(app.element("btn-casting-continue").disabled, true);
    assert.strictEqual(
      app.element("btn-casting-continue").textContent,
      "Analysing characters…"
    );

    // Another click while the analysis runs must not fire a second request.
    analyse(app);
    await flush();

    gate.resolve({});
    await flush();

    assert.strictEqual(
      app.calls("POST", "/projects/p1/analyze-characters").length,
      1,
      "analysis must run exactly once for the script"
    );
    assert.strictEqual(app.element("cast-progress").textContent, "Character analysis complete.");
    assert.strictEqual(app.element("mv-toggle").disabled, false);
    assert.strictEqual(
      app.element("btn-casting-continue").textContent,
      "Save & Approve Casting",
      "once the cast is ready the button saves + approves the casting"
    );
    assert.strictEqual(app.element("btn-casting-continue").disabled, false);
  });

  /* 4 — detection results are displayed from the real payload */
  await check("detected characters are displayed from the analysis payload", async () => {
    const app = await boot({
      localStorage: savedState(),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["POST", "/projects/p1/analyze-characters", { status: 200, body: ANALYSIS_P1 }],
        ["POST", "/projects/p1/voice-casting", { status: 200, body: makeCasting("p1") }],
        ["POST", "/projects/p1/voice-casting/select", (body) => selectInCasting(makeCasting("p1"), body)],
      ],
    });
    toggle(app.element, true);
    analyse(app);
    await flush();

    assert.strictEqual(
      app.element("cast-role-count").textContent,
      "Detected characters — 2 roles"
    );
    const rows = app.element("cast-roles-body").innerHTML;
    assert.ok(rows.includes("सुमन"), rows);
    assert.ok(rows.includes("Young female protagonist who warns the visitor."), rows);
    assert.ok(rows.includes("Narrator"), rows);
    assert.ok(rows.includes("Third-person storytelling voice"), rows);
    // Non-speaking person: never a speaking role…
    assert.ok(!rows.includes("राहुल"), rows);
    // …but honestly reported as not having a voice.
    assert.ok(
      app.element("cast-note").textContent.includes("राहुल"),
      app.element("cast-note").textContent
    );
    assert.ok(
      app.element("cast-note").textContent.includes("not given a speaking voice"),
      app.element("cast-note").textContent
    );
  });

  /* 5 — narrator separate + narrator-only script message */
  await check("narrator stays separate and narrator-only scripts are explained", async () => {
    const app = await boot({
      localStorage: savedState(),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["POST", "/projects/p1/analyze-characters", { status: 200, body: ANALYSIS_P1 }],
        ["POST", "/projects/p1/voice-casting", { status: 200, body: makeCasting("p1") }],
        ["POST", "/projects/p1/voice-casting/select", (body) => selectInCasting(makeCasting("p1"), body)],
      ],
    });
    toggle(app.element, true);
    analyse(app);
    await flush();

    const rows = app.element("cast-roles-body").innerHTML;
    assert.ok(rows.includes('data-role="narrator"'), rows);
    assert.ok(rows.includes('data-role="suman"'), rows);
    const cards = app.element("cast-cards").innerHTML;
    const articles = cards.match(/<article class="cast-card"/g) || [];
    assert.strictEqual(articles.length, 2, "narrator + speaking characters only");
    assert.ok(cards.includes("cast-badge"), "the narrator carries its own badge");
    assert.ok(cards.includes("Narrator</span>"), cards);

    // Narrator-only script (no speaking characters detected).
    const only = await boot({
      localStorage: savedState(),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["POST", "/projects/p1/analyze-characters", { status: 200, body: ANALYSIS_P2 }],
        [
          "POST",
          "/projects/p1/voice-casting",
          { status: 200, body: { project_id: "p1", approved: false, narrator: { candidates: NARRATOR_CANDIDATES, selected: null, status: "pending" }, characters: {}, missing_speakers: ["narrator"] } },
        ],
        [
          "POST",
          "/projects/p1/voice-casting/select",
          (body) => selectInCasting({ narrator: { candidates: NARRATOR_CANDIDATES, selected: null }, characters: {} }, body),
        ],
      ],
    });
    toggle(only.element, true);
    analyse(only);
    await flush();
    assert.strictEqual(
      only.element("cast-role-count").textContent,
      "Detected characters — narrator only"
    );
    assert.ok(
      only.element("cast-note").textContent.includes("narrator-only narration"),
      only.element("cast-note").textContent
    );
  });

  /* 6 — automatic casting uses the backend recommendations */
  await check("automatic casting locks the backend recommendations", async () => {
    const casting = makeCasting("p1");
    const app = await boot({
      localStorage: savedState(),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["POST", "/projects/p1/analyze-characters", { status: 200, body: ANALYSIS_P1 }],
        ["POST", "/projects/p1/voice-casting", { status: 200, body: casting }],
        ["POST", "/projects/p1/voice-casting/select", (body) => selectInCasting(casting, body)],
      ],
    });
    toggle(app.element, true);
    analyse(app);
    await flush();

    const selects = app.calls("POST", "/projects/p1/voice-casting/select");
    assert.strictEqual(selects.length, 2, JSON.stringify(selects.map((c) => c.body)));
    assert.deepStrictEqual(selects[0].body, {
      speaker_id: "narrator",
      voice_id: "voice-narrator-01",
    });
    assert.deepStrictEqual(selects[1].body, {
      speaker_id: "suman",
      voice_id: "voice-suman-02",
    });

    const cards = app.element("cast-cards").innerHTML;
    assert.ok(!cards.includes("<select"), "automatic mode needs no per-role control");
    assert.ok(cards.includes("Recommended voice:"), cards);
    assert.ok(cards.includes("Narrator Deep"), cards);
    assert.ok(cards.includes("Warm Soft"), cards);
    assert.strictEqual(app.element("btn-casting-continue").disabled, false);
  });

  /* 7 — manual selection updates the correct character */
  await check("manual selection updates the chosen character's voice", async () => {
    const casting = makeCasting("p1");
    const app = await boot({
      localStorage: savedState(),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["POST", "/projects/p1/analyze-characters", { status: 200, body: ANALYSIS_P1 }],
        ["POST", "/projects/p1/voice-casting", { status: 200, body: casting }],
        ["POST", "/projects/p1/voice-casting/select", (body) => selectInCasting(casting, body)],
        ["POST", "/projects/p1/voice-casting/approve", () => approveCasting(casting)],
      ],
    });
    toggle(app.element, true);
    analyse(app);
    await flush();

    app.element("cast-mode-manual").checked = true;
    fire(app.element("cast-mode-manual"), "change", { target: { checked: true } });
    await flush();

    const cards = app.element("cast-cards").innerHTML;
    assert.ok(cards.includes('<select id="cast-select-suman"'), cards);
    assert.ok(cards.includes('<select id="cast-select-narrator"'), cards);
    assert.ok(
      cards.includes('data-action="voice" data-speaker="suman"'),
      "the voice dropdown is wired for delegated change events"
    );
    assert.ok(cards.includes("voice-suman-02"), "the recommendation is the default");
    assert.ok(cards.includes("voice-hi-3"), "actual backend voices only");

    // Change ONLY सुमन, leave the narrator alone.
    fire(app.element("cast-cards"), "change", {
      target: { dataset: { action: "voice", speaker: "suman" }, value: "voice-hi-3" },
    });
    await flush();
    assert.ok(
      app.element("cast-status-suman").textContent.includes("switch to"),
      app.element("cast-status-suman").textContent
    );

    app.fetchCalls.length = 0; // only assert what "Continue" itself sends
    fire(app.element("btn-casting-continue"), "click", {});
    await flush();

    const selects = app.calls("POST", "/projects/p1/voice-casting/select");
    assert.strictEqual(selects.length, 1, JSON.stringify(selects.map((c) => c.body)));
    assert.deepStrictEqual(selects[0].body, {
      speaker_id: "suman",
      voice_id: "voice-hi-3",
    });
    assert.strictEqual(app.calls("POST", "/projects/p1/voice-casting/approve").length, 1);
    assert.strictEqual(app.element("btn-casting-continue").textContent, "Casting approved");
    assert.strictEqual(app.element("btn-casting-continue").disabled, true);
    assert.ok(
      app.element("status").textContent.includes("Casting approved"),
      app.element("status").textContent
    );
    assert.strictEqual(app.element("btn-generate").disabled, false);
  });

  /* 8 — preview calls the audition endpoint */
  await check("preview calls the existing audition endpoint", async () => {
    const casting = makeCasting("p1");
    const gate = deferred();
    const app = await boot({
      localStorage: savedState(),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["POST", "/projects/p1/analyze-characters", { status: 200, body: ANALYSIS_P1 }],
        ["POST", "/projects/p1/voice-casting", { status: 200, body: casting }],
        ["POST", "/projects/p1/voice-casting/select", (body) => selectInCasting(casting, body)],
        [
          "POST",
          "/projects/p1/voice-casting/audition",
          () => ({ status: 200, pending: gate.promise }),
        ],
      ],
    });
    toggle(app.element, true);
    analyse(app);
    await flush();

    // The rendered cards must carry the delegation attributes a real browser
    // needs (the fake DOM dispatches to the container directly).
    const rendered = app.element("cast-cards").innerHTML;
    assert.ok(rendered.includes('id="cast-audition-suman"'), rendered);
    assert.ok(
      rendered.includes('data-action="audition" data-speaker="suman"'),
      rendered
    );

    fire(app.element("cast-cards"), "click", {
      target: { dataset: { action: "audition", speaker: "suman" } },
    });
    await flush();
    assert.strictEqual(
      app.element("cast-audition-suman").textContent,
      "Generating preview…"
    );

    gate.resolve({});
    await flush();

    const auditions = app.calls("POST", "/projects/p1/voice-casting/audition");
    assert.strictEqual(auditions.length, 1);
    const body = auditions[0].body;
    assert.strictEqual(body.speaker_id, "suman");
    assert.strictEqual(body.voice_id, "voice-suman-02", "previews the selected voice");
    assert.ok(typeof body.text === "string" && body.text.length > 0);
    assert.ok(body.text.length <= 400, "audition lines are capped server-side");
    assert.strictEqual(app.element("cast-audition-suman").textContent, "■ Stop preview");
  });

  /* 9 — the returned preview audio plays */
  await check("the preview audio plays and can be stopped", async () => {
    const realNow = Date.now;
    let clockOffset = 0;
    Date.now = () => realNow() + clockOffset; // step past the preview cooldown
    try {
      const casting = makeCasting("p1");
      const app = await boot({
        localStorage: savedState(),
        routes: [
          ["GET", "/usage", { status: 200, body: USAGE_OK }],
          ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
          ["POST", "/projects/p1/analyze-characters", { status: 200, body: ANALYSIS_P1 }],
          ["POST", "/projects/p1/voice-casting", { status: 200, body: casting }],
          ["POST", "/projects/p1/voice-casting/select", (body) => selectInCasting(casting, body)],
          ["POST", "/projects/p1/voice-casting/audition", { status: 200 }],
        ],
      });
      toggle(app.element, true);
      analyse(app);
      await flush();

      fire(app.element("cast-cards"), "click", {
        target: { dataset: { action: "audition", speaker: "narrator" } },
      });
      await flush();

      const first = app.audio()[0];
      assert.ok(first, "one audio element for the preview");
      assert.ok(first.url.startsWith("blob:"), first.url);
      assert.strictEqual(first.playCalls, 1);
      assert.strictEqual(first.paused, false);
      assert.ok(
        app.element("status").textContent.includes("Playing the voice preview"),
        app.element("status").textContent
      );

      // The stream ends -> the button returns to its idle label.
      first.onended();
      await flush();
      assert.strictEqual(app.element("cast-audition-narrator").textContent, "▶ Preview Voice");

      // A later preview can be stopped mid-playback (pause + revoke).
      clockOffset += 11 * 1000;
      fire(app.element("cast-cards"), "click", {
        target: { dataset: { action: "audition", speaker: "narrator" } },
      });
      await flush();
      const second = app.audio()[1];
      assert.ok(second, "the second preview created its own audio element");
      assert.strictEqual(second.playCalls, 1);
      fire(app.element("cast-cards"), "click", {
        target: { dataset: { action: "audition", speaker: "narrator" } },
      });
      await flush();
      assert.strictEqual(second.pauseCalls, 1, "the second click stops the preview");
      assert.ok(app.revoked.length >= 1, "the object URL is released");
      assert.strictEqual(app.element("cast-audition-narrator").textContent, "▶ Preview Voice");
    } finally {
      Date.now = realNow;
    }
  });

  /* 10 — replacing the script invalidates analysis + assignments */
  await check("replacing the script invalidates the previous analysis", async () => {
    const castingP1 = makeCasting("p1");
    const app = await boot({
      localStorage: savedState(),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["POST", "/projects/p1/analyze-characters", { status: 200, body: ANALYSIS_P1 }],
        ["POST", "/projects/p1/voice-casting", { status: 200, body: castingP1 }],
        ["POST", "/projects/p1/voice-casting/select", (body) => selectInCasting(castingP1, body)],
        ["POST", "/projects/from-script", { status: 201, body: IMPORT_P2 }],
        ["POST", "/projects/p2/analyze-characters", { status: 200, body: ANALYSIS_P2 }],
        [
          "POST",
          "/projects/p2/voice-casting",
          { status: 200, body: { project_id: "p2", approved: false, narrator: { candidates: NARRATOR_CANDIDATES, selected: null, status: "pending" }, characters: {}, missing_speakers: ["narrator"] } },
        ],
        [
          "POST",
          "/projects/p2/voice-casting/select",
          (body) => selectInCasting({ narrator: { candidates: NARRATOR_CANDIDATES, selected: null }, characters: {} }, body),
        ],
      ],
    });
    toggle(app.element, true);
    analyse(app);
    await flush();
    assert.strictEqual(app.calls("POST", "/projects/p1/analyze-characters").length, 1);
    const p1Selects = app.calls("POST", "/projects/p1/voice-casting/select").length;
    assert.ok(p1Selects >= 1, "the first script's cast was assigned");

    upload(app.element, { name: "second.txt", size: 99 });
    await flush();
    assert.strictEqual(
      app.calls("POST", "/projects/p2/analyze-characters").length,
      0,
      "a replaced script is never analysed automatically"
    );
    analyse(app); // the Character Analyser button is clicked for the new script
    await flush();
    assert.strictEqual(
      app.calls("POST", "/projects/p2/analyze-characters").length,
      1,
      "the new script is analysed exactly once"
    );
    const saved = JSON.parse(app.store.get("audiobook.state.v1"));
    assert.strictEqual(saved.casting.analysisKey, "p2:400");
    assert.strictEqual(saved.casting.analysis.project_id, "p2");
    assert.strictEqual(saved.casting.mvPlanKey, null, "the old plan key is dropped");
    assert.strictEqual(
      app.element("cast-roles-body").innerHTML.includes("सुमन"),
      false,
      "the previous script's roles are gone"
    );
    assert.strictEqual(
      app.element("cast-role-count").textContent,
      "Detected characters — narrator only"
    );
    assert.strictEqual(
      app.calls("POST", "/projects/p1/voice-casting/select").length,
      p1Selects,
      "no stale assignment is re-sent for the replaced script"
    );
  });

  /* 11 — duplicate analysis and audition submissions are prevented */
  await check("duplicate analysis and audition requests are prevented", async () => {
    const casting = makeCasting("p1");
    const gate = deferred();
    const app = await boot({
      localStorage: savedState(),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        [
          "POST",
          "/projects/p1/analyze-characters",
          () => ({ status: 200, body: ANALYSIS_P1, pending: gate.promise }),
        ],
        ["POST", "/projects/p1/voice-casting", { status: 200, body: casting }],
        ["POST", "/projects/p1/voice-casting/select", (body) => selectInCasting(casting, body)],
        ["POST", "/projects/p1/voice-casting/audition", { status: 200 }],
      ],
    });

    // Two rapid button clicks while the analysis is in flight.
    toggle(app.element, true);
    analyse(app);
    analyse(app);
    await flush();
    gate.resolve({});
    await flush();
    assert.strictEqual(app.calls("POST", "/projects/p1/analyze-characters").length, 1);

    // Re-enabling an already analysed script must not analyse again.
    toggle(app.element, false);
    toggle(app.element, true);
    await flush();
    assert.strictEqual(app.calls("POST", "/projects/p1/analyze-characters").length, 1);
    assert.strictEqual(
      app.element("btn-casting-continue").textContent,
      "Save & Approve Casting",
      "the finished analysis is reused, not re-run"
    );

    // An audition blocks every other audition while it is running…
    const pendingAudition = deferred();
    app.fetchCalls.length = 0;
    // Route auditions through a gate by wrapping fetch for this part only.
    const realFetch = global.fetch;
    global.fetch = async (url, options = {}) => {
      if (String(url).endsWith("/voice-casting/audition") && options.method === "POST") {
        app.fetchCalls.push({ url, method: "POST", body: JSON.parse(options.body) });
        await pendingAudition.promise;
        return makeResponse({ status: 200 });
      }
      return realFetch(url, options);
    };
    fire(app.element("cast-cards"), "click", {
      target: { dataset: { action: "audition", speaker: "narrator" } },
    });
    fire(app.element("cast-cards"), "click", {
      target: { dataset: { action: "audition", speaker: "suman" } },
    });
    await flush();
    assert.strictEqual(
      app.fetchCalls.filter((c) => String(c.url).endsWith("/voice-casting/audition")).length,
      1,
      "only one audition runs at a time"
    );
    assert.ok(
      app.element("status").textContent.includes("already being generated"),
      app.element("status").textContent
    );
    pendingAudition.resolve({});
    await flush();
    assert.strictEqual(app.audio().length, 1);

    // …and a successful preview starts a short cooldown for the next one.
    fire(app.element("cast-cards"), "click", {
      target: { dataset: { action: "audition", speaker: "suman" } },
    });
    await flush();
    assert.strictEqual(
      app.fetchCalls.filter((c) => String(c.url).endsWith("/voice-casting/audition")).length,
      1,
      "the cooldown prevents an immediate duplicate submission"
    );
    assert.ok(
      app.element("status").textContent.includes("wait a moment"),
      app.element("status").textContent
    );
    assert.strictEqual(app.audio().length, 1, "no second preview was generated");
  });

  /* 12 — analysis and audition work at zero allowance */
  await check("analysis and audition work when the allowance is zero", async () => {
    const casting = makeCasting("p1");
    const app = await boot({
      localStorage: savedState(),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_EXHAUSTED }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["POST", "/projects/p1/analyze-characters", { status: 200, body: ANALYSIS_P1 }],
        ["POST", "/projects/p1/voice-casting", { status: 200, body: casting }],
        ["POST", "/projects/p1/voice-casting/select", (body) => selectInCasting(casting, body)],
        ["POST", "/projects/p1/voice-casting/audition", { status: 200 }],
      ],
    });
    assert.strictEqual(app.element("btn-generate").disabled, true, "generation is blocked");
    assert.strictEqual(
      app.element("quota-note").classList.contains("hidden"),
      false,
      "the exhaustion message stays visible"
    );

    toggle(app.element, true);
    analyse(app);
    await flush();
    assert.strictEqual(app.calls("POST", "/projects/p1/analyze-characters").length, 1);
    assert.strictEqual(
      app.element("cast-role-count").textContent,
      "Detected characters — 2 roles",
      "casting stays usable at zero allowance"
    );

    fire(app.element("cast-cards"), "click", {
      target: { dataset: { action: "audition", speaker: "suman" } },
    });
    await flush();
    assert.strictEqual(app.calls("POST", "/projects/p1/voice-casting/audition").length, 1);
    assert.strictEqual(app.audio().length, 1);
    assert.strictEqual(app.element("btn-generate").disabled, true);
    assert.strictEqual(
      app.element("mv-toggle").disabled,
      false,
      "the casting option is never gated by the allowance"
    );
  });

  /* 13 — multi-voice generation: no second plan call, honest mode check */
  await check("multi-voice generation skips analyze-narration and verifies the mode", async () => {
    const casting = makeCasting("p1");
    casting.approved = true;
    casting.missing_speakers = [];
    ["narrator"].concat(Object.keys(casting.characters)).forEach((id) => {
      const entry = entryOf(casting, id);
      entry.status = "approved";
      entry.selected = {
        character_id: id,
        voice_id: id === "narrator" ? "voice-narrator-01" : "voice-suman-02",
        voice_name: id === "narrator" ? "Narrator Deep" : "Warm Soft",
        provider: "elevenlabs",
        selected_at: "2026-11-07T00:00:00+00:00",
        locked: true,
      };
    });
    const mode = { mode: "multi_voice" };
    const app = await boot({
      localStorage: savedState({
        casting: {
          enabled: true,
          mode: "auto",
          mvPlanKey: null,
          analysis: ANALYSIS_P1,
          analysisKey: "p1:757",
        },
      }),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["GET", "/projects/p1/voice-casting", { status: 200, body: casting }],
        ...generateRoutes("p1", mode),
        ["POST", "/projects/p1/create-multi-voice-plan", { status: 200, body: { language: "hi" } }],
      ],
    });

    // Restoring shows the cast without re-analysing (analysis is cached).
    assert.strictEqual(
      app.calls("POST", "/projects/p1/analyze-characters").length,
      0,
      "a reload must not re-run the analysis"
    );
    assert.strictEqual(
      app.element("cast-role-count").textContent,
      "Detected characters — 2 roles"
    );
    assert.strictEqual(app.element("btn-generate").disabled, false, "cast already approved");

    fire(app.element("btn-generate"), "click", {});
    await flush();
    assert.ok(
      app.element("status").textContent.includes("Your audiobook is ready"),
      app.element("status").textContent
    );
    assert.strictEqual(app.calls("POST", "/projects/p1/analyze-narration").length, 0);
    assert.strictEqual(app.calls("POST", "/projects/p1/create-multi-voice-plan").length, 1);
    assert.strictEqual(app.calls("POST", "/projects/p1/generate-audio").length, 1);

    // Regenerating reuses the saved plan (no second planning call/spend).
    fire(app.element("btn-regenerate"), "click", {});
    await flush();
    assert.strictEqual(app.calls("POST", "/projects/p1/create-multi-voice-plan").length, 1);
    assert.strictEqual(app.calls("POST", "/projects/p1/generate-audio").length, 2);
    assert.strictEqual(app.calls("POST", "/projects/p1/analyze-narration").length, 0);

    // Honest failure: if the backend did NOT use the character voices, say so
    // and drop the plan key so the next attempt rebuilds the plan.
    mode.mode = "narration_plan";
    fire(app.element("btn-generate"), "click", {});
    await flush();
    assert.ok(
      app.element("status").textContent.includes("single-voice file"),
      app.element("status").textContent
    );
    mode.mode = "multi_voice";
    fire(app.element("btn-generate"), "click", {});
    await flush();
    assert.strictEqual(
      app.calls("POST", "/projects/p1/create-multi-voice-plan").length,
      2,
      "the plan key was cleared so the plan is rebuilt"
    );
    assert.strictEqual(app.calls("POST", "/projects/p1/analyze-narration").length, 0);
  });

  /* 14 — existing projects and generated audio stay accessible */
  await check("existing projects and generated audio remain accessible", async () => {
    const app = await boot({
      localStorage: savedState({
        audio: { url: AUDIO_URL, filename: "audiobook.mp3" },
        planKey: "{}",
      }),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["GET", "/projects/p1/audio", { status: 200, body: { audio_url: AUDIO_URL, audio_filename: "audiobook.mp3" } }],
      ],
    });
    assert.strictEqual(app.element("result-card").classList.contains("hidden"), false);
    assert.strictEqual(app.element("btn-download").href, AUDIO_URL);
    assert.strictEqual(app.element("audio-player").src, AUDIO_URL);
    assert.ok(
      app.element("status").textContent.length > 0,
      "a status line is shown after restoring"
    );
    // Switching the casting option on afterwards does not disturb the result.
    const casting = makeCasting("p1");
    const withCasting = await boot({
      localStorage: savedState({ audio: { url: AUDIO_URL, filename: "audiobook.mp3" } }),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["GET", "/projects/p1/audio", { status: 200, body: { audio_url: AUDIO_URL, audio_filename: "audiobook.mp3" } }],
        ["POST", "/projects/p1/analyze-characters", { status: 200, body: ANALYSIS_P1 }],
        ["POST", "/projects/p1/voice-casting", { status: 200, body: casting }],
        ["POST", "/projects/p1/voice-casting/select", (body) => selectInCasting(casting, body)],
      ],
    });
    toggle(withCasting.element, true);
    await flush();
    assert.strictEqual(
      withCasting.calls("POST", "/projects/p1/analyze-characters").length,
      0,
      "switching the option on does not analyse by itself"
    );
    analyse(withCasting);
    await flush();
    assert.strictEqual(withCasting.element("btn-download").href, AUDIO_URL);
    assert.strictEqual(withCasting.element("result-card").classList.contains("hidden"), false);
    assert.strictEqual(withCasting.calls("POST", "/projects/p1/analyze-characters").length, 1);
  });

  /* 15 — the script maximum comes from GET /usage, never from a literal */
  await check("the script maximum is read from the /usage response", async () => {
    const usage = {
      generation_limit: 5,
      generations_used: 1,
      generations_remaining: 4,
      max_script_characters: 4321,
    };
    const app = await boot({
      localStorage: savedState(),
      routes: [
        ["GET", "/usage", { status: 200, body: usage }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
      ],
    });
    assert.strictEqual(app.element("char-count").textContent, "757 / 4321 characters");
    assert.ok(app.element("usage-line").textContent.includes("4321"));
    assert.strictEqual(
      app.element("usage-line").textContent.includes("200000"),
      false
    );
  });

  /* 16 — upload reveals the visible, unchecked casting section (§13) */
  await check("script upload reveals the visible, unchecked casting section", async () => {
    const app = await boot({
      localStorage: {},
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["POST", "/projects/from-script", { status: 201, body: IMPORT_P1 }],
      ],
    });
    // Without a project the section is hidden and the badges skip it.
    assert.strictEqual(app.element("casting-card").classList.contains("hidden"), true);
    assert.strictEqual(app.element("step-badge-style").textContent, "2");
    assert.strictEqual(app.element("step-badge-generate").textContent, "3");

    upload(app.element);
    await flush();

    assert.strictEqual(
      app.element("casting-card").classList.contains("hidden"),
      false,
      "the casting section must be visible right after the upload"
    );
    assert.strictEqual(app.element("mv-toggle").checked, false, "unchecked by default");
    assert.strictEqual(app.element("casting-body").classList.contains("hidden"), true);
    assert.strictEqual(app.element("step-badge-style").textContent, "3");
    assert.strictEqual(app.element("step-badge-generate").textContent, "4");
    assert.strictEqual(
      app.calls("POST", "/projects/p1/analyze-characters").length,
      0,
      "nothing is analysed before the user opts in"
    );
    assert.strictEqual(
      app.fetchCalls.filter((c) => c.url.includes("voice-casting")).length,
      0,
      "no casting request while the option is off"
    );

    // Enabling reveals the panel — with the action button naming the next
    // step — and STILL nothing is analysed automatically.
    toggle(app.element, true);
    await flush();
    assert.strictEqual(
      app.element("btn-casting-continue").textContent,
      "Character Analyser",
      "the action button names the next step, not a dead end"
    );
    assert.strictEqual(
      app.calls("POST", "/projects/p1/analyze-characters").length,
      0,
      "enabling never starts the analysis"
    );
  });

  /* 17 — reload: the server plan is reused, never re-analysed (§10, §14.4) */
  await check("reload restores the server plan without re-analysis or a second plan", async () => {
    const casting = approvedCasting();
    const app = await boot({
      localStorage: savedState({
        casting: {
          enabled: true,
          mode: "auto",
          mvPlanKey: null,
          analysis: ANALYSIS_P1,
          analysisKey: "p1:757",
        },
      }),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["GET", "/projects/p1/voice-casting", { status: 200, body: casting }],
        ["GET", "/projects/p1/multi-voice-plan/status", { status: 200, body: planReadyStatus() }],
        ...generateRoutes("p1", { mode: "multi_voice" }),
      ],
    });

    assert.strictEqual(
      app.calls("POST", "/projects/p1/analyze-characters").length,
      0,
      "a reload must not re-run the analysis"
    );
    assert.strictEqual(app.element("mv-toggle").checked, true);
    assert.strictEqual(app.element("casting-card").classList.contains("hidden"), false);

    fire(app.element("btn-generate"), "click", {});
    await flush();

    assert.strictEqual(
      app.calls("POST", "/projects/p1/create-multi-voice-plan").length,
      0,
      "a valid server-side plan is reused — no new plan request"
    );
    assert.strictEqual(app.calls("POST", "/projects/p1/generate-audio").length, 1);
    assert.strictEqual(app.element("result-card").classList.contains("hidden"), false);
  });

  /* 18 — localStorage loss: project, cast and plan recovered (§10, §14.3) */
  await check("localStorage loss recovers project, cast and plan from the server", async () => {
    const casting = approvedCasting();
    const app = await boot({
      localStorage: {}, // the browser lost EVERYTHING
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        [
          "GET",
          "/projects",
          {
            status: 200,
            body: [
              {
                project_id: "p1",
                name: "Room 207",
                description: null,
                created_at: "2026-11-07T09:00:00+00:00",
                has_script: true,
                script: {
                  filename: "script.txt",
                  stored_as: "script.txt",
                  size_bytes: 1234,
                  character_count: 757,
                  uploaded_at: "2026-11-07T09:00:00+00:00",
                },
              },
            ],
          },
        ],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["GET", "/projects/p1/voice-casting", { status: 200, body: casting }],
        ["GET", "/projects/p1/multi-voice-plan/status", { status: 200, body: planReadyStatus() }],
        [
          "GET",
          "/media/projects/p1/script.txt",
          { status: 200, text: "Once upon a time.\n\nThe end." },
        ],
        ["GET", "/projects/p1/audio", { status: 404, body: { detail: "no audio" } }],
        ...generateRoutes("p1", { mode: "multi_voice" }),
      ],
    });

    assert.strictEqual(
      app.calls("POST", "/projects/p1/analyze-characters").length,
      0,
      "restore must never re-run character analysis"
    );
    assert.strictEqual(
      app.element("mv-toggle").checked,
      true,
      "a stored cast means the multi-character option was on"
    );
    assert.strictEqual(app.element("casting-card").classList.contains("hidden"), false);
    // Descriptions are rebuilt from the SERVER's stored analysis (story).
    assert.ok(
      app.element("cast-roles-body").innerHTML.includes("<tr"),
      "character rows restored from the server-side analysis"
    );
    assert.ok(
      app.element("cast-role-count").textContent.includes("2 roles"),
      app.element("cast-role-count").textContent
    );

    fire(app.element("btn-generate"), "click", {});
    await flush();

    assert.strictEqual(
      app.calls("POST", "/projects/p1/create-multi-voice-plan").length,
      0,
      "the valid server-side plan is reused — zero duplicate plan creation"
    );
    assert.strictEqual(app.calls("POST", "/projects/p1/generate-audio").length, 1);
    assert.strictEqual(app.element("result-card").classList.contains("hidden"), false);
  });

  /* 19 — no stored cast: only the Character Analyser button starts detection */
  await check("restore without a stored cast analyses only via the Character Analyser button", async () => {
    const app = await boot({
      localStorage: savedState({
        casting: {
          enabled: true,
          mode: "auto",
          mvPlanKey: null,
          analysis: null,
          analysisKey: null,
        },
      }),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["GET", "/projects/p1/voice-casting", { status: 404, body: { detail: "no cast" } }],
        ["GET", "/projects/p1/multi-voice-plan/status", { status: 200, body: planEmptyStatus() }],
        ["POST", "/projects/p1/analyze-characters", { status: 200, body: ANALYSIS_P1 }],
        ["POST", "/projects/p1/voice-casting", { status: 200, body: makeCasting("p1") }],
      ],
    });

    assert.strictEqual(
      app.calls("POST", "/projects/p1/analyze-characters").length,
      0,
      "restoring must not start the analysis automatically"
    );
    assert.strictEqual(
      app.element("btn-casting-continue").textContent,
      "Character Analyser",
      "the button offers the explicit action"
    );
    assert.strictEqual(app.element("btn-casting-continue").disabled, false);

    // The explicit button click starts detection — exactly once.
    analyse(app);
    await flush();
    assert.strictEqual(app.calls("POST", "/projects/p1/analyze-characters").length, 1);
    assert.ok(app.element("cast-roles-body").innerHTML.includes("<tr"));
  });

  /* 20 — switching back to "Let AI choose" re-selects the AI recommendation */
  await check("switching back to Let AI choose re-selects the AI recommendation", async () => {
    const casting = makeCasting("p1");
    const app = await boot({
      localStorage: savedState(),
      routes: [
        ["GET", "/usage", { status: 200, body: USAGE_OK }],
        ["GET", "/projects/p1", { status: 200, body: { project_id: "p1" } }],
        ["POST", "/projects/p1/analyze-characters", { status: 200, body: ANALYSIS_P1 }],
        ["POST", "/projects/p1/voice-casting", { status: 200, body: casting }],
        ["POST", "/projects/p1/voice-casting/select", (body) => selectInCasting(casting, body)],
        ["POST", "/projects/p1/voice-casting/approve", () => approveCasting(casting)],
      ],
    });
    toggle(app.element, true);
    analyse(app);
    await flush();

    // Manual pick for सुमन — locked in by Save & Approve Casting.
    app.element("cast-mode-manual").checked = true;
    fire(app.element("cast-mode-manual"), "change", { target: { checked: true } });
    await flush();
    fire(app.element("cast-cards"), "change", {
      target: { dataset: { action: "voice", speaker: "suman" }, value: "voice-hi-3" },
    });
    await flush();
    assert.ok(
      app.element("cast-status-suman").textContent.includes("switch to"),
      app.element("cast-status-suman").textContent
    );
    fire(app.element("btn-casting-continue"), "click", {});
    await flush();
    const manual = app.calls("POST", "/projects/p1/voice-casting/select");
    assert.deepStrictEqual(manual[manual.length - 1].body, {
      speaker_id: "suman",
      voice_id: "voice-hi-3",
    });
    assert.strictEqual(app.element("btn-casting-continue").textContent, "Casting approved");

    // Back to AI mode: the manual pick must NOT survive — the AI's
    // recommendation (per story + script) is selected again.
    app.element("cast-mode-auto").checked = true;
    fire(app.element("cast-mode-auto"), "change", { target: { checked: true } });
    await flush();

    const selects = app.calls("POST", "/projects/p1/voice-casting/select");
    assert.strictEqual(selects.length, 4, JSON.stringify(selects.map((c) => c.body)));
    // The recommendation is re-selected over the manual pick.
    assert.deepStrictEqual(selects[3].body, {
      speaker_id: "suman",
      voice_id: "voice-suman-02",
    });
    const cards = app.element("cast-cards").innerHTML;
    assert.ok(
      cards.includes("Selected: Warm Soft"),
      "the VOICE's name is shown — never the character's name"
    );
    assert.ok(!cards.includes("switch to"), "no manual pick survives the switch");
    assert.ok(cards.includes("Selected: Narrator Deep"), cards);
    // A changed cast needs to be saved + approved again.
    assert.strictEqual(app.element("btn-casting-continue").textContent, "Save & Approve Casting");
  });

  if (failures) {
    console.error(`${failures} casting frontend check(s) failed`);
    process.exit(1);
  }
  console.log("all casting frontend checks passed");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
