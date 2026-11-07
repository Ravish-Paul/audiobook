/* Frontend app behavior — run by pytest (see tests/test_generation_limits.py).
 *
 * Drives the REAL app.js in node against a minimal fake DOM and a stubbed
 * fetch, then asserts the UI contract of the generation-allowance stage:
 *   * the tester's own usage is displayed (remaining + total + script max);
 *   * Generate and Regenerate are disabled when no allowance remains;
 *   * the exhaustion message tells the tester to contact the administrator;
 *   * already generated audiobooks stay visible/playable/downloadable;
 *   * over-long scripts are rejected with a clear message before upload.
 *
 * No network, no browser, no providers — pure frontend behavior.
 */
"use strict";

const assert = require("assert");
const path = require("path");

const APP_PATH = path.join(__dirname, "..", "..", "app", "static", "app.js");
const LIMITS_PATH = path.join(__dirname, "..", "..", "app", "static", "limits.js");

/* --------------------------------------------------------------------- */
/* Minimal fake DOM                                                       */
/* --------------------------------------------------------------------- */
class FakeElement {
  constructor(id) {
    this.id = id;
    this.textContent = "";
    this.value = "";
    this.href = "";
    this.src = "";
    this.className = "";
    this.disabled = false;
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

function makeResponse(spec) {
  const status = spec.status || 200;
  return {
    ok: status < 400,
    status,
    json: async () => spec.body,
  };
}

function flush() {
  // Drain the microtask/macrotask queue so fire-and-forget UI work settles.
  let chain = Promise.resolve();
  for (let i = 0; i < 25; i += 1) chain = chain.then(() => new Promise((r) => setImmediate(r)));
  return chain;
}

async function runScenario(name, scenario, assertFn) {
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
    fetchCalls.push({ url, options });
    for (const [route, response] of scenario.routes) {
      if (url === route) return makeResponse(response);
    }
    return makeResponse({ status: 404, body: { detail: "not found" } });
  };
  global.FormData = class {
    append() {}
  };
  global.ABLimits = require(LIMITS_PATH);

  delete require.cache[require.resolve(APP_PATH)];
  require(APP_PATH);

  const element = (id) => global.document.getElementById(id);
  try {
    for (const fn of docListeners.DOMContentLoaded || []) await fn();
    await flush();
    await assertFn({ element, fetchCalls, store });
    console.log(`ok - ${name}`);
  } catch (error) {
    console.error(`NOT OK - ${name}: ${error.message}`);
    throw error;
  }
}

const USAGE_EXHAUSTED = {
  generation_limit: 1,
  generations_used: 1,
  generations_remaining: 0,
  max_script_characters: 3000,
};
const USAGE_AVAILABLE = {
  generation_limit: 3,
  generations_used: 1,
  generations_remaining: 2,
  max_script_characters: 3000,
};
const AUDIO_URL = "/media/projects/p1/audio/audiobook.mp3";

function savedState(overrides) {
  return {
    "audiobook.state.v1": JSON.stringify({
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
        characterCount: 5,
      },
      planKey: null,
      audio: { url: AUDIO_URL, filename: "audiobook.mp3" },
      ...(overrides || {}),
    }),
  };
}

const PROJECT_ROUTES = [
  ["/usage", { status: 200, body: USAGE_EXHAUSTED }],
  ["/projects/p1", { status: 200, body: { project_id: "p1" } }],
  ["/projects/p1/audio", { status: 200, body: { audio_url: AUDIO_URL, audio_filename: "audiobook.mp3" } }],
];

/* --------------------------------------------------------------------- */
/* Scenarios                                                              */
/* --------------------------------------------------------------------- */
(async () => {
  await runScenario(
    "exhausted allowance: usage shown, generation disabled, audiobook playable",
    { localStorage: savedState(), routes: PROJECT_ROUTES },
    ({ element }) => {
      const usageLine = element("usage-line").textContent;
      assert.ok(
        usageLine.includes("Audiobook generations remaining: 0 of 1"),
        `usage line: ${usageLine}`
      );
      assert.ok(usageLine.includes("3000"), `script maximum shown: ${usageLine}`);
      assert.strictEqual(element("char-count").textContent, "5 / 3000 characters");

      assert.strictEqual(element("btn-generate").disabled, true, "Generate must be disabled");
      assert.strictEqual(element("btn-regenerate").disabled, true, "Regenerate must be disabled");

      const note = element("quota-note");
      assert.strictEqual(note.classList.contains("hidden"), false, "message must be visible");
      assert.ok(/exhausted/i.test(note.textContent), note.textContent);
      assert.ok(/administrator/i.test(note.textContent), note.textContent);

      // Already generated audiobooks stay available.
      assert.strictEqual(element("btn-download").href, AUDIO_URL);
      assert.strictEqual(element("btn-download").classList.contains("disabled"), false);
      assert.strictEqual(element("audio-player").src, AUDIO_URL);
    }
  );

  await runScenario(
    "allowance available: generation enabled, no exhaustion message",
    {
      localStorage: savedState(),
      routes: PROJECT_ROUTES.map(([route, response]) =>
        route === "/usage" ? [route, { status: 200, body: USAGE_AVAILABLE }] : [route, response]
      ),
    },
    ({ element }) => {
      const usageLine = element("usage-line").textContent;
      assert.ok(
        usageLine.includes("Audiobook generations remaining: 2 of 3"),
        `usage line: ${usageLine}`
      );
      assert.strictEqual(element("btn-generate").disabled, false);
      assert.strictEqual(element("btn-regenerate").disabled, false);
      assert.strictEqual(element("quota-note").classList.contains("hidden"), true);
    }
  );

  await runScenario(
    "over-long script is rejected with a clear message before upload",
    {
      localStorage: {},
      routes: [
        ["/usage", { status: 200, body: { ...USAGE_AVAILABLE, max_script_characters: 3000 } }],
      ],
    },
    async ({ element, fetchCalls }) => {
      const file = {
        name: "big.txt",
        size: 4000,
        text: async () => "a".repeat(3001),
      };
      element("file-input").files = [file];
      for (const fn of element("file-input").listeners.change || []) fn();
      await flush();

      assert.ok(
        element("status").textContent.includes("too long"),
        element("status").textContent
      );
      assert.ok(
        !fetchCalls.some((call) => call.url === "/projects/from-script"),
        "an over-long script must never reach the backend from the UI"
      );
    }
  );

  console.log("all frontend app behavior checks passed");
})().catch(() => process.exit(1));
