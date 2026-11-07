/* Frontend limit helpers — run by pytest (see tests/test_generation_limits.py).
 *
 * Everything here is pure logic: no network, no browser, no providers.
 * The `SAMPLES=` line at the end lets the Python side compare these counts
 * against Python's `len()` for the exact same strings, proving that the
 * live frontend counter and the backend MAX_SCRIPT_CHARACTERS check can
 * never disagree (Unicode code points, not UTF-16 units or UTF-8 bytes).
 */
"use strict";

const assert = require("assert");
const path = require("path");

const limits = require(path.join(__dirname, "..", "..", "app", "static", "limits.js"));

let failures = 0;
function check(name, fn) {
  try {
    fn();
    console.log(`ok - ${name}`);
  } catch (error) {
    failures += 1;
    console.error(`NOT OK - ${name}: ${error.message}`);
  }
}

/* ---- countCharacters: code points, BOM, newlines ---------------------- */
check("ASCII counts one per character", () => {
  assert.strictEqual(limits.countCharacters("a".repeat(3000)), 3000);
});

check("exactly 3000 is accepted, 3001 exceeds", () => {
  assert.strictEqual(limits.exceedsCharacterLimit(3000, 3000), false);
  assert.strictEqual(limits.exceedsCharacterLimit(3001, 3000), true);
});

check("Hindi counts code points (never UTF-8 bytes)", () => {
  const hindi = "क".repeat(3000);
  assert.strictEqual(hindi.length, 3000); // JS units == code points here
  assert.strictEqual(limits.countCharacters(hindi), 3000);
  assert.strictEqual(Buffer.byteLength(hindi, "utf8"), 9000); // bytes would be wrong
});

check("emoji (supplementary plane) count as ONE character each", () => {
  const emoji = "😀".repeat(3000);
  assert.strictEqual(emoji.length, 6000); // UTF-16 units — the classic bug
  assert.strictEqual(limits.countCharacters(emoji), 3000); // code points
});

check("astral + BMP mix matches code-point counting", () => {
  assert.strictEqual(limits.countCharacters("a😀b"), 3);
  assert.strictEqual(limits.countCharacters("नमस्ते 😀"), 8);
});

check("newline handling: \r\n counts as two, like Python", () => {
  assert.strictEqual(limits.countCharacters("a".repeat(2998) + "\r\n"), 3000);
  assert.strictEqual(limits.countCharacters("a".repeat(2998) + "\n"), 2999);
});

check("a leading BOM is ignored (utf-8-sig parity)", () => {
  assert.strictEqual(limits.countCharacters("\uFEFF" + "a".repeat(3000)), 3000);
  assert.strictEqual(limits.countCharacters("a\uFEFFa"), 3); // only a leading BOM
});

check("null/undefined count as empty", () => {
  assert.strictEqual(limits.countCharacters(null), 0);
  assert.strictEqual(limits.countCharacters(undefined), 0);
});

/* ---- usageState / messages ------------------------------------------- */
check("remaining 0 is exhausted: generate must be disabled", () => {
  const s = limits.usageState({
    generation_limit: 1,
    generations_used: 1,
    generations_remaining: 0,
    max_script_characters: 3000,
  });
  assert.strictEqual(s.exhausted, true);
  assert.strictEqual(s.canGenerate, false);
});

check("remaining > 0 keeps generation enabled", () => {
  const s = limits.usageState({
    generation_limit: 3,
    generations_used: 1,
    generations_remaining: 2,
    max_script_characters: 3000,
  });
  assert.strictEqual(s.exhausted, false);
  assert.strictEqual(s.canGenerate, true);
});

check("zero limit is exhausted", () => {
  const s = limits.usageState({
    generation_limit: 0,
    generations_used: 0,
    generations_remaining: 0,
    max_script_characters: 3000,
  });
  assert.strictEqual(s.exhausted, true);
});

check("exhausted message explains the administrator route", () => {
  const message = limits.exhaustedMessage({
    generation_limit: 1,
    generations_used: 1,
    generations_remaining: 0,
    max_script_characters: 3000,
  });
  assert.ok(/exhausted/i.test(message), message);
  assert.ok(/administrator/i.test(message), message);
});

check("usage message shows remaining, limit and script maximum", () => {
  const message = limits.usageMessage({
    generation_limit: 3,
    generations_used: 1,
    generations_remaining: 2,
    max_script_characters: 3000,
  });
  assert.ok(message.includes("Audiobook generations remaining: 2 of 3"), message);
  assert.ok(message.includes("3000"), message);
});

check("counter text shows used / maximum", () => {
  assert.strictEqual(limits.counterText(42, 3000), "42 / 3000 characters");
  assert.strictEqual(limits.counterText(42, 0), "42 characters");
});

/* ---- machine-readable samples for the Python cross-check -------------- */
const SAMPLES = {
  ascii_3000: "a".repeat(3000),
  hindi_3000: "क".repeat(3000),
  emoji_3000: "😀".repeat(3000),
  mixed: "नमस्ते 😀 world — कहानी",
  crlf_3000: "a".repeat(2998) + "\r\n",
  lf_2999: "a".repeat(2998) + "\n",
  bom_3000: "\uFEFF" + "a".repeat(3000),
  zwj_family: "👨‍👩‍👧",
};

const counts = {};
for (const [name, value] of Object.entries(SAMPLES)) {
  counts[name] = limits.countCharacters(value);
}
console.log("SAMPLES=" + JSON.stringify(counts));

if (failures) {
  console.error(`${failures} frontend limit check(s) failed`);
  process.exit(1);
}
console.log("all frontend limit checks passed");
