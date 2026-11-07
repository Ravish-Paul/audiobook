/* Pure limit/usage helpers — shared by the page and the node tests.
 *
 * countCharacters() counts Unicode CODE POINTS (never UTF-16 units and
 * never UTF-8 bytes), matching Python's `len()` exactly so the live
 * frontend counter and the backend MAX_SCRIPT_CHARACTERS check always
 * agree — for Hindi, emoji and every other script. A leading BOM is
 * ignored, mirroring the backend's utf-8-sig decoding. Newlines are
 * counted as-is on both sides (no normalisation anywhere).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api; // node tests
  } else {
    root.ABLimits = api; // browser
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function countCharacters(text) {
    const clean = String(text == null ? "" : text).replace(/^\uFEFF/, "");
    return Array.from(clean).length; // code points, not UTF-16 units
  }

  function usageState(usage) {
    const safe = usage || {};
    const limit = Number.isFinite(safe.generation_limit) ? safe.generation_limit : null;
    const used = Number.isFinite(safe.generations_used) ? safe.generations_used : 0;
    const remaining = Number.isFinite(safe.generations_remaining)
      ? safe.generations_remaining
      : null;
    const maxCharacters = Number.isFinite(safe.max_script_characters)
      ? safe.max_script_characters
      : null;
    const exhausted = limit !== null && remaining === 0;
    return {
      known: limit !== null,
      limit,
      used,
      remaining,
      maxCharacters,
      exhausted,
      canGenerate: !exhausted,
    };
  }

  function exhaustedMessage(usage) {
    const s = usageState(usage);
    return (
      "Your generation allowance is exhausted (" +
      s.used +
      " of " +
      s.limit +
      " audiobook generations used). " +
      "Contact the administrator for an increased allowance."
    );
  }

  function usageMessage(usage) {
    const s = usageState(usage);
    if (!s.known) return "";
    const scriptPart =
      s.maxCharacters > 0
        ? `Script limit: ${s.maxCharacters} characters.`
        : "Script limit: unlimited.";
    return `Audiobook generations remaining: ${s.remaining} of ${s.limit}. ${scriptPart}`;
  }

  function counterText(count, maxCharacters) {
    if (Number.isFinite(maxCharacters) && maxCharacters > 0) {
      return `${count} / ${maxCharacters} characters`;
    }
    return `${count} characters`;
  }

  function exceedsCharacterLimit(count, maxCharacters) {
    return Number.isFinite(maxCharacters) && maxCharacters > 0 && count > maxCharacters;
  }

  return {
    countCharacters,
    usageState,
    exhaustedMessage,
    usageMessage,
    counterText,
    exceedsCharacterLimit,
  };
});
