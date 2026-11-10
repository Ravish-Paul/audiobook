"""User-controlled narration preferences (the anti-overacting contract).

These preferences are NOT a frontend-only setting: they are sent with
`POST /projects/{id}/analyze-narration`, persisted per project, embedded in
`narration_plan.json`, forwarded verbatim to MiMo, enforced afterwards by
`NarrationPolicyValidator` and included in the narration plan hash (so
changing them invalidates the previous plan's segments).

Defaults match the Narration Preferences panel:

    preset             natural
    acting_intensity   2        (1..5)
    whisper_policy     user_selected   ("only user-selected lines")
    tag_density        minimal
    custom_direction   ""  (free text, folded into the MiMo prompt)

Global narration profiles
-------------------------
The twelve profiles below are the *global creative direction*. They are
deliberately distinct: each one changes the MiMo prompt (scene priorities,
emotional boundaries and marker budgets), not just a label in the browser.
The legacy presets (`restrained_cinematic`, `expressive`, `dramatic`, `soft`,
`smooth`) stay accepted so saved plans and old `localStorage` selections keep
working; the UI only exposes the twelve profiles and automatic mode.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

NarrationPreset = Literal[
    # The twelve user-facing profiles.
    "natural",
    "cinematic",
    "horror",
    "romantic",
    "sensual",
    "emotional",
    "action",
    "mystery",
    "fantasy",
    "comedy",
    "professional",
    "children",
    # Automatic direction (MiMo analyses the whole story and picks a profile).
    "auto",
    # Legacy identifiers — accepted for backward compatibility only.
    "restrained_cinematic",
    "expressive",
    "dramatic",
    "soft",
    "smooth",
]
WhisperPolicy = Literal["never", "user_selected", "rare"]
TagDensity = Literal["minimal", "standard", "expressive"]

# The twelve profiles the Narration Director exposes.
NARRATION_PROFILES: tuple[str, ...] = (
    "natural",
    "cinematic",
    "horror",
    "romantic",
    "sensual",
    "emotional",
    "action",
    "mystery",
    "fantasy",
    "comedy",
    "professional",
    "children",
)

# Every accepted preset id (profiles + automatic + legacy).
NARRATION_PRESETS: tuple[str, ...] = NARRATION_PROFILES + (
    "auto",
    "restrained_cinematic",
    "expressive",
    "dramatic",
    "soft",
    "smooth",
)
WHISPER_POLICIES: tuple[str, ...] = ("never", "user_selected", "rare")
TAG_DENSITIES: tuple[str, ...] = ("minimal", "standard", "expressive")

# Human labels used by the UI and by the MiMo prompt.
PRESET_LABELS: dict[str, str] = {
    "natural": "Natural & Conversational",
    "cinematic": "Cinematic & Immersive",
    "horror": "Horror & Suspense",
    "romantic": "Romantic & Intimate",
    "sensual": "Sensual & Intimate",
    "emotional": "Emotional & Reflective",
    "action": "Action & Adventure",
    "mystery": "Mystery & Thriller",
    "fantasy": "Fantasy & Epic",
    "comedy": "Comedy & Lighthearted",
    "professional": "Calm & Professional",
    "children": "Children's Story",
    "auto": "AI Automatic",
    # Legacy labels (never shown as new profiles).
    "restrained_cinematic": "Restrained Cinematic",
    "expressive": "Expressive",
    "dramatic": "Dramatic",
    "soft": "Soft",
    "smooth": "Smooth",
}

# Short, user-facing descriptions for the profile cards.
PROFILE_DESCRIPTIONS: dict[str, str] = {
    "natural": "Clear, conversational and realistic storytelling.",
    "cinematic": "Atmospheric, immersive and dramatically controlled.",
    "horror": "Gradual tension, unease and suspense.",
    "romantic": "Warm, affectionate and emotionally close.",
    "sensual": "Close, subtle intimacy only where the scene allows it.",
    "emotional": "Sincere, reflective and gently emotional.",
    "action": "Energetic, urgent and sharply focused.",
    "mystery": "Measured, curious and quietly tense.",
    "fantasy": "Atmospheric, wondrous and epic in scale.",
    "comedy": "Natural timing with playful lightness.",
    "professional": "Clear, steady and composed.",
    "children": "Warm, clear and friendly storytelling.",
}

# The actual planning guidance. Each profile must produce a DIFFERENT plan:
# this text is rendered into the MiMo system prompt and reflected in the
# policy marker budgets (`PRESET_MARKER_FRACTION`).
PROFILE_GUIDANCE: dict[str, str] = {
    "natural": (
        "natural, conversational human storytelling; clear realistic phrasing "
        "and a moderate emotional range; do not perform at the listener. "
        "Ordinary narration stays ordinary and heightens only when the scene "
        "genuinely earns it; no theatrical extremes, shouting, trembling or "
        "breathless delivery"
    ),
    "cinematic": (
        "cinematic and immersive storytelling: atmosphere, controlled drama, "
        "meaningful pauses and strong scene transitions; build emotion "
        "gradually instead of peaking constantly. Exposition and quiet beats "
        "stay restrained; only key scenes carry larger dramatic weight"
    ),
    "horror": (
        "horror and suspense: begin grounded and calm, then let unease grow "
        "through pacing, silence and small details. Fear, dread or a rare "
        "hushed line belongs only where the scene earns it. Never make every "
        "paragraph frightening; no automatic whispering, screaming or "
        "theatrical dread"
    ),
    "romantic": (
        "romantic and intimate storytelling: warm, gentle, affectionate and "
        "emotionally close; allow playful lightness, tenderness, vulnerability "
        "and honest conflict. A romance must not turn every sentence into a "
        "declaration — neutral narration, action and ordinary dialogue stay "
        "natural"
    ),
    "sensual": (
        "adult romantic fiction with a close, warm, sensual, intimate, velvety "
        "delivery only where the text is genuinely intimate; subtle romantic "
        "tension, gentle controlled pacing and emotionally sincere expression. "
        "NEVER force whispering, breathiness or seductive delivery into "
        "unrelated or neutral scenes; never add explicit content that is "
        "absent from the source; ordinary romance, flirting, arguments and "
        "calm narration keep their natural delivery"
    ),
    "emotional": (
        "emotional and reflective storytelling: sincere, restrained emotion, "
        "tender pauses and controlled sadness or grief; allow hope and gradual "
        "recovery. Avoid melodrama, sobbing, exaggerated crying and a sad tone "
        "on every paragraph; quiet or neutral scenes stay quiet"
    ),
    "action": (
        "action and adventure storytelling: energetic and focused with real "
        "urgency during action and controlled intensity at peaks. Explanations, "
        "travel, planning and transitions stay calm and clear; avoid constant "
        "shouting or a permanently high intensity"
    ),
    "mystery": (
        "mystery and thriller storytelling: curious, measured and quietly "
        "tense; reveal information gradually and let suspense breathe. Not "
        "every line is a threat — ordinary investigation, dialogue and "
        "explanation remain calm and engaging"
    ),
    "fantasy": (
        "fantasy and epic storytelling: atmospheric, wonder-filled and large "
        "in scope, with expressive character dialogue and controlled grandeur. "
        "Everyday scenes stay grounded; do not apply epic intensity to every "
        "paragraph"
    ),
    "comedy": (
        "comedy and lighthearted storytelling: natural conversational timing, "
        "playful expression and well-placed comic pauses; keep the humour in "
        "the writing. Avoid cartoonish overacting, nonstop jokes and wacky "
        "voices unless the script explicitly asks for them"
    ),
    "professional": (
        "calm, professional storytelling for educational, documentary and "
        "nonfiction content: clear, steady, composed and consistent "
        "pronunciation with minimal unnecessary emotional variation; even "
        "emphasis and no performance"
    ),
    "children": (
        "children's storytelling: warm, friendly, clear and expressive with "
        "age-appropriate character delivery and gentle excitement; prioritise "
        "clear pronunciation. Avoid exaggerated cartoon voices unless the "
        "script explicitly asks for them"
    ),
    # Legacy profiles keep their historical meaning exactly.
    "restrained_cinematic": (
        "restrained cinematic tone: atmosphere over performance; measured, "
        "polished and still natural"
    ),
    "expressive": (
        "expressive and lively where the story earns it, but never overacted"
    ),
    "dramatic": (
        "heightened dramatic staging is welcome; strong emphasis is allowed "
        "where the scene supports it"
    ),
    "soft": (
        "soft, gentle, warm and intimate delivery; tender and quiet with a "
        "human closeness; restrained tags like [softly], [gently] or [tender] "
        "only when they genuinely help; never loud or theatrical"
    ),
    "smooth": (
        "smooth, flowing, even delivery; polished, effortless and unhurried; "
        "an even rhythm with graceful transitions; no sudden spikes in volume "
        "or emotion"
    ),
    "auto": (
        "AI-automatic mood — YOU are the director: first weigh the whole "
        "story, then choose each paragraph's mood and delivery to fit that "
        "specific moment. Choose the global direction that fits the story "
        "(natural, cinematic, horror, romantic, sensual, emotional, action, "
        "mystery, fantasy, comedy, professional or children) instead of "
        "applying one mood everywhere. Let the mood change as the story "
        "changes; keep neutral narration neutral, stay restrained by default "
        "and heighten only where the scene earns it"
    ),
}

WHISPER_POLICY_LABELS: dict[str, str] = {
    "never": "Never",
    "user_selected": "Only user-selected lines",
    "rare": "AI may use rarely when essential",
}
TAG_DENSITY_LABELS: dict[str, str] = {
    "minimal": "Minimal",
    "standard": "Standard",
    "expressive": "Expressive",
}

# Normalisation helpers: MiMo may echo the human label instead of the id.
_LABEL_TO_PRESET: dict[str, str] = {
    label.casefold(): preset for preset, label in PRESET_LABELS.items()
}
_LABEL_TO_PRESET.update(
    {
        "natural & conversational": "natural",
        "cinematic & immersive": "cinematic",
        "horror & suspense": "horror",
        "romantic & intimate": "romantic",
        "sensual & intimate": "sensual",
        "emotional & reflective": "emotional",
        "action & adventure": "action",
        "mystery & thriller": "mystery",
        "fantasy & epic": "fantasy",
        "comedy & lighthearted": "comedy",
        "calm & professional": "professional",
        "children's story": "children",
        "childrens story": "children",
    }
)


def normalize_profile_id(value: str) -> Optional[str]:
    """Map an id or a human label onto one of the twelve concrete profiles.

    Returns `None` for anything else, including `auto` and the legacy presets:
    a story analysis must recommend a real global profile, never itself.
    """
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.replace("_", " ").split()).strip()
    lowered = cleaned.casefold()
    if lowered in NARRATION_PROFILES:
        return lowered
    return _LABEL_TO_PRESET.get(lowered)


class NarrationPreferences(BaseModel):
    """The complete preference object sent by the frontend."""

    model_config = ConfigDict(extra="forbid")

    preset: NarrationPreset = Field(
        default="natural",
        description=(
            "Global creative direction: one of the twelve narration profiles "
            "or `auto` (MiMo analyses the story and chooses). Legacy preset "
            "ids remain accepted for older saved plans."
        ),
    )
    acting_intensity: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum intensity allowed for ANY paragraph (1..5).",
    )
    whisper_policy: WhisperPolicy = Field(
        default="user_selected",
        description=(
            "never = no whispering at all; user_selected = only the lines "
            "the user picked; rare = MiMo may whisper very occasionally."
        ),
    )
    whisper_paragraphs: list[int] = Field(
        default_factory=list,
        description=(
            "0-based paragraph indexes the user explicitly allows to be "
            "whispered (implements 'only user-selected lines')."
        ),
    )
    tag_density: TagDensity = Field(
        default="minimal",
        description=(
            "Maximum inline audio tags per paragraph: minimal=1, "
            "standard=3, expressive=8."
        ),
    )
    custom_direction: str = Field(
        default="",
        max_length=1000,
        description="Free-text direction added to the MiMo prompt verbatim.",
    )

    @model_validator(mode="after")
    def _clean_whisper_paragraphs(self) -> "NarrationPreferences":
        """Indexes are unique, non-negative and ordered (stable hashing)."""
        cleaned = sorted({int(index) for index in self.whisper_paragraphs})
        if any(index < 0 for index in cleaned):
            raise ValueError("whisper_paragraphs must not contain negative indexes")
        if cleaned != list(self.whisper_paragraphs):
            self.whisper_paragraphs = cleaned
        return self


class NarrationPreferencesResponse(BaseModel):
    """GET/PUT /projects/{project_id}/narration-preferences."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    preferences: NarrationPreferences
    saved: bool = Field(
        description=(
            "True when a preferences object was stored for this project; "
            "False when these are simply the defaults."
        )
    )
    saved_at: Optional[datetime] = None


class AnalyzeNarrationRequest(BaseModel):
    """Optional body of `POST /projects/{project_id}/analyze-narration`."""

    model_config = ConfigDict(extra="forbid")

    preferences: Optional[NarrationPreferences] = Field(
        default=None,
        description=(
            "Complete preferences object. When present it is persisted for "
            "the project and enforced on the generated plan."
        ),
    )


class PreviewNarrationRequest(BaseModel):
    """Body of `POST /projects/{project_id}/analyze-narration/preview`."""

    model_config = ConfigDict(extra="forbid")

    paragraph_index: int = Field(
        ge=0, description="0-based paragraph of the uploaded script to preview."
    )
    preferences: Optional[NarrationPreferences] = Field(
        default=None,
        description="Optional preferences to apply (and persist) for the preview.",
    )


class PreviewNarrationAudioRequest(BaseModel):
    """Body of `POST /projects/{project_id}/narration-preview/audio`.

    A short, representative line is synthesised on explicit user action. The
    result is returned directly (never written to the project's audiobook
    files) and the endpoint is classified as an auxiliary provider call, so
    it never consumes a complete-audiobook allowance.
    """

    model_config = ConfigDict(extra="forbid")

    paragraph_index: int = Field(
        ge=0, description="0-based paragraph of the uploaded script to preview."
    )
    preferences: Optional[NarrationPreferences] = Field(
        default=None,
        description="Optional preferences to apply (and persist) for the preview.",
    )
