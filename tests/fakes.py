"""Shared test doubles — no real MiMo / ElevenLabs / FFmpeg credits are spent.

* `FakeMiMo`      – scripted `MiMoClient` replacement (JSON strings or errors)
* `FakeComposer`  – replaces FFmpeg composition with a tiny deterministic file
* `make_plan_dict`– builds a *valid* MiMo-style plan for given paragraph texts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from app.schemas.narration import DELIVERY_STYLES, EMOTIONS
from app.services.errors import FFmpegCompositionError

FAKE_MP3 = b"ID3\x04\x00\x00fake-mp3-bytes-for-tests"
FAKE_MERGED_MP3 = b"ID3\x04\x00\x00fake-merged-audiobook"


class FakeMiMo:
    """Returns queued responses (or raises queued exceptions) in order.

    The last item is repeated, so a single-response fake behaves like a
    stable API and `[bad, good]` models a successful retry.
    """

    def __init__(self, responses: Sequence[Any]) -> None:
        if not responses:
            raise ValueError("FakeMiMo needs at least one response")
        self._responses = list(responses)
        self.calls: list[dict[str, str]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_user_prompt(self) -> str:
        return self.calls[-1]["user"]

    def complete_json(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        item = (
            self._responses.pop(0)
            if len(self._responses) > 1
            else self._responses[0]
        )
        if isinstance(item, Exception):
            raise item
        return item


class FakeComposer:
    """Writes deterministic bytes instead of invoking FFmpeg."""

    def __init__(self, payload: bytes = FAKE_MERGED_MP3, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        self.calls: list[list[Any]] = []

    @property
    def segment_counts(self) -> list[int]:
        return [len(call) for call in self.calls]

    def compose(self, segments: Sequence[Any], output_path: Path) -> None:
        self.calls.append(list(segments))
        if self.fail:
            raise FFmpegCompositionError("simulated ffmpeg failure")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(self.payload)


def make_plan_dict(
    texts: Sequence[str],
    *,
    tone: str = "suspenseful",
    language: str = "Hindi",
    base_pause_ms: int = 400,
    tts_texts: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """A valid, *varied* narration plan for exactly these paragraph texts.

    Every paragraph gets a different emotion, delivery style, intensity and
    voice settings — mirroring what a real director would produce and making
    it easy to assert that per-paragraph settings are honoured.

    `tts_text` is ALWAYS present (the planner requires it): by default it
    equals the original text (no expression needed). Pass `tts_texts` to
    attach Eleven v4 inline audio tags — each entry must reproduce the
    matching text byte-for-byte once its tags are stripped.
    """
    paragraphs: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        paragraph: dict[str, Any] = {
            "paragraph_index": index,
            "original_text": text,
            "tts_text": tts_texts[index] if tts_texts is not None else text,
            "emotion": EMOTIONS[index % len(EMOTIONS)],
            "delivery_style": DELIVERY_STYLES[index % len(DELIVERY_STYLES)],
            "intensity": (index % 5) + 1,
            "narration_direction": (
                f"Director note for paragraph {index}: vary the rhythm."
            ),
            "voice_settings": {
                "speed": round(min(1.2, 0.8 + 0.05 * index), 2),
                "stability": round(min(1.0, 0.3 + 0.1 * index), 2),
                "similarity_boost": 0.75,
                "style": round(min(1.0, 0.05 * index), 2),
                "use_speaker_boost": True,
            },
            "pause_after_ms": base_pause_ms + 100 * index,
        }
        paragraphs.append(paragraph)
    return {
        "overall_story_tone": tone,
        "language": language,
        "paragraphs": paragraphs,
    }


def plan_json(texts: Sequence[str], **kwargs: Any) -> str:
    """JSON string form of `make_plan_dict` (what MiMo would return)."""
    return json.dumps(make_plan_dict(texts, **kwargs), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Multi-voice / casting fixtures (spec §1, §7)
# --------------------------------------------------------------------------- #
# Script: paragraph 0 = narration + character dialogue, paragraph 1 = pure
# narration (with a NON-speaking character mentioned), paragraph 2 = a
# narration -> dialogue -> narration sandwich.
MV_SCRIPT = (
    "सुमन ने चाबी सौंपी और चेतावनी दी, कभी आईने में मत देखना।\n\n"
    "अचानक दरवाजे पर राहुल ने दस्तक दी।\n\n"
    "सुमन मुस्कुराई। उसने कहा, डरो मत। फिर वह चुप हो गई।"
)
MV_PARAGRAPHS = [
    "सुमन ने चाबी सौंपी और चेतावनी दी, कभी आईने में मत देखना।",
    "अचानक दरवाजे पर राहुल ने दस्तक दी।",
    "सुमन मुस्कुराई। उसने कहा, डरो मत। फिर वह चुप हो गई।",
]
NARRATOR_VOICE_ID = "voice-narrator-01"
SUMAN_VOICE_ID = "voice-suman-02"


def make_story_analysis_dict(**overrides: Any) -> dict[str, Any]:
    """A valid MiMo story/character analysis for `MV_SCRIPT` (spec §1/§2)."""
    story: dict[str, Any] = {
        "overall_tone": "suspenseful",
        "genre": "horror",
        "language": "hi",
        "locale": "hi-IN",
        "narrator": {
            "character_id": "narrator",
            "character_name": "Narrator",
            "role": "narrator",
            "description": "Omniscent storyteller.",
            "relationships": [],
            "language": "hi",
            "gender": None,
            "age": None,
            "accent": None,
            "uncertain_attributes": [],
            "voice_description": "Deep, calm, measured Hindi storytelling voice",
            "speaking_style": "Natural, restrained audiobook narration",
            "has_spoken_dialogue": True,
        },
        "characters": [
            {
                "character_id": "suman",
                "character_name": "सुमन",
                "role": "protagonist",
                "description": "Warns the visitor about the mirror.",
                "relationships": [],
                "language": "hi",
                # gender/age/accent NOT established by the script -> null.
                "gender": None,
                "age": None,
                "accent": None,
                "uncertain_attributes": [],
                "voice_description": "Soft, warm, grounded voice with quiet restraint",
                "speaking_style": "Natural, restrained, not theatrical",
                "has_spoken_dialogue": True,
            },
            {
                "character_id": "rahul",
                "character_name": "राहुल",
                "role": "minor_character",
                "description": "Knocks at the door; never speaks.",
                "relationships": [],
                "language": "hi",
                "gender": None,
                "age": None,
                "accent": None,
                "uncertain_attributes": [],
                "voice_description": "",
                "speaking_style": "",
                "has_spoken_dialogue": False,  # no speaking voice required
            },
        ],
    }
    story.update(overrides)
    return story


def make_multi_voice_plan_dict(**overrides: Any) -> dict[str, Any]:
    """A valid utterance plan for `MV_PARAGRAPHS` (spec §7 example shape).

    Words reconstruct each paragraph exactly; pauses: mid-paragraph <= 300ms,
    paragraph-final 300..1200ms; restrained tags on two dialogue utterances.
    """
    utterances = [
        {
            "utterance_id": "custom-0",
            "paragraph_index": 0,
            "sequence_index": 0,
            "speaker_id": "narrator",
            "speaker_type": "narrator",
            "original_text": "सुमन ने चाबी सौंपी और चेतावनी दी",
            "tts_text": "सुमन ने चाबी सौंपी और चेतावनी दी",
            "emotion": "neutral",
            "acting_intensity": 2,
            "delivery_style": "calm",
            "narration_direction": "Set the warning scene calmly.",
            "pause_after_ms": 200,
        },
        {
            "utterance_id": "custom-1",
            "paragraph_index": 0,
            "sequence_index": 1,
            "speaker_id": "suman",
            "speaker_type": "character",
            "original_text": "कभी आईने में मत देखना।",
            "tts_text": "[quiet, tense delivery] कभी आईने में मत देखना।",
            "emotion": "fear",
            "acting_intensity": 3,
            "delivery_style": "mysterious",
            "narration_direction": "Deliver the warning with quiet tension.",
            "pause_after_ms": 700,
        },
        {
            "utterance_id": "custom-2",
            "paragraph_index": 1,
            "sequence_index": 0,
            "speaker_id": "narrator",
            "speaker_type": "narrator",
            "original_text": "अचानक दरवाजे पर राहुल ने दस्तक दी।",
            "tts_text": "अचानक दरवाजे पर राहुल ने दस्तक दी।",
            "emotion": "suspense",
            "acting_intensity": 3,
            "delivery_style": "slow",
            "narration_direction": "Single narrator beat — keep it tense.",
            "pause_after_ms": 500,
        },
        {
            "utterance_id": "custom-3",
            "paragraph_index": 2,
            "sequence_index": 0,
            "speaker_id": "narrator",
            "speaker_type": "narrator",
            "original_text": "सुमन मुस्कुराई। उसने कहा,",
            "tts_text": "सुमन मुस्कुराई। उसने कहा,",
            "emotion": "neutral",
            "acting_intensity": 2,
            "delivery_style": "calm",
            "narration_direction": "Attribution stays with the narrator.",
            "pause_after_ms": 150,
        },
        {
            "utterance_id": "custom-4",
            "paragraph_index": 2,
            "sequence_index": 1,
            "speaker_id": "suman",
            "speaker_type": "character",
            "original_text": "डरो मत।",
            "tts_text": "[sad] डरो मत।",
            "emotion": "tenderness",
            "acting_intensity": 2,
            "delivery_style": "soft",
            "narration_direction": "Reassuring, soft.",
            "pause_after_ms": 200,
        },
        {
            "utterance_id": "custom-5",
            "paragraph_index": 2,
            "sequence_index": 2,
            "speaker_id": "narrator",
            "speaker_type": "narrator",
            "original_text": "फिर वह चुप हो गई।",
            "tts_text": "फिर वह चुप हो गई।",
            "emotion": "sadness",
            "acting_intensity": 2,
            "delivery_style": "reflective",
            "narration_direction": "Close the paragraph quietly.",
            "pause_after_ms": 700,
        },
    ]
    plan: dict[str, Any] = {
        "language": "hi",
        "overall_tone": "suspenseful",
        "utterances": utterances,
    }
    plan.update(overrides)
    return plan


def story_json(**overrides: Any) -> str:
    return json.dumps(make_story_analysis_dict(**overrides), ensure_ascii=False)


def multi_voice_plan_json(**overrides: Any) -> str:
    return json.dumps(make_multi_voice_plan_dict(**overrides), ensure_ascii=False)


class FakeCompletion:
    """Stand-in for `openai.types.chat.ChatCompletion`."""

    def __init__(self, content: str | None) -> None:
        message = type("Msg", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        self.choices = [choice]


class FakeOpenAIClient:
    """Duck-typed `OpenAI` client that records the request kwargs."""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result
        self._error = error
        outer = self

        class _Completions:
            def create(self, **kwargs: Any) -> Any:
                outer.calls.append(kwargs)
                if outer._error is not None:
                    raise outer._error
                return outer._result

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()
