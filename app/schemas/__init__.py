"""Pydantic schemas exposed by the API."""

from app.schemas.audio import AudioResponse
from app.schemas.casting import (
    AuditionRequest,
    CharacterAnalysisResponse,
    CharacterCast,
    CharacterProfile,
    SelectVoiceRequest,
    SpeakerCasting,
    StoryAnalysis,
    VoiceCandidate,
    VoiceCastingResponse,
    VoiceLock,
)
from app.schemas.elevenlabs import (
    AudioEffectsBackgroundNoise,
    AudioEffectsCatalog,
    AudioEffectsEnvironment,
    AudioEffectsFilterPreset,
)
from app.schemas.multi_voice import (
    MultiVoicePlan,
    MultiVoicePlanResponse,
    Utterance,
)
from app.schemas.narration import (
    DELIVERY_STYLES,
    EMOTIONS,
    DeliveryStyle,
    Emotion,
    NarrationPlan,
    NarrationPlanResponse,
    ParagraphPlan,
    VoiceSettings,
)
from app.schemas.project import (
    HealthResponse,
    ProjectCreateRequest,
    ProjectResponse,
    ScriptInfo,
    UploadScriptResponse,
    WelcomeResponse,
)

__all__ = [
    "DELIVERY_STYLES",
    "EMOTIONS",
    "AudioEffectsBackgroundNoise",
    "AudioEffectsCatalog",
    "AudioEffectsEnvironment",
    "AudioEffectsFilterPreset",
    "AudioResponse",
    "AuditionRequest",
    "CharacterAnalysisResponse",
    "CharacterCast",
    "CharacterProfile",
    "DeliveryStyle",
    "Emotion",
    "HealthResponse",
    "MultiVoicePlan",
    "MultiVoicePlanResponse",
    "NarrationPlan",
    "NarrationPlanResponse",
    "ParagraphPlan",
    "ProjectCreateRequest",
    "ProjectResponse",
    "ScriptInfo",
    "SelectVoiceRequest",
    "SpeakerCasting",
    "StoryAnalysis",
    "UploadScriptResponse",
    "Utterance",
    "VoiceCastingResponse",
    "VoiceCandidate",
    "VoiceLock",
    "VoiceSettings",
    "WelcomeResponse",
]
