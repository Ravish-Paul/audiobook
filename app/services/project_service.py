"""Project service: owns the on-disk layout and all business rules.

Storage layout:

    storage/projects/
      {project_id}/
        project.json     -> metadata (name, description, created_at, script, audio)
        script.txt       -> the uploaded script (always stored under this name)
        audio/
          audiobook.mp3  -> generated narration
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import Settings, get_settings
from app.schemas.casting import CharacterCast
from app.schemas.multi_voice import MultiVoicePlan
from app.schemas.narration import NarrationPlan
from app.schemas.narration_preferences import NarrationPreferences
from app.schemas.project import (
    ProjectCreateRequest,
    ProjectResponse,
    ScriptImportResponse,
    ScriptInfo,
    UploadScriptResponse,
)
from app.services.errors import (
    AudioSaveError,
    EmptyScriptError,
    ProjectError,
    ProjectNotFoundError,
    ScriptNotFoundError,
    ScriptTooLargeError,
    ScriptValidationError,
)
from app.services.script_import import (
    detect_script_language,
    detect_title,
    script_preview,
)
from app.services.script_limits import check_script_limits

METADATA_FILENAME = "project.json"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProjectService:
    """CRUD + script upload for projects, backed by local storage."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ #
    # Storage helpers
    # ------------------------------------------------------------------ #
    @property
    def storage_dir(self) -> Path:
        return self.settings.storage_dir

    def ensure_storage(self) -> Path:
        """Create the storage tree if it does not exist yet."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        return self.storage_dir

    def _project_dir(self, project_id: str) -> Path:
        return self.storage_dir / project_id

    def _metadata_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / METADATA_FILENAME

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _write_metadata(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # Projects
    # ------------------------------------------------------------------ #
    def create_project(
        self, payload: ProjectCreateRequest, owner: Optional[str] = None
    ) -> ProjectResponse:
        self.ensure_storage()

        project_id = str(uuid.uuid4())
        now = _utcnow()
        record: dict[str, Any] = {
            "project_id": project_id,
            "name": payload.name.strip(),
            "description": payload.description,
            "created_at": now.isoformat(),
            "script": None,
            # Private beta: which tester owns this project (None in dev mode).
            "owner": owner,
        }
        self._write_metadata(self._metadata_path(project_id), record)
        return self._to_response(record)

    def get_project(self, project_id: str) -> ProjectResponse:
        meta_path = self._metadata_path(project_id)
        if not project_id or not meta_path.is_file():
            raise ProjectNotFoundError(project_id)

        try:
            record = self._read_metadata(meta_path)
        except (json.JSONDecodeError, OSError) as exc:
            raise ProjectNotFoundError(project_id) from exc

        return self._to_response(record)

    def project_exists(self, project_id: str) -> bool:
        return bool(project_id) and self._metadata_path(project_id).is_file()

    # ------------------------------------------------------------------ #
    # Deletion (private-beta privacy: testers can delete their data)
    # ------------------------------------------------------------------ #
    def delete_project(self, project_id: str) -> None:
        """Remove one project and every file it owns (404 when unknown)."""
        if not project_id or not self._metadata_path(project_id).is_file():
            raise ProjectNotFoundError(project_id)

        project_dir = self._project_dir(project_id)
        try:
            shutil.rmtree(project_dir)
        except OSError as exc:
            raise ProjectError(
                f"Could not delete project '{project_id}': {exc}"
            ) from exc

    def delete_projects_owned_by(self, owner: Optional[str]) -> int:
        """Delete every project owned by `owner`; returns how many were removed."""
        if not owner:
            return 0
        self.ensure_storage()
        removed = 0
        for entry in self.storage_dir.iterdir():
            if not entry.is_dir():
                continue
            meta_path = entry / METADATA_FILENAME
            if not meta_path.is_file():
                continue
            try:
                record = self._read_metadata(meta_path)
            except (json.JSONDecodeError, OSError):
                continue
            if record.get("owner") != owner:
                continue
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        return removed

    def list_projects_owned_by(self, owner: Optional[str]) -> list[ProjectResponse]:
        """Every project owned by `owner`, newest first — never anyone else's.

        Lets the frontend restore the tester's own project after the browser
        lost its localStorage (same enumeration + owner filter as the
        deletion path above).
        """
        if not owner:
            return []
        self.ensure_storage()
        projects: list[ProjectResponse] = []
        for entry in self.storage_dir.iterdir():
            if not entry.is_dir():
                continue
            meta_path = entry / METADATA_FILENAME
            if not meta_path.is_file():
                continue
            try:
                record = self._read_metadata(meta_path)
            except (json.JSONDecodeError, OSError):
                continue
            if record.get("owner") != owner:
                continue
            try:
                projects.append(self._to_response(record))
            except (ValueError, TypeError, ProjectError):
                continue  # a corrupt record is never listed
        projects.sort(key=lambda p: p.created_at, reverse=True)
        return projects

    # ------------------------------------------------------------------ #
    # Script upload
    # ------------------------------------------------------------------ #
    def upload_script(
        self, project_id: str, filename: str, content: bytes
    ) -> UploadScriptResponse:
        """Validate and store an uploaded script for an existing project."""
        project = self.get_project(project_id)  # raises 404 when unknown

        text = self._validate_and_decode(filename, content)
        script_info = self._store_script(project_id, filename, text)

        script_path = (
            self._project_dir(project_id) / self.settings.stored_script_filename
        )
        return UploadScriptResponse(
            project_id=project.project_id,
            filename=script_info["filename"],
            character_count=script_info["character_count"],
            upload_status="success",
            size_bytes=script_info["size_bytes"],
            stored_path=str(script_path.relative_to(self.storage_dir.parent.parent)),
            uploaded_at=datetime.fromisoformat(script_info["uploaded_at"]),
        )

    def import_script(
        self, filename: str, content: bytes, owner: Optional[str] = None
    ) -> ScriptImportResponse:
        """One-step import for the simple creator UI (auto-created project).

        Validates the upload exactly like `upload_script`, derives the book
        title and language from the text (deterministic — never an LLM call),
        creates the project and stores the script. All done in that order so
        a rejected file never leaves an orphan project behind.
        """
        text = self._validate_and_decode(filename, content)

        title = detect_title(text, filename)
        language = detect_script_language(text)

        project = self.create_project(ProjectCreateRequest(name=title), owner=owner)
        script_info = self._store_script(project.project_id, filename, text)

        # Lazy import: narration_planner imports this module at import time.
        from app.services.narration_planner import split_paragraphs

        return ScriptImportResponse(
            project_id=project.project_id,
            title=title,
            language=language.name,
            language_code=language.code or None,
            paragraph_count=len(split_paragraphs(text)),
            filename=script_info["filename"],
            size_bytes=script_info["size_bytes"],
            character_count=script_info["character_count"],
            script_preview=script_preview(text),
            upload_status="success",
            uploaded_at=datetime.fromisoformat(script_info["uploaded_at"]),
        )

    def _validate_and_decode(self, filename: str, content: bytes) -> str:
        """Shared upload validation (size, extension, UTF-8) -> decoded text.

        Raises ScriptTooLargeError (413) / ScriptValidationError (400).
        """
        # 1) size limit (defence in depth: the route also checks Content-Length)
        max_bytes = self.settings.max_script_size_bytes
        if len(content) > max_bytes:
            raise ScriptTooLargeError(len(content), max_bytes)

        # 2) extension whitelist
        ext = Path(filename or "").suffix.lower()
        if ext not in self.settings.allowed_script_extensions:
            allowed = ", ".join(self.settings.allowed_script_extensions)
            raise ScriptValidationError(
                f"Unsupported file type '{ext or filename}'. Allowed: {allowed}"
            )

        # 3) UTF-8 decoding (utf-8-sig also tolerates an optional BOM)
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ScriptValidationError(
                "Script must be a valid UTF-8 encoded text file"
            ) from exc

        # 4) private-beta length / estimated-duration limits
        check_script_limits(self.settings, text)
        return text

    def _store_script(
        self, project_id: str, filename: str, text: str
    ) -> dict[str, Any]:
        """Persist `script.txt` + metadata; returns the `script_info` record."""
        project_dir = self._project_dir(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        script_path = project_dir / self.settings.stored_script_filename
        # Store the decoded text so the file on disk is guaranteed UTF-8.
        script_path.write_text(text, encoding="utf-8")

        uploaded_at = _utcnow()
        script_info: dict[str, Any] = {
            "filename": Path(filename).name,
            "stored_as": script_path.name,
            "size_bytes": script_path.stat().st_size,
            "character_count": len(text),
            "uploaded_at": uploaded_at.isoformat(),
        }

        meta_path = self._metadata_path(project_id)
        record = self._read_metadata(meta_path)
        record["script"] = script_info
        self._write_metadata(meta_path, record)
        return script_info

    # ------------------------------------------------------------------ #
    # Script reading (used by audio generation — no re-upload required)
    # ------------------------------------------------------------------ #
    def read_script(self, project_id: str) -> str:
        """Return the stored script text for TTS.

        Raises:
            ProjectNotFoundError  (404) unknown project
            ScriptNotFoundError  (404) no script.txt uploaded yet
            ScriptValidationError (400) stored file is not valid UTF-8
            EmptyScriptError      (400) script has no speakable text
        """
        self.get_project(project_id)  # raises 404 when unknown

        script_path = (
            self._project_dir(project_id) / self.settings.stored_script_filename
        )
        if not script_path.is_file():
            raise ScriptNotFoundError(project_id)

        try:
            text = script_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ScriptValidationError(
                "The stored script is not valid UTF-8 and cannot be synthesised"
            ) from exc
        except OSError as exc:
            raise ProjectError(f"Could not read the stored script: {exc}") from exc

        if not text.strip():
            raise EmptyScriptError(project_id)
        return text

    # ------------------------------------------------------------------ #
    # Audio (MP3) storage
    # ------------------------------------------------------------------ #
    def audio_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / self.settings.audio_dir_name

    def audio_path(self, project_id: str) -> Path:
        """storage/projects/{project_id}/audio/audiobook.mp3"""
        return self.audio_dir(project_id) / self.settings.audio_filename

    def get_audio_metadata(self, project_id: str) -> Optional[dict[str, Any]]:
        """Audio info recorded in project.json, or None when never generated."""
        meta_path = self._metadata_path(project_id)
        if not meta_path.is_file():
            return None
        try:
            record = self._read_metadata(meta_path)
        except (json.JSONDecodeError, OSError):
            return None
        audio = record.get("audio")
        return dict(audio) if isinstance(audio, dict) else None

    def save_audio_metadata(self, project_id: str, info: dict[str, Any]) -> None:
        """Persist audio info alongside the existing project metadata."""
        meta_path = self._metadata_path(project_id)
        if not meta_path.is_file():
            raise ProjectNotFoundError(project_id)
        record = self._read_metadata(meta_path)
        record["audio"] = info
        self._write_metadata(meta_path, record)

    # ------------------------------------------------------------------ #
    # Narration plan (MiMo output)
    # ------------------------------------------------------------------ #
    def narration_plan_path(self, project_id: str) -> Path:
        """storage/projects/{project_id}/narration_plan.json"""
        return (
            self._project_dir(project_id) / self.settings.narration_plan_filename
        )

    def save_narration_plan(
        self, project_id: str, plan: NarrationPlan
    ) -> datetime:
        """Write narration_plan.json and return when it was saved."""
        if not self.project_exists(project_id):
            raise ProjectNotFoundError(project_id)

        saved_at = _utcnow()
        payload: dict[str, Any] = dict(plan.model_dump(mode="json"))
        payload["project_id"] = project_id
        payload["saved_at"] = saved_at.isoformat()

        path = self.narration_plan_path(project_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise ProjectError(
                f"Could not save the narration plan: {exc}"
            ) from exc
        return saved_at

    def get_narration_plan(
        self, project_id: str
    ) -> Optional[tuple[NarrationPlan, Optional[datetime]]]:
        """Load narration_plan.json -> (plan, saved_at) or None when absent."""
        path = self.narration_plan_path(project_id)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

        saved_raw = data.get("saved_at") if isinstance(data, dict) else None
        saved_at = (
            datetime.fromisoformat(saved_raw) if isinstance(saved_raw, str) else None
        )
        try:
            # Extra keys (project_id, saved_at) are ignored by the schema.
            plan = NarrationPlan.model_validate(data)
        except Exception:
            return None
        return plan, saved_at

    # ------------------------------------------------------------------ #
    # Narration preferences (user controlled direction, per project)
    # ------------------------------------------------------------------ #
    def narration_preferences_path(self, project_id: str) -> Path:
        """storage/projects/{project_id}/narration_preferences.json"""
        return (
            self._project_dir(project_id)
            / self.settings.narration_preferences_filename
        )

    def save_narration_preferences(
        self, project_id: str, preferences: NarrationPreferences
    ) -> datetime:
        """Persist the user's narration preferences and return when saved."""
        if not self.project_exists(project_id):
            raise ProjectNotFoundError(project_id)

        saved_at = _utcnow()
        payload: dict[str, Any] = {
            # Nested (not flat): NarrationPreferences uses extra="forbid".
            "preferences": preferences.model_dump(mode="json"),
            "project_id": project_id,
            "saved_at": saved_at.isoformat(),
        }

        path = self.narration_preferences_path(project_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise ProjectError(
                f"Could not save the narration preferences: {exc}"
            ) from exc
        return saved_at

    def get_narration_preferences(
        self, project_id: str
    ) -> Optional[tuple[NarrationPreferences, Optional[datetime]]]:
        """Load narration_preferences.json -> (preferences, saved_at) or None."""
        path = self.narration_preferences_path(project_id)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

        saved_raw = data.get("saved_at") if isinstance(data, dict) else None
        saved_at = (
            datetime.fromisoformat(saved_raw) if isinstance(saved_raw, str) else None
        )
        try:
            preferences = NarrationPreferences.model_validate(
                data.get("preferences", data) if isinstance(data, dict) else data
            )
        except Exception:
            return None
        return preferences, saved_at

    # ------------------------------------------------------------------ #
    # Character cast (AI casting pipeline)
    # ------------------------------------------------------------------ #
    def character_cast_path(self, project_id: str) -> Path:
        """storage/projects/{project_id}/character_cast.json"""
        return self._project_dir(project_id) / self.settings.character_cast_filename

    def save_character_cast(
        self, project_id: str, cast: CharacterCast
    ) -> datetime:
        """Write character_cast.json and return when it was saved."""
        if not self.project_exists(project_id):
            raise ProjectNotFoundError(project_id)

        saved_at = _utcnow()
        payload: dict[str, Any] = dict(cast.model_dump(mode="json"))
        payload["project_id"] = project_id
        payload["created_at"] = (
            cast.created_at.isoformat() if cast.created_at else saved_at.isoformat()
        )
        payload["updated_at"] = saved_at.isoformat()

        path = self.character_cast_path(project_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise ProjectError(f"Could not save the character cast: {exc}") from exc
        return saved_at

    def get_character_cast(self, project_id: str) -> Optional[CharacterCast]:
        """Load character_cast.json or None when absent/corrupt."""
        path = self.character_cast_path(project_id)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
        try:
            return CharacterCast.model_validate(data)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Multi-voice (utterance) plan
    # ------------------------------------------------------------------ #
    def multi_voice_plan_path(self, project_id: str) -> Path:
        """storage/projects/{project_id}/multi_voice_plan.json"""
        return self._project_dir(project_id) / self.settings.multi_voice_plan_filename

    def save_multi_voice_plan(
        self, project_id: str, plan: MultiVoicePlan
    ) -> datetime:
        if not self.project_exists(project_id):
            raise ProjectNotFoundError(project_id)

        saved_at = _utcnow()
        payload: dict[str, Any] = dict(plan.model_dump(mode="json"))
        payload["project_id"] = project_id
        payload["saved_at"] = saved_at.isoformat()

        path = self.multi_voice_plan_path(project_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise ProjectError(
                f"Could not save the multi-voice plan: {exc}"
            ) from exc
        return saved_at

    def get_multi_voice_plan(
        self, project_id: str
    ) -> Optional[tuple[MultiVoicePlan, Optional[datetime]]]:
        """Load multi_voice_plan.json -> (plan, saved_at) or None."""
        path = self.multi_voice_plan_path(project_id)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

        saved_raw = data.get("saved_at") if isinstance(data, dict) else None
        saved_at = (
            datetime.fromisoformat(saved_raw) if isinstance(saved_raw, str) else None
        )
        try:
            plan = MultiVoicePlan.model_validate(data)
        except Exception:
            return None
        return plan, saved_at

    # ------------------------------------------------------------------ #
    # Audio segments + generation metadata
    # ------------------------------------------------------------------ #
    def segments_dir(self, project_id: str) -> Path:
        """storage/projects/{project_id}/audio/segments/"""
        return self.audio_dir(project_id) / self.settings.segment_dir_name

    def segment_path(
        self, project_id: str, paragraph_index: int, extension: str = ".mp3"
    ) -> Path:
        """paragraph_0001.mp3 for index 0 (1-based file names).

        `extension` follows the configured segment output format so PCM/WAV
        intermediates keep honest file names on disk.
        """
        if not extension.startswith("."):
            extension = f".{extension}"
        filename = f"paragraph_{paragraph_index + 1:04d}{extension}"
        return self.segments_dir(project_id) / filename

    def utterance_path(
        self,
        project_id: str,
        paragraph_index: int,
        sequence_index: int,
        extension: str = ".mp3",
    ) -> Path:
        """utterance_p0000_u0000.mp3 — one file per multi-voice utterance."""
        if not extension.startswith("."):
            extension = f".{extension}"
        filename = (
            f"utterance_p{paragraph_index:04d}_u{sequence_index:04d}{extension}"
        )
        return self.segments_dir(project_id) / filename

    def audio_metadata_path(self, project_id: str) -> Path:
        """storage/projects/{project_id}/audio/metadata.json"""
        return self.audio_dir(project_id) / self.settings.audio_metadata_filename

    def read_audio_metadata(self, project_id: str) -> Optional[dict[str, Any]]:
        path = self.audio_metadata_path(project_id)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def write_audio_metadata(
        self, project_id: str, data: dict[str, Any]
    ) -> None:
        path = self.audio_metadata_path(project_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise AudioSaveError(
                f"Could not save audio metadata: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Mapping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_response(record: dict[str, Any]) -> ProjectResponse:
        raw_script = record.get("script")
        script: Optional[ScriptInfo] = None
        if raw_script:
            script = ScriptInfo(
                filename=raw_script["filename"],
                stored_as=raw_script["stored_as"],
                size_bytes=raw_script["size_bytes"],
                character_count=raw_script["character_count"],
                uploaded_at=datetime.fromisoformat(raw_script["uploaded_at"]),
            )

        return ProjectResponse(
            project_id=record["project_id"],
            name=record["name"],
            description=record.get("description"),
            created_at=datetime.fromisoformat(record["created_at"]),
            has_script=script is not None,
            script=script,
        )
