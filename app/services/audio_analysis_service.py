"""Audio analysis service: voice-only audiobook -> pyannoteAI -> timeline.

Responsibilities (Stage 1 — analysis ONLY, no sound design/mixing):

1. locate the finished `audio/audiobook.mp3` and fingerprint it (SHA-256);
2. measure its REAL sample rate + duration with ffprobe;
3. upload it to pyannoteAI's temporary media storage (unique `media://`
   object key per analysis — never a fixed key);
4. submit an asynchronous `POST /v1/diarize` job and persist the job
   bookkeeping immediately (project_id, analysis_id, pyannote_job_id,
   audio_hash, model ids, statuses, timestamps);
5. poll `GET /v1/jobs/{job_id}` ON DEMAND with exponential backoff — the
   initial HTTP request never stays open while transcription runs;
6. the moment the provider reports `succeeded`, fetch the output and save
   `audio_analysis_raw.json` **immediately** (provider results are deleted
   24h after completion — we never depend on pyannote retaining them);
7. validate + normalize timestamps (Decimal ROUND_HALF_UP, real sample
   rate) into `audio_analysis_normalized.json`.

Storage layout (follows the existing per-project conventions):

    storage/projects/{project_id}/audio/analysis/
        audio_analysis_metadata.json      # index: all analysis versions
        {analysis_id}/
            audio_analysis_raw.json       # exact provider output (verbatim)
            audio_analysis_normalized.json

Duplicate-spend protection: an analysis is keyed by
(audio_hash, diarization model, transcription model). An identical
SUCCESSFUL analysis is reused; an identical IN-FLIGHT analysis is returned
instead of resubmitted. When the audio bytes change, a NEW analysis version
is created.

Background-processing reality check: this project has no durable worker or
queue (by design — no Redis/Celery). Polling therefore happens when the
status endpoint is called; job state lives on disk, so a server restart
loses only the *waiting*, never the job id or any saved result. If nobody
polls a running job, its output may expire at the provider after 24h —
then the record must be resubmitted (documented limitation, see README).
Webhooks are not enabled because the app has no stable public HTTPS
endpoint in development; they are the production upgrade path.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from app.config import Settings, get_settings
from app.schemas.audio_analysis import (
    AudioAnalysisNormalized,
    AudioAnalysisStartResponse,
    AudioAnalysisStatusResponse,
)
from app.services.audio_analysis_normalizer import normalize_analysis_output
from app.services.audio_probe import AudioProbe, probe_audio
from app.services.errors import (
    AudioAnalysisIncompleteError,
    AudioAnalysisNormalizationError,
    AudioAnalysisNotFoundError,
    AudioSaveError,
    ProjectError,
    ProjectNotFoundError,
    PyannoteConfigError,
)
from app.services.project_service import ProjectService
from app.services.pyannote_client import PyannoteClient, redact_url_secrets

# Local (our) statuses for an analysis version.
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELED = "canceled"
TERMINAL_STATUSES = frozenset({STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELED})

ANALYSIS_DIR_NAME = "analysis"
METADATA_FILENAME = "audio_analysis_metadata.json"
RAW_FILENAME = "audio_analysis_raw.json"
NORMALIZED_FILENAME = "audio_analysis_normalized.json"

_HASH_CHUNK = 1024 * 1024
_MAX_ERROR_CHARS = 500


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


class AudioAnalysisService:
    """Orchestrates the pyannoteAI analysis pipeline for one project."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        project_service: Optional[ProjectService] = None,
        client: Optional[PyannoteClient] = None,
        probe: Optional[Callable[[Path, Settings], AudioProbe]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.projects = project_service or ProjectService(self.settings)
        # DI seams: tests inject a mocked client + a fake probe so the
        # automated suite never spends pyannote credits.
        self._client = client
        self.probe = probe if probe is not None else probe_audio

    def _pyannote(self) -> PyannoteClient:
        return self._client if self._client is not None else PyannoteClient(
            self.settings
        )

    # ------------------------------------------------------------------ #
    # Storage paths
    # ------------------------------------------------------------------ #
    def analysis_root(self, project_id: str) -> Path:
        """storage/projects/{project_id}/audio/analysis/"""
        return self.projects.audio_dir(project_id) / ANALYSIS_DIR_NAME

    def metadata_path(self, project_id: str) -> Path:
        return self.analysis_root(project_id) / METADATA_FILENAME

    def version_dir(self, project_id: str, analysis_id: str) -> Path:
        return self.analysis_root(project_id) / analysis_id

    def raw_path(self, project_id: str, analysis_id: str) -> Path:
        return self.version_dir(project_id, analysis_id) / RAW_FILENAME

    def normalized_path(self, project_id: str, analysis_id: str) -> Path:
        return self.version_dir(project_id, analysis_id) / NORMALIZED_FILENAME

    # ------------------------------------------------------------------ #
    # Safe JSON persistence (atomic replace — a crash never truncates a file)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            raise AudioSaveError(
                f"Could not save {path.name}: {exc}"
            ) from exc

    @staticmethod
    def _read_json(path: Path) -> Optional[Any]:
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

    def _read_metadata(self, project_id: str) -> dict[str, Any]:
        """Load audio_analysis_metadata.json (never silently overwritten).

        A CORRUPT index is an error: resubmitting blindly could spend
        credits twice, so the operator must look at the file first.
        """
        path = self.metadata_path(project_id)
        if not path.is_file():
            now = _iso(_utcnow())
            return {
                "project_id": project_id,
                "current_analysis_id": None,
                "analyses": [],
                "created_at": now,
                "updated_at": now,
            }
        data = self._read_json(path)
        if not isinstance(data, dict):
            raise ProjectError(
                f"{METADATA_FILENAME} for project '{project_id}' is corrupt "
                "— fix or remove it before starting a new analysis "
                "(automatic resubmission is disabled to avoid double billing)"
            )
        data.setdefault("analyses", [])
        if not isinstance(data["analyses"], list):
            data["analyses"] = []
        return data

    def _save_metadata(self, project_id: str, data: dict[str, Any]) -> None:
        data["updated_at"] = _iso(_utcnow())
        self._write_json(self.metadata_path(project_id), data)

    def _save_record(
        self, project_id: str, metadata: dict[str, Any], record: dict[str, Any]
    ) -> dict[str, Any]:
        """Insert/replace one analysis record and mark it current."""
        record["updated_at"] = _iso(_utcnow())
        analyses = metadata["analyses"]
        for index, existing in enumerate(analyses):
            if isinstance(existing, dict) and existing.get(
                "analysis_id"
            ) == record["analysis_id"]:
                analyses[index] = record
                break
        else:
            analyses.append(record)
        metadata["current_analysis_id"] = record["analysis_id"]
        self._save_metadata(project_id, metadata)
        return record

    def _current_record(
        self, project_id: str, metadata: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        current_id = metadata.get("current_analysis_id")
        analyses = [
            item for item in metadata.get("analyses", []) if isinstance(item, dict)
        ]
        for record in reversed(analyses):
            if current_id and record.get("analysis_id") == current_id:
                return record
        return analyses[-1] if analyses else None

    # ------------------------------------------------------------------ #
    # Fingerprints
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _config_key(
        audio_hash: str, model_id: str, transcription_model: str
    ) -> str:
        """Fingerprint of everything that shapes the analysis result."""
        payload = json.dumps(
            {
                "audio_hash": audio_hash,
                "model_id": model_id,
                "transcription_model": transcription_model,
                "transcription": True,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    # ------------------------------------------------------------------ #
    # POST /projects/{project_id}/audio-analysis
    # ------------------------------------------------------------------ #
    async def start_analysis(
        self, project_id: str
    ) -> AudioAnalysisStartResponse:
        """Upload + submit (once) and return immediately with the job id."""
        # 1) project must exist -> 404
        if not self.projects.project_exists(project_id):
            raise ProjectNotFoundError(project_id)

        # 2) the voice-only audiobook must exist -> 404
        audio_path = self.projects.audio_path(project_id)
        if not audio_path.is_file():
            raise AudioAnalysisNotFoundError(
                project_id, reason="the audiobook has not been generated yet"
            )

        # 3) credentials -> 500 (before any file/network work)
        if not self.settings.pyannote_api_key:
            raise PyannoteConfigError(
                "PYANNOTE_API_KEY is not configured. Copy .env.example to "
                ".env and add your pyannoteAI API key."
            )

        # 4) fingerprint the audio + configuration
        audio_hash = self._sha256_file(audio_path)
        model_id = self.settings.pyannote_model_id
        transcription_model = self.settings.pyannote_transcription_model
        config_key = self._config_key(audio_hash, model_id, transcription_model)

        metadata = self._read_metadata(project_id)

        # 5) duplicate-spend protection: reuse identical finished/running work
        reusable = self._find_reusable(project_id, metadata, config_key, audio_hash)
        if reusable is not None:
            return self._start_response(
                project_id,
                reusable,
                reused=True,
                message=(
                    "An identical analysis already exists for this audio — "
                    "no new pyannoteAI job was submitted"
                ),
            )

        # 6) measure the REAL sample rate + duration (never hardcoded)
        probe = self.probe(audio_path, self.settings)

        # 7) unique media object key (project + analysis uuid — never fixed)
        analysis_id = str(uuid.uuid4())
        object_key = f"audiobook/{project_id}/{analysis_id}"

        client = self._pyannote()
        # 8) temporary media upload: declare key -> PUT the local bytes
        presigned_url = await client.create_media_upload_url(object_key)
        audio_bytes = audio_path.read_bytes()
        await client.upload_media(presigned_url, audio_bytes)
        # The pre-signed URL is dropped here — never stored, never logged.

        # 9) submit the async job (NOT auto-retried: duplicates cost money)
        job = await client.submit_diarization(object_key)
        job_id = str(job.get("jobId") or "").strip()
        job_status = str(job.get("status") or "created").strip().lower() or "created"

        now = _iso(_utcnow())
        record: dict[str, Any] = {
            "project_id": project_id,
            "analysis_id": analysis_id,
            "config_key": config_key,
            "audio_hash": audio_hash,
            "audio_filename": audio_path.name,
            "audio_size_bytes": audio_path.stat().st_size,
            "sample_rate": probe.sample_rate,
            "audio_duration_ms": probe.duration_ms,
            "media_object_key": object_key,
            "model_id": model_id,
            "transcription_model": transcription_model,
            "pyannote_job_id": job_id,
            "job_status": job_status,
            # Always start as "processing": even if the provider echoed an
            # immediately-succeeded job, the OUTPUT has not been fetched and
            # saved yet — the first status call performs that poll.
            "status": STATUS_PROCESSING,
            "poll_count": 0,
            "last_polled_at": None,
            "error": None,
            "word_count": None,
            "speaker_count": None,
            "raw_file": None,
            "normalized_file": None,
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
        }
        self._save_record(project_id, metadata, record)

        warning = job.get("warning")
        message = "pyannoteAI diarization job submitted"
        if isinstance(warning, str) and warning.strip():
            message = f"{message} (warning: {redact_url_secrets(warning)[:200]})"

        return self._start_response(
            project_id, record, reused=False, message=message
        )

    def _find_reusable(
        self,
        project_id: str,
        metadata: dict[str, Any],
        config_key: str,
        audio_hash: str,
    ) -> Optional[dict[str, Any]]:
        """Newest analysis with identical inputs that needs no new credits.

        * completed  -> its files must still exist on disk (else resubmit);
        * processing -> an in-flight provider job (never double-submit);
        * failed/canceled -> NOT reused, a fresh attempt is allowed
          (failed/canceled jobs are not billed by pyannoteAI).
        """
        for record in reversed(metadata.get("analyses", [])):
            if not isinstance(record, dict):
                continue
            if record.get("config_key") != config_key:
                continue
            if record.get("audio_hash") != audio_hash:
                continue
            status = record.get("status")
            analysis_id = record.get("analysis_id")
            if not isinstance(analysis_id, str):
                continue
            if status == STATUS_COMPLETED:
                if self.raw_path(project_id, analysis_id).is_file() and (
                    self.normalized_path(project_id, analysis_id).is_file()
                ):
                    return record
            elif status == STATUS_PROCESSING and record.get("pyannote_job_id"):
                return record
        return None

    # ------------------------------------------------------------------ #
    # GET /projects/{project_id}/audio-analysis/status
    # ------------------------------------------------------------------ #
    async def get_status(
        self, project_id: str
    ) -> AudioAnalysisStatusResponse:
        """Local + provider status. Polls the provider at most once per call."""
        if not self.projects.project_exists(project_id):
            raise ProjectNotFoundError(project_id)

        metadata = self._read_metadata(project_id)
        record = self._current_record(project_id, metadata)
        if record is None:
            raise AudioAnalysisNotFoundError(
                project_id,
                reason="no analysis has been started yet — run "
                "POST /projects/{project_id}/audio-analysis first",
            )

        if record.get("status") not in TERMINAL_STATUSES:
            if self._poll_due(record):
                record = await self._poll_provider(project_id, metadata, record)
            else:
                return self._status_response(
                    record,
                    message=(
                        "Analysis in progress — next provider poll is not due "
                        "yet (backoff); call status again shortly"
                    ),
                )

        status = record.get("status")
        if status == STATUS_COMPLETED:
            message = "Analysis completed — normalized timeline is available"
        elif status == STATUS_CANCELED:
            message = "The pyannoteAI job was canceled"
        else:
            message = (
                "Analysis failed — see `error` (the raw provider response "
                "is preserved when one was received)"
            )
        return self._status_response(record, message=message)

    def _poll_due(self, record: dict[str, Any]) -> bool:
        """Exponential backoff between provider polls (rate-limit friendly)."""
        last_raw = record.get("last_polled_at")
        if not last_raw:
            return True
        try:
            last = datetime.fromisoformat(str(last_raw))
        except ValueError:
            return True
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)

        base = max(0.0, self.settings.pyannote_poll_interval_seconds)
        cap = max(base, self.settings.pyannote_poll_max_interval_seconds)
        polls = max(0, int(record.get("poll_count") or 0))
        interval = min(base * (2 ** min(polls, 16)), cap)
        elapsed = (_utcnow() - last).total_seconds()
        return elapsed >= interval

    async def _poll_provider(
        self,
        project_id: str,
        metadata: dict[str, Any],
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """One GET /v1/jobs/{id}; persists results the moment it succeeds."""
        client = self._pyannote()
        job = await client.get_job(str(record.get("pyannote_job_id") or ""))

        job_status = str(job.get("status") or "").strip().lower()
        record["job_status"] = job_status or record.get("job_status")
        record["poll_count"] = int(record.get("poll_count") or 0) + 1
        record["last_polled_at"] = _iso(_utcnow())

        if job_status == "succeeded":
            output = job.get("output")
            # Saves raw + normalized and updates the record in place.
            return self._finalize_success(project_id, metadata, record, output, job)

        if job_status in ("failed", "canceled"):
            record["status"] = (
                STATUS_CANCELED if job_status == "canceled" else STATUS_FAILED
            )
            record["error"] = self._job_error(job) or (
                f"The pyannoteAI job {job_status}"
            )
            return self._save_record(project_id, metadata, record)

        # created / pending / running (unknown future statuses stay here too)
        record["status"] = STATUS_PROCESSING
        return self._save_record(project_id, metadata, record)

    @staticmethod
    def _job_error(job: dict[str, Any]) -> str:
        """Short, secret-free error text from a failed provider job."""
        candidates = [job.get("error")]
        output = job.get("output")
        if isinstance(output, dict):
            candidates.append(output.get("error"))
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                text = " ".join(redact_url_secrets(candidate).split())
                return text[:_MAX_ERROR_CHARS]
        return ""

    def _finalize_success(
        self,
        project_id: str,
        metadata: dict[str, Any],
        record: dict[str, Any],
        output: Any,
        job: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist the raw response FIRST, then normalize it.

        Provider outputs are deleted 24h after completion — this is the
        only moment the full result can be saved, so it happens before any
        validation and regardless of whether normalization succeeds.
        """
        analysis_id = record["analysis_id"]
        now = _iso(_utcnow())

        raw_doc: dict[str, Any] = {}
        if isinstance(output, dict):
            # Verbatim provider fields (diarization, wordLevelTranscription,
            # turnLevelTranscription, exclusiveDiarization, *Probability, …)
            # are copied unchanged — words are never rewritten.
            raw_doc.update(output)
        else:
            # Unexpected type: still preserved for debugging.
            raw_doc["_provider_output"] = output
        raw_doc["_analysis_metadata"] = {
            "project_id": project_id,
            "analysis_id": analysis_id,
            "pyannote_job_id": record.get("pyannote_job_id"),
            "audio_hash": record.get("audio_hash"),
            "audio_filename": record.get("audio_filename"),
            "model_id": record.get("model_id"),
            "transcription_model": record.get("transcription_model"),
            "job_status": "succeeded",
            "provider_created_at": job.get("createdAt"),
            "provider_updated_at": job.get("updatedAt"),
            "fetched_at": now,
            "created_at": record.get("created_at"),
            "updated_at": now,
        }
        raw_path = self.raw_path(project_id, analysis_id)
        self._write_json(raw_path, raw_doc)
        record["raw_file"] = str(raw_path.relative_to(self.projects.audio_dir(
            project_id
        )))

        # --- normalization (raw is already safe on disk) ----------------- #
        try:
            normalized = normalize_analysis_output(
                output,
                project_id=project_id,
                analysis_id=analysis_id,
                model_id=str(record.get("model_id") or ""),
                transcription_model=str(record.get("transcription_model") or ""),
                sample_rate=int(record.get("sample_rate") or 0),
                audio_duration_ms=int(record.get("audio_duration_ms") or 0),
                duration_tolerance_ms=self.settings.pyannote_duration_tolerance_ms,
            )
        except AudioAnalysisNormalizationError as exc:
            # Raw output is already saved above — keep it for debugging and
            # mark this analysis failed instead of losing the evidence.
            record["status"] = STATUS_FAILED
            record["error"] = str(exc)
            return self._save_record(project_id, metadata, record)

        normalized_path = self.normalized_path(project_id, analysis_id)
        self._write_json(normalized_path, normalized.model_dump(mode="json"))

        record["status"] = STATUS_COMPLETED
        record["error"] = None
        record["completed_at"] = now
        record["word_count"] = len(normalized.words)
        record["speaker_count"] = len(normalized.speakers)
        record["normalized_file"] = str(
            normalized_path.relative_to(self.projects.audio_dir(project_id))
        )
        return self._save_record(project_id, metadata, record)

    # ------------------------------------------------------------------ #
    # GET /projects/{project_id}/audio-analysis
    # ------------------------------------------------------------------ #
    def get_normalized(self, project_id: str) -> AudioAnalysisNormalized:
        """The finished normalized timeline (404/409 otherwise)."""
        if not self.projects.project_exists(project_id):
            raise ProjectNotFoundError(project_id)

        metadata = self._read_metadata(project_id)
        record = self._current_record(project_id, metadata)
        if record is None:
            raise AudioAnalysisNotFoundError(project_id)

        status = record.get("status")
        if status != STATUS_COMPLETED:
            raise AudioAnalysisIncompleteError(
                project_id,
                str(status or "unknown"),
                str(record.get("error") or ""),
            )

        path = self.normalized_path(project_id, str(record["analysis_id"]))
        data = self._read_json(path)
        if not isinstance(data, dict):
            raise AudioAnalysisNotFoundError(
                project_id,
                reason="the normalized analysis file is missing on disk",
            )
        try:
            return AudioAnalysisNormalized.model_validate(data)
        except Exception as exc:
            raise AudioAnalysisIncompleteError(
                project_id, "corrupt", f"normalized file unreadable: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Response mapping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _local_status(job_status: str) -> str:
        if job_status == "succeeded":
            return STATUS_COMPLETED
        if job_status == "failed":
            return STATUS_FAILED
        if job_status == "canceled":
            return STATUS_CANCELED
        return STATUS_PROCESSING

    @staticmethod
    def _start_response(
        project_id: str,
        record: dict[str, Any],
        *,
        reused: bool,
        message: str,
    ) -> AudioAnalysisStartResponse:
        return AudioAnalysisStartResponse(
            success=True,
            message=message,
            project_id=project_id,
            analysis_id=str(record.get("analysis_id") or ""),
            status=str(record.get("status") or STATUS_PROCESSING),
            job_status=str(record.get("job_status") or "created"),
            model_id=str(record.get("model_id") or ""),
            transcription_model=str(record.get("transcription_model") or ""),
            audio_hash=str(record.get("audio_hash") or ""),
            reused=reused,
            created_at=datetime.fromisoformat(
                str(record.get("created_at") or _iso(_utcnow()))
            ),
        )

    @staticmethod
    def _status_response(
        record: dict[str, Any], *, message: str
    ) -> AudioAnalysisStatusResponse:
        def _dt(value: Any) -> Optional[datetime]:
            if not value:
                return None
            try:
                return datetime.fromisoformat(str(value))
            except ValueError:
                return None

        return AudioAnalysisStatusResponse(
            success=True,
            message=message,
            project_id=str(record.get("project_id") or ""),
            analysis_id=str(record.get("analysis_id") or ""),
            status=str(record.get("status") or STATUS_PROCESSING),
            job_status=str(record.get("job_status") or "created"),
            model_id=str(record.get("model_id") or ""),
            transcription_model=str(record.get("transcription_model") or ""),
            error=record.get("error"),
            sample_rate=record.get("sample_rate"),
            audio_duration_ms=record.get("audio_duration_ms"),
            word_count=record.get("word_count"),
            speaker_count=record.get("speaker_count"),
            poll_count=int(record.get("poll_count") or 0),
            created_at=_dt(record.get("created_at")) or _utcnow(),
            updated_at=_dt(record.get("updated_at")) or _utcnow(),
            completed_at=_dt(record.get("completed_at")),
        )
