"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Generic
# --------------------------------------------------------------------------- #
class WelcomeResponse(BaseModel):
    message: str
    app: str
    version: str
    docs_url: str = "/docs"
    endpoints: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    storage_ready: bool
    timestamp: datetime


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #
class ProjectCreateRequest(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human readable project name.",
        examples=["Mere Sapno Ka Audiobook"],
    )
    description: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional notes about the project.",
        examples=["Chapter-wise narration of my draft."],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"name": "Mere Sapno Ka Audiobook", "description": "First draft."}
            ]
        }
    )


class ScriptInfo(BaseModel):
    filename: str = Field(description="Original name of the uploaded file.")
    stored_as: str = Field(description="Name the file is stored under on disk.")
    size_bytes: int
    character_count: int
    uploaded_at: datetime


class ProjectResponse(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    has_script: bool = False
    script: Optional[ScriptInfo] = None


class UploadScriptResponse(BaseModel):
    project_id: str
    filename: str
    character_count: int
    upload_status: str = Field(
        description="'success' when the script was stored on disk."
    )
    size_bytes: int
    stored_path: str
    uploaded_at: datetime


class ScriptImportResponse(BaseModel):
    """`POST /projects/from-script` — auto-created project + stored script.

    Everything the simple creator UI needs after an upload: the detected
    title and language, paragraph count and a short preview. `project_id`
    stays in the API for clients but is never shown to end users.
    """

    project_id: str
    title: str = Field(
        description="Detected book title (explicit heading, else the filename)."
    )
    language: str = Field(
        description="Detected script language (refined by the narration plan)."
    )
    language_code: Optional[str] = Field(
        default=None, description="ISO 639-1 code when known."
    )
    paragraph_count: int
    filename: str
    size_bytes: int
    character_count: int
    script_preview: str = Field(description="Short plain-text preview of the script.")
    upload_status: str = Field(
        description="'success' when the script was stored on disk."
    )
    uploaded_at: datetime


class DeleteResponse(BaseModel):
    """DELETE /projects and DELETE /projects/{project_id} (privacy tools)."""

    deleted: bool = True
    project_id: Optional[str] = None
    deleted_count: int = 0
