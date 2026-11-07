"""Project endpoints: create, read and upload a .txt script."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from app.config import get_settings
from app.schemas.project import (
    DeleteResponse,
    ProjectCreateRequest,
    ProjectResponse,
    ScriptImportResponse,
    UploadScriptResponse,
)
from app.security import current_tester
from app.services import ProjectService
from app.services.errors import ProjectError

router = APIRouter(prefix="/projects", tags=["Projects"])

# Read the upload in chunks so a huge body never sits fully in memory.
CHUNK_SIZE = 64 * 1024

# FastAPI renamed HTTP_413_REQUEST_ENTITY_TOO_LARGE -> HTTP_413_CONTENT_TOO_LARGE.
HTTP_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", None) or 413


def _service() -> ProjectService:
    return ProjectService(get_settings())


@router.post(
    "/from-script",
    response_model=ScriptImportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a .txt script and auto-create its project",
)
async def import_script(
    file: UploadFile = File(..., description="Plain .txt script (max 5 MB)"),
    content_length: Optional[int] = Header(default=None),
    tester: str = Depends(current_tester),
) -> ScriptImportResponse:
    """One-step import for the simple creator UI.

    Validates the upload exactly like `upload-script` (size -> 413, `.txt`
    only -> 400, valid UTF-8 -> 400), then derives the book title and
    language from the text, creates the project automatically and stores the
    script. Existing clients keep using `POST /projects` +
    `upload-script`; this endpoint only adds the automatic flow.
    """
    settings = get_settings()
    service = ProjectService(settings)

    # Cheap early rejection of oversized payloads before reading the body.
    if content_length is not None and content_length > settings.max_script_size_bytes:
        raise HTTPException(
            status_code=HTTP_413,
            detail=(
                f"Script is too large: {content_length} bytes "
                f"(limit: {settings.max_script_size_bytes} bytes)"
            ),
        )

    filename = file.filename or ""
    if Path(filename).suffix.lower() not in settings.allowed_script_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt files are supported",
        )

    # Stream the body and enforce the size limit while reading.
    limit = settings.max_script_size_bytes
    buffer = bytearray()
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise HTTPException(
                status_code=HTTP_413,
                detail=f"Script exceeds the {limit} byte ({settings.max_script_size_mb} MB) limit",
            )
    await file.close()

    try:
        return service.import_script(filename, bytes(buffer), owner=tester)
    except ProjectError as exc:
        # Domain errors carry their own HTTP status (400 / 404 / 413).
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post(
    "",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a project",
)
def create_project(
    payload: ProjectCreateRequest,
    tester: str = Depends(current_tester),
) -> ProjectResponse:
    """Create a new audiobook project and generate its UUID."""
    return _service().create_project(payload, owner=tester)


@router.get(
    "",
    response_model=list[ProjectResponse],
    summary="List the current tester's projects",
)
def list_my_projects(
    tester: str = Depends(current_tester),
) -> list[ProjectResponse]:
    """The AUTHENTICATED tester's own projects, newest first.

    Restore support: a browser that lost its localStorage recovers its
    project from the server instead of requiring a re-upload. Identity comes
    from the session only and another tester's projects are never listed —
    same isolation as every project route. Reuses `ProjectResponse`.
    """
    return _service().list_projects_owned_by(tester)


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Get project details",
)
def get_project(project_id: str) -> ProjectResponse:
    """Return project details, or 404 when the project does not exist."""
    return _service().get_project(project_id)


@router.delete(
    "",
    response_model=DeleteResponse,
    summary="Delete every project of the current tester",
)
def delete_my_projects(tester: str = Depends(current_tester)) -> DeleteResponse:
    """Privacy: delete the tester's projects and all associated files."""
    removed = _service().delete_projects_owned_by(tester)
    return DeleteResponse(deleted=True, deleted_count=removed)


@router.delete(
    "/{project_id}",
    response_model=DeleteResponse,
    summary="Delete one project and its files",
)
def delete_project(project_id: str) -> DeleteResponse:
    """Delete a project: metadata, script, plan, segments and audio."""
    _service().delete_project(project_id)
    return DeleteResponse(deleted=True, project_id=project_id)


@router.post(
    "/{project_id}/upload-script",
    response_model=UploadScriptResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a .txt script",
)
async def upload_script(
    project_id: str,
    file: UploadFile = File(..., description="Plain .txt script (max 5 MB)"),
    content_length: Optional[int] = Header(default=None),
) -> UploadScriptResponse:
    """Validate, decode and store a script as `storage/projects/{id}/script.txt`.

    Checks performed:
    * project must exist (otherwise 404)
    * size limit (5 MB) -> 413
    * `.txt` extension only -> 400
    * valid UTF-8 content -> 400
    """
    settings = get_settings()
    service = ProjectService(settings)

    # Cheap early rejection of oversized payloads before reading the body.
    if content_length is not None and content_length > settings.max_script_size_bytes:
        raise HTTPException(
            status_code=HTTP_413,
            detail=(
                f"Script is too large: {content_length} bytes "
                f"(limit: {settings.max_script_size_bytes} bytes)"
            ),
        )

    # 404 for an unknown project, before touching the file at all.
    if not service.project_exists(project_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found",
        )

    filename = file.filename or ""
    if Path(filename).suffix.lower() not in settings.allowed_script_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt files are supported",
        )

    # Stream the body and enforce the size limit while reading.
    limit = settings.max_script_size_bytes
    buffer = bytearray()
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise HTTPException(
                status_code=HTTP_413,
                detail=f"Script exceeds the {limit} byte ({settings.max_script_size_mb} MB) limit",
            )
    await file.close()

    try:
        return service.upload_script(project_id, filename, bytes(buffer))
    except ProjectError as exc:
        # Domain errors carry their own HTTP status (400 / 404 / 413).
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
