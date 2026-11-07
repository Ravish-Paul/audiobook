"""Media downloads: generated audio and stored project files.

Replaces the old bare `StaticFiles` mount over the whole projects folder.
URLs are unchanged (`/media/projects/{project_id}/audio/audiobook.mp3`), but:

* every request is validated against path traversal (the resolved file must
  stay inside its project folder);
* in beta mode `app.security.BetaSecurityMiddleware` additionally requires a
  tester token AND project ownership before this route is ever reached.

Browsers receive these files with the session cookie automatically (same
origin), so the one-page creator experience is unchanged.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.config import get_settings

router = APIRouter(tags=["Media"])


@router.get(
    "/media/projects/{project_id}/{file_path:path}",
    summary="Download a stored project file (audio, script, plan)",
    response_class=FileResponse,
)
def get_media_file(project_id: str, file_path: str) -> FileResponse:
    settings = get_settings()
    root = settings.storage_dir.resolve()
    project_dir = (root / project_id).resolve()
    if not project_dir.is_relative_to(root):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    target = (project_dir / file_path).resolve()
    if not target.is_relative_to(project_dir) or not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    return FileResponse(target)
