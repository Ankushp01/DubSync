from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from api.job_manager import JOBS_DIR

router = APIRouter(prefix="/api/v1/files", tags=["files"])


@router.api_route("/{job_id}/{file_path:path}", methods=["GET", "HEAD"])
async def get_job_file(job_id: str, file_path: str):
    """Serve job input and output files (video, audio, JSON) with range support."""
    # Resolve job path safely
    job_dir = (JOBS_DIR / job_id).resolve()
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job directory not found")

    target_file = (job_dir / file_path).resolve()

    # Prevent directory traversal attacks
    if not str(target_file).startswith(str(job_dir)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not target_file.is_file():
        raise HTTPException(status_code=404, detail="Requested file not found")

    mime_type, _ = mimetypes.guess_type(str(target_file))
    if not mime_type:
        if target_file.suffix == ".json":
            mime_type = "application/json"
        elif target_file.suffix == ".wav":
            mime_type = "audio/wav"
        elif target_file.suffix == ".mp4":
            mime_type = "video/mp4"
        else:
            mime_type = "application/octet-stream"

    return FileResponse(
        path=str(target_file),
        media_type=mime_type,
        filename=target_file.name,
    )
