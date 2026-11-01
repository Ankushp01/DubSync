from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from api.auth.dependencies import get_current_user
from api.job_manager import job_manager, JOBS_DIR
from database.connection import db_manager
from database.models import UserModel
from database.repositories.jobs import ProcessingJobsRepository

router = APIRouter(prefix="/api/v1/files", tags=["files"])


@router.api_route("/{job_id}/{file_path:path}", methods=["GET", "HEAD"])
async def get_job_file(
    job_id: str,
    file_path: str,
    current_user: UserModel = Depends(get_current_user),
):
    """Serve job input and output files (video, audio, JSON) with ownership enforcement."""
    # Verify ownership of the job
    job = job_manager.get_job(job_id)
    if job:
        if job.get("user_id") != current_user.user_id:
            raise HTTPException(status_code=404, detail="Requested file not found")
    elif db_manager.is_connected:
        db_job = await ProcessingJobsRepository().get_by_id_for_user(job_id, current_user.user_id)
        if not db_job:
            raise HTTPException(status_code=404, detail="Requested file not found")
    else:
        raise HTTPException(status_code=404, detail="Requested file not found")

    # Resolve job path safely
    job_dir = (JOBS_DIR / job_id).resolve()
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Requested file not found")

    target_file = (job_dir / file_path).resolve()

    # Prevent directory traversal attacks
    if not str(target_file).startswith(str(job_dir)):
        raise HTTPException(status_code=404, detail="Requested file not found")

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
