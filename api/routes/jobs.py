from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from api.job_manager import job_manager, JOBS_DIR
from api.schemas import (
    CreateJobResponse,
    DialogueSegmentResponse,
    JobResultResponse,
    JobStatusResponse,
)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.post("", response_model=CreateJobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    video: UploadFile = File(...),
    reference_audio: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    source_language: str = Form("en"),
    target_language: str = Form("hi"),
    voice_mode: str = Form("original"),
    speech_speed: float = Form(1.0),
    enable_lip_sync: bool = Form(True),
    output_quality: str = Form("high"),
):
    """Create and start a real dubbing pipeline job."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing video filename")

    # Read video content
    video_bytes = await video.read()
    if len(video_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty")

    # Explicit preflight check: ensure models & runtimes are ready before accepting job
    from dubsync.readiness import check_system_readiness
    is_ready, details, error_msg = check_system_readiness(
        target_language=target_language,
        enable_lip_sync=enable_lip_sync,
    )
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Preflight model check failed: {error_msg}",
        )

    ref_bytes = None
    ref_filename = None
    if reference_audio and reference_audio.filename:
        ref_bytes = await reference_audio.read()
        ref_filename = reference_audio.filename

    job_id = job_manager.create_job(
        video_filename=video.filename,
        video_bytes=video_bytes,
        title=title,
        source_language=source_language,
        target_language=target_language,
        voice_mode=voice_mode,
        reference_audio_bytes=ref_bytes,
        reference_audio_filename=ref_filename,
        speech_speed=speech_speed,
        enable_lip_sync=enable_lip_sync,
        output_quality=output_quality,
    )

    return CreateJobResponse(job_id=job_id, status="queued")


@router.get("", response_model=List[JobStatusResponse])
async def list_jobs():
    """List all projects/jobs in order of creation."""
    jobs = job_manager.list_jobs()
    return [
        JobStatusResponse(
            job_id=j["job_id"],
            title=j["title"],
            status=j["status"],
            current_stage=j["current_stage"],
            overall_progress=j["overall_progress"],
            current_operation=j["current_operation"],
            created_at=j["created_at"],
            started_at=j.get("started_at"),
            completed_at=j.get("completed_at"),
            source_language=j["source_language"],
            target_language=j["target_language"],
            duration_seconds=j.get("duration_seconds"),
            enable_lip_sync=j.get("enable_lip_sync", True),
            voice_mode=j.get("voice_mode", "original"),
            stages=j["stages"],
            error=j.get("error"),
        )
        for j in jobs
    ]


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    """Retrieve full status of a specific job."""
    j = job_manager.get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=j["job_id"],
        title=j["title"],
        status=j["status"],
        current_stage=j["current_stage"],
        overall_progress=j["overall_progress"],
        current_operation=j["current_operation"],
        created_at=j["created_at"],
        started_at=j.get("started_at"),
        completed_at=j.get("completed_at"),
        source_language=j["source_language"],
        target_language=j["target_language"],
        duration_seconds=j.get("duration_seconds"),
        enable_lip_sync=j.get("enable_lip_sync", True),
        voice_mode=j.get("voice_mode", "original"),
        stages=j["stages"],
        error=j.get("error"),
    )


@router.get("/{job_id}/events")
async def stream_job_events(job_id: str):
    """Server-Sent Events (SSE) stream for real-time stage progress."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    queue = job_manager.subscribe(job_id)

    async def event_generator():
        # First emit initial state
        initial_event = {
            "job_id": job_id,
            "stage": job["current_stage"],
            "status": job["status"],
            "progress": job["overall_progress"],
            "message": job["current_operation"],
            "error": job.get("error"),
        }
        yield f"data: {json.dumps(initial_event)}\n\n"

        try:
            while True:
                # Wait for next event or heartbeat timeout
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("status") in ("completed", "failed", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    # Heartbeat keepalive comment
                    yield ": ping\n\n"
        finally:
            job_manager.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(job_id: str):
    """Retrieve completed artifacts, video URLs, and dialogue segments."""
    j = job_manager.get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = Path(j["output_dir"])
    dialogue_segments: List[DialogueSegmentResponse] = []

    # Parse stage 1 and stage 2 JSON artifacts if present
    stage1_file = output_dir / "stage1_transcript.json"
    stage2_file = output_dir / "stage2_translation.json"

    if stage2_file.is_file():
        try:
            s2_data = json.loads(stage2_file.read_text(encoding="utf-8"))
            for idx, seg in enumerate(s2_data.get("dialogue_segments", []), start=1):
                start = float(seg.get("start", 0))
                end = float(seg.get("end", 0))
                dialogue_segments.append(
                    DialogueSegmentResponse(
                        index=idx,
                        start=start,
                        end=end,
                        source_text=seg.get("source_text", ""),
                        translated_text=seg.get("translated_text", ""),
                        target_duration=round(end - start, 2),
                    )
                )
        except Exception:
            pass
    elif stage1_file.is_file():
        try:
            s1_data = json.loads(stage1_file.read_text(encoding="utf-8"))
            for idx, seg in enumerate(s1_data.get("dialogue_segments", []), start=1):
                start = float(seg.get("start", 0))
                end = float(seg.get("end", 0))
                dialogue_segments.append(
                    DialogueSegmentResponse(
                        index=idx,
                        start=start,
                        end=end,
                        source_text=seg.get("text", ""),
                        target_duration=round(end - start, 2),
                    )
                )
        except Exception:
            pass

    artifacts = j.get("artifacts", {})
    return JobResultResponse(
        job_id=j["job_id"],
        title=j["title"],
        status=j["status"],
        video_url=artifacts.get("dubbed_video_url"),
        audio_url=artifacts.get("dubbed_audio_url"),
        original_video_url=artifacts.get("original_video_url"),
        transcript_url=artifacts.get("transcript_json_url"),
        translation_url=artifacts.get("translation_json_url"),
        source_language=j["source_language"],
        target_language=j["target_language"],
        duration_seconds=j.get("duration_seconds"),
        voice_mode=j.get("voice_mode", "original"),
        enable_lip_sync=j.get("enable_lip_sync", True),
        dialogue_segments=dialogue_segments,
        completed_at=j.get("completed_at"),
    )


@router.get("/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    """Retrieve actual stage 3 and 4 evaluation metrics if available."""
    j = job_manager.get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = Path(j["output_dir"])
    metrics_data: Dict[str, Any] = {
        "job_id": job_id,
        "status": j["status"],
    }

    s3_metrics = output_dir / "evaluation" / "stage3_metrics.json"
    s4_metrics = output_dir / "evaluation" / "stage4_metrics.json"

    if s3_metrics.is_file():
        try:
            metrics_data["stage3_tts"] = json.loads(s3_metrics.read_text(encoding="utf-8"))
        except Exception:
            pass

    if s4_metrics.is_file():
        try:
            metrics_data["stage4_assembly"] = json.loads(s4_metrics.read_text(encoding="utf-8"))
        except Exception:
            pass

    return metrics_data


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running or queued job."""
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not cancel job (may already be finished or invalid)")
    return {"status": "cancelled", "job_id": job_id}


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """Delete a job record and its output folder."""
    success = job_manager.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted", "job_id": job_id}
