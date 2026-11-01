from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from api.auth.dependencies import get_current_user, require_verified_user
from api.job_manager import job_manager, JOBS_DIR
from api.schemas import (
    CreateJobResponse,
    DialogueSegmentResponse,
    EvaluationResponse,
    JobResultResponse,
    JobStatusResponse,
    PaginatedJobsResponse,
    SegmentDetailResponse,
)
from database.connection import db_manager
from database.models import UserModel
from database.repositories.evaluations import EvaluationsRepository
from database.repositories.segments import SegmentsRepository

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.post("", response_model=CreateJobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    video: UploadFile = File(...),
    reference_audio: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    project_id: Optional[str] = Form(None),
    source_language: str = Form("en"),
    target_language: str = Form("hi"),
    voice_mode: str = Form("original"),
    speech_speed: float = Form(1.0),
    enable_lip_sync: bool = Form(True),
    output_quality: str = Form("high"),
    current_user: UserModel = Depends(require_verified_user),
):
    """Create and start a real dubbing pipeline job with optional project association."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing video filename")

    # Read video content
    video_bytes = await video.read()
    if len(video_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty")

    # Explicit language capability check: ensure requested language pair is supported
    from dubsync.languages import validate_language_pair
    is_pair_valid, pair_error = validate_language_pair(source_language, target_language)
    if not is_pair_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported language pair: {pair_error}",
        )

    # Explicit preflight check: ensure models & runtimes are ready before accepting job
    from dubsync.readiness import check_system_readiness
    is_ready, details, error_msg = check_system_readiness(
        source_language=source_language,
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
        project_id=project_id,
        user_id=current_user.user_id,
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


@router.get("", response_model=Union[PaginatedJobsResponse, List[JobStatusResponse]])
async def list_jobs(
    project_id: Optional[str] = None,
    page: Optional[int] = None,
    limit: Optional[int] = None,
    current_user: UserModel = Depends(get_current_user),
):
    """List all projects/jobs, optionally paginated or filtered by project."""
    jobs = job_manager.list_jobs(project_id=project_id, user_id=current_user.user_id)
    items = [
        JobStatusResponse(
            job_id=j["job_id"],
            title=j["title"],
            project_id=j.get("project_id"),
            status=j["status"],
            current_stage=j["current_stage"],
            overall_progress=j["overall_progress"],
            current_operation=j["current_operation"],
            created_at=j["created_at"],
            started_at=j.get("started_at"),
            completed_at=j.get("completed_at"),
            source_language=j["source_language"],
            detected_source_language=j.get("detected_source_language"),
            target_language=j["target_language"],
            duration_seconds=j.get("duration_seconds"),
            enable_lip_sync=j.get("enable_lip_sync", True),
            voice_mode=j.get("voice_mode", "original"),
            stages=j["stages"],
            error=j.get("error"),
        )
        for j in jobs
    ]

    if page is not None:
        p = max(1, page)
        lim = max(1, limit or 10)
        total = len(items)
        total_pages = max(1, (total + lim - 1) // lim) if total > 0 else 0
        start = (p - 1) * lim
        paged_items = items[start : start + lim]
        return PaginatedJobsResponse(
            items=paged_items,
            page=p,
            limit=lim,
            total=total,
            total_pages=total_pages,
        )

    return items


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, current_user: UserModel = Depends(get_current_user)):
    """Retrieve full status of a specific job."""
    j = job_manager.get_job_for_user(job_id, current_user.user_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=j["job_id"],
        title=j["title"],
        project_id=j.get("project_id"),
        status=j["status"],
        current_stage=j["current_stage"],
        overall_progress=j["overall_progress"],
        current_operation=j["current_operation"],
        created_at=str(j["created_at"]),
        started_at=str(j["started_at"]) if j.get("started_at") else None,
        completed_at=str(j["completed_at"]) if j.get("completed_at") else None,
        source_language=j["source_language"],
        detected_source_language=j.get("detected_source_language"),
        target_language=j["target_language"],
        duration_seconds=j.get("duration_seconds"),
        enable_lip_sync=j.get("enable_lip_sync", True),
        voice_mode=j.get("voice_mode", "original"),
        stages=j["stages"],
        error=j.get("error"),
    )


@router.get("/{job_id}/segments", response_model=List[SegmentDetailResponse])
async def get_job_segments(job_id: str, current_user: UserModel = Depends(get_current_user)):
    """Retrieve detailed segment records from MongoDB or local pipeline artifacts."""
    j = job_manager.get_job_for_user(job_id, current_user.user_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")

    # 1. Try querying MongoDB if connected
    if db_manager.is_connected:
        try:
            repo = SegmentsRepository()
            db_segments = await repo.get_by_job_for_user(job_id, current_user.user_id)
            if db_segments:
                return [
                    SegmentDetailResponse(
                        segment_id=str(s.get("segment_id", s.get("_id", f"seg-{s.get('segment_index', idx)}"))),
                        segment_index=s.get("segment_index", idx),
                        start_time=float(s.get("start_time", 0.0)),
                        end_time=float(s.get("end_time", 0.0)),
                        original_text=s.get("original_text", ""),
                        translated_text=s.get("translated_text", ""),
                        raw_tts_duration=float(s.get("raw_tts_duration", 0.0)),
                        adapted_tts_duration=float(s.get("adapted_tts_duration", 0.0)),
                        speed_factor=float(s.get("speed_factor", 1.0)),
                        audio_path=s.get("audio_path"),
                        final_placement_start=float(s.get("final_placement_start", 0.0)),
                        final_placement_end=float(s.get("final_placement_end", 0.0)),
                        trimming_seconds=float(s.get("trimming_seconds", 0.0)),
                        status=s.get("status", "completed"),
                        created_at=s.get("created_at"),
                    )
                    for idx, s in enumerate(db_segments, start=1)
                ]
        except Exception:
            pass

    # 2. Local fallback from job directory artifacts
    output_dir = Path(j["output_dir"])
    stage2_file = output_dir / "stage2_translation.json"
    timing_file = output_dir / "timing_accounting.json"

    timing_info: Dict[int, Any] = {}
    if timing_file.is_file():
        try:
            tdata = json.loads(timing_file.read_text(encoding="utf-8"))
            for s in tdata.get("segments", []):
                idx = s.get("index") or s.get("segment_index")
                if idx is not None:
                    timing_info[int(idx)] = s
        except Exception:
            pass

    segments_out: List[SegmentDetailResponse] = []
    if stage2_file.is_file():
        try:
            s2_data = json.loads(stage2_file.read_text(encoding="utf-8"))
            for idx, seg in enumerate(s2_data.get("dialogue_segments", []), start=1):
                t_seg = timing_info.get(idx, {})
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                raw_dur = float(t_seg.get("raw_duration_sec", 0.0))
                proc_dur = float(t_seg.get("processed_duration_sec", end - start))
                speed = float(t_seg.get("atempo_factor", 1.0))
                trim = float(t_seg.get("trimmed_duration_sec", 0.0))

                segments_out.append(
                    SegmentDetailResponse(
                        segment_id=f"seg-{job_id}-{idx}",
                        segment_index=idx,
                        start_time=start,
                        end_time=end,
                        original_text=seg.get("source_text", seg.get("text", "")),
                        translated_text=seg.get("translated_text", ""),
                        raw_tts_duration=raw_dur,
                        adapted_tts_duration=proc_dur,
                        speed_factor=speed,
                        audio_path=f"/api/v1/files/{job_id}/tts_segment_{idx:03d}.wav",
                        final_placement_start=start,
                        final_placement_end=start + proc_dur,
                        trimming_seconds=trim,
                        status="completed",
                    )
                )
        except Exception:
            pass

    return segments_out


@router.get("/{job_id}/evaluation", response_model=EvaluationResponse)
async def get_job_evaluation(job_id: str, current_user: UserModel = Depends(get_current_user)):
    """Retrieve full evaluation metrics from MongoDB or local evaluation artifacts."""
    j = job_manager.get_job_for_user(job_id, current_user.user_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")

    # 1. Try querying MongoDB if connected
    if db_manager.is_connected:
        try:
            repo = EvaluationsRepository()
            db_eval = await repo.get_by_job_for_user(job_id, current_user.user_id)
            if db_eval:
                return EvaluationResponse(
                    evaluation_id=str(db_eval.get("evaluation_id", db_eval.get("_id", f"eval-{job_id}"))),
                    job_id=job_id,
                    project_id=db_eval.get("project_id"),
                    output_id=db_eval.get("output_id"),
                    created_at=str(db_eval.get("created_at", "")),
                    video_duration=db_eval.get("video_duration"),
                    audio_duration=db_eval.get("audio_duration"),
                    video_frame_count=db_eval.get("video_frame_count"),
                    frame_rate=db_eval.get("frame_rate"),
                    speech_preservation_percentage=db_eval.get("speech_preservation_percentage"),
                    trimmed_audio_seconds=db_eval.get("trimmed_audio_seconds"),
                    timeline_extension_seconds=db_eval.get("timeline_extension_seconds"),
                    lip_sync_metrics=db_eval.get("lip_sync_metrics", {}),
                    media_integrity=db_eval.get("media_integrity", {}),
                    segment_metrics=db_eval.get("segment_metrics", []),
                    unmeasured_or_null_metrics=db_eval.get("unmeasured_or_null_metrics", {}),
                    warnings=db_eval.get("warnings", []),
                )
        except Exception:
            pass

    # 2. Local fallback from evaluation_summary.json or timing_accounting.json
    output_dir = Path(j["output_dir"])
    eval_summary_file = output_dir / "evaluation" / "evaluation_summary.json"
    if eval_summary_file.is_file():
        try:
            edata = json.loads(eval_summary_file.read_text(encoding="utf-8"))
            return EvaluationResponse(
                evaluation_id=f"eval-{job_id}",
                job_id=job_id,
                project_id=j.get("project_id"),
                created_at=j.get("completed_at") or j.get("created_at", ""),
                video_duration=edata.get("video_duration_seconds"),
                audio_duration=edata.get("audio_duration_seconds"),
                video_frame_count=edata.get("video_frame_count"),
                frame_rate=edata.get("frame_rate"),
                speech_preservation_percentage=edata.get("speech_preservation_percentage", 100.0),
                trimmed_audio_seconds=edata.get("trimmed_audio_seconds", 0.0),
                timeline_extension_seconds=edata.get("timeline_extension_seconds", 0.0),
                lip_sync_metrics=edata.get("lip_sync_metrics", {}),
                media_integrity=edata.get("media_integrity", {}),
                segment_metrics=edata.get("segment_metrics", []),
                unmeasured_or_null_metrics=edata.get("unmeasured_or_null_metrics", {}),
                warnings=edata.get("warnings", []),
            )
        except Exception:
            pass

    # If no evaluation artifact exists, construct from job details
    dur = j.get("duration_seconds") or 0.0
    return EvaluationResponse(
        evaluation_id=f"eval-{job_id}",
        job_id=job_id,
        project_id=j.get("project_id"),
        created_at=j.get("completed_at") or j.get("created_at", ""),
        video_duration=dur,
        audio_duration=dur,
        speech_preservation_percentage=100.0 if j.get("status") == "completed" else 0.0,
        trimmed_audio_seconds=0.0,
        timeline_extension_seconds=0.0,
        lip_sync_metrics={},
        media_integrity={"status": j.get("status")},
        segment_metrics=[],
        unmeasured_or_null_metrics={"note": "Evaluation artifacts not generated or unmeasured"},
        warnings=[],
    )


@router.get("/{job_id}/events")
async def stream_job_events(job_id: str, current_user: UserModel = Depends(get_current_user)):
    """Server-Sent Events (SSE) stream for real-time stage progress."""
    job = job_manager.get_job_for_user(job_id, current_user.user_id)
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
        if job.get("status") in ("completed", "failed", "cancelled"):
            return

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
async def get_job_result(job_id: str, current_user: UserModel = Depends(get_current_user)):
    """Retrieve completed artifacts, video URLs, and dialogue segments."""
    j = job_manager.get_job_for_user(job_id, current_user.user_id)
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
        project_id=j.get("project_id"),
        status=j["status"],
        video_url=artifacts.get("dubbed_video_url"),
        audio_url=artifacts.get("dubbed_audio_url"),
        original_video_url=artifacts.get("original_video_url"),
        transcript_url=artifacts.get("transcript_json_url"),
        translation_url=artifacts.get("translation_json_url"),
        source_language=j["source_language"],
        detected_source_language=j.get("detected_source_language"),
        target_language=j["target_language"],
        duration_seconds=j.get("duration_seconds"),
        voice_mode=j.get("voice_mode", "original"),
        enable_lip_sync=j.get("enable_lip_sync", True),
        dialogue_segments=dialogue_segments,
        completed_at=j.get("completed_at"),
    )


@router.get("/{job_id}/metrics")
async def get_job_metrics(job_id: str, current_user: UserModel = Depends(get_current_user)):
    """Retrieve actual stage 3 and 4 evaluation metrics if available."""
    j = job_manager.get_job_for_user(job_id, current_user.user_id)
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
async def cancel_job(job_id: str, current_user: UserModel = Depends(get_current_user)):
    """Cancel a running or queued job."""
    j = job_manager.get_job_for_user(job_id, current_user.user_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not cancel job (may already be finished or invalid)")
    return {"status": "cancelled", "job_id": job_id}


@router.delete("/{job_id}")
async def delete_job(job_id: str, current_user: UserModel = Depends(get_current_user)):
    """Delete a job record and its output folder."""
    j = job_manager.get_job_for_user(job_id, current_user.user_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    success = job_manager.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted", "job_id": job_id}
