from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from api.auth.dependencies import get_current_user, require_verified_user
from api.job_manager import job_manager, JOBS_DIR, DEFAULT_STAGES
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
from database.models import UserModel, resolve_overall_progress, utc_now
from database.repositories.evaluations import EvaluationsRepository
from database.repositories.segments import SegmentsRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def resolve_created_at(j: Dict[str, Any]) -> Optional[str]:
    """Resolve job creation timestamp without fabricating current UTC time for historical records.

    Resolution order:
    1. Direct 'created_at' field (datetime or ISO string).
    2. MongoDB ObjectId generation_time extracted from '_id' if present.
    3. Historical timestamps ('started_at' or 'completed_at') if created_at is omitted.
    4. None (preserves nullability without fabricating current time).
    """
    val = j.get("created_at")
    if isinstance(val, datetime):
        return val.isoformat()
    if val:
        return str(val)

    # Fallback to MongoDB ObjectId timestamp if present
    raw_id = j.get("_id")
    if raw_id is not None:
        if hasattr(raw_id, "generation_time"):
            try:
                return raw_id.generation_time.isoformat()
            except Exception:
                pass
        if isinstance(raw_id, str):
            try:
                from bson import ObjectId
                if ObjectId.is_valid(raw_id):
                    return ObjectId(raw_id).generation_time.isoformat()
            except Exception:
                pass

    # Fallback to historical started_at or completed_at
    for fallback_field in ("started_at", "completed_at"):
        fallback_val = j.get(fallback_field)
        if isinstance(fallback_val, datetime):
            return fallback_val.isoformat()
        if fallback_val:
            return str(fallback_val)

    return None


def build_job_status_response(j: Dict[str, Any]) -> JobStatusResponse:
    """Safely construct a JobStatusResponse with backward-compatible schema defaults.
    
    Raises:
        ValueError: If record is not a dict or lacks a valid job identifier ('job_id' or '_id').
    """
    if not isinstance(j, dict):
        raise ValueError("Invalid job record: expected dictionary")

    raw_id = j.get("job_id") or j.get("_id")
    if not raw_id or not str(raw_id).strip():
        raise ValueError("Invalid job record: missing required identifier ('job_id' or '_id')")
    job_id = str(raw_id).strip()

    status = str(j.get("status") or "queued")
    prog = resolve_overall_progress(j)

    # Normalize stages
    raw_stages = j.get("stages")
    stages: List[Dict[str, Any]] = []
    if isinstance(raw_stages, list) and len(raw_stages) > 0:
        for idx, s in enumerate(raw_stages, start=1):
            if isinstance(s, dict):
                stages.append({
                    "id": str(s.get("id") or f"stage_{idx}"),
                    "name": str(s.get("name") or f"Stage {idx}"),
                    "stage_number": int(s.get("stage_number") or idx),
                    "status": str(s.get("status") or ("completed" if status == "completed" else "queued")),
                    "duration_seconds": float(s["duration_seconds"]) if s.get("duration_seconds") is not None else None,
                    "details": s.get("details"),
                })
    else:
        for s in DEFAULT_STAGES:
            stage_copy = dict(s)
            if status == "completed":
                stage_copy["status"] = "completed"
            stages.append(stage_copy)

    # Format timestamp strings safely (preserves nullability without fabricating current time)
    created_at_str = resolve_created_at(j)

    started_at_val = j.get("started_at")
    started_at_str = (
        started_at_val.isoformat()
        if isinstance(started_at_val, datetime)
        else (str(started_at_val) if started_at_val else None)
    )

    completed_at_val = j.get("completed_at")
    completed_at_str = (
        completed_at_val.isoformat()
        if isinstance(completed_at_val, datetime)
        else (str(completed_at_val) if completed_at_val else None)
    )

    dur = j.get("duration_seconds")
    if dur is not None:
        try:
            dur = float(dur)
        except (ValueError, TypeError):
            dur = None

    title = str(j.get("title") or job_id or "Untitled Dubbing Job")
    current_stage = str(j.get("current_stage") or ("finalizing" if status == "completed" else "extracting_audio"))
    current_op = str(j.get("current_operation") or ("Dubbing completed successfully" if status == "completed" else "Job queued"))

    return JobStatusResponse(
        job_id=job_id,
        title=title,
        project_id=j.get("project_id"),
        status=status,
        current_stage=current_stage,
        overall_progress=prog,
        current_operation=current_op,
        created_at=created_at_str,
        started_at=started_at_str,
        completed_at=completed_at_str,
        source_language=str(j.get("source_language") or "en"),
        detected_source_language=j.get("detected_source_language"),
        target_language=str(j.get("target_language") or "hi"),
        duration_seconds=dur,
        enable_lip_sync=bool(j.get("enable_lip_sync", True)),
        voice_mode=str(j.get("voice_mode") or "original"),
        stages=stages,
        error=j.get("error"),
    )


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
    items: List[JobStatusResponse] = []
    for j in jobs:
        try:
            items.append(build_job_status_response(j))
        except (ValueError, TypeError) as err:
            logger.warning("Skipping invalid job document: %s; error: %s", j, err)

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

    try:
        return build_job_status_response(j)
    except ValueError as err:
        logger.error("Job record corrupted for job_id %s: %s", job_id, err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Corrupted job record: {err}",
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
    output_dir_str = j.get("output_dir") or str(JOBS_DIR / (j.get("job_id") or job_id) / "output")
    output_dir = Path(output_dir_str)
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
    output_dir_str = j.get("output_dir") or str(JOBS_DIR / (j.get("job_id") or job_id) / "output")
    output_dir = Path(output_dir_str)
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
            "job_id": str(job.get("job_id") or job_id),
            "stage": str(job.get("current_stage") or "extracting_audio"),
            "status": str(job.get("status") or "queued"),
            "progress": resolve_overall_progress(job),
            "message": str(job.get("current_operation") or "Job queued"),
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

    output_dir_str = j.get("output_dir") or str(JOBS_DIR / (j.get("job_id") or job_id) / "output")
    output_dir = Path(output_dir_str)
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
    completed_at_val = j.get("completed_at")
    completed_at_str = (
        completed_at_val.isoformat()
        if isinstance(completed_at_val, datetime)
        else (str(completed_at_val) if completed_at_val else None)
    )
    dur = j.get("duration_seconds")
    if dur is not None:
        try:
            dur = float(dur)
        except (ValueError, TypeError):
            dur = None

    return JobResultResponse(
        job_id=str(j.get("job_id") or j.get("_id") or job_id),
        title=str(j.get("title") or j.get("job_id") or "Untitled Dubbing Job"),
        project_id=j.get("project_id"),
        status=str(j.get("status") or "completed"),
        video_url=artifacts.get("dubbed_video_url"),
        audio_url=artifacts.get("dubbed_audio_url"),
        original_video_url=artifacts.get("original_video_url"),
        transcript_url=artifacts.get("transcript_json_url"),
        translation_url=artifacts.get("translation_json_url"),
        source_language=str(j.get("source_language") or "en"),
        detected_source_language=j.get("detected_source_language"),
        target_language=str(j.get("target_language") or "hi"),
        duration_seconds=dur,
        voice_mode=str(j.get("voice_mode") or "original"),
        enable_lip_sync=bool(j.get("enable_lip_sync", True)),
        dialogue_segments=dialogue_segments,
        completed_at=completed_at_str,
    )


@router.get("/{job_id}/metrics")
async def get_job_metrics(job_id: str, current_user: UserModel = Depends(get_current_user)):
    """Retrieve actual stage 3 and 4 evaluation metrics if available."""
    j = job_manager.get_job_for_user(job_id, current_user.user_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir_str = j.get("output_dir") or str(JOBS_DIR / (j.get("job_id") or job_id) / "output")
    output_dir = Path(output_dir_str)
    metrics_data: Dict[str, Any] = {
        "job_id": job_id,
        "status": str(j.get("status") or "unknown"),
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
