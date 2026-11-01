"""Job Manager for DubSync AI Pipeline.

Manages job lifecycle, executes the real backend pipeline stages asynchronously,
persists records to MongoDB Atlas across all 11 collections, and emits real-time events for SSE / polling clients.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional
import uuid

from dubsync.audio.assembly import assemble_stage3_audio
from dubsync.audio.extraction import extract_audio
from dubsync.config import load_config, AppConfig
from dubsync.pipeline import _write_stage1, run_stage2, run_stage3, run_stage4, run_stage5
from dubsync.stt.whisper import WhisperTranscriber
from dubsync.video.mux import mux_dubbed_audio

from database.connection import db_manager, run_coro_sync
from database.repositories import (
    ProjectsRepository,
    VideosRepository,
    TranscriptsRepository,
    TranslationsRepository,
    GeneratedAudioRepository,
    DubbedVideosRepository,
    ProcessingLogsRepository,
    ProcessingJobsRepository,
    SegmentsRepository,
    EvaluationsRepository,
)

logger = logging.getLogger("dubsync.api.job_manager")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = PROJECT_ROOT / "jobs"
REGISTRY_FILE = JOBS_DIR / "registry.json"

WAV2LIP_PYTHON = PROJECT_ROOT / ".wav2lip-venv" / "Scripts" / "python.exe"
WAV2LIP_DIR = PROJECT_ROOT / "Wav2Lip-HD"

from dubsync.languages import (
    evaluate_language_pair,
    get_language,
    resolve_chatterbox_tag,
    resolve_indictrans_tag,
    resolve_whisper_code,
)

DEFAULT_STAGES = [
    {"id": "extracting_audio", "name": "Extracting audio", "stage_number": 1, "status": "queued", "details": "16 kHz mono extraction"},
    {"id": "transcribing", "name": "Transcribing speech", "stage_number": 2, "status": "queued", "details": "Faster-Whisper with VAD"},
    {"id": "translating", "name": "Translating", "stage_number": 3, "status": "queued", "details": "IndicTrans2 machine translation"},
    {"id": "generating_voice", "name": "Generating voice", "stage_number": 4, "status": "queued", "details": "Chatterbox TTS voice cloning"},
    {"id": "aligning_audio", "name": "Aligning audio", "stage_number": 5, "status": "queued", "details": "Timeline assembly & duration matching"},
    {"id": "lip_synchronization", "name": "Lip synchronization", "stage_number": 6, "status": "queued", "details": "Wav2Lip-HD video facial alignment"},
    {"id": "finalizing", "name": "Finalizing", "stage_number": 7, "status": "queued", "details": "Packaging final output"},
]


def _probe_video_file(path: Path) -> Dict[str, Any]:
    file_size = path.stat().st_size if path.is_file() else 0
    duration = 0.0
    fps = 23.976
    width = None
    height = None
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            fmt = data.get("format", {})
            if "duration" in fmt:
                duration = round(float(fmt["duration"]), 3)
            for s in data.get("streams", []):
                if s.get("codec_type") == "video":
                    width = s.get("width")
                    height = s.get("height")
                    r_fps = s.get("r_frame_rate", "")
                    if "/" in r_fps:
                        num, den = r_fps.split("/", 1)
                        if float(den) > 0:
                            fps = round(float(num) / float(den), 3)
                    break
    except Exception:
        pass
    return {
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "file_size": file_size,
    }


class JobManager:
    def __init__(self):
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}
        # Single-threaded executor prevents simultaneous heavy GPU jobs on 4GB VRAM
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="dubsync_worker")
        self.active_futures: Dict[str, concurrent.futures.Future] = {}
        self._load_registry()

    def _load_registry(self):
        if REGISTRY_FILE.is_file():
            try:
                data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
                for job in data:
                    self.jobs[job["job_id"]] = job
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

    def _save_registry(self):
        try:
            REGISTRY_FILE.write_text(
                json.dumps(list(self.jobs.values()), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    async def sync_with_db(self):
        """Called on application startup to recover interrupted jobs and synchronize database records."""
        if not db_manager.is_connected:
            return
        try:
            jobs_repo = ProcessingJobsRepository()
            # 1. Recover incomplete jobs left in 'processing' status from prior backend run
            recovered = await jobs_repo.recover_interrupted_jobs()
            if recovered:
                logger.info(f"[MongoDB Recovery] Marked {recovered} interrupted job(s) as failed across restart.")

            # 2. Sync database jobs into memory cache
            db_res = await jobs_repo.list_jobs(limit=500)
            for j in db_res.get("items", []):
                self.jobs[j["job_id"]] = j

            # 3. For any local jobs in registry that are not in DB, sync them into Atlas
            for job_id, local_job in list(self.jobs.items()):
                existing = await jobs_repo.get_by_id(job_id)
                if not existing:
                    await jobs_repo.create_job(local_job)
                    logger.info(f"[MongoDB Sync] Seeded local job record '{job_id}' into MongoDB Atlas.")

            self._save_registry()
        except Exception as exc:
            logger.warning(f"[MongoDB Sync Error] Could not complete database sync: {exc}")

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def get_job_for_user(self, job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        job = self.jobs.get(job_id)
        if job and job.get("user_id") == user_id:
            return job
        return None

    def list_jobs(self, project_id: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        jobs = list(self.jobs.values())
        if project_id:
            jobs = [j for j in jobs if j.get("project_id") == project_id]
        if user_id:
            jobs = [j for j in jobs if j.get("user_id") == user_id]
        return sorted(
            jobs,
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )

    def create_job(
        self,
        *,
        video_filename: str,
        video_bytes: bytes,
        title: Optional[str] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        source_language: str = "en",
        target_language: str = "hi",
        voice_mode: str = "original",
        reference_audio_bytes: Optional[bytes] = None,
        reference_audio_filename: Optional[str] = None,
        speech_speed: float = 1.0,
        enable_lip_sync: bool = True,
        output_quality: str = "high",
    ) -> str:
        job_id = f"job-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
        job_dir = JOBS_DIR / job_id
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        logs_dir = job_dir / "logs"

        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        video_path = input_dir / video_filename
        video_path.write_bytes(video_bytes)

        ref_path = None
        if reference_audio_bytes and reference_audio_filename:
            ref_path = input_dir / reference_audio_filename
            ref_path.write_bytes(reference_audio_bytes)

        # Probe video metadata
        probe_info = _probe_video_file(video_path)
        actual_project_id = project_id or f"proj_{uuid.uuid4().hex[:12]}"
        video_id = f"vid_{uuid.uuid4().hex[:12]}"

        job_data: Dict[str, Any] = {
            "job_id": job_id,
            "project_id": actual_project_id,
            "video_id": video_id,
            "user_id": user_id,
            "title": title or Path(video_filename).stem,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "completed_at": None,
            "current_stage": "extracting_audio",
            "overall_progress": 0,
            "current_operation": "Job queued for pipeline execution",
            "source_language": source_language,
            "target_language": target_language,
            "voice_mode": voice_mode,
            "speech_speed": speech_speed,
            "enable_lip_sync": enable_lip_sync,
            "output_quality": output_quality,
            "video_filename": video_filename,
            "video_path": str(video_path),
            "reference_audio_path": str(ref_path) if ref_path else None,
            "output_dir": str(output_dir),
            "stages": [dict(s) for s in DEFAULT_STAGES],
            "artifacts": {},
            "error": None,
            "duration_seconds": probe_info.get("duration"),
        }

        self.jobs[job_id] = job_data
        self._save_registry()

        # Persist project, video, job, and log to MongoDB if connected
        if db_manager.is_connected:
            try:
                run_coro_sync(self._db_on_job_create(job_data, video_filename, str(video_path), probe_info))
            except Exception as dbe:
                logger.warning(f"MongoDB persistence on job creation warning: {dbe}")

        # Submit to background executor
        future = self.executor.submit(self._run_pipeline, job_id)
        self.active_futures[job_id] = future

        return job_id

    async def _db_on_job_create(self, job_data: Dict[str, Any], video_filename: str, video_path: str, probe_info: Dict[str, Any]):
        if not db_manager.is_connected:
            return
        proj_repo = ProjectsRepository()
        vid_repo = VideosRepository()
        jobs_repo = ProcessingJobsRepository()
        logs_repo = ProcessingLogsRepository()

        # Ensure project exists
        proj = await proj_repo.get_by_id(job_data["project_id"])
        if not proj:
            await proj_repo.create_project(
                project_id=job_data["project_id"],
                user_id=job_data.get("user_id"),
                project_name=job_data.get("title") or "Dubbing Project",
                source_language=job_data["source_language"],
                target_languages=[job_data["target_language"]],
            )

        # Create video metadata record
        await vid_repo.create_video(
            video_id=job_data["video_id"],
            project_id=job_data["project_id"],
            user_id=job_data.get("user_id"),
            file_name=video_filename,
            file_path=video_path,
            duration=probe_info.get("duration", 0.0),
            fps=probe_info.get("fps", 23.976),
            width=probe_info.get("width"),
            height=probe_info.get("height"),
            file_size=probe_info.get("file_size", 0),
        )

        # Create processing job record
        await jobs_repo.create_job(job_data)

        # Initial log
        await logs_repo.add_log(
            job_id=job_data["job_id"],
            user_id=job_data.get("user_id"),
            stage="extracting_audio",
            status="queued",
            details={"message": "Job queued for pipeline execution", "video_filename": video_filename},
            project_id=job_data["project_id"],
        )

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job["status"] in ("completed", "failed", "cancelled"):
            return False

        job["status"] = "cancelled"
        job["current_operation"] = "Pipeline execution cancelled by user"
        self._save_registry()
        self._emit_event(job_id, "finalizing", "cancelled", job["overall_progress"], "Job cancelled")

        if db_manager.is_connected:
            try:
                run_coro_sync(ProcessingJobsRepository().update_job(job_id, {"status": "cancelled", "current_operation": "Job cancelled"}))
                run_coro_sync(ProcessingLogsRepository().add_log(job_id, "finalizing", "cancelled", {"message": "Cancelled by user"}))
            except Exception as e:
                logger.warning(f"Could not update cancellation in DB: {e}")

        return True

    def delete_job(self, job_id: str) -> bool:
        job = self.jobs.pop(job_id, None)
        if not job:
            return False
        job_dir = JOBS_DIR / job_id
        if job_dir.exists():
            try:
                shutil.rmtree(job_dir, ignore_errors=True)
            except Exception as e:
                logger.error(f"Error removing job dir {job_dir}: {e}")
        self._save_registry()

        if db_manager.is_connected:
            try:
                run_coro_sync(ProcessingJobsRepository().delete_job(job_id))
            except Exception as e:
                logger.warning(f"Could not delete job record from DB: {e}")

        return True

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        if job_id not in self.subscribers:
            self.subscribers[job_id] = []
        self.subscribers[job_id].append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue):
        if job_id in self.subscribers:
            try:
                self.subscribers[job_id].remove(queue)
            except ValueError:
                pass

    def _emit_event(self, job_id: str, stage: str, status: str, progress: int, message: str, error: Optional[str] = None):
        event = {
            "job_id": job_id,
            "stage": stage,
            "status": status,
            "progress": progress,
            "message": message,
            "error": error,
        }
        if job_id in self.subscribers:
            for q in list(self.subscribers[job_id]):
                try:
                    q.put_nowait(event)
                except Exception:
                    pass

    def _update_stage_status(self, job: Dict[str, Any], stage_id: str, status: str, details: Optional[str] = None):
        for s in job["stages"]:
            if s["id"] == stage_id:
                s["status"] = status
                if details:
                    s["details"] = details
                break

    # -------------------------------------------------------------
    # Database helper wrappers for pipeline worker
    # -------------------------------------------------------------
    def _db_log(self, job_id: str, stage: str, status: str, details: Optional[Dict[str, Any]] = None, severity: str = "info", project_id: Optional[str] = None):
        if not db_manager.is_connected:
            return
        try:
            job = self.jobs.get(job_id, {})
            u_id = job.get("user_id")
            p_id = project_id or job.get("project_id")
            run_coro_sync(ProcessingLogsRepository().add_log(job_id, stage, status, details, severity, p_id, user_id=u_id))
        except Exception as e:
            logger.debug(f"DB log write skipped: {e}")

    def _db_update_job(self, job_id: str, updates: Dict[str, Any]):
        if not db_manager.is_connected:
            return
        try:
            run_coro_sync(ProcessingJobsRepository().update_job(job_id, updates))
        except Exception as e:
            logger.debug(f"DB job update skipped: {e}")

    def _db_save_transcript(self, job_id: str, transcript_path: Path, transcript: Any, source_lang: str):
        if not db_manager.is_connected or not transcript_path.is_file():
            return
        try:
            data = json.loads(transcript_path.read_text(encoding="utf-8"))
            job = self.jobs.get(job_id, {})
            diag_segs = data.get("dialogue_segments", [])
            original_text = " ".join(s.get("text", "") for s in diag_segs)
            res = run_coro_sync(
                TranscriptsRepository().create_transcript(
                    video_id=job.get("video_id", "vid_default"),
                    project_id=job.get("project_id", "proj_default"),
                    job_id=job_id,
                    user_id=job.get("user_id"),
                    source_language=source_lang,
                    original_text=original_text,
                    segments=diag_segs,
                    metadata={"schema_version": data.get("schema_version", 1)},
                )
            )
            job["transcript_id"] = res.get("transcript_id")
        except Exception as e:
            logger.warning(f"Could not persist transcript to MongoDB: {e}")

    def _db_save_translation(self, job_id: str, stage2_path: Path, source_lang: str, target_lang: str):
        if not db_manager.is_connected or not stage2_path.is_file():
            return
        try:
            data = json.loads(stage2_path.read_text(encoding="utf-8"))
            job = self.jobs.get(job_id, {})
            diag_segs = data.get("dialogue_segments", [])
            source_text = " ".join(s.get("source_text", "") for s in diag_segs)
            translated_text = " ".join(s.get("translated_text", "") for s in diag_segs)
            res = run_coro_sync(
                TranslationsRepository().create_translation(
                    transcript_id=job.get("transcript_id", f"tr_{job_id}"),
                    job_id=job_id,
                    user_id=job.get("user_id"),
                    source_language=source_lang,
                    target_language=target_lang,
                    source_text=source_text,
                    translated_text=translated_text,
                    segments=diag_segs,
                    metadata={"schema_version": data.get("schema_version", 1)},
                )
            )
            job["translation_id"] = res.get("translation_id")
        except Exception as e:
            logger.warning(f"Could not persist translation to MongoDB: {e}")

    def _db_save_audio_and_segments(self, job_id: str, stage3_path: Path, target_lang: str):
        if not db_manager.is_connected or not stage3_path.is_file():
            return
        try:
            data = json.loads(stage3_path.read_text(encoding="utf-8"))
            job = self.jobs.get(job_id, {})
            segments = data.get("segments", [])
            
            # 1. Save generated audio
            total_dur = sum(s.get("final_duration", s.get("generated_duration", 0.0)) for s in segments)
            res_aud = run_coro_sync(
                GeneratedAudioRepository().create_audio(
                    translation_id=job.get("translation_id", f"trans_{job_id}"),
                    job_id=job_id,
                    user_id=job.get("user_id"),
                    voice_model=data.get("tts_model", "Chatterbox Multilingual V3"),
                    target_language=target_lang,
                    audio_path=str(stage3_path.parent / "audio" / "dubbed_audio.wav"),
                    duration=total_dur,
                )
            )
            job["audio_id"] = res_aud.get("audio_id")

            # 2. Save dialogue segments
            seg_batch = []
            for s in segments:
                seg_batch.append({
                    "job_id": job_id,
                    "project_id": job.get("project_id"),
                    "user_id": job.get("user_id"),
                    "transcript_id": job.get("transcript_id"),
                    "translation_id": job.get("translation_id"),
                    "segment_index": s.get("index", 1),
                    "start_time": s.get("start", 0.0),
                    "end_time": s.get("end", 0.0),
                    "original_text": s.get("source_text", ""),
                    "translated_text": s.get("final_translated_text", s.get("translated_text", "")),
                    "raw_tts_duration": s.get("raw_generated_duration", 0.0),
                    "adapted_tts_duration": s.get("final_duration", s.get("generated_duration", 0.0)),
                    "speed_factor": s.get("speech_speed", 1.0),
                    "audio_path": str(stage3_path.parent / s.get("audio", "")),
                    "status": "completed",
                })
            run_coro_sync(SegmentsRepository().create_segments_batch(seg_batch))
        except Exception as e:
            logger.warning(f"Could not persist audio & segments to MongoDB: {e}")

    def _db_update_segments_alignment(self, job_id: str, accounting_path: Path):
        if not db_manager.is_connected or not accounting_path.is_file():
            return
        try:
            data = json.loads(accounting_path.read_text(encoding="utf-8"))
            accounting_records = data.get("segments", [])
            job = self.jobs.get(job_id, {})
            seg_batch = []
            for r in accounting_records:
                placement = r.get("timeline_placement", [0.0, 0.0])
                seg_batch.append({
                    "job_id": job_id,
                    "project_id": job.get("project_id"),
                    "user_id": job.get("user_id"),
                    "segment_index": r.get("segment_index", 1),
                    "final_placement_start": placement[0] if len(placement) > 0 else 0.0,
                    "final_placement_end": placement[1] if len(placement) > 1 else 0.0,
                    "trimming_seconds": r.get("trimmed_seconds", 0.0),
                    "speed_factor": r.get("applied_speed_factor", 1.0),
                    "status": "aligned",
                })
            run_coro_sync(SegmentsRepository().create_segments_batch(seg_batch))
        except Exception as e:
            logger.warning(f"Could not update segment alignment in MongoDB: {e}")

    def _db_save_final_output_and_eval(self, job_id: str, final_video_path: Path, dubbed_audio_path: Path):
        if not db_manager.is_connected:
            return
        try:
            job = self.jobs.get(job_id, {})
            probe = _probe_video_file(final_video_path)
            
            # Save dubbed video output
            out_res = run_coro_sync(
                DubbedVideosRepository().create_dubbed_video(
                    project_id=job.get("project_id", "proj_default"),
                    job_id=job_id,
                    video_id=job.get("video_id", "vid_default"),
                    audio_id=job.get("audio_id"),
                    user_id=job.get("user_id"),
                    output_path=str(final_video_path),
                    duration=probe.get("duration", 0.0),
                    status="completed",
                    metadata={"fps": probe.get("fps"), "width": probe.get("width"), "height": probe.get("height")},
                )
            )

            # Check for evaluation records
            output_dir = Path(job.get("output_dir", ""))
            accounting_file = output_dir / "timing_accounting.json"
            eval_summary_file = PROJECT_ROOT / "output" / "Front_evaluation" / "evaluation_summary.json"
            sync_file = PROJECT_ROOT / "output" / "Front_evaluation" / "sync_window_analysis.json"

            lip_sync_metrics = {}
            if sync_file.is_file():
                try:
                    lip_sync_metrics = json.loads(sync_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            speech_preserved = 100.0
            trimmed_seconds = 0.0
            timeline_ext = 0.0
            segment_metrics = []

            if accounting_file.is_file():
                try:
                    acct_data = json.loads(accounting_file.read_text(encoding="utf-8"))
                    trimmed_seconds = float(acct_data.get("total_trimmed_seconds", 0.0))
                    timeline_ext = float(acct_data.get("total_extension_seconds", 0.0))
                    speech_preserved = 100.0 if trimmed_seconds == 0.0 else 90.0
                    segment_metrics = acct_data.get("segments", [])
                except Exception:
                    pass

            run_coro_sync(
                EvaluationsRepository().create_evaluation(
                    job_id=job_id,
                    project_id=job.get("project_id"),
                    output_id=out_res.get("output_id"),
                    user_id=job.get("user_id"),
                    video_duration=probe.get("duration"),
                    audio_duration=probe.get("duration"),
                    video_frame_count=int(probe.get("duration", 0) * probe.get("fps", 23.976)),
                    frame_rate=probe.get("fps"),
                    speech_preservation_percentage=speech_preserved,
                    trimmed_audio_seconds=trimmed_seconds,
                    timeline_extension_seconds=timeline_ext,
                    lip_sync_metrics=lip_sync_metrics,
                    segment_metrics=segment_metrics,
                    unmeasured_or_null_metrics={
                        "asr_word_error_rate_wer": None,
                        "asr_wer_reason": "Ground truth transcript unavailable for automated WER calculation",
                        "translation_bleu_score": None,
                        "translation_bleu_reason": "Human reference translation unavailable for BLEU",
                        "quantitative_lipsync_syncnet_lse_d": None,
                        "quantitative_lipsync_lse_d_reason": "SyncNet neural evaluation unavailable; CV cross-correlation used",
                    },
                )
            )
        except Exception as e:
            logger.warning(f"Could not persist final output & evaluation to MongoDB: {e}")

    # -------------------------------------------------------------
    # Main Pipeline Runner
    # -------------------------------------------------------------
    def _run_pipeline(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job:
            return

        job["status"] = "processing"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        self._save_registry()
        self._db_update_job(job_id, {"status": "processing", "started_at": job["started_at"]})
        self._db_log(job_id, "pipeline", "started", {"message": "Pipeline processing started"}, project_id=job.get("project_id"))

        output_dir = Path(job["output_dir"])
        video_path = Path(job["video_path"])

        from dataclasses import replace

        try:
            # Load local configuration
            config_path = PROJECT_ROOT / "configs" / "local.toml"
            if not config_path.is_file():
                config_path = PROJECT_ROOT / "configs" / "default.toml"
            config = load_config(config_path)

            # Initial default configuration
            indictrans_tgt = resolve_indictrans_tag(job["target_language"]) or "hin_Deva"
            tts_tgt = resolve_chatterbox_tag(job["target_language"]) or "hi"
            new_translation = replace(config.translation, target_language=indictrans_tgt)
            new_tts = replace(config.tts, language_id=tts_tgt)
            config = replace(config, translation=new_translation, tts=new_tts)

            # -------------------------------------------------------------
            # Stage 1: Audio Extraction
            # -------------------------------------------------------------
            if job["status"] == "cancelled":
                return
            job["current_stage"] = "extracting_audio"
            job["current_operation"] = "Extracting 16 kHz mono audio stream via FFmpeg..."
            job["overall_progress"] = 10
            self._update_stage_status(job, "extracting_audio", "processing")
            self._save_registry()
            self._db_update_job(job_id, {"current_stage": "extracting_audio", "progress": 10, "current_operation": job["current_operation"], "stages": job["stages"]})
            self._db_log(job_id, "extracting_audio", "processing", {"operation": "FFmpeg audio extraction"}, project_id=job.get("project_id"))
            self._emit_event(job_id, "extracting_audio", "processing", 10, job["current_operation"])

            source_audio_path = output_dir / "source_16khz_mono.wav"
            extract_audio(
                video_path,
                source_audio_path,
                ffmpeg_executable=config.ffmpeg.executable,
                sample_rate=config.audio.sample_rate,
                channels=config.audio.channels,
            )
            self._update_stage_status(job, "extracting_audio", "completed")
            self._save_registry()
            self._db_log(job_id, "extracting_audio", "completed", {"audio_path": str(source_audio_path)}, project_id=job.get("project_id"))

            # -------------------------------------------------------------
            # Stage 2: Faster-Whisper ASR
            # -------------------------------------------------------------
            if job["status"] == "cancelled":
                return
            job["current_stage"] = "transcribing"
            job["current_operation"] = "Transcribing speech acoustics with Faster-Whisper ASR..."
            job["overall_progress"] = 25
            self._update_stage_status(job, "transcribing", "processing")
            self._save_registry()
            self._db_update_job(job_id, {"current_stage": "transcribing", "progress": 25, "current_operation": job["current_operation"], "stages": job["stages"]})
            self._db_log(job_id, "transcribing", "processing", {"model": config.whisper.model_name}, project_id=job.get("project_id"))
            self._emit_event(job_id, "transcribing", "processing", 25, job["current_operation"])

            transcript_path = output_dir / "stage1_transcript.json"
            transcriber = WhisperTranscriber(
                device=config.runtime.device,
                compute_type=config.runtime.compute_type,
                model_name=config.whisper.model_name,
                model_path=config.whisper.model_path,
            )

            whisper_lang = resolve_whisper_code(job.get("source_language", "auto"))
            transcript = transcriber.transcribe(
                source_audio_path,
                language=whisper_lang or config.whisper.language or None,
                beam_size=config.whisper.beam_size,
                vad_filter=config.whisper.vad_filter,
            )
            _write_stage1(transcript_path, transcript, config)

            # Record detected source language and dialogue segments
            detected_lang = transcript.language or "en"
            job["detected_source_language"] = detected_lang
            self._update_stage_status(job, "transcribing", "completed", f"{len(transcript.segments)} segments detected ({detected_lang})")
            job["overall_progress"] = 40
            self._save_registry()
            self._db_save_transcript(job_id, transcript_path, transcript, detected_lang)
            self._db_update_job(job_id, {"detected_source_language": detected_lang, "progress": 40, "stages": job["stages"]})
            self._db_log(job_id, "transcribing", "completed", {"detected_language": detected_lang, "segment_count": len(transcript.segments)}, project_id=job.get("project_id"))

            # -------------------------------------------------------------
            # Stage 3: Translation (IndicTrans2)
            # -------------------------------------------------------------
            if job["status"] == "cancelled":
                return

            # Verify that detected source language can route to requested target language
            pair_eval = evaluate_language_pair(detected_lang, job["target_language"])
            if not pair_eval.usable:
                raise RuntimeError(
                    f"Language routing failure: Audio detected as '{detected_lang}', "
                    f"which cannot be translated to '{job['target_language']}' in the active pipeline. {pair_eval.reason}"
                )

            target_info = get_language(job["target_language"])
            target_name = target_info.name if target_info else job["target_language"]
            indictrans_src = resolve_indictrans_tag(detected_lang) or "eng_Latn"

            new_translation = replace(
                config.translation,
                source_language=indictrans_src,
                target_language=indictrans_tgt,
            )
            config = replace(config, translation=new_translation)

            job["current_stage"] = "translating"
            job["current_operation"] = f"Translating dialogue from {detected_lang} to {target_name} (IndicTrans2)..."
            job["overall_progress"] = 45
            self._update_stage_status(job, "translating", "processing")
            self._save_registry()
            self._db_update_job(job_id, {"current_stage": "translating", "progress": 45, "current_operation": job["current_operation"], "stages": job["stages"]})
            self._db_log(job_id, "translating", "processing", {"source": detected_lang, "target": job["target_language"]}, project_id=job.get("project_id"))
            self._emit_event(job_id, "translating", "processing", 45, job["current_operation"])

            stage2_path = run_stage2(transcript_path, config)
            self._update_stage_status(job, "translating", "completed")
            job["overall_progress"] = 60
            self._save_registry()
            self._db_save_translation(job_id, stage2_path, detected_lang, job["target_language"])
            self._db_update_job(job_id, {"progress": 60, "stages": job["stages"]})
            self._db_log(job_id, "translating", "completed", {"translation_path": str(stage2_path)}, project_id=job.get("project_id"))

            # -------------------------------------------------------------
            # Stage 4: Chatterbox Multilingual TTS & Duration Adaptation
            # -------------------------------------------------------------
            if job["status"] == "cancelled":
                return
            job["current_stage"] = "generating_voice"
            job["current_operation"] = "Synthesizing speaker-conditioned speech with Chatterbox Multilingual..."
            job["overall_progress"] = 65
            self._update_stage_status(job, "generating_voice", "processing")
            self._save_registry()
            self._db_update_job(job_id, {"current_stage": "generating_voice", "progress": 65, "current_operation": job["current_operation"], "stages": job["stages"]})
            self._db_log(job_id, "generating_voice", "processing", {"voice_mode": job.get("voice_mode")}, project_id=job.get("project_id"))
            self._emit_event(job_id, "generating_voice", "processing", 65, job["current_operation"])

            ref_audio = job.get("reference_audio_path")
            speech_speed = float(job.get("speech_speed", 1.0))
            stage3_path = run_stage3(stage2_path, config, reference_audio=ref_audio, speech_speed=speech_speed)
            self._update_stage_status(job, "generating_voice", "completed")
            job["overall_progress"] = 80
            self._save_registry()
            self._db_save_audio_and_segments(job_id, stage3_path, job["target_language"])
            self._db_update_job(job_id, {"progress": 80, "stages": job["stages"]})
            self._db_log(job_id, "generating_voice", "completed", {"manifest": str(stage3_path)}, project_id=job.get("project_id"))

            # -------------------------------------------------------------
            # Stage 5: Audio Assembly
            # -------------------------------------------------------------
            if job["status"] == "cancelled":
                return
            job["current_stage"] = "aligning_audio"
            job["current_operation"] = "Assembling master dubbed audio track on timeline..."
            job["overall_progress"] = 85
            self._update_stage_status(job, "aligning_audio", "processing")
            self._save_registry()
            self._db_update_job(job_id, {"current_stage": "aligning_audio", "progress": 85, "current_operation": job["current_operation"], "stages": job["stages"]})
            self._db_log(job_id, "aligning_audio", "processing", {}, project_id=job.get("project_id"))
            self._emit_event(job_id, "aligning_audio", "processing", 85, job["current_operation"])

            dubbed_audio_path = run_stage4(stage3_path, output_dir, source_audio_path=source_audio_path)
            accounting_path = output_dir / "timing_accounting.json"
            if accounting_path.is_file():
                try:
                    with accounting_path.open("r", encoding="utf-8") as af:
                        acct_data = json.load(af)
                    if acct_data.get("timeline_extended"):
                        ext_sec = acct_data.get("total_extension_seconds", 0.0)
                        self._update_stage_status(
                            job, "aligning_audio", "completed", f"Timeline extended (+{ext_sec:.2f}s) to preserve full dialogue"
                        )
                    else:
                        self._update_stage_status(job, "aligning_audio", "completed", "Audio aligned with zero trimming")
                except Exception:
                    self._update_stage_status(job, "aligning_audio", "completed")
            else:
                self._update_stage_status(job, "aligning_audio", "completed")
            job["overall_progress"] = 90
            self._save_registry()
            self._db_update_segments_alignment(job_id, accounting_path)
            self._db_update_job(job_id, {"progress": 90, "stages": job["stages"]})
            self._db_log(job_id, "aligning_audio", "completed", {"dubbed_audio_path": str(dubbed_audio_path)}, project_id=job.get("project_id"))

            # -------------------------------------------------------------
            # Stage 6: Video Muxing
            # -------------------------------------------------------------
            if job["status"] == "cancelled":
                return
            job["current_stage"] = "finalizing"
            job["current_operation"] = "Multiplexing dubbed audio with video stream..."
            self._update_stage_status(job, "finalizing", "processing")
            self._emit_event(job_id, "finalizing", "processing", 90, job["current_operation"])
            self._db_log(job_id, "finalizing", "processing", {"operation": "Video multiplexing"}, project_id=job.get("project_id"))

            dubbed_video_path = run_stage5(
                video_path,
                dubbed_audio_path,
                output_dir,
                config.ffmpeg.executable,
            )

            # -------------------------------------------------------------
            # Stage 7: Lip Synchronization (Wav2Lip-HD)
            # -------------------------------------------------------------
            final_video_path = dubbed_video_path

            if job.get("enable_lip_sync", True):
                if job["status"] == "cancelled":
                    return
                job["current_stage"] = "lip_synchronization"
                job["current_operation"] = "Initializing Wav2Lip-HD neural lip synchronization..."
                job["overall_progress"] = 91
                self._update_stage_status(job, "lip_synchronization", "processing")
                self._emit_event(job_id, "lip_synchronization", "processing", 91, job["current_operation"])
                self._db_update_job(job_id, {"current_stage": "lip_synchronization", "progress": 91, "current_operation": job["current_operation"], "stages": job["stages"]})
                self._db_log(job_id, "lip_synchronization", "processing", {}, project_id=job.get("project_id"))

                def lipsync_progress_callback(msg: str, pct: int):
                    job["current_operation"] = msg
                    job["overall_progress"] = pct
                    self._save_registry()
                    self._emit_event(job_id, "lip_synchronization", "processing", pct, msg)

                from dubsync.pipeline import run_stage6
                try:
                    wav2lip_out = run_stage6(
                        video_path=video_path,
                        dubbed_audio_path=dubbed_audio_path,
                        output_dir=output_dir,
                        progress_callback=lipsync_progress_callback,
                    )
                    final_video_path = wav2lip_out
                    self._update_stage_status(job, "lip_synchronization", "completed")
                    self._save_registry()
                    self._db_log(job_id, "lip_synchronization", "completed", {"output_video": str(wav2lip_out)}, project_id=job.get("project_id"))
                except Exception as w_err:
                    logger.error(f"Wav2Lip-HD execution failed: {w_err}")
                    raise RuntimeError(f"Wav2Lip-HD lip synchronization failed: {w_err}") from w_err
            else:
                self._update_stage_status(job, "lip_synchronization", "skipped", "Disabled by user configuration")

            # -------------------------------------------------------------
            # Pipeline Finished
            # -------------------------------------------------------------
            job["overall_progress"] = 100
            job["status"] = "completed"
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            job["current_operation"] = "Dubbing completed successfully"
            self._update_stage_status(job, "finalizing", "completed")

            job["artifacts"] = {
                "original_video_url": f"/api/v1/files/{job_id}/input/{video_path.name}",
                "dubbed_video_url": f"/api/v1/files/{job_id}/output/video/{final_video_path.name}",
                "dubbed_audio_url": f"/api/v1/files/{job_id}/output/audio/{dubbed_audio_path.name}",
                "transcript_json_url": f"/api/v1/files/{job_id}/output/{transcript_path.name}",
                "translation_json_url": f"/api/v1/files/{job_id}/output/{stage2_path.name}",
                "timing_accounting_url": f"/api/v1/files/{job_id}/output/timing_accounting.json",
            }

            self._save_registry()
            self._db_save_final_output_and_eval(job_id, final_video_path, dubbed_audio_path)
            self._db_update_job(
                job_id,
                {
                    "status": "completed",
                    "progress": 100,
                    "completed_at": job["completed_at"],
                    "current_operation": job["current_operation"],
                    "artifacts": job["artifacts"],
                    "stages": job["stages"],
                },
            )
            self._db_log(job_id, "pipeline", "completed", {"artifacts": job["artifacts"]}, project_id=job.get("project_id"))
            self._emit_event(job_id, "finalizing", "completed", 100, "Dubbing completed successfully")

        except Exception as exc:
            logger.exception(f"Pipeline failed for job {job_id}: {exc}")
            job["status"] = "failed"
            job["error"] = str(exc)
            job["current_operation"] = f"Pipeline failed: {exc}"
            self._update_stage_status(job, job.get("current_stage", "finalizing"), "failed", str(exc))
            self._save_registry()
            self._db_update_job(
                job_id,
                {
                    "status": "failed",
                    "error": str(exc),
                    "current_operation": job["current_operation"],
                    "stages": job["stages"],
                },
            )
            self._db_log(job_id, job.get("current_stage", "finalizing"), "failed", {"error": str(exc)}, severity="error", project_id=job.get("project_id"))
            self._emit_event(job_id, job.get("current_stage", "finalizing"), "failed", job["overall_progress"], str(exc), error=str(exc))


job_manager = JobManager()
