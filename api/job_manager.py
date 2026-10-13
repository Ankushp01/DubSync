"""Job Manager for DubSync AI Pipeline.

Manages job lifecycle, executes the real backend pipeline stages asynchronously,
and emits real-time events for SSE / polling clients.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from dubsync.audio.assembly import assemble_stage3_audio
from dubsync.audio.extraction import extract_audio
from dubsync.config import load_config, AppConfig
from dubsync.pipeline import _write_stage1, run_stage2, run_stage3, run_stage4, run_stage5
from dubsync.stt.whisper import WhisperTranscriber
from dubsync.video.mux import mux_dubbed_audio

logger = logging.getLogger("dubsync.api.job_manager")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = PROJECT_ROOT / "jobs"
REGISTRY_FILE = JOBS_DIR / "registry.json"

WAV2LIP_PYTHON = PROJECT_ROOT / ".wav2lip-venv" / "Scripts" / "python.exe"
WAV2LIP_DIR = PROJECT_ROOT / "Wav2Lip-HD"

LANGUAGE_MAP: Dict[str, Dict[str, str]] = {
    "hi": {"indictrans": "hin_Deva", "tts": "hi", "name": "Hindi"},
    "mr": {"indictrans": "mar_Deva", "tts": "mr", "name": "Marathi"},
    "bn": {"indictrans": "ben_Beng", "tts": "bn", "name": "Bengali"},
    "ta": {"indictrans": "tam_Taml", "tts": "ta", "name": "Tamil"},
    "te": {"indictrans": "tel_Telu", "tts": "te", "name": "Telugu"},
    "gu": {"indictrans": "guj_Gujr", "tts": "gu", "name": "Gujarati"},
    "kn": {"indictrans": "kan_Knda", "tts": "kn", "name": "Kannada"},
    "ml": {"indictrans": "mal_Mlym", "tts": "ml", "name": "Malayalam"},
    "pa": {"indictrans": "pan_Guru", "tts": "pa", "name": "Punjabi"},
    "es": {"indictrans": "spa_Latn", "tts": "es", "name": "Spanish"},
    "fr": {"indictrans": "fra_Latn", "tts": "fr", "name": "French"},
    "de": {"indictrans": "deu_Latn", "tts": "de", "name": "German"},
}

DEFAULT_STAGES = [
    {"id": "extracting_audio", "name": "Extracting audio", "stage_number": 1, "status": "queued", "details": "16 kHz mono extraction"},
    {"id": "transcribing", "name": "Transcribing speech", "stage_number": 2, "status": "queued", "details": "Faster-Whisper with VAD"},
    {"id": "translating", "name": "Translating", "stage_number": 3, "status": "queued", "details": "IndicTrans2 machine translation"},
    {"id": "generating_voice", "name": "Generating voice", "stage_number": 4, "status": "queued", "details": "Chatterbox TTS voice cloning"},
    {"id": "aligning_audio", "name": "Aligning audio", "stage_number": 5, "status": "queued", "details": "Timeline assembly & duration matching"},
    {"id": "lip_synchronization", "name": "Lip synchronization", "stage_number": 6, "status": "queued", "details": "Wav2Lip-HD video facial alignment"},
    {"id": "finalizing", "name": "Finalizing", "stage_number": 7, "status": "queued", "details": "Packaging final output"},
]


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

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        return sorted(
            list(self.jobs.values()),
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )

    def create_job(
        self,
        *,
        video_filename: str,
        video_bytes: bytes,
        title: Optional[str] = None,
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

        job_data: Dict[str, Any] = {
            "job_id": job_id,
            "title": title or Path(video_filename).stem,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
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
        }

        self.jobs[job_id] = job_data
        self._save_registry()

        # Submit to background executor
        future = self.executor.submit(self._run_pipeline, job_id)
        self.active_futures[job_id] = future

        return job_id

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

    def _run_pipeline(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job:
            return

        job["status"] = "processing"
        job["started_at"] = datetime.now().isoformat()
        self._save_registry()
        output_dir = Path(job["output_dir"])
        video_path = Path(job["video_path"])

        from dataclasses import replace

        try:
            # Load local configuration
            config_path = PROJECT_ROOT / "configs" / "local.toml"
            if not config_path.is_file():
                config_path = PROJECT_ROOT / "configs" / "default.toml"
            config = load_config(config_path)

            # Map language codes using dataclasses.replace on frozen configs
            lang_info = LANGUAGE_MAP.get(job["target_language"], {"indictrans": "hin_Deva", "tts": "hi"})
            new_translation = replace(config.translation, target_language=lang_info["indictrans"])
            new_tts = replace(config.tts, language_id=lang_info["tts"])
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
            self._emit_event(job_id, "transcribing", "processing", 25, job["current_operation"])

            transcript_path = output_dir / "stage1_transcript.json"
            transcriber = WhisperTranscriber(
                device=config.runtime.device,
                compute_type=config.runtime.compute_type,
                model_name=config.whisper.model_name,
                model_path=config.whisper.model_path,
            )
            transcript = transcriber.transcribe(
                source_audio_path,
                language=config.whisper.language,
                beam_size=config.whisper.beam_size,
                vad_filter=config.whisper.vad_filter,
            )
            _write_stage1(transcript_path, transcript, config)

            # Record detected source language and dialogue segments
            job["detected_source_language"] = transcript.language
            self._update_stage_status(job, "transcribing", "completed", f"{len(transcript.segments)} segments detected")
            job["overall_progress"] = 40
            self._save_registry()

            # -------------------------------------------------------------
            # Stage 3: Translation (IndicTrans2)
            # -------------------------------------------------------------
            if job["status"] == "cancelled":
                return
            job["current_stage"] = "translating"
            job["current_operation"] = f"Translating dialogue to {lang_info.get('name', 'target language')} (IndicTrans2)..."
            job["overall_progress"] = 45
            self._update_stage_status(job, "translating", "processing")
            self._save_registry()
            self._emit_event(job_id, "translating", "processing", 45, job["current_operation"])

            stage2_path = run_stage2(transcript_path, config)
            self._update_stage_status(job, "translating", "completed")
            job["overall_progress"] = 60
            self._save_registry()

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
            self._emit_event(job_id, "generating_voice", "processing", 65, job["current_operation"])

            ref_audio = job.get("reference_audio_path")
            speech_speed = float(job.get("speech_speed", 1.0))
            stage3_path = run_stage3(stage2_path, config, reference_audio=ref_audio, speech_speed=speech_speed)
            self._update_stage_status(job, "generating_voice", "completed")
            job["overall_progress"] = 80
            self._save_registry()

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
            self._emit_event(job_id, "aligning_audio", "processing", 85, job["current_operation"])

            dubbed_audio_path = run_stage4(stage3_path, output_dir, source_audio_path=source_audio_path)
            self._update_stage_status(job, "aligning_audio", "completed")
            job["overall_progress"] = 90
            self._save_registry()

            # -------------------------------------------------------------
            # Stage 6: Video Muxing
            # -------------------------------------------------------------
            if job["status"] == "cancelled":
                return
            job["current_stage"] = "finalizing"
            job["current_operation"] = "Multiplexing dubbed audio with video stream..."
            self._update_stage_status(job, "finalizing", "processing")
            self._emit_event(job_id, "finalizing", "processing", 90, job["current_operation"])

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
            job["completed_at"] = datetime.now().isoformat()
            job["current_operation"] = "Dubbing completed successfully"
            self._update_stage_status(job, "finalizing", "completed")

            job["artifacts"] = {
                "original_video_url": f"/api/v1/files/{job_id}/input/{video_path.name}",
                "dubbed_video_url": f"/api/v1/files/{job_id}/output/video/{final_video_path.name}",
                "dubbed_audio_url": f"/api/v1/files/{job_id}/output/audio/{dubbed_audio_path.name}",
                "transcript_json_url": f"/api/v1/files/{job_id}/output/{transcript_path.name}",
                "translation_json_url": f"/api/v1/files/{job_id}/output/{stage2_path.name}",
            }

            self._save_registry()
            self._emit_event(job_id, "finalizing", "completed", 100, "Dubbing completed successfully")

        except Exception as exc:
            logger.exception(f"Pipeline failed for job {job_id}: {exc}")
            job["status"] = "failed"
            job["error"] = str(exc)
            job["current_operation"] = f"Pipeline failed: {exc}"
            self._update_stage_status(job, job.get("current_stage", "finalizing"), "failed", str(exc))
            self._save_registry()
            self._emit_event(job_id, job.get("current_stage", "finalizing"), "failed", job["overall_progress"], str(exc), error=str(exc))


job_manager = JobManager()
