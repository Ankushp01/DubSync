from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CreateJobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str = "Job created and queued for execution"


class StageInfo(BaseModel):
    id: str
    name: str
    stage_number: int
    status: str  # queued, processing, completed, failed, skipped
    duration_seconds: Optional[float] = None
    details: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    title: str
    status: str  # queued, processing, completed, failed, cancelled
    current_stage: str
    overall_progress: int  # 0 to 100
    current_operation: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    source_language: str
    target_language: str
    duration_seconds: Optional[float] = None
    enable_lip_sync: bool
    voice_mode: str
    stages: List[StageInfo]
    error: Optional[str] = None


class DialogueSegmentResponse(BaseModel):
    index: int
    start: float
    end: float
    source_text: str
    translated_text: Optional[str] = None
    target_duration: Optional[float] = None
    actual_duration: Optional[float] = None


class JobResultResponse(BaseModel):
    job_id: str
    title: str
    status: str
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    original_video_url: Optional[str] = None
    transcript_url: Optional[str] = None
    translation_url: Optional[str] = None
    source_language: str
    target_language: str
    duration_seconds: Optional[float] = None
    voice_mode: str
    enable_lip_sync: bool
    dialogue_segments: List[DialogueSegmentResponse] = []
    completed_at: Optional[str] = None


class JobEvent(BaseModel):
    job_id: str
    stage: str
    status: str
    progress: int
    message: str
    error: Optional[str] = None
