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
    project_id: Optional[str] = None
    status: str  # queued, processing, completed, failed, cancelled, interrupted
    current_stage: str
    overall_progress: int  # 0 to 100
    current_operation: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    source_language: str
    detected_source_language: Optional[str] = None
    target_language: str
    duration_seconds: Optional[float] = None
    enable_lip_sync: bool
    voice_mode: str
    stages: List[StageInfo]
    error: Optional[str] = None


class PaginatedJobsResponse(BaseModel):
    items: List[JobStatusResponse]
    page: int
    limit: int
    total: int
    total_pages: int


class DialogueSegmentResponse(BaseModel):
    index: int
    start: float
    end: float
    source_text: str
    translated_text: Optional[str] = None
    target_duration: Optional[float] = None
    actual_duration: Optional[float] = None


class SegmentDetailResponse(BaseModel):
    segment_id: str
    segment_index: int
    start_time: float
    end_time: float
    original_text: str
    translated_text: str = ""
    raw_tts_duration: float = 0.0
    adapted_tts_duration: float = 0.0
    speed_factor: float = 1.0
    audio_path: Optional[str] = None
    final_placement_start: float = 0.0
    final_placement_end: float = 0.0
    trimming_seconds: float = 0.0
    status: str = "completed"
    created_at: Optional[str] = None


class EvaluationResponse(BaseModel):
    evaluation_id: str
    job_id: str
    project_id: Optional[str] = None
    output_id: Optional[str] = None
    created_at: str
    video_duration: Optional[float] = None
    audio_duration: Optional[float] = None
    video_frame_count: Optional[int] = None
    frame_rate: Optional[float] = None
    speech_preservation_percentage: Optional[float] = None
    trimmed_audio_seconds: Optional[float] = None
    timeline_extension_seconds: Optional[float] = None
    lip_sync_metrics: Dict[str, Any] = Field(default_factory=dict)
    media_integrity: Dict[str, Any] = Field(default_factory=dict)
    segment_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    unmeasured_or_null_metrics: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class JobResultResponse(BaseModel):
    job_id: str
    title: str
    project_id: Optional[str] = None
    status: str
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    original_video_url: Optional[str] = None
    transcript_url: Optional[str] = None
    translation_url: Optional[str] = None
    source_language: str
    detected_source_language: Optional[str] = None
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
