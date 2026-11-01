"""Pydantic schemas and serialization helpers for MongoDB models."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pydantic import BaseModel, Field, ConfigDict


def utc_now() -> datetime:
    """Return current UTC datetime with timezone awareness."""
    return datetime.now(timezone.utc)


def serialize_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Recursively convert MongoDB document BSON types (like ObjectId and datetime) to JSON-serializable types."""
    if doc is None:
        return None
    res = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            res[k] = str(v)
        elif isinstance(v, datetime):
            res[k] = v.isoformat()
        elif isinstance(v, list):
            res[k] = [
                serialize_doc(item) if isinstance(item, dict)
                else (str(item) if isinstance(item, ObjectId)
                      else (item.isoformat() if isinstance(item, datetime) else item))
                for item in v
            ]
        elif isinstance(v, dict):
            res[k] = serialize_doc(v)
        else:
            res[k] = v
    return res


# =========================================================================
# Collection 1: users
# =========================================================================
class UserPreferences(BaseModel):
    default_source_language: str = "en"
    default_target_language: str = "hi"
    default_voice: Optional[str] = None


class UserModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: str
    name: str
    display_name: Optional[str] = None
    email: str
    role: str = "user"
    password_hash: str = ""
    email_verified: bool = False
    is_active: bool = True
    verification_token_hash: Optional[str] = None
    verification_token_expires_at: Optional[datetime] = None
    reset_token_hash: Optional[str] = None
    reset_token_expires_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    preferences: UserPreferences = Field(default_factory=UserPreferences)


# =========================================================================
# Collection: sessions
# =========================================================================
class SessionModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str
    user_id: str
    refresh_token_hash: str
    expires_at: datetime
    revoked_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utc_now)
    last_used_at: datetime = Field(default_factory=utc_now)
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None


# =========================================================================
# Collection 2: dubbing_projects
# =========================================================================
class ProjectModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    project_id: str
    user_id: Optional[str] = None
    project_name: str
    source_language: str = "en"
    target_languages: List[str] = Field(default_factory=lambda: ["hi"])
    status: str = "created"  # created, processing, completed, failed
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =========================================================================
# Collection 3: videos
# =========================================================================
class VideoModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    video_id: str
    project_id: str
    user_id: Optional[str] = None
    file_name: str
    file_path: str
    format: str = "mp4"
    duration: float = 0.0
    file_size: int = 0
    fps: float = 23.976
    width: Optional[int] = None
    height: Optional[int] = None
    uploaded_at: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =========================================================================
# Collection 4: transcripts
# =========================================================================
class TranscriptModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    transcript_id: str
    video_id: str
    project_id: str
    job_id: str
    user_id: Optional[str] = None
    source_language: str
    original_text: str
    segments: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =========================================================================
# Collection 5: translations
# =========================================================================
class TranslationModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    translation_id: str
    transcript_id: str
    job_id: str
    user_id: Optional[str] = None
    source_language: str
    target_language: str
    source_text: str
    translated_text: str
    segments: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =========================================================================
# Collection 6: generated_audio
# =========================================================================
class GeneratedAudioModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    audio_id: str
    translation_id: str
    job_id: str
    user_id: Optional[str] = None
    voice_model: str = "Chatterbox Multilingual V3"
    target_language: str = "hi"
    audio_path: str
    duration: float = 0.0
    sample_rate: int = 24000
    channels: int = 1
    created_at: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =========================================================================
# Collection 7: dubbed_videos
# =========================================================================
class DubbedVideoModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    output_id: str
    project_id: str
    job_id: str
    video_id: str
    user_id: Optional[str] = None
    audio_id: Optional[str] = None
    output_path: str
    format: str = "mp4"
    duration: float = 0.0
    status: str = "completed"
    created_at: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =========================================================================
# Collection 8: processing_logs
# =========================================================================
class ProcessingLogModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    log_id: str
    project_id: Optional[str] = None
    job_id: str
    user_id: Optional[str] = None
    stage: str
    status: str
    timestamp: datetime = Field(default_factory=utc_now)
    details: Dict[str, Any] = Field(default_factory=dict)
    severity: str = "info"  # info, warning, error


# =========================================================================
# Collection 9: processing_jobs
# =========================================================================
class ProcessingJobModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str
    project_id: str
    video_id: Optional[str] = None
    user_id: Optional[str] = None
    title: Optional[str] = None
    source_language: str = "en"
    detected_source_language: Optional[str] = None
    target_language: str = "hi"
    voice_mode: str = "original"
    status: str = "queued"  # queued, processing, completed, failed, cancelled, interrupted
    current_stage: str = "extracting_audio"
    progress: int = 0
    current_operation: str = "Job queued"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    configuration: Dict[str, Any] = Field(default_factory=dict)
    stages: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


# =========================================================================
# Collection 10: segments
# =========================================================================
class DialogueSegmentModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    segment_id: str
    job_id: str
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    transcript_id: Optional[str] = None
    translation_id: Optional[str] = None
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
    created_at: datetime = Field(default_factory=utc_now)


# =========================================================================
# Collection 11: evaluations
# =========================================================================
class EvaluationModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    evaluation_id: str
    job_id: str
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    output_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
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


# Compatibility aliases
UserDoc = UserModel
SessionDoc = SessionModel
ProjectDoc = ProjectModel
VideoDoc = VideoModel
TranscriptDoc = TranscriptModel
TranslationDoc = TranslationModel
AudioDoc = GeneratedAudioModel
DubbedVideoDoc = DubbedVideoModel
LogDoc = ProcessingLogModel
JobDoc = ProcessingJobModel
SegmentDoc = DialogueSegmentModel
EvaluationDoc = EvaluationModel
