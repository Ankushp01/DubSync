"""Evaluations repository."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from database.collections import EVALUATIONS_COLLECTION
from database.models import EvaluationModel, utc_now
from database.repositories.base import BaseRepository


class EvaluationsRepository(BaseRepository):
    def __init__(self, db=None):
        super().__init__(EVALUATIONS_COLLECTION, db=db)

    async def create_evaluation(
        self,
        job_id: str,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        output_id: Optional[str] = None,
        video_duration: Optional[float] = None,
        audio_duration: Optional[float] = None,
        video_frame_count: Optional[int] = None,
        frame_rate: Optional[float] = None,
        speech_preservation_percentage: Optional[float] = None,
        trimmed_audio_seconds: Optional[float] = None,
        timeline_extension_seconds: Optional[float] = None,
        lip_sync_metrics: Optional[Dict[str, Any]] = None,
        media_integrity: Optional[Dict[str, Any]] = None,
        segment_metrics: Optional[List[Dict[str, Any]]] = None,
        unmeasured_or_null_metrics: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        evaluation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        col = self.get_collection()
        eid = evaluation_id or f"eval_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        doc = {
            "evaluation_id": eid,
            "job_id": job_id,
            "project_id": project_id,
            "user_id": user_id,
            "output_id": output_id,
            "created_at": now,
            "video_duration": video_duration,
            "audio_duration": audio_duration,
            "video_frame_count": video_frame_count,
            "frame_rate": frame_rate,
            "speech_preservation_percentage": speech_preservation_percentage,
            "trimmed_audio_seconds": trimmed_audio_seconds,
            "timeline_extension_seconds": timeline_extension_seconds,
            "lip_sync_metrics": lip_sync_metrics or {},
            "media_integrity": media_integrity or {},
            "segment_metrics": segment_metrics or [],
            "unmeasured_or_null_metrics": unmeasured_or_null_metrics or {},
            "warnings": warnings or [],
        }
        await col.insert_one(doc)
        return self.serialize(doc)

    async def get_by_id(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"evaluation_id": evaluation_id})
        return self.serialize(doc)

    async def get_by_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"job_id": job_id}, sort=[("created_at", -1)])
        return self.serialize(doc)

    async def get_by_job_for_user(self, job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"job_id": job_id, "user_id": user_id}, sort=[("created_at", -1)])
        return self.serialize(doc)
