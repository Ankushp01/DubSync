"""Dialogue segments repository."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from database.collections import SEGMENTS_COLLECTION
from database.models import DialogueSegmentModel, utc_now
from database.repositories.base import BaseRepository


class SegmentsRepository(BaseRepository):
    def __init__(self, db=None):
        super().__init__(SEGMENTS_COLLECTION, db=db)

    async def create_segments_batch(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not segments:
            return []
        col = self.get_collection()
        docs = []
        now = utc_now()
        for s in segments:
            doc = {
                "segment_id": s.get("segment_id") or f"seg_{uuid.uuid4().hex[:12]}",
                "job_id": s["job_id"],
                "project_id": s.get("project_id"),
                "user_id": s.get("user_id"),
                "transcript_id": s.get("transcript_id"),
                "translation_id": s.get("translation_id"),
                "segment_index": int(s.get("segment_index", s.get("index", 1))),
                "start_time": round(float(s.get("start_time", s.get("start", 0.0))), 3),
                "end_time": round(float(s.get("end_time", s.get("end", 0.0))), 3),
                "original_text": s.get("original_text", s.get("source_text", s.get("text", ""))),
                "translated_text": s.get("translated_text", ""),
                "raw_tts_duration": round(float(s.get("raw_tts_duration", 0.0)), 3),
                "adapted_tts_duration": round(float(s.get("adapted_tts_duration", 0.0)), 3),
                "speed_factor": round(float(s.get("speed_factor", s.get("speech_speed", 1.0))), 2),
                "audio_path": s.get("audio_path"),
                "final_placement_start": round(float(s.get("final_placement_start", s.get("start", 0.0))), 3),
                "final_placement_end": round(float(s.get("final_placement_end", s.get("end", 0.0))), 3),
                "trimming_seconds": round(float(s.get("trimming_seconds", 0.0)), 3),
                "status": s.get("status", "completed"),
                "created_at": s.get("created_at") or now,
            }
            docs.append(doc)
        
        # Batch insert or replace if existing
        for doc in docs:
            await col.replace_one(
                {"job_id": doc["job_id"], "segment_index": doc["segment_index"]},
                doc,
                upsert=True,
            )
        return self.serialize_list(docs)

    async def get_by_job(self, job_id: str) -> List[Dict[str, Any]]:
        col = self.get_collection()
        cursor = col.find({"job_id": job_id}).sort("segment_index", 1)
        items = await cursor.to_list(length=500)
        return self.serialize_list(items)

    async def get_by_job_for_user(self, job_id: str, user_id: str) -> List[Dict[str, Any]]:
        col = self.get_collection()
        cursor = col.find({"job_id": job_id, "user_id": user_id}).sort("segment_index", 1)
        items = await cursor.to_list(length=500)
        return self.serialize_list(items)

    async def get_by_id(self, segment_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"segment_id": segment_id})
        return self.serialize(doc)

    async def update_segment(self, segment_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        updates.pop("segment_id", None)
        updates.pop("_id", None)
        res = await col.find_one_and_update(
            {"segment_id": segment_id},
            {"$set": updates},
            return_document=True,
        )
        return self.serialize(res)
