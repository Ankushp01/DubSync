"""Transcripts repository."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from database.collections import TRANSCRIPTS_COLLECTION
from database.models import TranscriptModel, utc_now
from database.repositories.base import BaseRepository


class TranscriptsRepository(BaseRepository):
    def __init__(self, db=None):
        super().__init__(TRANSCRIPTS_COLLECTION, db=db)

    async def create_transcript(
        self,
        video_id: str,
        project_id: str,
        job_id: str,
        source_language: str,
        original_text: str,
        user_id: Optional[str] = None,
        segments: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        transcript_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        col = self.get_collection()
        tr_id = transcript_id or f"tr_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        doc = {
            "transcript_id": tr_id,
            "video_id": video_id,
            "project_id": project_id,
            "job_id": job_id,
            "user_id": user_id,
            "source_language": source_language,
            "original_text": original_text,
            "segments": segments or [],
            "created_at": now,
            "metadata": metadata or {},
        }
        await col.insert_one(doc)
        return self.serialize(doc)

    async def get_by_id(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"transcript_id": transcript_id})
        return self.serialize(doc)

    async def get_by_id_for_user(self, transcript_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"transcript_id": transcript_id, "user_id": user_id})
        return self.serialize(doc)

    async def get_by_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"job_id": job_id})
        return self.serialize(doc)
