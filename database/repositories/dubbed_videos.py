"""Dubbed videos output metadata repository."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from database.collections import DUBBED_VIDEOS_COLLECTION
from database.models import DubbedVideoModel, utc_now
from database.repositories.base import BaseRepository


class DubbedVideosRepository(BaseRepository):
    def __init__(self, db=None):
        super().__init__(DUBBED_VIDEOS_COLLECTION, db=db)

    async def create_dubbed_video(
        self,
        project_id: str,
        job_id: str,
        video_id: str,
        output_path: str,
        user_id: Optional[str] = None,
        audio_id: Optional[str] = None,
        format: str = "mp4",
        duration: float = 0.0,
        status: str = "completed",
        metadata: Optional[Dict[str, Any]] = None,
        output_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        col = self.get_collection()
        out_id = output_id or f"out_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        doc = {
            "output_id": out_id,
            "project_id": project_id,
            "job_id": job_id,
            "user_id": user_id,
            "video_id": video_id,
            "audio_id": audio_id,
            "output_path": output_path,
            "format": format,
            "duration": round(float(duration), 3),
            "status": status,
            "created_at": now,
            "metadata": metadata or {},
        }
        await col.insert_one(doc)
        return self.serialize(doc)

    async def get_by_id(self, output_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"output_id": output_id})
        return self.serialize(doc)

    async def get_by_id_for_user(self, output_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"output_id": output_id, "user_id": user_id})
        return self.serialize(doc)

    async def get_by_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"job_id": job_id})
        return self.serialize(doc)

    async def get_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        col = self.get_collection()
        cursor = col.find({"project_id": project_id}).sort("created_at", -1)
        items = await cursor.to_list(length=100)
        return self.serialize_list(items)
