"""Videos metadata repository."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from database.collections import VIDEOS_COLLECTION
from database.models import VideoModel, utc_now
from database.repositories.base import BaseRepository


class VideosRepository(BaseRepository):
    def __init__(self, db=None):
        super().__init__(VIDEOS_COLLECTION, db=db)

    async def create_video(
        self,
        project_id: str,
        file_name: str,
        file_path: str,
        user_id: Optional[str] = None,
        format: str = "mp4",
        duration: float = 0.0,
        file_size: int = 0,
        fps: float = 23.976,
        width: Optional[int] = None,
        height: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        video_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        col = self.get_collection()
        vid = video_id or f"vid_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        doc = {
            "video_id": vid,
            "project_id": project_id,
            "user_id": user_id,
            "file_name": file_name,
            "file_path": file_path,
            "format": format,
            "duration": round(float(duration), 3),
            "file_size": int(file_size),
            "fps": round(float(fps), 3),
            "width": width,
            "height": height,
            "uploaded_at": now,
            "metadata": metadata or {},
        }
        await col.insert_one(doc)
        return self.serialize(doc)

    async def get_by_id(self, video_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"video_id": video_id})
        return self.serialize(doc)

    async def get_by_id_for_user(self, video_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"video_id": video_id, "user_id": user_id})
        return self.serialize(doc)

    async def get_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        col = self.get_collection()
        cursor = col.find({"project_id": project_id}).sort("uploaded_at", -1)
        items = await cursor.to_list(length=100)
        return self.serialize_list(items)
