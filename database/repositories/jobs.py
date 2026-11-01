"""Processing jobs repository."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from database.collections import PROCESSING_JOBS_COLLECTION
from database.models import ProcessingJobModel, utc_now
from database.repositories.base import BaseRepository


class ProcessingJobsRepository(BaseRepository):
    def __init__(self, db=None):
        super().__init__(PROCESSING_JOBS_COLLECTION, db=db)

    async def create_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        col = self.get_collection()
        now = utc_now()
        doc = {
            "job_id": job_data["job_id"],
            "project_id": job_data.get("project_id", "default_project"),
            "video_id": job_data.get("video_id"),
            "user_id": job_data.get("user_id"),
            "title": job_data.get("title") or job_data["job_id"],
            "source_language": job_data.get("source_language", "en"),
            "detected_source_language": job_data.get("detected_source_language"),
            "target_language": job_data.get("target_language", "hi"),
            "voice_mode": job_data.get("voice_mode", "original"),
            "status": job_data.get("status", "queued"),
            "current_stage": job_data.get("current_stage", "extracting_audio"),
            "progress": int(job_data.get("overall_progress", job_data.get("progress", 0))),
            "current_operation": job_data.get("current_operation", "Job queued"),
            "started_at": job_data.get("started_at"),
            "completed_at": job_data.get("completed_at"),
            "error": job_data.get("error"),
            "configuration": job_data.get("configuration", {}),
            "stages": job_data.get("stages", []),
            "artifacts": job_data.get("artifacts", {}),
            "created_at": job_data.get("created_at") or now,
            "updated_at": now,
        }
        await col.replace_one({"job_id": doc["job_id"]}, doc, upsert=True)
        return self.serialize(doc)

    async def get_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"job_id": job_id})
        return self.serialize(doc)

    async def get_by_id_for_user(self, job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"job_id": job_id, "user_id": user_id})
        return self.serialize(doc)

    async def list_jobs(
        self,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        limit: int = 10,
    ) -> Dict[str, Any]:
        col = self.get_collection()
        query: Dict[str, Any] = {}
        if project_id:
            query["project_id"] = project_id
        if user_id:
            query["user_id"] = user_id
        if status:
            query["status"] = status

        total = await col.count_documents(query)
        skip = (page - 1) * limit
        cursor = col.find(query).sort("created_at", -1).skip(skip).limit(limit)
        items = await cursor.to_list(length=limit)

        total_pages = math.ceil(total / limit) if total > 0 else 0
        return {
            "items": self.serialize_list(items),
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages,
        }

    async def update_job(self, job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        clean_updates = dict(updates)
        clean_updates.pop("job_id", None)
        clean_updates.pop("_id", None)
        clean_updates["updated_at"] = utc_now()

        # Prevent completed jobs from accidentally returning to processing
        existing = await col.find_one({"job_id": job_id}, {"status": 1})
        if existing and existing.get("status") in ("completed", "failed", "cancelled"):
            if clean_updates.get("status") in ("queued", "processing"):
                clean_updates.pop("status", None)

        res = await col.find_one_and_update(
            {"job_id": job_id},
            {"$set": clean_updates},
            return_document=True,
        )
        return self.serialize(res)

    async def delete_job(self, job_id: str) -> bool:
        col = self.get_collection()
        res = await col.delete_one({"job_id": job_id})
        return res.deleted_count > 0

    async def recover_interrupted_jobs(self) -> int:
        """Find active jobs left incomplete across backend process restarts and mark them as interrupted."""
        col = self.get_collection()
        now = utc_now()
        res = await col.update_many(
            {"status": "processing"},
            {
                "$set": {
                    "status": "failed",
                    "error": "Process was interrupted by backend restart before completion. Intermediate artifacts preserved.",
                    "current_operation": "Interrupted by server restart",
                    "updated_at": now,
                    "completed_at": now,
                }
            },
        )
        return res.modified_count
