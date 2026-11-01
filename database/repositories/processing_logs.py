"""Processing execution logs repository."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from database.collections import PROCESSING_LOGS_COLLECTION
from database.models import ProcessingLogModel, utc_now
from database.repositories.base import BaseRepository


class ProcessingLogsRepository(BaseRepository):
    def __init__(self, db=None):
        super().__init__(PROCESSING_LOGS_COLLECTION, db=db)

    async def add_log(
        self,
        job_id: str,
        stage: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info",
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        log_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        col = self.get_collection()
        lid = log_id or f"log_{uuid.uuid4().hex[:12]}"
        doc = {
            "log_id": lid,
            "project_id": project_id,
            "job_id": job_id,
            "user_id": user_id,
            "stage": stage,
            "status": status,
            "timestamp": utc_now(),
            "details": details or {},
            "severity": severity,
        }
        await col.insert_one(doc)
        return self.serialize(doc)

    async def get_logs_for_job(self, job_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        col = self.get_collection()
        cursor = col.find({"job_id": job_id}).sort("timestamp", 1).limit(limit)
        items = await cursor.to_list(length=limit)
        return self.serialize_list(items)

    async def get_logs_for_job_for_user(self, job_id: str, user_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        col = self.get_collection()
        cursor = col.find({"job_id": job_id, "user_id": user_id}).sort("timestamp", 1).limit(limit)
        items = await cursor.to_list(length=limit)
        return self.serialize_list(items)
