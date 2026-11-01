"""Dubbing projects repository."""
from __future__ import annotations

import math
import uuid
from typing import Any, Dict, List, Optional
from database.collections import PROJECTS_COLLECTION
from database.models import ProjectModel, utc_now
from database.repositories.base import BaseRepository


class ProjectsRepository(BaseRepository):
    def __init__(self, db=None):
        super().__init__(PROJECTS_COLLECTION, db=db)

    async def create_project(
        self,
        project_name: str,
        user_id: Optional[str] = None,
        source_language: str = "en",
        target_languages: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        col = self.get_collection()
        pid = project_id or f"proj_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        doc = {
            "project_id": pid,
            "user_id": user_id,
            "project_name": project_name,
            "source_language": source_language,
            "target_languages": target_languages or ["hi"],
            "status": "created",
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {},
        }
        await col.insert_one(doc)
        return self.serialize(doc)

    async def get_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"project_id": project_id})
        return self.serialize(doc)

    async def get_by_id_for_user(self, project_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"project_id": project_id, "user_id": user_id})
        return self.serialize(doc)

    async def list_projects(
        self,
        user_id: Optional[str] = None,
        page: int = 1,
        limit: int = 10,
    ) -> Dict[str, Any]:
        col = self.get_collection()
        query: Dict[str, Any] = {}
        if user_id:
            query["user_id"] = user_id

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

    async def update_project(self, project_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        updates["updated_at"] = utc_now()
        # Avoid overriding project_id
        updates.pop("project_id", None)
        updates.pop("_id", None)
        res = await col.find_one_and_update(
            {"project_id": project_id},
            {"$set": updates},
            return_document=True,
        )
        return self.serialize(res)

    async def update_project_for_user(self, project_id: str, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        updates["updated_at"] = utc_now()
        updates.pop("project_id", None)
        updates.pop("user_id", None)
        updates.pop("_id", None)
        res = await col.find_one_and_update(
            {"project_id": project_id, "user_id": user_id},
            {"$set": updates},
            return_document=True,
        )
        return self.serialize(res)

    async def delete_project(self, project_id: str) -> bool:
        col = self.get_collection()
        res = await col.delete_one({"project_id": project_id})
        return res.deleted_count > 0

    async def delete_project_for_user(self, project_id: str, user_id: str) -> bool:
        col = self.get_collection()
        res = await col.delete_one({"project_id": project_id, "user_id": user_id})
        return res.deleted_count > 0
