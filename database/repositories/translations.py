"""Translations repository."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from database.collections import TRANSLATIONS_COLLECTION
from database.models import TranslationModel, utc_now
from database.repositories.base import BaseRepository


class TranslationsRepository(BaseRepository):
    def __init__(self, db=None):
        super().__init__(TRANSLATIONS_COLLECTION, db=db)

    async def create_translation(
        self,
        transcript_id: str,
        job_id: str,
        source_language: str,
        target_language: str,
        source_text: str,
        translated_text: str,
        user_id: Optional[str] = None,
        segments: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        translation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        col = self.get_collection()
        trans_id = translation_id or f"trans_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        doc = {
            "translation_id": trans_id,
            "transcript_id": transcript_id,
            "job_id": job_id,
            "user_id": user_id,
            "source_language": source_language,
            "target_language": target_language,
            "source_text": source_text,
            "translated_text": translated_text,
            "segments": segments or [],
            "created_at": now,
            "metadata": metadata or {},
        }
        await col.insert_one(doc)
        return self.serialize(doc)

    async def get_by_id(self, translation_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"translation_id": translation_id})
        return self.serialize(doc)

    async def get_by_id_for_user(self, translation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"translation_id": translation_id, "user_id": user_id})
        return self.serialize(doc)

    async def get_by_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"job_id": job_id})
        return self.serialize(doc)

    async def get_by_transcript(self, transcript_id: str) -> List[Dict[str, Any]]:
        col = self.get_collection()
        cursor = col.find({"transcript_id": transcript_id}).sort("created_at", -1)
        items = await cursor.to_list(length=100)
        return self.serialize_list(items)
