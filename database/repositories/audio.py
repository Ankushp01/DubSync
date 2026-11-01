"""Generated audio metadata repository."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from database.collections import GENERATED_AUDIO_COLLECTION
from database.models import GeneratedAudioModel, utc_now
from database.repositories.base import BaseRepository


class GeneratedAudioRepository(BaseRepository):
    def __init__(self, db=None):
        super().__init__(GENERATED_AUDIO_COLLECTION, db=db)

    async def create_audio(
        self,
        translation_id: str,
        job_id: str,
        audio_path: str,
        user_id: Optional[str] = None,
        voice_model: str = "Chatterbox Multilingual V3",
        target_language: str = "hi",
        duration: float = 0.0,
        sample_rate: int = 24000,
        channels: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        audio_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        col = self.get_collection()
        aid = audio_id or f"aud_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        doc = {
            "audio_id": aid,
            "translation_id": translation_id,
            "job_id": job_id,
            "user_id": user_id,
            "voice_model": voice_model,
            "target_language": target_language,
            "audio_path": audio_path,
            "duration": round(float(duration), 3),
            "sample_rate": sample_rate,
            "channels": channels,
            "created_at": now,
            "metadata": metadata or {},
        }
        await col.insert_one(doc)
        return self.serialize(doc)

    async def get_by_id(self, audio_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"audio_id": audio_id})
        return self.serialize(doc)

    async def get_by_id_for_user(self, audio_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"audio_id": audio_id, "user_id": user_id})
        return self.serialize(doc)

    async def get_by_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"job_id": job_id})
        return self.serialize(doc)
