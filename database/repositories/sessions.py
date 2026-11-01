"""Sessions repository for refresh token tracking and session revocation."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from database.collections import SESSIONS_COLLECTION
from database.models import SessionModel, utc_now
from database.repositories.base import BaseRepository


class SessionsRepository(BaseRepository):
    def __init__(self, db=None):
        super().__init__(SESSIONS_COLLECTION, db=db)

    async def create_session(
        self,
        user_id: str,
        refresh_token_hash: str,
        expires_at: datetime,
        session_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        col = self.get_collection()
        sid = session_id or f"sess_{uuid.uuid4().hex[:16]}"
        now = utc_now()
        doc = {
            "session_id": sid,
            "user_id": user_id,
            "refresh_token_hash": refresh_token_hash,
            "expires_at": expires_at,
            "revoked_at": None,
            "created_at": now,
            "last_used_at": now,
            "user_agent": user_agent,
            "ip_address": ip_address,
        }
        await col.insert_one(doc)
        return self.serialize(doc)

    async def get_by_token_hash(self, refresh_token_hash: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"refresh_token_hash": refresh_token_hash})
        return self.serialize(doc)

    async def get_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"session_id": session_id})
        return self.serialize(doc)

    async def revoke_session(self, session_id: str) -> bool:
        col = self.get_collection()
        res = await col.update_one(
            {"session_id": session_id},
            {"$set": {"revoked_at": utc_now()}},
        )
        return res.modified_count > 0

    async def revoke_all_user_sessions(self, user_id: str) -> int:
        col = self.get_collection()
        res = await col.update_many(
            {"user_id": user_id, "revoked_at": None},
            {"$set": {"revoked_at": utc_now()}},
        )
        return res.modified_count

    async def touch_session(self, session_id: str) -> bool:
        col = self.get_collection()
        res = await col.update_one(
            {"session_id": session_id},
            {"$set": {"last_used_at": utc_now()}},
        )
        return res.modified_count > 0

    async def list_active_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        col = self.get_collection()
        now = utc_now()
        cursor = col.find({
            "user_id": user_id,
            "revoked_at": None,
            "expires_at": {"$gt": now},
        }).sort("last_used_at", -1)
        items = await cursor.to_list(length=100)
        return self.serialize_list(items)
