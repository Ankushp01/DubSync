"""Users repository with authentication, verification, and recovery support."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from database.collections import USERS_COLLECTION
from database.models import UserModel, utc_now
from database.repositories.base import BaseRepository


class UsersRepository(BaseRepository):
    def __init__(self, db=None):
        super().__init__(USERS_COLLECTION, db=db)

    async def create_user(
        self,
        name: str,
        email: str,
        password_hash: str = "",
        preferences: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        email_verified: bool = False,
        is_active: bool = True,
    ) -> Dict[str, Any]:
        col = self.get_collection()
        uid = user_id or f"usr_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        normalized_email = email.strip().lower()
        doc = {
            "user_id": uid,
            "name": name.strip(),
            "email": normalized_email,
            "password_hash": password_hash,
            "email_verified": email_verified,
            "is_active": is_active,
            "verification_token_hash": None,
            "verification_token_expires_at": None,
            "reset_token_hash": None,
            "reset_token_expires_at": None,
            "last_login_at": None,
            "created_at": now,
            "updated_at": now,
            "preferences": preferences or {
                "default_source_language": "en",
                "default_target_language": "hi",
                "default_voice": None,
            },
        }
        await col.insert_one(doc)
        return self.serialize(doc)

    async def get_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        doc = await col.find_one({"user_id": user_id})
        return self.serialize(doc)

    async def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        normalized = email.strip().lower()
        doc = await col.find_one({"email": normalized})
        return self.serialize(doc)

    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        clean = dict(updates)
        clean.pop("user_id", None)
        clean.pop("_id", None)
        clean["updated_at"] = utc_now()
        res = await col.find_one_and_update(
            {"user_id": user_id},
            {"$set": clean},
            return_document=True,
        )
        return self.serialize(res)

    async def update_password(self, user_id: str, new_password_hash: str) -> bool:
        col = self.get_collection()
        now = utc_now()
        res = await col.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "password_hash": new_password_hash,
                    "reset_token_hash": None,
                    "reset_token_expires_at": None,
                    "updated_at": now,
                }
            },
        )
        return res.modified_count > 0

    async def update_last_login(self, user_id: str) -> bool:
        col = self.get_collection()
        now = utc_now()
        res = await col.update_one(
            {"user_id": user_id},
            {"$set": {"last_login_at": now, "updated_at": now}},
        )
        return res.modified_count > 0

    async def set_verification_token(self, user_id: str, token_hash: str, expires_at: datetime) -> bool:
        col = self.get_collection()
        res = await col.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "verification_token_hash": token_hash,
                    "verification_token_expires_at": expires_at,
                    "updated_at": utc_now(),
                }
            },
        )
        return res.modified_count > 0

    async def verify_email_token(self, token_hash: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        now = utc_now()
        user = await col.find_one({
            "verification_token_hash": token_hash,
            "verification_token_expires_at": {"$gt": now},
        })
        if not user:
            return None
        res = await col.find_one_and_update(
            {"user_id": user["user_id"]},
            {
                "$set": {
                    "email_verified": True,
                    "verification_token_hash": None,
                    "verification_token_expires_at": None,
                    "updated_at": now,
                }
            },
            return_document=True,
        )
        return self.serialize(res)

    async def set_reset_token(self, user_id: str, token_hash: str, expires_at: datetime) -> bool:
        col = self.get_collection()
        res = await col.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "reset_token_hash": token_hash,
                    "reset_token_expires_at": expires_at,
                    "updated_at": utc_now(),
                }
            },
        )
        return res.modified_count > 0

    async def get_by_reset_token_hash(self, token_hash: str) -> Optional[Dict[str, Any]]:
        col = self.get_collection()
        now = utc_now()
        user = await col.find_one({
            "reset_token_hash": token_hash,
            "reset_token_expires_at": {"$gt": now},
        })
        return self.serialize(user)

    async def delete_user(self, user_id: str) -> bool:
        col = self.get_collection()
        res = await col.delete_one({"user_id": user_id})
        return res.deleted_count > 0
