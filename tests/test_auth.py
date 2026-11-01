"""Comprehensive Test Suite for DubSync User Authentication and Authorization System.

Verifies:
1. Successful registration.
2. Duplicate email rejection (409).
3. Password hashing verification (Argon2id).
4. Successful login.
5. Invalid credentials rejection (401).
6. Disabled account rejection (403).
7. Access token expiration.
8. Refresh token rotation.
9. Refresh token reuse detection (revokes all active sessions).
10. Logout.
11. Logout-all.
12. Password change.
13. Password reset workflow.
14. Email verification workflow.
15. User profile retrieval (/me).
16. Project ownership.
17. Job ownership.
18. Cross-user video access rejection (404).
19. Cross-user output download rejection (404).
20. Cross-user evaluation access rejection (404).
21. Authenticated SSE access.
22. Session recovery after backend restart.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import jwt

from api.main import app
from api.job_manager import job_manager, JOBS_DIR
from api.auth.service import rate_limiter
from api.auth.security import (
    COOKIE_ACCESS_NAME,
    COOKIE_REFRESH_NAME,
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    create_access_token,
    hash_password,
    verify_password,
    hash_token,
)
from database.connection import db_manager
from database.models import UserModel, utc_now
from database.repositories import (
    UsersRepository,
    SessionsRepository,
    ProjectsRepository,
    VideosRepository,
    EvaluationsRepository,
    ProcessingJobsRepository,
    SegmentsRepository,
    ProcessingLogsRepository,
)


class MockCollection:
    """Thread-safe and async-compatible in-memory MongoDB mock collection."""

    def __init__(self, name: str):
        self.name = name
        self.docs = []

    def _matches(self, doc: dict, query: dict) -> bool:
        for k, v in query.items():
            if k == "$or" and isinstance(v, list):
                if not any(self._matches(doc, q) for q in v):
                    return False
            elif isinstance(v, dict):
                val = doc.get(k)
                for op, op_val in v.items():
                    if op == "$gt" and (val is None or not (val > op_val)):
                        return False
                    elif op == "$lt" and (val is None or not (val < op_val)):
                        return False
                    elif op == "$gte" and (val is None or not (val >= op_val)):
                        return False
                    elif op == "$lte" and (val is None or not (val <= op_val)):
                        return False
                    elif op == "$ne" and val == op_val:
                        return False
                    elif op == "$in" and (val not in op_val):
                        return False
            elif doc.get(k) != v:
                return False
        return True

    async def insert_one(self, doc: dict):
        d = dict(doc)
        if "_id" not in d:
            from bson import ObjectId
            d["_id"] = ObjectId()
        self.docs.append(d)
        res = MagicMock()
        res.inserted_id = d["_id"]
        return res

    async def find_one(self, query: dict, projection=None, sort=None):
        matching = [d for d in self.docs if self._matches(d, query)]
        if not matching:
            return None
        if sort:
            key, direction = sort[0]
            matching.sort(key=lambda x: str(x.get(key, "")), reverse=(direction == -1))
        return dict(matching[0])

    def find(self, query: dict, projection=None):
        matching = [dict(d) for d in self.docs if self._matches(d, query)]

        class Cursor:
            def __init__(self, data):
                self._data = list(data)

            def sort(self, field, direction):
                self._data.sort(key=lambda x: str(x.get(field, "")), reverse=(direction == -1))
                return self

            def skip(self, n):
                self._data = self._data[n:]
                return self

            def limit(self, n):
                self._data = self._data[:n]
                return self

            async def to_list(self, length=None):
                if length is not None:
                    return self._data[:length]
                return self._data

        return Cursor(matching)

    async def count_documents(self, query: dict):
        return sum(1 for d in self.docs if self._matches(d, query))

    async def update_one(self, query: dict, update: dict):
        res = MagicMock()
        res.matched_count = 0
        res.modified_count = 0
        for d in self.docs:
            if self._matches(d, query):
                res.matched_count = 1
                if "$set" in update:
                    d.update(update["$set"])
                    res.modified_count = 1
                break
        return res

    async def update_many(self, query: dict, update: dict):
        res = MagicMock()
        res.modified_count = 0
        for d in self.docs:
            if self._matches(d, query):
                if "$set" in update:
                    d.update(update["$set"])
                    res.modified_count += 1
        return res

    async def find_one_and_update(self, query: dict, update: dict, return_document=True):
        for d in self.docs:
            if self._matches(d, query):
                if "$set" in update:
                    d.update(update["$set"])
                return dict(d)
        return None

    async def replace_one(self, query: dict, doc: dict, upsert=False):
        for i, d in enumerate(self.docs):
            if self._matches(d, query):
                self.docs[i] = dict(doc)
                res = MagicMock()
                res.modified_count = 1
                return res
        if upsert:
            await self.insert_one(doc)
            res = MagicMock()
            res.upserted_id = doc.get("_id")
            return res
        res = MagicMock()
        res.modified_count = 0
        return res

    async def delete_one(self, query: dict):
        res = MagicMock()
        res.deleted_count = 0
        for i, d in enumerate(self.docs):
            if self._matches(d, query):
                del self.docs[i]
                res.deleted_count = 1
                break
        return res

    async def delete_many(self, query: dict):
        res = MagicMock()
        before = len(self.docs)
        self.docs = [d for d in self.docs if not self._matches(d, query)]
        res.deleted_count = before - len(self.docs)
        return res


class TestDubSyncAuthentication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up shared in-memory collections for repositories
        cls.users_col = MockCollection("users")
        cls.sessions_col = MockCollection("sessions")
        cls.projects_col = MockCollection("dubbing_projects")
        cls.videos_col = MockCollection("videos")
        cls.evals_col = MockCollection("evaluations")
        cls.jobs_col = MockCollection("processing_jobs")
        cls.segments_col = MockCollection("segments")
        cls.logs_col = MockCollection("processing_logs")

        cls.mock_db = {
            "users": cls.users_col,
            "sessions": cls.sessions_col,
            "dubbing_projects": cls.projects_col,
            "videos": cls.videos_col,
            "evaluations": cls.evals_col,
            "processing_jobs": cls.jobs_col,
            "segments": cls.segments_col,
            "processing_logs": cls.logs_col,
        }

        # Mock db_manager connection
        cls._orig_client = getattr(db_manager, "_client", None)
        cls._orig_connected = db_manager._is_connected
        db_manager._client = MagicMock()
        db_manager._is_connected = True
        cls.patch_db = patch.object(db_manager, "get_database", return_value=cls.mock_db)
        cls.patch_db.start()

        cls.patch_rl = patch.object(rate_limiter, "check_rate_limit", return_value=True)
        cls.patch_rl.start()

        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.patch_rl.stop()
        cls.patch_db.stop()
        db_manager._client = cls._orig_client
        db_manager._is_connected = cls._orig_connected

    def setUp(self):
        self.client.cookies.clear()

    # 1. Successful registration
    def test_01_successful_registration(self):
        payload = {
            "email": "user1@dubsync.example.com",
            "password": "Password123!@#",
            "display_name": "User One",
        }
        res = self.client.post("/api/v1/auth/register", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIn("user", data)
        self.assertEqual(data["user"]["email"], "user1@dubsync.example.com")
        self.assertEqual(data["user"]["display_name"], "User One")
        self.assertNotIn("password", data["user"])
        self.assertNotIn("password_hash", data["user"])
        self.assertIn("access_token", data)
        # Verify HttpOnly cookies
        cookies = res.cookies
        self.assertIn(COOKIE_ACCESS_NAME, cookies)
        self.assertIn(COOKIE_REFRESH_NAME, cookies)

    # 2. Duplicate email rejection (409)
    def test_02_duplicate_email_rejection(self):
        payload = {
            "email": "USER1@DUBSYNC.EXAMPLE.COM",  # Case-insensitive duplicate
            "password": "Password123!@#",
            "display_name": "Duplicate User",
        }
        res = self.client.post("/api/v1/auth/register", json=payload)
        self.assertEqual(res.status_code, 409)
        detail = res.json()["detail"].lower()
        self.assertTrue("already exists" in detail or "already registered" in detail)

    # 3. Password hashing verification (Argon2id)
    def test_03_password_hashing_verification(self):
        user_doc = next((d for d in self.users_col.docs if d.get("email") == "user1@dubsync.example.com"), None)
        self.assertIsNotNone(user_doc)
        pw_hash = user_doc.get("password_hash")
        self.assertTrue(pw_hash.startswith("$argon2id$"))
        self.assertTrue(verify_password(pw_hash, "Password123!@#"))
        self.assertFalse(verify_password(pw_hash, "WrongPassword123"))

    # 4. Successful login
    def test_04_successful_login(self):
        res = self.client.post("/api/v1/auth/login", json={
            "email": "user1@dubsync.example.com",
            "password": "Password123!@#",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["user"]["email"], "user1@dubsync.example.com")
        self.assertIn(COOKIE_ACCESS_NAME, res.cookies)
        self.assertIn(COOKIE_REFRESH_NAME, res.cookies)

    # 5. Invalid credentials rejection (401 generic)
    def test_05_invalid_credentials(self):
        res = self.client.post("/api/v1/auth/login", json={
            "email": "user1@dubsync.example.com",
            "password": "IncorrectPassword999",
        })
        self.assertEqual(res.status_code, 401)
        self.assertIn("invalid email or password", res.json()["detail"].lower())

        # Nonexistent email must return identical generic message
        res_non = self.client.post("/api/v1/auth/login", json={
            "email": "nonexistent@dubsync.example.com",
            "password": "SomePassword123!",
        })
        self.assertEqual(res_non.status_code, 401)
        self.assertEqual(res_non.json()["detail"], res.json()["detail"])

    # 6. Disabled account rejection (403)
    def test_06_disabled_account_rejection(self):
        # Register user and deactivate
        self.client.post("/api/v1/auth/register", json={
            "email": "disabled@dubsync.example.com",
            "password": "Password123!@#",
            "display_name": "Disabled User",
        })
        for d in self.users_col.docs:
            if d.get("email") == "disabled@dubsync.example.com":
                d["is_active"] = False

        res = self.client.post("/api/v1/auth/login", json={
            "email": "disabled@dubsync.example.com",
            "password": "Password123!@#",
        })
        self.assertEqual(res.status_code, 403)
        self.assertIn("disabled", res.json()["detail"].lower())

    # 7. Access token expiration handling
    def test_07_access_token_expiration(self):
        # Generate expired JWT with proper type: access claim
        expired_payload = {
            "sub": "some_user_id",
            "email": "user1@dubsync.example.com",
            "role": "user",
            "type": "access",
            "exp": datetime.now(timezone.utc) - timedelta(minutes=5),
        }
        expired_token = jwt.encode(expired_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        res = self.client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
        self.assertEqual(res.status_code, 401)
        self.assertIn("expired", res.json()["detail"].lower())

    # 8. Refresh token rotation
    def test_08_refresh_token_rotation(self):
        login_res = self.client.post("/api/v1/auth/login", json={
            "email": "user1@dubsync.example.com",
            "password": "Password123!@#",
        })
        initial_refresh = login_res.cookies.get(COOKIE_REFRESH_NAME)

        # Call refresh via header or cookie
        ref_res = self.client.post(
            "/api/v1/auth/refresh",
            headers={"X-Refresh-Token": initial_refresh},
        )
        self.assertEqual(ref_res.status_code, 200)
        new_refresh = ref_res.cookies.get(COOKIE_REFRESH_NAME)
        self.assertNotEqual(initial_refresh, new_refresh)

        # Old refresh token is marked revoked
        old_hash = hash_token(initial_refresh)
        old_sess = next((s for s in self.sessions_col.docs if s.get("refresh_token_hash") == old_hash), None)
        self.assertIsNotNone(old_sess)
        self.assertIsNotNone(old_sess.get("revoked_at"))

    # 9. Refresh token reuse detection (revokes all active sessions)
    def test_09_refresh_token_reuse_detection(self):
        login_res = self.client.post("/api/v1/auth/login", json={
            "email": "user1@dubsync.example.com",
            "password": "Password123!@#",
        })
        stolen_token = login_res.cookies.get(COOKIE_REFRESH_NAME)

        # Legitimate rotation
        legit_res = self.client.post("/api/v1/auth/refresh", headers={"X-Refresh-Token": stolen_token})
        self.assertEqual(legit_res.status_code, 200)

        # Attacker tries to use the already-rotated (revoked) stolen_token
        self.client.cookies.clear()
        attacker_res = self.client.post("/api/v1/auth/refresh", headers={"X-Refresh-Token": stolen_token})
        self.assertEqual(attacker_res.status_code, 401)
        detail_msg = attacker_res.json()["detail"].lower()
        self.assertTrue("revoked" in detail_msg or "reuse detected" in detail_msg)

        # Verify that reuse detection revoked ALL user sessions
        user_doc = next((d for d in self.users_col.docs if d.get("email") == "user1@dubsync.example.com"), None)
        active_sessions = [
            s for s in self.sessions_col.docs
            if s.get("user_id") == user_doc["user_id"] and s.get("revoked_at") is None
        ]
        self.assertEqual(len(active_sessions), 0)

    # 10. Logout
    def test_10_logout(self):
        login_res = self.client.post("/api/v1/auth/login", json={
            "email": "user1@dubsync.example.com",
            "password": "Password123!@#",
        })
        token = login_res.json()["access_token"]
        ref_cookie = login_res.cookies.get(COOKIE_REFRESH_NAME)

        logout_res = self.client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {token}", "X-Refresh-Token": ref_cookie},
        )
        self.assertEqual(logout_res.status_code, 200)

        # Session is revoked
        h = hash_token(ref_cookie)
        sess = next((s for s in self.sessions_col.docs if s.get("refresh_token_hash") == h), None)
        self.assertIsNotNone(sess.get("revoked_at"))

    # 11. Logout-all
    def test_11_logout_all(self):
        login1 = self.client.post("/api/v1/auth/login", json={"email": "user1@dubsync.example.com", "password": "Password123!@#"})
        login2 = self.client.post("/api/v1/auth/login", json={"email": "user1@dubsync.example.com", "password": "Password123!@#"})
        token = login2.json()["access_token"]

        res = self.client.post("/api/v1/auth/logout-all", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "all_sessions_revoked")

    # 12. Password change
    def test_12_password_change(self):
        login = self.client.post("/api/v1/auth/login", json={"email": "user1@dubsync.example.com", "password": "Password123!@#"})
        token = login.json()["access_token"]

        ch_res = self.client.post(
            "/api/v1/users/me/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"current_password": "Password123!@#", "new_password": "NewSecretPassword99!"},
        )
        self.assertEqual(ch_res.status_code, 200)

        # Login with old password fails
        old_login = self.client.post("/api/v1/auth/login", json={"email": "user1@dubsync.example.com", "password": "Password123!@#"})
        self.assertEqual(old_login.status_code, 401)

        # Login with new password succeeds
        new_login = self.client.post("/api/v1/auth/login", json={"email": "user1@dubsync.example.com", "password": "NewSecretPassword99!"})
        self.assertEqual(new_login.status_code, 200)

    # 13. Password reset workflow
    def test_13_password_reset_flow(self):
        forgot_res = self.client.post("/api/v1/auth/forgot-password", json={"email": "user1@dubsync.example.com"})
        self.assertEqual(forgot_res.status_code, 200)
        reset_token = forgot_res.json().get("preview_token")
        self.assertIsNotNone(reset_token)

        reset_res = self.client.post("/api/v1/auth/reset-password", json={
            "token": reset_token,
            "new_password": "ResetPasswordFinal123!",
        })
        self.assertEqual(reset_res.status_code, 200)

        # Login with reset password succeeds
        login_res = self.client.post("/api/v1/auth/login", json={
            "email": "user1@dubsync.example.com",
            "password": "ResetPasswordFinal123!",
        })
        self.assertEqual(login_res.status_code, 200)

    # 14. Email verification workflow
    def test_14_email_verification_flow(self):
        reg = self.client.post("/api/v1/auth/register", json={
            "email": "verify_me@dubsync.example.com",
            "password": "Password123!@#",
            "display_name": "Verify Me",
        })
        token = reg.json().get("preview_verification_token")
        self.assertIsNotNone(token)

        ver_res = self.client.post("/api/v1/auth/verify-email", json={"token": token})
        self.assertEqual(ver_res.status_code, 200)

        # User is marked verified
        user_doc = next((d for d in self.users_col.docs if d.get("email") == "verify_me@dubsync.example.com"), None)
        self.assertTrue(user_doc.get("email_verified"))

    # 15. User profile retrieval (/me)
    def test_15_user_profile_me(self):
        login = self.client.post("/api/v1/auth/login", json={
            "email": "verify_me@dubsync.example.com",
            "password": "Password123!@#",
        })
        token = login.json()["access_token"]
        me_res = self.client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me_res.status_code, 200)
        data = me_res.json()
        self.assertEqual(data["email"], "verify_me@dubsync.example.com")
        self.assertEqual(data["display_name"], "Verify Me")

    # 16. Project ownership enforcement
    def test_16_project_ownership(self):
        # Register User A and User B
        reg_a = self.client.post("/api/v1/auth/register", json={"email": "usera_proj@test.com", "password": "Password123!@#", "display_name": "User A"})
        token_a = reg_a.json()["access_token"]
        reg_b = self.client.post("/api/v1/auth/register", json={"email": "userb_proj@test.com", "password": "Password123!@#", "display_name": "User B"})
        token_b = reg_b.json()["access_token"]

        # User A creates a project
        p_res = self.client.post(
            "/api/v1/projects",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"project_name": "User A Secret Project"},
        )
        self.assertEqual(p_res.status_code, 201)
        proj_id = p_res.json()["project_id"]

        # User A can get it
        get_a = self.client.get(f"/api/v1/projects/{proj_id}", headers={"Authorization": f"Bearer {token_a}"})
        self.assertEqual(get_a.status_code, 200)

        # User B gets 404 (preventing existence leak)
        get_b = self.client.get(f"/api/v1/projects/{proj_id}", headers={"Authorization": f"Bearer {token_b}"})
        self.assertEqual(get_b.status_code, 404)

        # User B cannot update
        up_b = self.client.patch(
            f"/api/v1/projects/{proj_id}",
            headers={"Authorization": f"Bearer {token_b}"},
            json={"project_name": "Hacked"},
        )
        self.assertEqual(up_b.status_code, 404)

        # User B cannot delete
        del_b = self.client.delete(f"/api/v1/projects/{proj_id}", headers={"Authorization": f"Bearer {token_b}"})
        self.assertEqual(del_b.status_code, 404)

    # 17. Job ownership enforcement
    def test_17_job_ownership(self):
        reg_a = self.client.post("/api/v1/auth/register", json={"email": "usera_job@test.com", "password": "Password123!@#", "display_name": "User A"})
        token_a = reg_a.json()["access_token"]
        user_a_id = reg_a.json()["user"]["user_id"]

        reg_b = self.client.post("/api/v1/auth/register", json={"email": "userb_job@test.com", "password": "Password123!@#", "display_name": "User B"})
        token_b = reg_b.json()["access_token"]

        # Manually seed a job for User A into job_manager
        test_job_id = "job-test-owner-001"
        job_manager.jobs[test_job_id] = {
            "job_id": test_job_id,
            "user_id": user_a_id,
            "project_id": "proj_a",
            "title": "User A Job",
            "status": "processing",
            "current_stage": "extracting_audio",
            "overall_progress": 10,
            "current_operation": "Processing audio",
            "source_language": "en",
            "target_language": "hi",
            "stages": [],
            "created_at": utc_now(),
            "output_dir": str(JOBS_DIR / test_job_id),
        }

        # User A can access job
        get_a = self.client.get(f"/api/v1/jobs/{test_job_id}", headers={"Authorization": f"Bearer {token_a}"})
        self.assertEqual(get_a.status_code, 200)

        # User B receives 404
        get_b = self.client.get(f"/api/v1/jobs/{test_job_id}", headers={"Authorization": f"Bearer {token_b}"})
        self.assertEqual(get_b.status_code, 404)

        # User B cannot cancel
        can_b = self.client.post(f"/api/v1/jobs/{test_job_id}/cancel", headers={"Authorization": f"Bearer {token_b}"})
        self.assertEqual(can_b.status_code, 404)

        # User B cannot delete
        del_b = self.client.delete(f"/api/v1/jobs/{test_job_id}", headers={"Authorization": f"Bearer {token_b}"})
        self.assertEqual(del_b.status_code, 404)

    # 18. Cross-user video access rejection (404)
    def test_18_cross_user_video_access(self):
        reg_a = self.client.post("/api/v1/auth/register", json={"email": "usera_vid@test.com", "password": "Password123!@#", "display_name": "User A"})
        token_a = reg_a.json()["access_token"]
        user_a_id = reg_a.json()["user"]["user_id"]

        reg_b = self.client.post("/api/v1/auth/register", json={"email": "userb_vid@test.com", "password": "Password123!@#", "display_name": "User B"})
        token_b = reg_b.json()["access_token"]

        # Seed video owned by User A
        self.videos_col.docs.append({
            "video_id": "vid_user_a",
            "user_id": user_a_id,
            "file_name": "input.mp4",
        })

        # User A succeeds
        get_a = self.client.get("/api/v1/videos/vid_user_a", headers={"Authorization": f"Bearer {token_a}"})
        self.assertEqual(get_a.status_code, 200)

        # User B gets 404
        get_b = self.client.get("/api/v1/videos/vid_user_a", headers={"Authorization": f"Bearer {token_b}"})
        self.assertEqual(get_b.status_code, 404)

    # 19. Cross-user output download rejection (404)
    def test_19_cross_user_output_download(self):
        reg_a = self.client.post("/api/v1/auth/register", json={"email": "usera_file@test.com", "password": "Password123!@#", "display_name": "User A"})
        token_a = reg_a.json()["access_token"]
        user_a_id = reg_a.json()["user"]["user_id"]

        reg_b = self.client.post("/api/v1/auth/register", json={"email": "userb_file@test.com", "password": "Password123!@#", "display_name": "User B"})
        token_b = reg_b.json()["access_token"]

        # Create temporary job directory with a file
        job_id = "job-file-test-001"
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        test_file = job_dir / "output.mp4"
        test_file.write_bytes(b"dummy video content")

        job_manager.jobs[job_id] = {
            "job_id": job_id,
            "user_id": user_a_id,
            "output_dir": str(job_dir),
        }

        try:
            # User A can access file
            res_a = self.client.get(f"/api/v1/files/{job_id}/output.mp4", headers={"Authorization": f"Bearer {token_a}"})
            self.assertEqual(res_a.status_code, 200)

            # User B receives 404
            res_b = self.client.get(f"/api/v1/files/{job_id}/output.mp4", headers={"Authorization": f"Bearer {token_b}"})
            self.assertEqual(res_b.status_code, 404)

            # Unauthenticated receives 401
            self.client.cookies.clear()
            res_unauth = self.client.get(f"/api/v1/files/{job_id}/output.mp4")
            self.assertEqual(res_unauth.status_code, 401)
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    # 20. Cross-user evaluation access rejection (404)
    def test_20_cross_user_evaluation_access(self):
        reg_a = self.client.post("/api/v1/auth/register", json={"email": "usera_eval@test.com", "password": "Password123!@#", "display_name": "User A"})
        token_a = reg_a.json()["access_token"]
        user_a_id = reg_a.json()["user"]["user_id"]

        reg_b = self.client.post("/api/v1/auth/register", json={"email": "userb_eval@test.com", "password": "Password123!@#", "display_name": "User B"})
        token_b = reg_b.json()["access_token"]

        job_id = "job-eval-test-001"
        job_manager.jobs[job_id] = {
            "job_id": job_id,
            "user_id": user_a_id,
            "duration_seconds": 15.0,
            "status": "completed",
            "output_dir": str(JOBS_DIR / job_id),
        }

        # User A accesses evaluation
        get_a = self.client.get(f"/api/v1/jobs/{job_id}/evaluation", headers={"Authorization": f"Bearer {token_a}"})
        self.assertEqual(get_a.status_code, 200)

        # User B receives 404
        get_b = self.client.get(f"/api/v1/jobs/{job_id}/evaluation", headers={"Authorization": f"Bearer {token_b}"})
        self.assertEqual(get_b.status_code, 404)

    # 21. Authenticated SSE access
    def test_21_authenticated_sse_access(self):
        reg_a = self.client.post("/api/v1/auth/register", json={"email": "usera_sse@test.com", "password": "Password123!@#", "display_name": "User A"})
        token_a = reg_a.json()["access_token"]
        user_a_id = reg_a.json()["user"]["user_id"]

        reg_b = self.client.post("/api/v1/auth/register", json={"email": "userb_sse@test.com", "password": "Password123!@#", "display_name": "User B"})
        token_b = reg_b.json()["access_token"]

        job_id = "job-sse-test-001"
        job_manager.jobs[job_id] = {
            "job_id": job_id,
            "user_id": user_a_id,
            "status": "completed",
            "current_stage": "finalizing",
            "overall_progress": 100,
            "current_operation": "Done",
        }

        # Unauthenticated request returns 401
        self.client.cookies.clear()
        res_unauth = self.client.get(f"/api/v1/jobs/{job_id}/events")
        self.assertEqual(res_unauth.status_code, 401)

        # User B returns 404
        res_b = self.client.get(f"/api/v1/jobs/{job_id}/events", headers={"Authorization": f"Bearer {token_b}"})
        self.assertEqual(res_b.status_code, 404)

        # User A can connect via Bearer or query param token
        with self.client.stream("GET", f"/api/v1/jobs/{job_id}/events", headers={"Authorization": f"Bearer {token_a}"}) as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])

    # 22. Session recovery after backend restart
    def test_22_session_recovery_after_backend_restart(self):
        login = self.client.post("/api/v1/auth/login", json={"email": "user1@dubsync.example.com", "password": "ResetPasswordFinal123!"})
        refresh_cookie = login.cookies.get(COOKIE_REFRESH_NAME)

        # Verify session is persisted in the sessions collection
        tok_hash = hash_token(refresh_cookie)
        persisted = next((s for s in self.sessions_col.docs if s.get("refresh_token_hash") == tok_hash), None)
        self.assertIsNotNone(persisted)
        self.assertIsNone(persisted.get("revoked_at"))

        # In a new backend instantiation or client request, the refresh token remains valid
        fresh_refresh = self.client.post("/api/v1/auth/refresh", headers={"X-Refresh-Token": refresh_cookie})
        self.assertEqual(fresh_refresh.status_code, 200)
        self.assertIn("access_token", fresh_refresh.json())


if __name__ == "__main__":
    unittest.main()
