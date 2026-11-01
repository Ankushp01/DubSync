"""Test suite for Jobs API schema compatibility, legacy document fallback, and isolation.

Covers:
1. Legacy job without `overall_progress`.
2. Completed legacy job (derives 100%).
3. Failed legacy job (preserves recorded progress).
4. Processing job (uses stage/progress benchmarks).
5. Newly created job (initializes overall_progress and progress to 0).
6. Multiple jobs with mixed old and new schemas (all serialize cleanly).
7. User A cannot see User B's jobs (cross-user isolation).
8. Pagination remains functional.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from bson import ObjectId
from fastapi.testclient import TestClient

from api.main import app
from api.job_manager import job_manager, DEFAULT_STAGES
from api.auth.security import create_access_token
from api.routes.jobs import build_job_status_response, resolve_created_at
from database.models import resolve_overall_progress, utc_now
from database.repositories.jobs import ProcessingJobsRepository


class TestJobsSchemaCompatibility(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.user_a_id = "usr_schema_test_user_a"
        self.user_b_id = "usr_schema_test_user_b"

        self.token_user_a = create_access_token(
            user_id=self.user_a_id,
            email="user_a@example.com",
            expires_delta=timedelta(hours=1),
        )
        self.token_user_b = create_access_token(
            user_id=self.user_b_id,
            email="user_b@example.com",
            expires_delta=timedelta(hours=1),
        )

        # Clear in-memory jobs for predictable testing
        self._orig_jobs = dict(job_manager.jobs)
        job_manager.jobs.clear()

    def tearDown(self):
        job_manager.jobs.clear()
        job_manager.jobs.update(self._orig_jobs)

    # -------------------------------------------------------------------------
    # Unit Tests for resolve_overall_progress
    # -------------------------------------------------------------------------
    def test_01_legacy_job_without_overall_progress_fallback(self):
        """Legacy document with only 'progress' field resolves correctly."""
        doc = {
            "job_id": "job-legacy-1",
            "status": "processing",
            "progress": 42,
        }
        self.assertEqual(resolve_overall_progress(doc), 42)

    def test_02_completed_legacy_job_derives_100_percent(self):
        """Completed legacy job without any progress fields derives 100%."""
        doc = {
            "job_id": "job-completed-legacy",
            "status": "completed",
        }
        self.assertEqual(resolve_overall_progress(doc), 100)

    def test_03_failed_legacy_job_preserves_recorded_progress(self):
        """Failed legacy job preserves its recorded progress percentage."""
        doc = {
            "job_id": "job-failed-legacy",
            "status": "failed",
            "progress": 65,
        }
        self.assertEqual(resolve_overall_progress(doc), 65)

    def test_04_processing_job_uses_stage_information(self):
        """Processing job without progress fields uses stage benchmark fallback."""
        doc_extracting = {
            "job_id": "job-proc-extracting",
            "status": "processing",
            "current_stage": "extracting_audio",
        }
        self.assertEqual(resolve_overall_progress(doc_extracting), 10)

        doc_aligning = {
            "job_id": "job-proc-aligning",
            "status": "processing",
            "current_stage": "aligning_audio",
        }
        self.assertEqual(resolve_overall_progress(doc_aligning), 85)

        # Queued job without progress info resolves to 0%
        doc_queued = {
            "job_id": "job-queued",
            "status": "queued",
        }
        self.assertEqual(resolve_overall_progress(doc_queued), 0)

    # -------------------------------------------------------------------------
    # API Integration Tests for GET /api/v1/jobs
    # -------------------------------------------------------------------------
    def test_05_newly_created_job_in_memory_has_overall_progress(self):
        """Newly created job initializes overall_progress and progress to 0."""
        job_id = "job-new-test-05"
        job_manager.jobs[job_id] = {
            "job_id": job_id,
            "title": "New Test Job",
            "user_id": self.user_a_id,
            "status": "queued",
            "current_stage": "extracting_audio",
            "progress": 0,
            "overall_progress": 0,
            "current_operation": "Job queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_language": "en",
            "target_language": "hi",
            "stages": [dict(s) for s in DEFAULT_STAGES],
        }

        resp = self.client.get(
            "/api/v1/jobs",
            headers={"Authorization": f"Bearer {self.token_user_a}"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["job_id"], job_id)
        self.assertEqual(data[0]["overall_progress"], 0)

    def test_06_multiple_jobs_with_mixed_old_and_new_schemas(self):
        """GET /api/v1/jobs succeeds with mixed legacy and new schemas without KeyError."""
        # 1. Legacy doc from MongoDB (has 'progress', no 'overall_progress')
        job_manager.jobs["job-legacy-mongo"] = {
            "job_id": "job-legacy-mongo",
            "title": "Legacy Mongo Job",
            "user_id": self.user_a_id,
            "status": "completed",
            "current_stage": "finalizing",
            "progress": 100,
            "current_operation": "Dubbing completed successfully",
            "created_at": "2026-10-01T10:00:00Z",
            "source_language": "en",
            "target_language": "hi",
            "stages": [dict(s) for s in DEFAULT_STAGES],
        }

        # 2. Legacy doc without any progress fields at all
        job_manager.jobs["job-legacy-bare"] = {
            "job_id": "job-legacy-bare",
            "title": "Bare Legacy Job",
            "user_id": self.user_a_id,
            "status": "completed",
            "created_at": "2026-10-02T10:00:00Z",
            "source_language": "en",
            "target_language": "hi",
        }

        # 3. New schema doc (has both 'progress' and 'overall_progress')
        job_manager.jobs["job-new-schema"] = {
            "job_id": "job-new-schema",
            "title": "Modern Schema Job",
            "user_id": self.user_a_id,
            "status": "processing",
            "current_stage": "translating",
            "progress": 45,
            "overall_progress": 45,
            "current_operation": "Translating segments",
            "created_at": "2026-10-03T10:00:00Z",
            "source_language": "en",
            "target_language": "hi",
            "stages": [dict(s) for s in DEFAULT_STAGES],
        }

        # 4. Failed legacy job
        job_manager.jobs["job-failed-legacy"] = {
            "job_id": "job-failed-legacy",
            "title": "Failed Legacy Job",
            "user_id": self.user_a_id,
            "status": "failed",
            "progress": 25,
            "current_stage": "transcribing",
            "error": "Transcription timeout",
            "created_at": "2026-10-04T10:00:00Z",
            "source_language": "en",
            "target_language": "hi",
        }

        resp = self.client.get(
            "/api/v1/jobs?page=1&limit=10",
            headers={"Authorization": f"Bearer {self.token_user_a}"},
        )
        self.assertEqual(resp.status_code, 200, f"Expected 200, got: {resp.text}")
        data = resp.json()
        self.assertIn("items", data)
        self.assertEqual(data["total"], 4)

        items_by_id = {item["job_id"]: item for item in data["items"]}
        self.assertEqual(items_by_id["job-legacy-mongo"]["overall_progress"], 100)
        self.assertEqual(items_by_id["job-legacy-bare"]["overall_progress"], 100)
        self.assertEqual(items_by_id["job-new-schema"]["overall_progress"], 45)
        self.assertEqual(items_by_id["job-failed-legacy"]["overall_progress"], 25)

    def test_07_user_isolation_user_a_cannot_see_user_b_jobs(self):
        """User A only sees User A's jobs; User B only sees User B's jobs."""
        job_manager.jobs["job-user-a"] = {
            "job_id": "job-user-a",
            "title": "User A Job",
            "user_id": self.user_a_id,
            "status": "completed",
            "progress": 100,
            "created_at": "2026-10-01T10:00:00Z",
            "source_language": "en",
            "target_language": "hi",
        }
        job_manager.jobs["job-user-b"] = {
            "job_id": "job-user-b",
            "title": "User B Job",
            "user_id": self.user_b_id,
            "status": "completed",
            "progress": 100,
            "created_at": "2026-10-01T11:00:00Z",
            "source_language": "en",
            "target_language": "hi",
        }

        # User A requests jobs
        resp_a = self.client.get(
            "/api/v1/jobs",
            headers={"Authorization": f"Bearer {self.token_user_a}"},
        )
        self.assertEqual(resp_a.status_code, 200)
        jobs_a = resp_a.json()
        self.assertEqual(len(jobs_a), 1)
        self.assertEqual(jobs_a[0]["job_id"], "job-user-a")

        # User B requests jobs
        resp_b = self.client.get(
            "/api/v1/jobs",
            headers={"Authorization": f"Bearer {self.token_user_b}"},
        )
        self.assertEqual(resp_b.status_code, 200)
        jobs_b = resp_b.json()
        self.assertEqual(len(jobs_b), 1)
        self.assertEqual(jobs_b[0]["job_id"], "job-user-b")

    def test_08_pagination_remains_functional(self):
        """Pagination returns correct items, page, limit, total, and total_pages."""
        # Create 7 jobs for User A
        for i in range(1, 8):
            job_manager.jobs[f"job-page-{i:02d}"] = {
                "job_id": f"job-page-{i:02d}",
                "title": f"Page Job {i}",
                "user_id": self.user_a_id,
                "status": "completed",
                "progress": 100,
                "created_at": f"2026-10-01T10:{i:02d}:00Z",
                "source_language": "en",
                "target_language": "hi",
            }

        # Page 1, limit 3
        resp1 = self.client.get(
            "/api/v1/jobs?page=1&limit=3",
            headers={"Authorization": f"Bearer {self.token_user_a}"},
        )
        self.assertEqual(resp1.status_code, 200)
        p1_data = resp1.json()
        self.assertEqual(p1_data["page"], 1)
        self.assertEqual(p1_data["limit"], 3)
        self.assertEqual(p1_data["total"], 7)
        self.assertEqual(p1_data["total_pages"], 3)
        self.assertEqual(len(p1_data["items"]), 3)

        # Page 2, limit 3
        resp2 = self.client.get(
            "/api/v1/jobs?page=2&limit=3",
            headers={"Authorization": f"Bearer {self.token_user_a}"},
        )
        self.assertEqual(resp2.status_code, 200)
        p2_data = resp2.json()
        self.assertEqual(p2_data["page"], 2)
        self.assertEqual(len(p2_data["items"]), 3)

        # Page 3, limit 3 (last page with remaining 1 item)
        resp3 = self.client.get(
            "/api/v1/jobs?page=3&limit=3",
            headers={"Authorization": f"Bearer {self.token_user_a}"},
        )
        self.assertEqual(resp3.status_code, 200)
        p3_data = resp3.json()
        self.assertEqual(p3_data["page"], 3)
        self.assertEqual(len(p3_data["items"]), 1)

    # -------------------------------------------------------------------------
    # Strict Review Focus Tests
    # -------------------------------------------------------------------------
    def test_09_invalid_job_records_handled_explicitly(self):
        """Invalid job records without job_id/_id raise ValueError and are safely skipped in list."""
        # 1. Direct call raises ValueError (no 'unknown_job' substitution)
        with self.assertRaises(ValueError) as ctx:
            build_job_status_response({"status": "processing"})
        self.assertIn("missing required identifier", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            build_job_status_response({})
        self.assertIn("missing required identifier", str(ctx2.exception))

        # 2. In list_jobs, invalid documents are skipped without crashing the endpoint
        job_manager.jobs["valid-job"] = {
            "job_id": "valid-job",
            "title": "Valid Job",
            "user_id": self.user_a_id,
            "status": "completed",
            "progress": 100,
        }
        job_manager.jobs["invalid-job"] = {
            # Intentionally missing job_id and _id
            "title": "Corrupt Job",
            "user_id": self.user_a_id,
            "status": "processing",
        }

        resp = self.client.get(
            "/api/v1/jobs",
            headers={"Authorization": f"Bearer {self.token_user_a}"},
        )
        self.assertEqual(resp.status_code, 200)
        items = resp.json()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["job_id"], "valid-job")

    def test_10_missing_created_at_preserves_nullability_and_fallbacks(self):
        """Missing historical created_at preserves nullability or resolves from ObjectId/started_at."""
        # 1. Purely missing without fallbacks returns None (does NOT invent current UTC time)
        doc_bare = {
            "job_id": "job-no-dates",
            "status": "queued",
        }
        self.assertIsNone(resolve_created_at(doc_bare))
        resp_bare = build_job_status_response(doc_bare)
        self.assertIsNone(resp_bare.created_at)

        # 2. Derives from ObjectId generation time if present in _id
        oid = ObjectId("654321000000000000000001")
        doc_oid = {
            "job_id": "job-with-oid",
            "_id": oid,
            "status": "completed",
        }
        resolved_oid_time = resolve_created_at(doc_oid)
        self.assertIsNotNone(resolved_oid_time)
        self.assertEqual(resolved_oid_time, oid.generation_time.isoformat())
        resp_oid = build_job_status_response(doc_oid)
        self.assertEqual(resp_oid.created_at, oid.generation_time.isoformat())

        # 3. Derives from started_at when created_at is omitted
        doc_started = {
            "job_id": "job-started",
            "status": "processing",
            "started_at": "2026-09-15T08:30:00Z",
        }
        self.assertEqual(resolve_created_at(doc_started), "2026-09-15T08:30:00Z")
        resp_started = build_job_status_response(doc_started)
        self.assertEqual(resp_started.created_at, "2026-09-15T08:30:00Z")

    def test_11_progress_resolution_never_overwrites_legitimate_progress(self):
        """Legitimate recorded progress is preserved for processing, failed, cancelled jobs."""
        # Processing job with recorded progress > 0 does NOT get overridden by stage benchmark
        doc_proc = {
            "job_id": "job-proc-77",
            "status": "processing",
            "current_stage": "extracting_audio",  # benchmark is 10
            "overall_progress": 77,
        }
        self.assertEqual(resolve_overall_progress(doc_proc), 77)

        # Cancelled job with recorded progress preserves it
        doc_cancelled = {
            "job_id": "job-cancelled-33",
            "status": "cancelled",
            "progress": 33,
        }
        self.assertEqual(resolve_overall_progress(doc_cancelled), 33)

        # Failed job with recorded progress preserves it
        doc_failed = {
            "job_id": "job-failed-58",
            "status": "failed",
            "overall_progress": 58,
        }
        self.assertEqual(resolve_overall_progress(doc_failed), 58)

        # Completed job always resolves to 100% even if document recorded 0
        doc_completed_zero = {
            "job_id": "job-completed-zero",
            "status": "completed",
            "progress": 0,
        }
        self.assertEqual(resolve_overall_progress(doc_completed_zero), 100)

        # Queued job always resolves to 0%
        doc_queued = {
            "job_id": "job-queued",
            "status": "queued",
            "progress": 50,  # invalid/premature recorded progress for queued
        }
        self.assertEqual(resolve_overall_progress(doc_queued), 0)


if __name__ == "__main__":
    unittest.main()
