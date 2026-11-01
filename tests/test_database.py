import asyncio
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId

from fastapi.testclient import TestClient
from api.auth.security import create_access_token
from api.main import app
from database.collections import (
    COLLECTIONS,
    COLLECTION_USERS,
    COLLECTION_PROJECTS,
    COLLECTION_VIDEOS,
    COLLECTION_TRANSCRIPTS,
    COLLECTION_TRANSLATIONS,
    COLLECTION_AUDIO,
    COLLECTION_DUBBED_VIDEOS,
    COLLECTION_LOGS,
    COLLECTION_JOBS,
    COLLECTION_SEGMENTS,
    COLLECTION_EVALUATIONS,
)
from database.models import (
    serialize_doc,
    UserDoc,
    ProjectDoc,
    VideoDoc,
    TranscriptDoc,
    TranslationDoc,
    AudioDoc,
    DubbedVideoDoc,
    LogDoc,
    JobDoc,
    SegmentDoc,
    EvaluationDoc,
)
from database.connection import DatabaseManager, db_manager
from database.repositories import (
    UsersRepository,
    DubbingProjectsRepository,
    VideosRepository,
    TranscriptsRepository,
    TranslationsRepository,
    GeneratedAudioRepository,
    DubbedVideosRepository,
    ProcessingLogsRepository,
    ProcessingJobsRepository,
    SegmentsRepository,
    EvaluationsRepository,
)


class TestDatabaseModelsAndSerialization(unittest.TestCase):
    """Test serialization helpers and Pydantic database models."""

    def test_serialize_doc_with_objectid_and_datetime(self):
        obj_id = ObjectId()
        now = datetime.utcnow()
        doc = {
            "_id": obj_id,
            "project_id": "proj-123",
            "created_at": now,
            "nested": {
                "sub_id": ObjectId(),
                "time": now,
                "numbers": [1, 2, 3],
            },
            "array_of_objects": [{"oid": ObjectId()}],
        }

        serialized = serialize_doc(doc)
        self.assertEqual(serialized["_id"], str(obj_id))
        self.assertIsInstance(serialized["created_at"], str)
        self.assertEqual(serialized["created_at"], now.isoformat())
        self.assertIsInstance(serialized["nested"]["sub_id"], str)
        self.assertIsInstance(serialized["nested"]["time"], str)
        self.assertIsInstance(serialized["array_of_objects"][0]["oid"], str)

    def test_all_11_collections_defined(self):
        expected_collections = {
            "users",
            "dubbing_projects",
            "videos",
            "transcripts",
            "translations",
            "generated_audio",
            "dubbed_videos",
            "processing_logs",
            "processing_jobs",
            "segments",
            "evaluations",
            "sessions",
        }
        self.assertEqual(set(COLLECTIONS), expected_collections)

    def test_model_instantiation(self):
        user = UserDoc(user_id="user_1", email="test@dubsync.ai", name="Test User")
        self.assertEqual(user.user_id, "user_1")

        project = ProjectDoc(project_id="proj_1", project_name="Demo Project", status="created")
        self.assertEqual(project.status, "created")

        video = VideoDoc(video_id="vid_1", project_id="proj_1", file_name="front.mp4", duration=16.3, file_path="/path/front.mp4")
        self.assertEqual(video.duration, 16.3)

        job = JobDoc(job_id="job-1", project_id="proj_1", title="Job 1", source_language="en", target_language="hi")
        self.assertEqual(job.status, "queued")

        segment = SegmentDoc(
            segment_id="seg_1",
            job_id="job-1",
            segment_index=1,
            start_time=0.0,
            end_time=8.88,
            original_text="Hello world",
        )
        self.assertEqual(segment.segment_index, 1)

        eval_doc = EvaluationDoc(
            evaluation_id="eval_1",
            job_id="job-1",
            speech_preservation_percentage=100.0,
            trimmed_audio_seconds=0.0,
        )
        self.assertEqual(eval_doc.speech_preservation_percentage, 100.0)


class MockAsyncCursor:
    def __init__(self, docs):
        self.docs = docs

    def sort(self, *args, **kwargs):
        return self

    def skip(self, n):
        self.docs = self.docs[n:]
        return self

    def limit(self, n):
        self.docs = self.docs[:n]
        return self

    async def to_list(self, length=None):
        return list(self.docs)

    def __aiter__(self):
        self._iter = iter(self.docs)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class MockAsyncCollection:
    """In-memory async mock collection to test repository operations without live MongoDB."""

    def __init__(self, name="mock_col"):
        self.name = name
        self.docs = []

    async def insert_one(self, doc):
        stored = dict(doc)
        if "_id" not in stored:
            stored["_id"] = ObjectId()
        self.docs.append(stored)
        mock_result = MagicMock()
        mock_result.inserted_id = stored["_id"]
        return mock_result

    async def insert_many(self, docs):
        for d in docs:
            await self.insert_one(d)
        mock_result = MagicMock()
        return mock_result

    async def find_one(self, filter_dict, *args, **kwargs):
        for d in self.docs:
            match = True
            for k, v in filter_dict.items():
                if d.get(k) != v:
                    match = False
                    break
            if match:
                return dict(d)
        return None

    def find(self, filter_dict=None):
        filter_dict = filter_dict or {}
        matches = []
        for d in self.docs:
            match = True
            for k, v in filter_dict.items():
                if d.get(k) != v:
                    match = False
                    break
            if match:
                matches.append(dict(d))
        return MockAsyncCursor(matches)

    async def count_documents(self, filter_dict=None):
        cursor = self.find(filter_dict)
        return len(cursor.docs)

    async def update_one(self, filter_dict, update_dict):
        mock_res = MagicMock()
        mock_res.modified_count = 0
        set_vals = update_dict.get("$set", {})
        for d in self.docs:
            match = True
            for k, v in filter_dict.items():
                if d.get(k) != v:
                    match = False
                    break
            if match:
                d.update(set_vals)
                mock_res.modified_count = 1
                break
        return mock_res

    async def update_many(self, filter_dict, update_dict):
        mock_res = MagicMock()
        count = 0
        set_vals = update_dict.get("$set", {})
        for d in self.docs:
            match = True
            for k, v in filter_dict.items():
                if d.get(k) != v:
                    match = False
                    break
            if match:
                d.update(set_vals)
                count += 1
        mock_res.modified_count = count
        return mock_res

    async def find_one_and_update(self, filter_dict, update_dict, return_document=True):
        set_vals = update_dict.get("$set", {})
        for d in self.docs:
            match = True
            for k, v in filter_dict.items():
                if d.get(k) != v:
                    match = False
                    break
            if match:
                d.update(set_vals)
                return dict(d)
        return None

    async def replace_one(self, filter_dict, doc, upsert=False):
        for i, d in enumerate(self.docs):
            match = True
            for k, v in filter_dict.items():
                if d.get(k) != v:
                    match = False
                    break
            if match:
                self.docs[i] = dict(doc)
                mock_res = MagicMock()
                mock_res.modified_count = 1
                return mock_res
        if upsert:
            await self.insert_one(doc)
            mock_res = MagicMock()
            mock_res.upserted_id = doc.get("_id")
            return mock_res
        mock_res = MagicMock()
        mock_res.modified_count = 0
        return mock_res

    async def delete_one(self, filter_dict):
        mock_res = MagicMock()
        mock_res.deleted_count = 0
        for i, d in enumerate(self.docs):
            match = True
            for k, v in filter_dict.items():
                if d.get(k) != v:
                    match = False
                    break
            if match:
                del self.docs[i]
                mock_res.deleted_count = 1
                break
        return mock_res


class TestRepositoriesWithMockCollection(unittest.IsolatedAsyncioTestCase):
    """Test all repositories against mock async collections."""

    async def test_projects_repository(self):
        repo = DubbingProjectsRepository()
        mock_col = MockAsyncCollection("dubbing_projects")
        repo.get_collection = MagicMock(return_value=mock_col)

        # 1. Create project
        p = await repo.create_project(project_name="Test Project", project_id="proj_abc")
        self.assertEqual(p["project_name"], "Test Project")
        self.assertEqual(p["project_id"], "proj_abc")

        # 2. Get by ID
        fetched = await repo.get_by_id("proj_abc")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["project_name"], "Test Project")

        # 3. List projects
        paged = await repo.list_projects(page=1, limit=10)
        self.assertEqual(paged["total"], 1)
        self.assertEqual(len(paged["items"]), 1)

        # 4. Update project
        up = await repo.update_project("proj_abc", {"metadata": {"desc": "Updated description"}})
        self.assertEqual(up["metadata"]["desc"], "Updated description")

        # 5. Delete project
        deleted = await repo.delete_project("proj_abc")
        self.assertTrue(deleted)
        self.assertIsNone(await repo.get_by_id("proj_abc"))

    async def test_jobs_recovery_interrupted(self):
        repo = ProcessingJobsRepository()
        mock_col = MockAsyncCollection("processing_jobs")
        repo.get_collection = MagicMock(return_value=mock_col)

        # Seed jobs: one processing, one completed
        await mock_col.insert_one({"job_id": "job_1", "status": "processing", "title": "Incomplete Job"})
        await mock_col.insert_one({"job_id": "job_2", "status": "completed", "title": "Done Job"})

        recovered = await repo.recover_interrupted_jobs()
        self.assertEqual(recovered, 1)

        job1 = await repo.get_by_id("job_1")
        self.assertEqual(job1["status"], "failed")
        self.assertIn("interrupted", job1["error"].lower())

        job2 = await repo.get_by_id("job_2")
        self.assertEqual(job2["status"], "completed")

    async def test_segments_repository(self):
        repo = SegmentsRepository()
        mock_col = MockAsyncCollection("segments")
        repo.get_collection = MagicMock(return_value=mock_col)

        segs = [
            {"segment_id": "s1", "job_id": "job-x", "segment_index": 1, "start_time": 0.0, "end_time": 5.0},
            {"segment_id": "s2", "job_id": "job-x", "segment_index": 2, "start_time": 5.0, "end_time": 10.0},
        ]
        created = await repo.create_segments_batch(segs)
        self.assertEqual(len(created), 2)

        by_job = await repo.get_by_job("job-x")
        self.assertEqual(len(by_job), 2)
        self.assertEqual(by_job[0]["segment_index"], 1)

    async def test_evaluations_repository(self):
        repo = EvaluationsRepository()
        mock_col = MockAsyncCollection("evaluations")
        repo.get_collection = MagicMock(return_value=mock_col)

        res = await repo.create_evaluation(
            job_id="job-y",
            evaluation_id="eval-1",
            speech_preservation_percentage=100.0,
            timeline_extension_seconds=0.0,
        )
        self.assertEqual(res["speech_preservation_percentage"], 100.0)

        fetched = await repo.get_by_job("job-y")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["evaluation_id"], "eval-1")


class TestFastAPIDatabaseEndpoints(unittest.TestCase):
    """Test database endpoints via FastAPI TestClient."""

    def setUp(self):
        self.client = TestClient(app)
        self._orig_connected = db_manager._is_connected

    def tearDown(self):
        db_manager._is_connected = self._orig_connected

    def test_database_health_endpoint_when_disconnected(self):
        """When database is disconnected, must return HTTP 503 with truthful details."""
        db_manager._is_connected = False
        with patch.object(db_manager, "health_check", AsyncMock(return_value={
            "status": "disconnected",
            "database": "dubsync",
            "error": "Database is not connected",
        })):
            res = self.client.get("/api/v1/health/database")
            self.assertEqual(res.status_code, 503)
            data = res.json()
            self.assertEqual(data["status"], "disconnected")
            self.assertEqual(data["database"], "dubsync")

    def test_database_health_endpoint_when_connected(self):
        """When database is connected, must return HTTP 200 with connection metadata."""
        db_manager._is_connected = True
        with patch.object(db_manager, "health_check", AsyncMock(return_value={
            "status": "connected",
            "database": "dubsync",
            "server_version": "7.0.5",
            "connection": "atlas",
            "ping_ms": 42.5,
        })):
            res = self.client.get("/api/v1/health/database")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "connected")
            self.assertEqual(data["connection"], "atlas")
            self.assertEqual(data["ping_ms"], 42.5)

    def test_projects_crud_endpoint_returns_503_when_offline(self):
        """Project operations return 503 Service Unavailable when MongoDB is not connected."""
        db_manager._is_connected = False
        auth_h = {"Authorization": f"Bearer {create_access_token('usr_offline', 'offline@example.com')}"}
        res = self.client.get("/api/v1/projects", headers=auth_h)
        self.assertEqual(res.status_code, 503)
        self.assertIn("not connected", res.json()["detail"].lower())

        create_res = self.client.post("/api/v1/projects", json={"name": "Offline Project"}, headers=auth_h)
        self.assertEqual(create_res.status_code, 503)


if __name__ == "__main__":
    unittest.main()
