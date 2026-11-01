"""Index creation for DubSync MongoDB collections."""
from __future__ import annotations

import logging
from pymongo import ASCENDING, DESCENDING, IndexModel

from database.collections import (
    USERS_COLLECTION,
    SESSIONS_COLLECTION,
    PROJECTS_COLLECTION,
    VIDEOS_COLLECTION,
    TRANSCRIPTS_COLLECTION,
    TRANSLATIONS_COLLECTION,
    GENERATED_AUDIO_COLLECTION,
    DUBBED_VIDEOS_COLLECTION,
    PROCESSING_LOGS_COLLECTION,
    PROCESSING_JOBS_COLLECTION,
    SEGMENTS_COLLECTION,
    EVALUATIONS_COLLECTION,
)

logger = logging.getLogger("dubsync.database.indexes")

INDEX_DEFINITIONS = {
    USERS_COLLECTION: [
        IndexModel([("user_id", ASCENDING)], unique=True, name="idx_users_user_id"),
        IndexModel([("email", ASCENDING)], unique=True, name="idx_users_email"),
    ],
    SESSIONS_COLLECTION: [
        IndexModel([("session_id", ASCENDING)], unique=True, name="idx_sessions_session_id"),
        IndexModel([("user_id", ASCENDING)], name="idx_sessions_user_id"),
        IndexModel([("refresh_token_hash", ASCENDING)], unique=True, name="idx_sessions_token_hash"),
        IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0, name="idx_sessions_ttl_expires"),
    ],
    PROJECTS_COLLECTION: [
        IndexModel([("project_id", ASCENDING)], unique=True, name="idx_projects_project_id"),
        IndexModel([("user_id", ASCENDING)], name="idx_projects_user_id"),
        IndexModel([("created_at", DESCENDING)], name="idx_projects_created_at"),
    ],
    VIDEOS_COLLECTION: [
        IndexModel([("video_id", ASCENDING)], unique=True, name="idx_videos_video_id"),
        IndexModel([("project_id", ASCENDING)], name="idx_videos_project_id"),
        IndexModel([("user_id", ASCENDING)], name="idx_videos_user_id"),
    ],
    PROCESSING_JOBS_COLLECTION: [
        IndexModel([("job_id", ASCENDING)], unique=True, name="idx_jobs_job_id"),
        IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)], name="idx_jobs_user_created"),
        IndexModel([("project_id", ASCENDING), ("created_at", DESCENDING)], name="idx_jobs_project_created"),
        IndexModel([("status", ASCENDING)], name="idx_jobs_status"),
    ],
    TRANSCRIPTS_COLLECTION: [
        IndexModel([("transcript_id", ASCENDING)], unique=True, name="idx_transcripts_transcript_id"),
        IndexModel([("job_id", ASCENDING)], name="idx_transcripts_job_id"),
        IndexModel([("user_id", ASCENDING)], name="idx_transcripts_user_id"),
        IndexModel([("video_id", ASCENDING)], name="idx_transcripts_video_id"),
    ],
    TRANSLATIONS_COLLECTION: [
        IndexModel([("translation_id", ASCENDING)], unique=True, name="idx_translations_translation_id"),
        IndexModel([("transcript_id", ASCENDING)], name="idx_translations_transcript_id"),
        IndexModel([("job_id", ASCENDING)], name="idx_translations_job_id"),
        IndexModel([("user_id", ASCENDING)], name="idx_translations_user_id"),
    ],
    GENERATED_AUDIO_COLLECTION: [
        IndexModel([("audio_id", ASCENDING)], unique=True, name="idx_audio_audio_id"),
        IndexModel([("translation_id", ASCENDING)], name="idx_audio_translation_id"),
        IndexModel([("job_id", ASCENDING)], name="idx_audio_job_id"),
        IndexModel([("user_id", ASCENDING)], name="idx_audio_user_id"),
    ],
    DUBBED_VIDEOS_COLLECTION: [
        IndexModel([("output_id", ASCENDING)], unique=True, name="idx_outputs_output_id"),
        IndexModel([("job_id", ASCENDING)], name="idx_outputs_job_id"),
        IndexModel([("project_id", ASCENDING)], name="idx_outputs_project_id"),
        IndexModel([("user_id", ASCENDING)], name="idx_outputs_user_id"),
    ],
    SEGMENTS_COLLECTION: [
        IndexModel([("segment_id", ASCENDING)], unique=True, name="idx_segments_segment_id"),
        IndexModel([("job_id", ASCENDING), ("segment_index", ASCENDING)], unique=True, name="idx_segments_job_index"),
        IndexModel([("project_id", ASCENDING)], name="idx_segments_project_id"),
        IndexModel([("user_id", ASCENDING)], name="idx_segments_user_id"),
    ],
    PROCESSING_LOGS_COLLECTION: [
        IndexModel([("job_id", ASCENDING), ("timestamp", ASCENDING)], name="idx_logs_job_timestamp"),
        IndexModel([("log_id", ASCENDING)], unique=True, name="idx_logs_log_id"),
        IndexModel([("user_id", ASCENDING)], name="idx_logs_user_id"),
    ],
    EVALUATIONS_COLLECTION: [
        IndexModel([("evaluation_id", ASCENDING)], unique=True, name="idx_evaluations_evaluation_id"),
        IndexModel([("job_id", ASCENDING)], name="idx_evaluations_job_id"),
        IndexModel([("user_id", ASCENDING)], name="idx_evaluations_user_id"),
    ],
}


async def ensure_indexes(db) -> dict[str, list[str]]:
    """Idempotently create indexes across all collections on the target async database."""
    created = {}
    for col_name, indexes in INDEX_DEFINITIONS.items():
        try:
            col = db[col_name]
            result = await col.create_indexes(indexes)
            created[col_name] = result
        except Exception as e:
            logger.warning(f"Could not create indexes for {col_name}: {e}")
            created[col_name] = [f"FAILED: {e}"]
    return created
