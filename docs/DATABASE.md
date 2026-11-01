# DubSync MongoDB Atlas Architecture & Integration Guide

This document details the persistent database architecture implemented for the DubSync multilingual neural video-dubbing pipeline.

---

## 1. Architecture Overview

DubSync utilizes **MongoDB Atlas** as its centralized, asynchronous persistence engine, managed via `pymongo.AsyncMongoClient`. 

### Key Principles:
1. **Separation of Concerns:**
   - **MongoDB Atlas** stores all structured workflow records: users, projects, video metadata, speech recognition transcripts, machine translations, generated audio metadata, dialogue segments, processing logs, job execution lifecycles, and evaluation metrics.
   - **Local File System** continues to store all heavy media binaries (source MP4 files, extracted audio WAVs, TTS audio chunks, dubbed output videos, and Wav2Lip model checkpoints). Document records in MongoDB store file paths, relative URLs, duration, dimensions, and file sizes.
2. **Graceful Degradation & Truthful Health Checks:**
   - Database availability does not crash or block local dubbing workflows.
   - `GET /api/v1/health/database` returns genuine HTTP 200 with server metadata and ping latency when Atlas is reachable, and HTTP 503 (`status: disconnected`) when unreachable.
   - Database operations (such as project creation) return HTTP 503 Service Unavailable when the database is offline, rather than pretending to succeed or fabricating fake data.
3. **Automatic Startup Job Recovery:**
   - When the backend restarts, `ProcessingJobsRepository.recover_interrupted_jobs()` queries for any jobs stranded in `"processing"` state and transitions them to `"failed"` with descriptive diagnostics: `"Process was interrupted by backend restart before completion. Intermediate artifacts preserved."`
   - Completed jobs, transcripts, translations, generated audio, and dialogue segments remain intact.
4. **Thread-Safe Asynchronous Dispatching:**
   - The dubbing pipeline executes neural inference on worker threads (`ThreadPoolExecutor`). To avoid event loop conflicts with `AsyncMongoClient`, worker threads invoke database mutations via `run_coro_sync()`, which safely dispatches coroutines to the main FastAPI event loop using `asyncio.run_coroutine_threadsafe()`.

---

## 2. Collections & Schemas

The database schema encompasses **11 dedicated collections**:

### 1. `users`
- **Purpose:** User profiles and pipeline configuration preferences.
- **Fields:**
  - `user_id` (string, unique)
  - `name` (string)
  - `email` (string, unique)
  - `created_at` (ISO datetime)
  - `updated_at` (ISO datetime)
  - `preferences` (object: `default_source_language`, `default_target_language`, `default_voice`)

### 2. `dubbing_projects`
- **Purpose:** Top-level project entities grouping video assets and dubbing jobs.
- **Fields:**
  - `project_id` (string, unique)
  - `user_id` (string, optional)
  - `project_name` (string)
  - `source_language` (string, e.g. `"en"`)
  - `target_languages` (array of strings, e.g. `["hi"]`)
  - `status` (string: `"created"`, `"processing"`, `"completed"`, `"failed"`)
  - `created_at`, `updated_at` (ISO datetimes)
  - `metadata` (object)

### 3. `videos`
- **Purpose:** Source video metadata and storage locations.
- **Fields:**
  - `video_id` (string, unique)
  - `project_id` (string)
  - `file_name` (string)
  - `file_path` (string, local disk location)
  - `format` (string, e.g. `"mp4"`)
  - `duration` (float, seconds)
  - `file_size` (integer, bytes)
  - `fps` (float)
  - `width`, `height` (integers)
  - `uploaded_at` (ISO datetime)

### 4. `transcripts`
- **Purpose:** Automatic Speech Recognition (ASR) outputs from Faster-Whisper.
- **Fields:**
  - `transcript_id` (string, unique)
  - `video_id` (string)
  - `project_id` (string)
  - `job_id` (string)
  - `source_language` (string)
  - `original_text` (string)
  - `segments` (array of segment objects with timestamps)
  - `created_at` (ISO datetime)

### 5. `translations`
- **Purpose:** Stage 2 translated dialogue and pacing adaptation results.
- **Fields:**
  - `translation_id` (string, unique)
  - `transcript_id` (string)
  - `job_id` (string)
  - `source_language` (string)
  - `target_language` (string)
  - `source_text` (string)
  - `translated_text` (string)
  - `segments` (array of translated dialogue segments)
  - `created_at` (ISO datetime)

### 6. `generated_audio`
- **Purpose:** Stage 3 voice synthesis and assembled audio stream metadata.
- **Fields:**
  - `audio_id` (string, unique)
  - `translation_id` (string)
  - `job_id` (string)
  - `voice_model` (string, e.g. `"Chatterbox Multilingual V3"`)
  - `target_language` (string)
  - `audio_path` (string, local path to final assembled WAV)
  - `duration` (float, seconds)
  - `sample_rate` (integer, e.g. 24000)
  - `channels` (integer, 1 for mono)
  - `created_at` (ISO datetime)

### 7. `dubbed_videos`
- **Purpose:** Stage 4 Wav2Lip facial synchronization output records.
- **Fields:**
  - `output_id` (string, unique)
  - `project_id` (string)
  - `job_id` (string)
  - `video_id` (string)
  - `audio_id` (string, optional)
  - `output_path` (string, local path to final dubbed MP4)
  - `format` (string, e.g. `"mp4"`)
  - `duration` (float, seconds)
  - `status` (string: `"completed"`, `"failed"`)
  - `created_at` (ISO datetime)

### 8. `processing_logs`
- **Purpose:** Structured execution logs and stage transition telemetry.
- **Fields:**
  - `log_id` (string, unique)
  - `project_id` (string, optional)
  - `job_id` (string)
  - `stage` (string, e.g. `"extracting_audio"`, `"transcribing"`, etc.)
  - `status` (string: `"queued"`, `"processing"`, `"completed"`, `"failed"`)
  - `timestamp` (ISO datetime)
  - `details` (object)
  - `severity` (string: `"info"`, `"warning"`, `"error"`)

### 9. `processing_jobs`
- **Purpose:** Master state machine tracking for all dubbing jobs.
- **Fields:**
  - `job_id` (string, unique)
  - `project_id` (string)
  - `video_id` (string, optional)
  - `user_id` (string, optional)
  - `title` (string)
  - `source_language` (string)
  - `detected_source_language` (string, optional)
  - `target_language` (string)
  - `voice_mode` (string: `"original"`, `"custom"`)
  - `status` (string: `"queued"`, `"processing"`, `"completed"`, `"failed"`, `"cancelled"`, `"interrupted"`)
  - `current_stage` (string)
  - `progress` (integer, 0-100)
  - `current_operation` (string)
  - `started_at`, `completed_at` (ISO datetimes)
  - `error` (string, optional)
  - `configuration` (object)
  - `stages` (array of stage tracking items)
  - `artifacts` (object: relative URLs to media assets)
  - `created_at`, `updated_at` (ISO datetimes)

### 10. `segments`
- **Purpose:** Granular, segment-by-segment alignment and duration adaptation accounting.
- **Fields:**
  - `segment_id` (string, unique)
  - `job_id` (string)
  - `project_id` (string, optional)
  - `transcript_id`, `translation_id` (strings, optional)
  - `segment_index` (integer, 1-indexed)
  - `start_time`, `end_time` (floats, seconds)
  - `original_text`, `translated_text` (strings)
  - `raw_tts_duration`, `adapted_tts_duration` (floats)
  - `speed_factor` (float, e.g. 1.10)
  - `audio_path` (string, optional)
  - `final_placement_start`, `final_placement_end` (floats)
  - `trimming_seconds` (float, 0.0 for preserved speech)
  - `status` (string: `"completed"`, `"failed"`)
  - `created_at` (ISO datetime)

### 11. `evaluations`
- **Purpose:** Stage 3 and Stage 4 empirical duration, speech preservation, and media integrity benchmarks.
- **Fields:**
  - `evaluation_id` (string, unique)
  - `job_id` (string)
  - `project_id`, `output_id` (strings, optional)
  - `created_at` (ISO datetime)
  - `video_duration`, `audio_duration` (floats)
  - `video_frame_count` (integer)
  - `frame_rate` (float)
  - `speech_preservation_percentage` (float, e.g. 100.0)
  - `trimmed_audio_seconds` (float)
  - `timeline_extension_seconds` (float)
  - `lip_sync_metrics` (object)
  - `media_integrity` (object: decodability, RMS, peak amplitude)
  - `segment_metrics` (array of segment evaluation details)
  - `unmeasured_or_null_metrics` (object)
  - `warnings` (array of strings)

---

## 3. Database Indexes

Automatic idempotent index creation is triggered on initial connection in `database.indexes.ensure_indexes()`.

| Collection | Field(s) | Type | Purpose |
|------------|----------|------|---------|
| `users` | `user_id` | Unique | Primary key lookup |
| `users` | `email` | Unique | Login & uniqueness |
| `dubbing_projects` | `project_id` | Unique | Primary key lookup |
| `dubbing_projects` | `user_id`, `created_at: -1` | Compound | User project history listing |
| `videos` | `video_id` | Unique | Video metadata retrieval |
| `videos` | `project_id` | Standard | Project video listing |
| `transcripts` | `transcript_id` | Unique | Transcript lookup |
| `transcripts` | `job_id` | Standard | Job transcript binding |
| `translations` | `translation_id` | Unique | Translation lookup |
| `translations` | `job_id` | Standard | Job translation binding |
| `generated_audio` | `audio_id` | Unique | Audio record lookup |
| `generated_audio` | `job_id` | Standard | Job audio retrieval |
| `dubbed_videos` | `output_id` | Unique | Video output lookup |
| `dubbed_videos` | `job_id` | Standard | Job output retrieval |
| `processing_logs` | `job_id`, `timestamp: -1` | Compound | Fast reverse chronological log stream |
| `processing_jobs` | `job_id` | Unique | Primary job identification |
| `processing_jobs` | `status` | Standard | Active job queries & startup recovery |
| `processing_jobs` | `project_id`, `created_at: -1` | Compound | Filtered workspace queries |
| `segments` | `job_id`, `segment_index: 1` | Compound Unique | Segment sequence integrity |
| `evaluations` | `job_id` | Standard | Evaluation lookups by job |

---

## 4. API Endpoints

### Database Health Check
- **`GET /api/v1/health/database`** (and `/health/database`):
  - Returns `200 OK` when connected:
    ```json
    {
      "status": "connected",
      "database": "dubsync",
      "server_version": "7.0.5",
      "connection": "atlas",
      "ping_ms": 38.2
    }
    ```
  - Returns `503 Service Unavailable` when offline:
    ```json
    {
      "status": "disconnected",
      "database": "dubsync",
      "error": "The DNS query name does not exist: _mongodb._tcp.herocluster..."
    }
    ```

### Projects CRUD (`/api/v1/projects`)
- `POST /api/v1/projects` — Create project (accepts `name` or `project_name`).
- `GET /api/v1/projects?page=1&limit=20` — Paginated projects list.
- `GET /api/v1/projects/{project_id}` — Retrieve project metadata.
- `PATCH /api/v1/projects/{project_id}` — Update project metadata.
- `DELETE /api/v1/projects/{project_id}` — Delete project.

### Pipeline Entity Endpoints (`/api/v1/...`)
- `GET /api/v1/videos/{video_id}` — Video record.
- `GET /api/v1/transcripts/{transcript_id}` — Transcript record.
- `GET /api/v1/translations/{translation_id}` — Translation record.
- `GET /api/v1/outputs/{output_id}` — Dubbed video output record.
- `GET /api/v1/jobs/{job_id}/segments` — Segment records with duration adaptation details.
- `GET /api/v1/jobs/{job_id}/evaluation` — Job evaluation metrics.

---

## 5. Environment Configuration

In `.env` located at `DubSync-Backend/.env`:
```bash
# MongoDB Atlas Connection URI
MONGODB_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=dubsync
```
When configured with a live cluster, the backend connects automatically upon startup during FastAPI lifespan. If the URI is invalid, unreachable, or DNS fails, the system logs a descriptive diagnostic and operates in safe offline fallback mode.
