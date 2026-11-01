"""API routes for accessing Videos, Transcripts, Translations, and Output records."""
from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.auth.dependencies import get_current_user
from database.connection import db_manager
from database.models import UserModel
from database.repositories.videos import VideosRepository
from database.repositories.transcripts import TranscriptsRepository
from database.repositories.translations import TranslationsRepository
from database.repositories.dubbed_videos import DubbedVideosRepository

router = APIRouter(prefix="/api/v1", tags=["entities"])


def _check_db():
    if not db_manager.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MongoDB Atlas is currently unavailable. ({db_manager.last_error or 'Not connected'})",
        )


@router.get("/videos/{video_id}")
async def get_video(video_id: str, current_user: UserModel = Depends(get_current_user)):
    """Retrieve uploaded source video metadata."""
    _check_db()
    repo = VideosRepository()
    doc = await repo.get_by_id_for_user(video_id, current_user.user_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")
    return doc


@router.get("/transcripts/{transcript_id}")
async def get_transcript(transcript_id: str, current_user: UserModel = Depends(get_current_user)):
    """Retrieve original speech transcription record."""
    _check_db()
    repo = TranscriptsRepository()
    doc = await repo.get_by_id_for_user(transcript_id, current_user.user_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Transcript '{transcript_id}' not found")
    return doc


@router.get("/translations/{translation_id}")
async def get_translation(translation_id: str, current_user: UserModel = Depends(get_current_user)):
    """Retrieve translated transcript version."""
    _check_db()
    repo = TranslationsRepository()
    doc = await repo.get_by_id_for_user(translation_id, current_user.user_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Translation '{translation_id}' not found")
    return doc


@router.get("/outputs/{output_id}")
async def get_output(output_id: str, current_user: UserModel = Depends(get_current_user)):
    """Retrieve rendered dubbed video output record."""
    _check_db()
    repo = DubbedVideosRepository()
    doc = await repo.get_by_id_for_user(output_id, current_user.user_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Output '{output_id}' not found")
    return doc
