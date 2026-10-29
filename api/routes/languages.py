"""Language capability and routing endpoints for DubSync API."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter

from dubsync.languages import (
    LANGUAGES,
    evaluate_language_pair,
    get_capabilities_payload,
    list_languages,
)

router = APIRouter(prefix="/api/v1/languages", tags=["languages"])


@router.get("", response_model=Dict[str, Any])
async def get_languages():
    """Retrieve list of recognized languages and basic capability status."""
    return get_capabilities_payload()


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_language_capabilities():
    """Retrieve comprehensive language capability registry and pair evaluation."""
    return get_capabilities_payload()


@router.get("/pairs", response_model=List[Dict[str, Any]])
async def get_supported_pairs():
    """Retrieve explicitly usable language pairs in the active pipeline."""
    payload = get_capabilities_payload()
    return payload["supported_pairs"]
