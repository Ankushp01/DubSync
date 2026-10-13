"""Explicit preflight readiness checks for DubSync AI Pipeline.

Validates that all required models, virtual environments, and GPU runtimes are
available and healthy before accepting or launching a dubbing job.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from dubsync.stt.whisper import ensure_whisper_model_cached

logger = logging.getLogger("dubsync.readiness")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Python venvs
MAIN_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
INDICTRANS_PYTHON = PROJECT_ROOT / ".indictrans-venv" / "Scripts" / "python.exe"
TTS_PYTHON = PROJECT_ROOT / ".tts-venv" / "Scripts" / "python.exe"
WAV2LIP_PYTHON = PROJECT_ROOT / ".wav2lip-venv" / "Scripts" / "python.exe"

# Model paths
WAV2LIP_CHECKPOINT = PROJECT_ROOT / "Wav2Lip-HD" / "checkpoints" / "wav2lip.pth"
S3FD_CHECKPOINT = PROJECT_ROOT / "Wav2Lip-HD" / "face_detection" / "detection" / "sfd" / "s3fd.pth"


def get_hf_hub_cache_dir() -> Path:
    """Return local HuggingFace cache directory."""
    return Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))


def check_cuda_readiness() -> Dict[str, Any]:
    """Verify CUDA availability and VRAM."""
    available = torch.cuda.is_available()
    if not available:
        return {
            "status": "error",
            "ready": False,
            "device": "cpu",
            "error": "CUDA is not available. DubSync requires an NVIDIA GPU for real-time neural pipeline execution.",
        }

    device_name = torch.cuda.get_device_name(0)
    free_gb, total_gb = None, None
    try:
        free_b, total_b = torch.cuda.mem_get_info(0)
        free_gb = round(free_b / (1024**3), 2)
        total_gb = round(total_b / (1024**3), 2)
    except Exception as exc:
        logger.warning(f"Could not read CUDA memory info: {exc}")

    return {
        "status": "ready",
        "ready": True,
        "device": device_name,
        "total_vram_gb": total_gb,
        "free_vram_gb": free_gb,
    }


def check_whisper_readiness(model_name: str = "small") -> Dict[str, Any]:
    """Verify Faster-Whisper model availability in local cache."""
    cache_dir = get_hf_hub_cache_dir()
    repo_folder = f"models--Systran--faster-whisper-{model_name}"
    snapshots_dir = cache_dir / repo_folder / "snapshots"

    if snapshots_dir.is_dir():
        for s in sorted(list(snapshots_dir.iterdir()), key=lambda p: p.stat().st_mtime, reverse=True):
            if s.is_dir() and (s / "model.bin").exists() and (s / "config.json").exists():
                return {
                    "status": "ready",
                    "ready": True,
                    "model": f"faster-whisper-{model_name}",
                    "cached_path": str(s),
                    "local_files_only": True,
                }

    return {
        "status": "not_cached",
        "ready": False,
        "model": f"faster-whisper-{model_name}",
        "error": f"Faster-Whisper model '{model_name}' is not cached locally. Run startup preflight to download.",
    }


def check_indictrans_readiness() -> Dict[str, Any]:
    """Verify IndicTrans2 virtual environment and cached model weights."""
    if not INDICTRANS_PYTHON.is_file():
        return {
            "status": "error",
            "ready": False,
            "error": f"IndicTrans2 virtual environment not found at: {INDICTRANS_PYTHON}",
        }

    worker_script = PROJECT_ROOT / "dubsync" / "translation" / "indictrans_worker.py"
    if not worker_script.is_file():
        return {
            "status": "error",
            "ready": False,
            "error": f"IndicTrans2 worker script not found at: {worker_script}",
        }

    cache_dir = get_hf_hub_cache_dir()
    repo_folder = "models--ai4bharat--indictrans2-en-indic-dist-200M"
    snapshots_dir = cache_dir / repo_folder / "snapshots"

    cached_snapshot: Optional[Path] = None
    if snapshots_dir.is_dir():
        for s in sorted(list(snapshots_dir.iterdir()), key=lambda p: p.stat().st_mtime, reverse=True):
            if s.is_dir() and (s / "config.json").exists() and (s / "model.safetensors").exists():
                cached_snapshot = s
                break

    if not cached_snapshot:
        return {
            "status": "not_cached",
            "ready": False,
            "model": "ai4bharat/indictrans2-en-indic-dist-200M",
            "error": "IndicTrans2 model weights are not cached locally. Download is required before running jobs.",
        }

    return {
        "status": "ready",
        "ready": True,
        "model": "ai4bharat/indictrans2-en-indic-dist-200M",
        "cached_path": str(cached_snapshot),
        "venv": str(INDICTRANS_PYTHON),
    }


def check_chatterbox_readiness(target_language: str = "hi") -> Dict[str, Any]:
    """Verify Chatterbox TTS virtual environment and model availability."""
    if not TTS_PYTHON.is_file():
        return {
            "status": "error",
            "ready": False,
            "error": f"Chatterbox virtual environment not found at: {TTS_PYTHON}",
        }

    worker_script = PROJECT_ROOT / "dubsync" / "tts" / "chatterbox_worker.py"
    if not worker_script.is_file():
        return {
            "status": "error",
            "ready": False,
            "error": f"Chatterbox worker script not found at: {worker_script}",
        }

    cache_dir = get_hf_hub_cache_dir()
    chatterbox_repo = cache_dir / "models--ResembleAI--chatterbox"
    multilingual_repo = cache_dir / "models--ResembleAI--Chatterbox-Multilingual-hi"

    if not chatterbox_repo.is_dir() and not multilingual_repo.is_dir():
        return {
            "status": "warning",
            "ready": True,
            "note": "Chatterbox model repos not found in standard cache, relying on .tts-venv bundled weights.",
        }

    return {
        "status": "ready",
        "ready": True,
        "venv": str(TTS_PYTHON),
        "language": target_language,
    }


def check_wav2lip_readiness() -> Dict[str, Any]:
    """Verify Wav2Lip-HD virtual environment and pretrained checkpoints."""
    if not WAV2LIP_PYTHON.is_file():
        return {
            "status": "error",
            "ready": False,
            "error": f"Wav2Lip virtual environment not found at: {WAV2LIP_PYTHON}",
        }

    if not WAV2LIP_CHECKPOINT.is_file():
        return {
            "status": "error",
            "ready": False,
            "error": f"Wav2Lip checkpoint not found at: {WAV2LIP_CHECKPOINT}",
        }

    if not S3FD_CHECKPOINT.is_file():
        return {
            "status": "error",
            "ready": False,
            "error": f"S3FD face detection checkpoint not found at: {S3FD_CHECKPOINT}",
        }

    return {
        "status": "ready",
        "ready": True,
        "checkpoint": str(WAV2LIP_CHECKPOINT),
        "face_detector": str(S3FD_CHECKPOINT),
        "venv": str(WAV2LIP_PYTHON),
    }


def check_system_readiness(
    target_language: str = "hi",
    enable_lip_sync: bool = True,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Run full preflight checks across all components required for a dubbing job.

    Returns:
        (is_ready, details_dict, error_message_if_any)
    """
    cuda = check_cuda_readiness()
    whisper = check_whisper_readiness()
    indictrans = check_indictrans_readiness()
    chatterbox = check_chatterbox_readiness(target_language)
    wav2lip = check_wav2lip_readiness() if enable_lip_sync else {"status": "skipped", "ready": True}

    details = {
        "cuda": cuda,
        "whisper": whisper,
        "indictrans": indictrans,
        "chatterbox": chatterbox,
        "wav2lip": wav2lip,
    }

    if not cuda["ready"]:
        return False, details, cuda.get("error", "CUDA runtime check failed.")

    if not whisper["ready"]:
        return False, details, whisper.get("error", "Faster-Whisper model readiness check failed.")

    if not indictrans["ready"]:
        return False, details, indictrans.get("error", "IndicTrans2 model readiness check failed.")

    if not chatterbox["ready"]:
        return False, details, chatterbox.get("error", "Chatterbox TTS readiness check failed.")

    if enable_lip_sync and not wav2lip.get("ready", False):
        return False, details, wav2lip.get("error", "Wav2Lip readiness check failed.")

    return True, details, None
