from __future__ import annotations

import os
import sys
from pathlib import Path
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.jobs import router as jobs_router
from api.routes.files import router as files_router
from dubsync.readiness import check_system_readiness
from dubsync.stt.whisper import ensure_whisper_model_cached


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preflight check on startup: verify models and runtimes
    try:
        # Ensure Whisper model is cached before jobs run
        ensure_whisper_model_cached("small")
        ready, details, error = check_system_readiness()
        if not ready:
            print(f"[PREFLIGHT WARNING] Model readiness check failed: {error}", file=sys.stderr)
        else:
            print("[PREFLIGHT SUCCESS] All neural pipeline models & environments are verified ready.", file=sys.stderr)
    except Exception as exc:
        print(f"[PREFLIGHT ERROR] Startup readiness verification encountered error: {exc}", file=sys.stderr)
    yield


app = FastAPI(
    title="DubSync AI Engine API",
    description="REST & SSE API layer for DubSync local neural video dubbing pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for Next.js frontend
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(jobs_router)
app.include_router(files_router)


@app.get("/health")
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "online",
        "service": "DubSync AI Engine",
        "version": "1.0.0",
    }


@app.get("/api/v1/system/status")
async def system_status():
    is_ready, details, error = check_system_readiness()
    cuda_info = details.get("cuda", {})

    return {
        "status": "ready" if is_ready else "degraded",
        "ready": is_ready,
        "cuda_available": cuda_info.get("ready", False),
        "device_name": cuda_info.get("device", "CPU"),
        "total_vram_gb": cuda_info.get("total_vram_gb"),
        "free_vram_gb": cuda_info.get("free_vram_gb"),
        "pipeline_version": "v1.0",
        "components": details,
        "error": error,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
