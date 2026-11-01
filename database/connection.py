"""Centralized MongoDB Atlas connection manager for DubSync."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from pymongo import AsyncMongoClient
from pymongo.errors import PyMongoError

from database.indexes import ensure_indexes

logger = logging.getLogger("dubsync.database")

# Environment resolution
DEFAULT_DATABASE_NAME = "dubsync"


def _load_env_if_missing():
    """Ensure environment variables from .env are loaded into os.environ if not already present."""
    if not os.getenv("MONGODB_URI"):
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.is_file():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    clean_line = line.strip()
                    if clean_line and not clean_line.startswith("#") and "=" in clean_line:
                        k, v = clean_line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception as e:
                logger.warning(f"Could not read .env file for database config: {e}")


class DatabaseUnavailableError(RuntimeError):
    """Raised when an operation requires MongoDB Atlas but the database is disconnected."""
    pass


class DatabaseManager:
    _instance: Optional[DatabaseManager] = None

    def __init__(self):
        _load_env_if_missing()
        raw_name = os.getenv("MONGODB_DATABASE") or os.getenv("MONGODB_DB_NAME") or DEFAULT_DATABASE_NAME
        self._client: Optional[AsyncMongoClient] = None
        self._database_name: str = raw_name.lower()
        self._is_connected: bool = False
        self._last_error: Optional[str] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def get_instance(cls) -> DatabaseManager:
        if cls._instance is None:
            cls._instance = DatabaseManager()
        return cls._instance

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def database_name(self) -> str:
        return self._database_name

    def set_main_loop(self, loop: asyncio.AbstractEventLoop):
        self._main_loop = loop

    def get_main_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._main_loop

    async def connect(self, uri: Optional[str] = None, database_name: Optional[str] = None) -> bool:
        """Initialize the single application-level MongoDB client and verify with ping."""
        _load_env_if_missing()
        connection_uri = uri or os.getenv("MONGODB_URI")
        raw_name = database_name or os.getenv("MONGODB_DATABASE") or os.getenv("MONGODB_DB_NAME") or DEFAULT_DATABASE_NAME
        self._database_name = raw_name.lower()

        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._main_loop = None

        if not connection_uri:
            self._is_connected = False
            self._last_error = "MONGODB_URI environment variable is not configured."
            logger.warning("[MongoDB] MONGODB_URI not found. Running in disconnected database mode.")
            return False

        try:
            # Mask URI when logging
            logger.info(f"[MongoDB] Initializing connection to database '{self._database_name}'...")
            self._client = AsyncMongoClient(
                connection_uri,
                serverSelectionTimeoutMS=4000,
                connectTimeoutMS=4000,
                socketTimeoutMS=5000,
            )
            db = self._client[self._database_name]
            # Perform genuine ping
            await db.command("ping")
            self._is_connected = True
            self._last_error = None
            logger.info(f"[MongoDB] Successfully connected and pinged database '{self._database_name}'.")

            # Ensure all collection indexes
            await ensure_indexes(db)
            return True

        except Exception as e:
            self._is_connected = False
            err_msg = f"{type(e).__name__}: {str(e)}"
            self._last_error = err_msg
            logger.warning(f"[MongoDB] Atlas connection could not be established: {err_msg}")
            return False

    async def disconnect(self):
        """Close the MongoDB client cleanly."""
        if self._client:
            try:
                await self._client.close()
                logger.info("[MongoDB] Connection closed cleanly.")
            except Exception as e:
                logger.warning(f"[MongoDB] Error closing connection: {e}")
            finally:
                self._client = None
                self._is_connected = False

    def get_database(self):
        """Return the active database instance. Raises DatabaseUnavailableError if disconnected."""
        if not self._is_connected or not self._client:
            raise DatabaseUnavailableError(
                f"MongoDB Atlas is currently unavailable. ({self._last_error or 'Not connected'})"
            )
        return self._client[self._database_name]

    def get_client(self) -> Optional[AsyncMongoClient]:
        return self._client

    async def health_check(self) -> Dict[str, Any]:
        """Perform a genuine ping to check Atlas health and measure round-trip latency."""
        if not self._client or not self._is_connected:
            return {
                "status": "disconnected",
                "database": self._database_name,
                "error": self._last_error or "Client not connected",
            }
        try:
            import time
            start = time.perf_counter()
            db = self._client[self._database_name]
            await db.command("ping")
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            build_info = {}
            try:
                build_info = await self._client.admin.command("buildInfo")
            except Exception:
                pass

            return {
                "status": "connected",
                "database": self._database_name,
                "connection": "atlas",
                "server_version": build_info.get("version"),
                "ping_ms": latency_ms,
            }
        except Exception as e:
            self._is_connected = False
            self._last_error = f"{type(e).__name__}: {e}"
            return {
                "status": "disconnected",
                "database": self._database_name,
                "error": self._last_error,
            }


# Singleton helpers
db_manager = DatabaseManager.get_instance()


def get_db():
    return db_manager.get_database()


def run_coro_sync(coro):
    """Run an async database coroutine synchronously, routing through the main event loop if available."""
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    loop = db_manager.get_main_loop()
    if loop and loop.is_running():
        if current_loop == loop:
            # Already running on the main event loop; schedule as task to avoid deadlocking the loop
            return current_loop.create_task(coro)
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result()
    else:
        # Fallback to local asyncio.run if no running main loop
        return asyncio.run(coro)
