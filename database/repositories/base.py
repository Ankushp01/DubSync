"""Base repository for DubSync MongoDB collections."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from database.connection import db_manager, DatabaseUnavailableError
from database.models import serialize_doc


class BaseRepository:
    def __init__(self, collection_name: str, db=None):
        self.collection_name = collection_name
        self._db_override = db

    def get_collection(self):
        """Get the active collection. Uses db_override if set (useful for testing), or default db."""
        if self._db_override is not None:
            return self._db_override[self.collection_name]
        return db_manager.get_database()[self.collection_name]

    def serialize(self, doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return serialize_doc(doc)

    def serialize_list(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [serialize_doc(d) for d in docs]
