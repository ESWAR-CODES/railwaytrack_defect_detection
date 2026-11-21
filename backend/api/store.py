"""Thread-Safe In-Memory Inspection History Store.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 4: FastAPI Inference & Inspection Backend.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from backend.api.schemas import InspectionHistoryItem


class InspectionStore:
    """Thread-safe storage manager for recent railway defect inspection summaries."""

    def __init__(self, max_items: int = 500) -> None:
        self.max_items = max_items
        self._items: List[InspectionHistoryItem] = []
        self._lock = threading.Lock()

    def add(self, item: InspectionHistoryItem) -> None:
        """Add a new inspection record to the front of history, evicting oldest if capacity reached."""
        with self._lock:
            self._items.insert(0, item)
            if len(self._items) > self.max_items:
                self._items = self._items[: self.max_items]

    def list_recent(self, limit: int = 50) -> List[InspectionHistoryItem]:
        """Return the most recent inspection records (newest first)."""
        with self._lock:
            safe_limit = max(1, min(limit, self.max_items))
            return list(self._items[:safe_limit])

    def get(self, inspection_id: str) -> Optional[InspectionHistoryItem]:
        """Look up an inspection record by its unique identifier."""
        with self._lock:
            for item in self._items:
                if item.inspection_id == inspection_id:
                    return item
            return None

    def count(self) -> int:
        """Return the total number of records currently stored."""
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        """Clear all stored records (useful for test isolation)."""
        with self._lock:
            self._items.clear()


# Global singleton instance
_store_instance: Optional[InspectionStore] = None
_store_lock = threading.Lock()


def get_inspection_store() -> InspectionStore:
    """Access the global inspection history store singleton."""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = InspectionStore()
    return _store_instance
