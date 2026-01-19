# backend/study_srs.py
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

DEFAULT_PROGRESS_PATH = "data/processed/study_progress.json"
INTERVALS = [1, 3, 7, 14, 30]

class SRSManager:
    def __init__(self, path: str = DEFAULT_PROGRESS_PATH):
        self.path = path
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError, OSError) as e:
                # Log error but return empty dict to allow recovery
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load SRS data from {self.path}: {e}")
                return {}
        return {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            # Use atomic write to prevent corruption
            tmp_path = self.path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            # Atomic replace
            if os.path.exists(self.path):
                os.replace(tmp_path, self.path)
            else:
                os.rename(tmp_path, self.path)
        except (IOError, OSError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to save SRS data to {self.path}: {e}")
            # Try to clean up temp file
            try:
                if os.path.exists(self.path + ".tmp"):
                    os.remove(self.path + ".tmp")
            except Exception:
                pass
            raise

    def list_due_cards(self, as_of: Optional[datetime] = None) -> List[str]:
        """Return list of card_ids that are due as of `as_of` (UTC)."""
        return self.get_due_cards(as_of=as_of)

    def get_due_cards(self, as_of: datetime = None):
        as_of = as_of or datetime.utcnow()
        due = []
        for card_id, meta in self._data.items():
            try:
                next_due = datetime.fromisoformat(meta["next_due"])
            except Exception:
                # bad format to treat as due
                due.append(card_id)
                continue
            if next_due <= as_of:
                due.append(card_id)
        return due

    def ensure_card(self, card_id: str):
        if card_id not in self._data:
            self._data[card_id] = {
                "interval_index": 0,
                "next_due": datetime.utcnow().isoformat(),
                "review_count": 0,
            }
            self._save()

    def reset_card(self, card_id: str):
        """Reset a card's scheduling to initial state."""
        self._data[card_id] = {
            "interval_index": 0,
            "next_due": datetime.utcnow().isoformat(),
            "review_count": 0,
        }
        self._save()

    def mark_review(self, card_id: str, correct: bool):
        self.ensure_card(card_id)
        meta = self._data[card_id]
        if correct:
            meta["interval_index"] = min(meta["interval_index"] + 1, len(INTERVALS) - 1)
        else:
            meta["interval_index"] = max(meta.get("interval_index", 0) - 1, 0)
        days = INTERVALS[meta["interval_index"]]
        meta["next_due"] = (datetime.utcnow() + timedelta(days=days)).isoformat()
        meta["review_count"] = meta.get("review_count", 0) + 1
        self._save()

    def get_card_meta(self, card_id: str):
        return self._data.get(card_id)
