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

    def ensure_card(self, card_id: str, meta: Optional[Dict[str, Any]] = None):
        created = False
        if card_id not in self._data:
            self._data[card_id] = {
                "interval_index": 0,
                "interval_days": INTERVALS[0],
                "next_due": datetime.utcnow().isoformat(),
                "last_reviewed": None,
                "review_count": 0,
            }
            created = True
        updated = False
        if meta:
            card_meta = self._data.get(card_id, {})
            for key, value in meta.items():
                if value is None or value == "":
                    continue
                if card_meta.get(key) != value:
                    card_meta[key] = value
                    updated = True
            self._data[card_id] = card_meta
        if created or updated:
            self._save()

    def reset_card(self, card_id: str):
        """Reset a card's scheduling to initial state."""
        existing = self._data.get(card_id, {})
        base = {
            "interval_index": 0,
            "interval_days": INTERVALS[0],
            "next_due": datetime.utcnow().isoformat(),
            "last_reviewed": None,
            "review_count": 0,
        }
        for key, value in existing.items():
            if key not in base:
                base[key] = value
        self._data[card_id] = base
        self._save()

    def mark_review(self, card_id: str, correct: bool):
        """
        Backward-compatible entry point:
          - correct=False behaves like "hard"
          - correct=True behaves like "good"
        """
        rating = "good" if bool(correct) else "hard"
        self.mark_review_with_rating(card_id, rating)

    def mark_review_with_rating(self, card_id: str, rating: str):
        """
        Update schedule with explicit SRS rating:
          - hard: short interval (at least 1 day, no promotion)
          - good: normal promotion (+1 step)
          - easy: larger promotion (+2 steps)
        """
        self.ensure_card(card_id)
        meta = self._data[card_id]
        now = datetime.utcnow()

        current_idx = int(meta.get("interval_index", 0))
        current_idx = max(0, min(current_idx, len(INTERVALS) - 1))
        r = str(rating or "").strip().lower()

        if r == "hard":
            next_idx = max(current_idx, 0)
            interval_days = max(1.0, float(INTERVALS[next_idx]))
        elif r == "easy":
            next_idx = min(current_idx + 2, len(INTERVALS) - 1)
            interval_days = float(INTERVALS[next_idx])
        else:
            # Default and "good"
            next_idx = min(current_idx + 1, len(INTERVALS) - 1)
            interval_days = float(INTERVALS[next_idx])

        meta["interval_index"] = next_idx
        meta["interval_days"] = interval_days
        meta["last_reviewed"] = now.isoformat()
        meta["next_due"] = (now + timedelta(days=interval_days)).isoformat()
        meta["review_count"] = int(meta.get("review_count", 0)) + 1
        self._save()

    def get_card_meta(self, card_id: str):
        return self._data.get(card_id)


    def update_card_meta(self, card_id: str, **meta: Any):
        """Update metadata for a card (question text, stem, etc.)."""
        self.ensure_card(card_id)
        updated = False
        card_meta = self._data.get(card_id, {})
        for key, value in meta.items():
            if value is None or value == "":
                continue
            if card_meta.get(key) != value:
                card_meta[key] = value
                updated = True
        if updated:
            self._data[card_id] = card_meta
            self._save()

    def delete_card(self, card_id: str):
        """Delete a card from SRS data."""
        if card_id in self._data:
            del self._data[card_id]
            self._save()