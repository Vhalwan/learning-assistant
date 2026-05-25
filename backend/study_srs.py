# backend/study_srs.py
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

from backend.persistence import NAMESPACE_SRS, use_remote_store

DEFAULT_PROGRESS_PATH = "data/processed/study_progress.json"
INTERVALS = [1, 3, 7, 14, 30, 60, 120]


def _default_path() -> Path:
    try:
        from backend.user_context import get_srs_path

        return get_srs_path()
    except Exception:
        return Path(DEFAULT_PROGRESS_PATH)


def _persist_srs_remotely(path: Optional[str] = None) -> bool:
    if not use_remote_store():
        return False
    if path is None:
        return True
    try:
        return Path(path).resolve() == _default_path().resolve()
    except Exception:
        return False


def _load_srs_raw(path: Optional[str] = None) -> Dict[str, Any]:
    if _persist_srs_remotely(path):
        from backend.supabase_store import load_namespace_dict

        raw = load_namespace_dict(NAMESPACE_SRS)
        return raw if isinstance(raw, dict) else {}

    p = Path(path or _default_path())
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except (json.JSONDecodeError, IOError, OSError) as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load SRS data from {p}: {e}")
        return {}


def _save_srs_raw(data: Dict[str, Any], path: Optional[str] = None) -> None:
    if _persist_srs_remotely(path):
        from backend.supabase_store import save_namespace_dict

        save_namespace_dict(NAMESPACE_SRS, data)
        return

    p = Path(path or _default_path())
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = p.with_suffix(p.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if p.exists():
            tmp_path.replace(p)
        else:
            tmp_path.rename(p)
    except (IOError, OSError) as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Failed to save SRS data to {p}: {e}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


class SRSManager:
    def __init__(self, path: str = None):
        if path is None:
            self.path = str(_default_path())
        else:
            self.path = path
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        return _load_srs_raw(self.path)

    def _save(self):
        _save_srs_raw(self._data, self.path)

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
          - hard: pull the card back into active rotation
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
            next_idx = max(current_idx - 2, 0)
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
        meta["last_rating"] = r
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
