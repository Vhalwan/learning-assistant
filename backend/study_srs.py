# backend/study_srs.py
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any

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
            except Exception:
                return {}
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get_due_cards(self, as_of: datetime = None):
        as_of = as_of or datetime.utcnow()
        due = []
        for card_id, meta in self._data.items():
            next_due = datetime.fromisoformat(meta["next_due"])
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

    def mark_review(self, card_id: str, correct: bool):
        self.ensure_card(card_id)
        meta = self._data[card_id]
        if correct:
            meta["interval_index"] = min(meta["interval_index"] + 1, len(INTERVALS) - 1)
        else:
            meta["interval_index"] = max(meta["interval_index"] - 1, 0)
        days = INTERVALS[meta["interval_index"]]
        meta["next_due"] = (datetime.utcnow() + timedelta(days=days)).isoformat()
        meta["review_count"] = meta.get("review_count", 0) + 1
        self._save()

    def get_card_meta(self, card_id: str):
        return self._data.get(card_id)
