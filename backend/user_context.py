"""Per-request user scope for data paths and Supabase reads/writes."""
from __future__ import annotations

import contextvars
import os
from pathlib import Path
from typing import Optional

USE_REMOTE_STORAGE = os.getenv("USE_REMOTE_STORAGE", "").lower() in ("1", "true", "yes")

_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("user_id", default=None)


def set_user_id(user_id: Optional[str]) -> None:
    _user_id.set((user_id or "").strip() or None)


def get_user_id() -> Optional[str]:
    return _user_id.get()


def require_user_id() -> str:
    uid = get_user_id()
    if not uid:
        raise RuntimeError("Authentication required")
    return uid


def get_raw_dir() -> Path:
    return Path("data/raw") / require_user_id()


def get_processed_dir() -> Path:
    d = Path("data/processed") / require_user_id()
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_embeddings_path(stem: str) -> Path:
    return get_processed_dir() / f"{stem}_embeddings.json"


def get_index_path(stem: str) -> Path:
    return get_processed_dir() / f"{stem}_embeddings.index"


def get_confusion_path() -> Path:
    return get_processed_dir() / "confusion.json"


def get_srs_path() -> Path:
    return get_processed_dir() / "study_progress.json"


def get_quiz_dir() -> str:
    return str(get_processed_dir())


def get_concept_dir() -> str:
    return str(get_processed_dir())
