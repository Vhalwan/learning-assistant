"""Named saved conversations per lecture (user_data blob, RLS-scoped)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.persistence import NAMESPACE_SAVED_CHATS, use_remote_store
from backend.supabase_store import load_blob, save_blob
from backend.user_context import require_user_id


def _data_key(stem: str, doc_id: str = "") -> str:
    doc_id_clean = (doc_id or "").strip()
    return f"{doc_id_clean}|{stem}" if doc_id_clean else stem


def load_saved_chats(stem: str, doc_id: str = "") -> Optional[List[Dict[str, Any]]]:
    if not use_remote_store():
        return None
    require_user_id()
    raw = load_blob(NAMESPACE_SAVED_CHATS, _data_key(stem, doc_id))
    if isinstance(raw, list):
        return raw
    return None


def save_saved_chats(stem: str, items: List[Dict[str, Any]], doc_id: str = "") -> None:
    if not use_remote_store():
        return
    require_user_id()
    save_blob(NAMESPACE_SAVED_CHATS, _data_key(stem, doc_id), items)
