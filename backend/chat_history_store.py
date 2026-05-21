"""Persist active lecture chat threads in Supabase (filtered by auth.uid() via RLS)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.persistence import use_remote_store
from backend.supabase_client import get_supabase_client
from backend.user_context import get_user_id, require_user_id


def _table():
    return get_supabase_client().table("chat_history")


def save_chat_history(stem: str, messages: List[Dict[str, Any]], doc_id: str = "") -> None:
    """Upsert the active chat thread for user_id + doc_id + stem."""
    doc_id_clean = (doc_id or "").strip()
    user_id = get_user_id() or ""
    print(
        f"[chat_history] saving user_id={user_id} stem={stem} "
        f"doc_id={doc_id_clean} messages={len(messages)}"
    )
    if not use_remote_store():
        return
    try:
        row = {
            "user_id": require_user_id(),
            "doc_id": doc_id_clean,
            "stem": stem,
            "messages": messages,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        result = _table().upsert(row, on_conflict="user_id,doc_id,stem").execute()
        print(f"[chat_history] save result: {result}")
    except Exception as e:
        print(f"[chat_history] save error: {e}")


def load_chat_history(stem: str, doc_id: str = "") -> Optional[List[Dict[str, Any]]]:
    """Load saved messages for user_id + doc_id + stem, or None if missing."""
    if not use_remote_store():
        return None
    user_id = get_user_id()
    if not user_id:
        return None
    resp = (
        _table()
        .select("messages")
        .eq("user_id", user_id)
        .eq("stem", stem)
        .eq("doc_id", (doc_id or "").strip())
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None
    messages = rows[0].get("messages")
    return messages if isinstance(messages, list) else None


def delete_chat_history(stem: str, doc_id: str = "") -> None:
    """Remove persisted chat for user_id + doc_id + stem (explicit clear only)."""
    if not use_remote_store():
        return
    user_id = require_user_id()
    (
        _table()
        .delete()
        .eq("user_id", user_id)
        .eq("stem", stem)
        .eq("doc_id", (doc_id or "").strip())
        .execute()
    )
