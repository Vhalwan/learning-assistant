"""User-scoped JSON persistence in Supabase (filtered by auth.uid() via RLS)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.supabase_client import get_supabase_client
from backend.user_context import get_user_id, require_user_id


def _table():
    return get_supabase_client().table("user_data")


def load_blob(namespace: str, data_key: str) -> Optional[Any]:
    user_id = get_user_id()
    if not user_id:
        return None
    resp = (
        _table()
        .select("payload")
        .eq("user_id", user_id)
        .eq("namespace", namespace)
        .eq("data_key", data_key)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None
    return rows[0].get("payload")


def save_blob(namespace: str, data_key: str, payload: Any) -> None:
    user_id = require_user_id()
    row = {
        "user_id": user_id,
        "namespace": namespace,
        "data_key": data_key,
        "payload": payload,
    }
    _table().upsert(row, on_conflict="user_id,namespace,data_key").execute()


def delete_blob(namespace: str, data_key: str) -> None:
    user_id = require_user_id()
    (
        _table()
        .delete()
        .eq("user_id", user_id)
        .eq("namespace", namespace)
        .eq("data_key", data_key)
        .execute()
    )


def list_keys(namespace: str) -> List[str]:
    user_id = require_user_id()
    resp = (
        _table()
        .select("data_key")
        .eq("user_id", user_id)
        .eq("namespace", namespace)
        .execute()
    )
    return [str(r.get("data_key") or "") for r in (resp.data or []) if r.get("data_key")]


def load_namespace_dict(namespace: str, default_key: str = "default") -> Dict[str, Any]:
    """Load a dict stored under a single blob key (e.g. SRS, confusion)."""
    raw = load_blob(namespace, default_key)
    if isinstance(raw, dict):
        return raw
    return {}


def save_namespace_dict(namespace: str, data: Dict[str, Any], default_key: str = "default") -> None:
    save_blob(namespace, default_key, data)
