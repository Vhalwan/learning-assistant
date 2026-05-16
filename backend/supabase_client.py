"""Supabase client factory and session helpers."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def get_supabase_url() -> str:
    url = (os.getenv("SUPABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("SUPABASE_URL is not set. Add it to your .env file.")
    return url


def get_supabase_anon_key() -> str:
    key = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
    if not key:
        raise RuntimeError("SUPABASE_ANON_KEY is not set. Add it to your .env file.")
    return key


@lru_cache(maxsize=1)
def get_supabase_client():
    from supabase import create_client

    return create_client(get_supabase_url(), get_supabase_anon_key())


def apply_session(access_token: str, refresh_token: str) -> None:
    client = get_supabase_client()
    client.auth.set_session(access_token, refresh_token)


def session_to_dict(session: Any) -> Dict[str, str]:
    return {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
    }


def user_to_dict(user: Any) -> Dict[str, str]:
    return {
        "id": str(user.id),
        "email": str(user.email or ""),
    }
