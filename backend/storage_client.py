"""Supabase Storage helpers for lecture PDFs and embeddings."""
from __future__ import annotations

from backend.supabase_client import get_supabase_client

LECTURES_BUCKET = "lectures"
EMBEDDINGS_BUCKET = "embeddings"


def upload_pdf(user_id: str, filename: str, file_bytes: bytes) -> None:
    client = get_supabase_client()
    client.storage.from_(LECTURES_BUCKET).upload(
        f"{user_id}/{filename}",
        file_bytes,
        file_options={"content-type": "application/pdf", "upsert": "true"},
    )


def download_pdf(user_id: str, filename: str) -> bytes:
    client = get_supabase_client()
    return client.storage.from_(LECTURES_BUCKET).download(f"{user_id}/{filename}")


def upload_embeddings(user_id: str, stem: str, json_str: str) -> None:
    client = get_supabase_client()
    client.storage.from_(EMBEDDINGS_BUCKET).upload(
        f"{user_id}/{stem}_embeddings.json",
        json_str.encode("utf-8"),
        file_options={"content-type": "application/json", "upsert": "true"},
    )


def download_embeddings(user_id: str, stem: str) -> str | None:
    try:
        client = get_supabase_client()
        data = client.storage.from_(EMBEDDINGS_BUCKET).download(
            f"{user_id}/{stem}_embeddings.json"
        )
        return data.decode("utf-8")
    except Exception:
        return None


def embeddings_exist(user_id: str, stem: str) -> bool:
    try:
        client = get_supabase_client()
        name = f"{stem}_embeddings.json"
        items = client.storage.from_(EMBEDDINGS_BUCKET).list(user_id) or []
        return any((item.get("name") or "") == name for item in items)
    except Exception:
        return False
