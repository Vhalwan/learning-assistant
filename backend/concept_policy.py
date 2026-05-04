"""
Chunk-anchored helpers for stable confusion tracking.

Core rule: confusion grouping must be based on chunk identity only.
Question wording, quiz ids, and runtime regrouping are never used to decide
which weakness card an MCQ belongs to.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Optional


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _shorten(value: str, limit: int = 120) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."


def slugify_concept_id(label: str, fallback: str = "concept") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _clean_text(label).lower()).strip("_")
    return slug or fallback


def _slugify_namespace(value: Any) -> str:
    namespace = re.sub(r"[^a-z0-9]+", "_", _clean_text(value).lower()).strip("_")
    return namespace


def derive_source_chunk_id(source_chunk: str) -> str:
    text = _clean_text(source_chunk)
    if not text:
        return ""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"chunk_{digest}"


def build_chunk_namespace(*, stem: str = "", doc_id: str = "") -> str:
    return _slugify_namespace(doc_id) or _slugify_namespace(stem)


def canonicalize_chunk_id(chunk_id: Any, *, stem: str = "", doc_id: str = "") -> str:
    clean = _clean_text(chunk_id).lower()
    if not clean:
        return ""
    if ":" in clean:
        return clean
    namespace = build_chunk_namespace(stem=stem, doc_id=doc_id)
    if namespace:
        return f"{namespace}:{clean}"
    return clean


def build_chunk_card_id(chunk_id: str) -> str:
    chunk = _clean_text(chunk_id).lower()
    if not chunk:
        return ""
    return f"chunk:{chunk}"


def _extract_source_chunk(question_item: Optional[Dict[str, Any]], existing: Optional[Dict[str, Any]]) -> str:
    detailed = question_item.get("detailed_explanation") if isinstance(question_item, dict) else {}
    if not isinstance(detailed, dict):
        detailed = {}
    candidates = [
        question_item.get("source_chunk") if isinstance(question_item, dict) else "",
        question_item.get("chunk_text") if isinstance(question_item, dict) else "",
        question_item.get("chunk") if isinstance(question_item, dict) else "",
        detailed.get("source_chunk"),
        existing.get("source_chunk") if isinstance(existing, dict) else "",
        existing.get("source_chunk_preview") if isinstance(existing, dict) else "",
    ]
    for candidate in candidates:
        clean = _clean_text(candidate)
        if clean:
            return clean
    return ""


def resolve_concept_bucket(
    *,
    qid: str = "",
    stem: str = "",
    question: str = "",
    question_item: Optional[Dict[str, Any]] = None,
    existing: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Resolve a stable concept bucket for a quiz result.

    Grouping priority:
      1. Explicit `chunk_id` / `source_chunk_id`
      2. Deterministic chunk fingerprint derived from the source chunk text

    There is intentionally no fallback to quiz ids or question wording.
    """
    question_item = question_item or {}
    existing = existing or {}

    concept_label = _clean_text(
        question_item.get("concept_label")
        or question_item.get("concept")
        or existing.get("concept_label")
        or existing.get("concept")
    )
    concept_id = _clean_text(question_item.get("concept_id") or existing.get("concept_id")).lower()

    source_chunk = _extract_source_chunk(question_item, existing)
    source_chunk_id = canonicalize_chunk_id(
        question_item.get("source_chunk_id")
        or question_item.get("chunk_id")
        or existing.get("source_chunk_id")
        or existing.get("chunk_id"),
        stem=_clean_text(question_item.get("stem") or existing.get("stem") or stem),
        doc_id=_clean_text(question_item.get("doc_id") or existing.get("doc_id")),
    )
    if not source_chunk_id and source_chunk:
        source_chunk_id = canonicalize_chunk_id(
            derive_source_chunk_id(source_chunk),
            stem=_clean_text(question_item.get("stem") or existing.get("stem") or stem),
            doc_id=_clean_text(question_item.get("doc_id") or existing.get("doc_id")),
        )

    if not concept_label and source_chunk:
        concept_label = _shorten(source_chunk, limit=90)
    if not concept_id and concept_label:
        concept_id = slugify_concept_id(concept_label, fallback="concept")
    if not concept_id and source_chunk_id:
        concept_id = source_chunk_id

    bucket_key = build_chunk_card_id(source_chunk_id)

    return {
        "concept_id": concept_id,
        "concept_label": concept_label,
        "source_chunk": source_chunk,
        "source_chunk_id": source_chunk_id,
        "source_chunk_preview": _shorten(source_chunk, limit=180) if source_chunk else "",
        "concept_bucket_key": bucket_key,
    }
