"""
Concept storage utilities for persisting and loading canonical concept lists.
"""
import json
import os
from typing import Dict, List, Optional

from backend.user_context import get_concept_dir, get_user_id

DEFAULT_CONCEPT_DIR = "data/processed"


def _default_concept_dir() -> str:
    if get_user_id():
        return get_concept_dir()
    return DEFAULT_CONCEPT_DIR


def get_concept_path(stem: str, concept_dir: Optional[str] = None) -> str:
    """Get the path for a concept file given a document stem."""
    if concept_dir is None:
        concept_dir = _default_concept_dir()
    return os.path.join(concept_dir, f"{stem}_concepts.json")


def save_concepts(stem: str, concepts: List[Dict], concept_dir: Optional[str] = None, doc_id: str = "") -> str:
    """
    Save concepts to disk using atomic write to prevent corruption.
    """
    if concept_dir is None:
        concept_dir = _default_concept_dir()
    concept_path = get_concept_path(stem, concept_dir)
    os.makedirs(os.path.dirname(concept_path) or ".", exist_ok=True)
    data = {
        "stem": stem,
        "concepts": concepts,
        "count": len(concepts),
        "doc_id": doc_id or "",
    }
    tmp_path = concept_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if os.path.exists(concept_path):
            os.replace(tmp_path, concept_path)
        else:
            os.rename(tmp_path, concept_path)
    except (IOError, OSError):
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise
    return concept_path


def load_concepts(stem: str, concept_dir: Optional[str] = None) -> Optional[List[Dict]]:
    """
    Load concepts from disk. Returns list or None if not found.
    """
    if concept_dir is None:
        concept_dir = _default_concept_dir()
    concept_path = get_concept_path(stem, concept_dir)
    if not os.path.exists(concept_path):
        return None
    try:
        with open(concept_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            concepts = data.get("concepts", [])
            if isinstance(concepts, list) and concepts:
                return concepts
    except Exception:
        pass
    return None


def load_concepts_with_meta(stem: str, concept_dir: Optional[str] = None) -> Optional[Dict]:
    """
    Load concepts plus metadata. Returns dict with keys: concepts, doc_id, stem, count.
    """
    if concept_dir is None:
        concept_dir = _default_concept_dir()
    concept_path = get_concept_path(stem, concept_dir)
    if not os.path.exists(concept_path):
        return None
    try:
        with open(concept_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and data.get("concepts") is not None:
                return data
    except Exception:
        pass
    return None
