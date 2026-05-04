"""
Quiz storage utilities for persisting and loading quiz items.
Quiz items are saved per document and can be loaded for SRS review.
"""
import json
import os
from typing import Dict, List, Optional
from pathlib import Path

from backend.concept_policy import resolve_concept_bucket

DEFAULT_QUIZ_DIR = "data/processed"


def get_quiz_path(stem: str, quiz_dir: str = DEFAULT_QUIZ_DIR) -> str:
    """Get the path for a quiz file given a document stem."""
    return os.path.join(quiz_dir, f"{stem}_quiz_items.json")


def _normalize_quiz_item(item: Dict) -> Dict:
    normalized = dict(item or {})
    chunk_meta = resolve_concept_bucket(
        question_item=normalized,
        existing=normalized,
    )
    chunk_id = (chunk_meta.get("source_chunk_id") or "").strip()
    if chunk_id and not str(normalized.get("chunk_id") or "").strip():
        normalized["chunk_id"] = chunk_id
    if chunk_id and not str(normalized.get("source_chunk_id") or "").strip():
        normalized["source_chunk_id"] = chunk_id
    source_chunk = (chunk_meta.get("source_chunk") or "").strip()
    if source_chunk and not str(normalized.get("chunk_text") or "").strip():
        normalized["chunk_text"] = source_chunk
    if source_chunk and not str(normalized.get("source_chunk") or "").strip():
        normalized["source_chunk"] = source_chunk
    if not str(normalized.get("mcq_id") or "").strip() and str(normalized.get("id") or "").strip():
        normalized["mcq_id"] = normalized.get("id")
    return normalized


def save_quiz_items(stem: str, quiz_items: List[Dict], quiz_dir: str = DEFAULT_QUIZ_DIR) -> str:
    """
    Save quiz items to disk using atomic write to prevent corruption.
    
    Args:
        stem: Document stem (e.g., "lecture2")
        quiz_items: List of quiz item dicts with id, question, choices, answer, explanation
        quiz_dir: Directory to save quiz files
    
    Returns:
        Path to saved file
    """
    quiz_path = get_quiz_path(stem, quiz_dir)
    os.makedirs(os.path.dirname(quiz_path) or ".", exist_ok=True)
    
    # Store in a format that's easy to load
    data = {
        "stem": stem,
        "quiz_items": quiz_items,
        "count": len(quiz_items)
    }
    
    # Use atomic write to prevent corruption
    tmp_path = quiz_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Atomic replace
        if os.path.exists(quiz_path):
            os.replace(tmp_path, quiz_path)
        else:
            os.rename(tmp_path, quiz_path)
    except (IOError, OSError) as e:
        # Clean up temp file on error
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise
    
    return quiz_path


def load_quiz_items(stem: str, quiz_dir: str = DEFAULT_QUIZ_DIR) -> Optional[List[Dict]]:
    """
    Load quiz items from disk.
    
    Args:
        stem: Document stem (e.g., "lecture2")
        quiz_dir: Directory containing quiz files
    
    Returns:
        List of quiz items or None if not found
    """
    # Try new format first
    quiz_path = get_quiz_path(stem, quiz_dir)
    if os.path.exists(quiz_path):
        try:
            with open(quiz_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                items = data.get("quiz_items", [])
                if items:
                    return [_normalize_quiz_item(item) for item in items if isinstance(item, dict)]
        except Exception:
            pass
    
    # Try old format (legacy: stem_quiz.json)
    old_quiz_path = os.path.join(quiz_dir, f"{stem}_quiz.json")
    if os.path.exists(old_quiz_path):
        try:
            with open(old_quiz_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Old format has "questions" array
                questions = data.get("questions", [])
                if questions:
                    # Convert old format to new format
                    converted = []
                    for q in questions:
                        # Old format: {id, question, answer, difficulty}
                        # New format: {id, question, choices, answer, explanation}
                        converted.append({
                            "id": q.get("id"),
                            "question": q.get("question", ""),
                            "choices": {},  # Old format doesn't have choices
                            "answer": q.get("answer", ""),
                            "explanation": ""
                        })
                    return [_normalize_quiz_item(item) for item in converted]
        except Exception:
            pass
    
    return None


def load_quiz_item_by_id(card_id: str, quiz_dir: str = DEFAULT_QUIZ_DIR) -> Optional[Dict]:
    """
    Load a specific quiz item by its card ID.
    Optimized to search files directly instead of loading all items.
    
    Args:
        card_id: Card ID (e.g., "lecture2_q1" or "lecture1_OS_FUNCTIONS_001")
        quiz_dir: Directory containing quiz files
    
    Returns:
        Quiz item dict or None if not found
    """
    if not card_id or "_" not in card_id:
        return None
    
    quiz_dir_path = Path(quiz_dir)
    if not quiz_dir_path.exists():
        return None
    
    # Try new format files first (most common)
    for quiz_file in quiz_dir_path.glob("*_quiz_items.json"):
        try:
            with open(quiz_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                quiz_items = data.get("quiz_items", [])
                for item in quiz_items:
                    if item.get("id") == card_id:
                        return _normalize_quiz_item(item)
        except (json.JSONDecodeError, IOError, OSError):
            continue
    
    # Fallback to old format files
    for quiz_file in quiz_dir_path.glob("*_quiz.json"):
        try:
            with open(quiz_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                questions = data.get("questions", [])
                for q in questions:
                    if q.get("id") == card_id:
                        question_text = q.get("question", "")
                        # Skip placeholder questions
                        if "placeholder" in question_text.lower():
                            return None
                        # Convert old format to new format
                        return _normalize_quiz_item({
                            "id": q.get("id"),
                            "question": question_text,
                            "choices": {},
                            "answer": q.get("answer", ""),
                            "explanation": ""
                        })
        except (json.JSONDecodeError, IOError, OSError):
            continue
    
    return None


def load_all_quiz_items(quiz_dir: str = DEFAULT_QUIZ_DIR) -> Dict[str, Dict]:
    """
    Load all quiz items from all quiz files (both new and old formats).
    Optimized to cache glob results and avoid redundant file operations.
    
    Returns:
        Dict mapping card_id -> quiz_item
    """
    all_items = {}
    quiz_dir_path = Path(quiz_dir)
    
    if not quiz_dir_path.exists():
        return all_items
    
    # Cache glob results to avoid multiple filesystem scans
    new_format_files = list(quiz_dir_path.glob("*_quiz_items.json"))
    old_format_files = list(quiz_dir_path.glob("*_quiz.json"))
    new_format_stems = {f.stem.replace("_quiz_items", "") for f in new_format_files}
    
    # Load new format files first (they take priority)
    for quiz_file in new_format_files:
        try:
            with open(quiz_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                quiz_items = data.get("quiz_items", [])
                for item in quiz_items:
                    item_id = item.get("id")
                    if item_id:
                        all_items[item_id] = _normalize_quiz_item(item)
        except (json.JSONDecodeError, IOError, OSError) as e:
            # Log specific errors but continue processing other files
            continue
    
    # Load old format files, skipping those that have new format equivalents
    for quiz_file in old_format_files:
        stem = quiz_file.stem.replace("_quiz", "")
        if stem in new_format_stems:
            continue  # Already loaded from new format
        
        try:
            with open(quiz_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                questions = data.get("questions", [])
                for q in questions:
                    item_id = q.get("id")
                    question_text = q.get("question", "")
                    # Skip placeholder questions
                    if "placeholder" in question_text.lower():
                        continue
                    if item_id and item_id not in all_items:  # Don't overwrite new format items
                        # Convert old format to new format
                        all_items[item_id] = _normalize_quiz_item({
                            "id": q.get("id"),
                            "question": question_text,
                            "choices": {},
                            "answer": q.get("answer", ""),
                            "explanation": ""
                        })
        except (json.JSONDecodeError, IOError, OSError):
            continue
    
    return all_items
