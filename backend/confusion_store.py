# backend/confusion_store.py
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

DEFAULT_PATH = Path("data/processed/confusion.json")

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ensure_parent(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

def load_confusion(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or DEFAULT_PATH)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Corrupt or unreadable file -> return empty to avoid crashing
        return {}

def save_confusion(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    p = Path(path or DEFAULT_PATH)
    _ensure_parent(p)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)

def _infer_stem_from_qid(qid: str) -> str:
    if not qid:
        return ""
    if "_" in qid:
        return qid.split("_", 1)[0]
    return ""


def record_quiz_result(
    qid: str,
    question: str,
    is_correct: bool,
    stem: str = "",
    concept: str = "",
    path: Optional[Path] = None,
) -> None:
    """
    Record that the user answered a quiz item. If incorrect, increment wrong_count and update timestamps.
    If correct, increment correct_count (optional).
    """
    p = Path(path or DEFAULT_PATH)
    data = load_confusion(p)
    key = qid or (question[:120] if question else "unknown")
    inferred_stem = stem or _infer_stem_from_qid(qid)
    entry = data.get(key, {
        "qid": qid,
        "question": question,
        "stem": inferred_stem,
        "wrong_count": 0,
        "correct_count": 0,
        "first_wrong": None,
        "last_wrong": None,
        "first_correct": None,
        "last_correct": None,
    })
    t = _now_iso()
    if is_correct:
        entry["correct_count"] = entry.get("correct_count", 0) + 1
        entry["last_correct"] = t
        if not entry.get("first_correct"):
            entry["first_correct"] = t
    else:
        entry["wrong_count"] = entry.get("wrong_count", 0) + 1
        entry["last_wrong"] = t
        if not entry.get("first_wrong"):
            entry["first_wrong"] = t
    # store latest question text (in case id is same but text updated)
    entry["question"] = question or entry.get("question", "")
    entry["concept"] = concept or entry.get("concept", "")
    entry["stem"] = inferred_stem or entry.get("stem", "")
    data[key] = entry
    save_confusion(data, p)

def get_top_confusions(limit: int = 10, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Return top confusions sorted by wrong_count desc (simple ranking).
    """
    p = Path(path or DEFAULT_PATH)
    data = load_confusion(p)
    items = list(data.values())
    # sort: highest wrong_count first, tie-break by recent last_wrong
    def sort_key(it):
        wc = int(it.get("wrong_count", 0))
        last_wrong = it.get("last_wrong") or ""
        return (-wc, last_wrong)
    items.sort(key=sort_key)
    return items[:limit]
