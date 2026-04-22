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
    # Common qid formats use underscore as separator (e.g. "lecture1_Q1").
    # Some generators may use spaces or hyphens in stems (e.g. "lecture 3_P1_Q1").
    # Be tolerant: split from the right on known suffix separators.
    if "_" in qid:
        return qid.rsplit("_", 1)[0]
    if " " in qid:
        return qid.rsplit(" ", 1)[0]
    if "-" in qid:
        return qid.rsplit("-", 1)[0]
    return qid


def record_quiz_result(
    qid: str,
    question: str,
    is_correct: bool,
    stem: str = "",
    concept: str = "",
    concept_id: str = "",
    concept_label: str = "",
    question_item: Optional[Dict[str, Any]] = None,
    chosen_answer: str = "",
    doc_id: str = "",
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
    entry["store_key"] = key
    resolved_label = concept_label or concept or entry.get("concept_label") or entry.get("concept", "")
    resolved_id = concept_id or entry.get("concept_id", "")
    entry["concept"] = resolved_label
    entry["concept_label"] = resolved_label
    entry["concept_id"] = resolved_id
    entry["item_type"] = "mcq"
    entry["origin"] = "quiz_mcq"
    entry["quiz_question_id"] = qid
    entry["last_is_correct"] = bool(is_correct)
    if chosen_answer:
        entry["last_chosen_answer"] = str(chosen_answer).strip().upper()
    mcq_source = question_item if isinstance(question_item, dict) else {}
    question_text = (question or mcq_source.get("question") or "").strip()
    choices = mcq_source.get("choices", {}) if isinstance(mcq_source.get("choices"), dict) else {}
    answer = (mcq_source.get("answer") or "").strip().upper() if isinstance(mcq_source, dict) else ""
    explanation = (mcq_source.get("explanation") or "").strip() if isinstance(mcq_source, dict) else ""
    if question_text:
        entry["question"] = question_text
    if choices:
        entry["choices"] = {
            "A": str(choices.get("A", "")),
            "B": str(choices.get("B", "")),
            "C": str(choices.get("C", "")),
            "D": str(choices.get("D", "")),
        }
    if answer:
        entry["answer"] = answer
    if explanation:
        entry["explanation"] = explanation
    if doc_id:
        entry["doc_id"] = doc_id
    entry["stem"] = inferred_stem or entry.get("stem", "")
    data[key] = entry
    save_confusion(data, p)

def get_top_confusions(limit: int = 10, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Return top confusions sorted by wrong_count desc (simple ranking).
    """
    p = Path(path or DEFAULT_PATH)
    data = load_confusion(p)
    items: List[Dict[str, Any]] = []
    for k, v in data.items():
        item = dict(v or {})
        if not item.get("store_key"):
            item["store_key"] = k
        items.append(item)
    # sort: highest wrong_count first, tie-break by recent last_wrong
    def sort_key(it):
        wc = int(it.get("wrong_count", 0))
        last_wrong = it.get("last_wrong") or ""
        return (-wc, last_wrong)
    items.sort(key=sort_key)
    return items[:limit]


def delete_confusion_entries(keys: List[str], path: Optional[Path] = None) -> int:
    """
    Delete confusion entries by their store keys. Returns number removed.
    """
    p = Path(path or DEFAULT_PATH)
    data = load_confusion(p)
    if not data:
        return 0
    removed = 0
    for k in keys or []:
        if k in data:
            del data[k]
            removed += 1
    if removed:
        save_confusion(data, p)
    return removed
