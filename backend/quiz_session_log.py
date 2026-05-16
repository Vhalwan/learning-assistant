"""
Append-only quiz session snapshots per lecture (stem) for progress UX:
last N completed quiz batches with accuracy, simple trend vs prior sessions.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.user_context import get_processed_dir, get_user_id

DEFAULT_DIR = Path("data/processed")
MAX_SESSIONS = 30


def _default_dir() -> Path:
    if get_user_id():
        return get_processed_dir()
    return DEFAULT_DIR


def _path_for_stem(stem: str) -> Path:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (stem or "").strip()) or "lecture"
    return _default_dir() / f"{safe}_quiz_sessions.json"


def _load_raw(stem: str) -> Dict[str, Any]:
    p = _path_for_stem(stem)
    if not p.exists():
        return {"stem": stem, "sessions": []}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"stem": stem, "sessions": []}
        sess = data.get("sessions")
        if not isinstance(sess, list):
            data["sessions"] = []
        return data
    except Exception:
        return {"stem": stem, "sessions": []}


def append_quiz_session(
    stem: str,
    doc_id: str,
    correct: int,
    total: int,
    ts_iso: str,
) -> None:
    if not (stem or "").strip() or total <= 0:
        return
    p = _path_for_stem(stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _load_raw(stem)
    acc = (float(correct) / float(total)) * 100.0 if total else 0.0
    row = {
        "ts": (ts_iso or "").strip(),
        "doc_id": (doc_id or "").strip(),
        "correct": int(correct),
        "total": int(total),
        "accuracy_pct": round(acc, 1),
    }
    sessions: List[Dict[str, Any]] = list(data.get("sessions") or [])
    sessions.append(row)
    if len(sessions) > MAX_SESSIONS:
        sessions = sessions[-MAX_SESSIONS:]
    data["stem"] = stem
    data["sessions"] = sessions
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def recent_sessions(stem: str, limit: int = 3) -> List[Dict[str, Any]]:
    data = _load_raw(stem)
    sess = list(data.get("sessions") or [])
    if limit <= 0:
        return []
    return sess[-limit:]


def trend_vs_prior(stem: str) -> Tuple[str, Optional[float]]:
    """
    Compare last session accuracy to mean of all prior sessions (same stem).
    Returns (label, delta_pct) where delta is last minus prior mean, or None if unknown.
    """
    data = _load_raw(stem)
    sess = [s for s in (data.get("sessions") or []) if isinstance(s, dict)]
    if len(sess) < 2:
        return ("Not enough quiz rounds yet — finish two full quizzes to see a trend.", None)
    last = sess[-1]
    prior = sess[:-1]
    try:
        last_acc = float(last.get("accuracy_pct", 0))
    except Exception:
        last_acc = 0.0
    prevs = []
    for s in prior:
        try:
            prevs.append(float(s.get("accuracy_pct", 0)))
        except Exception:
            continue
    if not prevs:
        return ("Not enough data for a trend yet.", None)
    mean_prior = sum(prevs) / len(prevs)
    delta = last_acc - mean_prior
    if delta >= 8:
        label = "Trend: improving — your last quiz beat your earlier average."
    elif delta <= -8:
        label = "Trend: down from your earlier average — worth a quick review pass."
    else:
        label = "Trend: steady — performance is in line with your recent quizzes."
    return (label, round(delta, 1))
