# backend/confusion_store.py
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.concept_policy import build_chunk_card_id, resolve_concept_bucket
from backend.quiz_storage import load_quiz_item_by_id
from backend.persistence import NAMESPACE_CONFUSION, use_remote_store
from backend.user_context import get_confusion_path, get_user_id

_FALLBACK_PATH = Path("data/processed/confusion.json")


def _default_path() -> Path:
    if get_user_id():
        return get_confusion_path()
    return _FALLBACK_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalized_stem(value: str) -> str:
    return str(value or "").strip()


def _stem_variants(stem: str) -> List[str]:
    base = _normalized_stem(stem)
    if not base:
        return []
    variants = {
        base,
        base.replace(" ", "_"),
        base.replace("_", " "),
    }
    return [v for v in variants if v]


def _entry_matches_stem(entry: Dict[str, Any], stem: str) -> bool:
    variants = _stem_variants(stem)
    if not variants:
        return True
    entry_stem = _normalized_stem(entry.get("stem"))
    for variant in variants:
        if entry_stem == variant:
            return True
    return False


def _normalize_choices(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in raw.items():
        clean_key = str(key or "").strip().upper()
        if not clean_key:
            continue
        out[clean_key] = str(value or "").strip()
    return out


def _min_non_empty(*values: str) -> str:
    present = [str(v).strip() for v in values if str(v or "").strip()]
    if not present:
        return ""
    return min(present)


def _max_non_empty(*values: str) -> str:
    present = [str(v).strip() for v in values if str(v or "").strip()]
    if not present:
        return ""
    return max(present)


def _card_title(concept_label: str, chunk_preview: str, chunk_id: str) -> str:
    title = _clean_text(concept_label) or _clean_text(chunk_preview) or _clean_text(chunk_id)
    return title


def _linked_chunks(*values: Any) -> List[str]:
    out: List[str] = []
    for value in values:
        if isinstance(value, list):
            candidates = value
        else:
            candidates = [value]
        for candidate in candidates:
            clean = _clean_text(candidate)
            if clean and clean not in out:
                out.append(clean)
    return out


def _coerce_mcq_history(
    raw_history: Any,
    *,
    card_question: str = "",
    last_qid: str = "",
    default_choices: Optional[Dict[str, str]] = None,
    default_answer: str = "",
) -> List[Dict[str, Any]]:
    """
    Normalize mcq_history to deduplicated dict rows with resolvable question text.
    """
    out: List[Dict[str, Any]] = []
    index_by_qid: Dict[str, int] = {}

    def _upsert(
        qid: str,
        question: str,
        entry_choices: Any = None,
        entry_answer: str = "",
    ) -> None:
        if not qid:
            return
        choices = _normalize_choices(entry_choices or {})
        answer = _clean_text(entry_answer).upper()
        if qid in index_by_qid:
            row = out[index_by_qid[qid]]
            incoming_question = _clean_text(question)
            existing_question = _clean_text(row.get("question"))
            if incoming_question and (
                not existing_question or len(incoming_question) > len(existing_question)
            ):
                row["question"] = incoming_question
            if not row.get("choices") and choices:
                row["choices"] = choices
            if not _clean_text(row.get("answer")) and answer:
                row["answer"] = answer
            return
        index_by_qid[qid] = len(out)
        out.append(
            {
                "qid": qid,
                "question": _clean_text(question),
                "choices": choices,
                "answer": answer,
            }
        )

    for entry in (raw_history or []):
        if isinstance(entry, dict):
            qid = _clean_text(entry.get("qid") or entry.get("id") or entry.get("mcq_id"))
            question = _clean_text(entry.get("question"))
            entry_choices = entry.get("choices")
            entry_answer = entry.get("answer") or entry.get("correct_answer")
        else:
            qid = _clean_text(entry)
            question = ""
            entry_choices = {}
            entry_answer = ""

        if not qid:
            continue

        if not question:
            quiz_item = load_quiz_item_by_id(qid) or {}
            question = _clean_text(quiz_item.get("question"))
            if not entry_choices and isinstance(quiz_item.get("choices"), dict):
                entry_choices = quiz_item.get("choices")
            if not entry_answer:
                entry_answer = quiz_item.get("answer") or quiz_item.get("correct_answer") or ""

        _upsert(qid, question, entry_choices, entry_answer)

    for row in out:
        if _clean_text(row.get("question")):
            continue
        qid = _clean_text(row.get("qid"))
        if qid == _clean_text(last_qid):
            row["question"] = _clean_text(card_question)
        if not _clean_text(row.get("question")) and qid:
            quiz_item = load_quiz_item_by_id(qid) or {}
            row["question"] = _clean_text(quiz_item.get("question"))
            if not row.get("choices") and isinstance(quiz_item.get("choices"), dict):
                row["choices"] = _normalize_choices(quiz_item.get("choices"))
            if not _clean_text(row.get("answer")):
                row["answer"] = _clean_text(quiz_item.get("answer") or quiz_item.get("correct_answer")).upper()

    if not out and _clean_text(last_qid):
        question = _clean_text(card_question)
        if not question:
            quiz_item = load_quiz_item_by_id(last_qid) or {}
            question = _clean_text(quiz_item.get("question"))
        _upsert(
            last_qid,
            question,
            default_choices,
            default_answer,
        )

    return out


def _new_card(chunk_id: str, stem: str = "", doc_id: str = "", card_id: str = "") -> Dict[str, Any]:
    card_id = _clean_text(card_id) or build_chunk_card_id(chunk_id)
    return {
        "card_id": card_id,
        "store_key": card_id,
        "chunk_id": chunk_id,
        "stem": _normalized_stem(stem),
        "doc_id": _clean_text(doc_id),
        "title": "",
        "concept": "",
        "concept_label": "",
        "concept_id": "",
        "source_chunk": "",
        "source_chunk_preview": "",
        "linked_chunk_ids": [chunk_id] if chunk_id else [],
        "total_attempts": 0,
        "wrong_attempts": 0,
        "correct_attempts": 0,
        "mcq_history": [],
        "first_seen": "",
        "last_seen": "",
        "last_updated": "",
        "last_wrong": "",
        "last_correct": "",
        "last_mcq_id": "",
        "quiz_question_id": "",
        "question": "",
        "last_question": "",
        "choices": {},
        "answer": "",
        "correct_answer": "",
        "explanation": "",
        "last_chosen_answer": "",
        "last_is_correct": False,
        "item_type": "mcq",
        "origin": "quiz_mcq",
        "version": 2,
    }


def _apply_chunk_meta(card: Dict[str, Any], meta: Dict[str, str]) -> None:
    chunk_id = _clean_text(meta.get("source_chunk_id"))
    bucket_key = _clean_text(meta.get("concept_bucket_key"))
    concept_label = _clean_text(meta.get("concept_label"))
    concept_id = _clean_text(meta.get("concept_id")).lower()
    source_chunk = _clean_text(meta.get("source_chunk"))
    source_chunk_preview = _clean_text(meta.get("source_chunk_preview"))

    if bucket_key:
        card["card_id"] = bucket_key
        card["store_key"] = bucket_key
    elif chunk_id and not _clean_text(card.get("card_id")):
        card["card_id"] = build_chunk_card_id(chunk_id)
        card["store_key"] = card["card_id"]
    if chunk_id:
        if not _clean_text(card.get("chunk_id")):
            card["chunk_id"] = chunk_id
        card["linked_chunk_ids"] = _linked_chunks(card.get("linked_chunk_ids"), chunk_id)
    if concept_label and not _clean_text(card.get("concept_label")):
        card["concept_label"] = concept_label
    if concept_id and not _clean_text(card.get("concept_id")):
        card["concept_id"] = concept_id
    if source_chunk and not _clean_text(card.get("source_chunk")):
        card["source_chunk"] = source_chunk
    if source_chunk_preview and not _clean_text(card.get("source_chunk_preview")):
        card["source_chunk_preview"] = source_chunk_preview

    title = _card_title(
        card.get("concept_label", ""),
        card.get("source_chunk_preview", "") or card.get("source_chunk", ""),
        card.get("chunk_id", ""),
    )
    if title:
        card["title"] = title
        card["concept"] = title


def _normalize_card_entry(entry: Dict[str, Any], store_key: str) -> Optional[Dict[str, Any]]:
    chunk_meta = resolve_concept_bucket(
        stem=entry.get("stem", ""),
        question_item=entry,
        existing=entry,
    )
    chunk_id = _clean_text(entry.get("chunk_id") or entry.get("source_chunk_id") or chunk_meta.get("source_chunk_id"))
    card_key = _clean_text(chunk_meta.get("concept_bucket_key") or store_key)
    if not chunk_id and not card_key:
        return None

    card = _new_card(
        chunk_id=chunk_id,
        stem=entry.get("stem", ""),
        doc_id=entry.get("doc_id", ""),
        card_id=card_key,
    )
    _apply_chunk_meta(card, chunk_meta)
    card["linked_chunk_ids"] = _linked_chunks(entry.get("linked_chunk_ids"), card.get("linked_chunk_ids"), chunk_id)

    wrong_attempts = _as_int(entry.get("wrong_attempts", entry.get("wrong_count", 0)))
    correct_attempts = _as_int(entry.get("correct_attempts", entry.get("correct_count", 0)))
    total_attempts = _as_int(entry.get("total_attempts", wrong_attempts + correct_attempts))
    total_attempts = max(total_attempts, wrong_attempts + correct_attempts)
    if correct_attempts <= 0 and total_attempts > wrong_attempts:
        correct_attempts = total_attempts - wrong_attempts

    card["total_attempts"] = total_attempts
    card["wrong_attempts"] = wrong_attempts
    card["correct_attempts"] = max(correct_attempts, 0)

    last_mcq_id = _clean_text(entry.get("last_mcq_id") or entry.get("quiz_question_id") or entry.get("qid"))
    card_question = _clean_text(entry.get("question") or entry.get("last_question"))
    card["mcq_history"] = _coerce_mcq_history(
        entry.get("mcq_history"),
        card_question=card_question,
        last_qid=last_mcq_id,
        default_choices=entry.get("choices"),
        default_answer=_clean_text(entry.get("correct_answer") or entry.get("answer")),
    )

    card["first_seen"] = _clean_text(entry.get("first_seen"))
    card["last_wrong"] = _clean_text(entry.get("last_wrong"))
    card["last_correct"] = _clean_text(entry.get("last_correct"))
    card["last_seen"] = _max_non_empty(
        entry.get("last_seen", ""),
        entry.get("last_updated", ""),
        card["last_wrong"],
        card["last_correct"],
    )
    card["last_updated"] = _max_non_empty(
        entry.get("last_updated", ""),
        card["last_seen"],
    )
    card["last_mcq_id"] = last_mcq_id
    card["quiz_question_id"] = last_mcq_id
    card["question"] = _clean_text(entry.get("question") or entry.get("last_question"))
    card["last_question"] = card["question"]
    card["choices"] = _normalize_choices(entry.get("choices"))
    answer = _clean_text(entry.get("correct_answer") or entry.get("answer")).upper()
    card["answer"] = answer
    card["correct_answer"] = answer
    card["explanation"] = _clean_text(entry.get("explanation"))
    card["last_chosen_answer"] = _clean_text(entry.get("last_chosen_answer") or entry.get("chosen_answer")).upper()
    card["last_is_correct"] = bool(entry.get("last_is_correct", False))
    card["item_type"] = _clean_text(entry.get("item_type") or "mcq") or "mcq"
    card["origin"] = _clean_text(entry.get("origin") or "quiz_mcq") or "quiz_mcq"
    card["version"] = 2

    if not card["first_seen"]:
        card["first_seen"] = _min_non_empty(card["last_wrong"], card["last_correct"], card["last_seen"])
    return card


def _build_card_from_legacy_entry(entry: Dict[str, Any], store_key: str) -> Optional[Dict[str, Any]]:
    qid = _clean_text(entry.get("qid") or entry.get("quiz_question_id") or store_key)
    quiz_item = load_quiz_item_by_id(qid) if qid else None

    question_item = dict(quiz_item or {})
    for key, value in (entry or {}).items():
        if key not in question_item or value not in ("", None, [], {}):
            question_item[key] = value

    chunk_meta = resolve_concept_bucket(
        qid=qid,
        stem=entry.get("stem", ""),
        question=entry.get("question", ""),
        question_item=question_item,
        existing=entry,
    )
    chunk_id = _clean_text(chunk_meta.get("source_chunk_id"))
    card_key = _clean_text(chunk_meta.get("concept_bucket_key") or store_key)
    if not chunk_id and not card_key:
        return None

    card = _new_card(
        chunk_id=chunk_id,
        stem=entry.get("stem", ""),
        doc_id=entry.get("doc_id", ""),
        card_id=card_key,
    )
    _apply_chunk_meta(card, chunk_meta)

    wrong_attempts = _as_int(entry.get("wrong_count", 0))
    correct_attempts = _as_int(entry.get("correct_count", 0))
    total_attempts = _as_int(entry.get("total_attempts", wrong_attempts + correct_attempts))
    total_attempts = max(total_attempts, wrong_attempts + correct_attempts)
    if correct_attempts <= 0 and total_attempts > wrong_attempts:
        correct_attempts = total_attempts - wrong_attempts

    card["total_attempts"] = total_attempts
    card["wrong_attempts"] = wrong_attempts
    card["correct_attempts"] = max(correct_attempts, 0)
    card["mcq_history"] = _coerce_mcq_history(
        [qid] * wrong_attempts if qid and wrong_attempts > 0 else [],
        card_question=_clean_text(question_item.get("question") or entry.get("question")),
        last_qid=qid,
        default_choices=question_item.get("choices") or entry.get("choices"),
        default_answer=_clean_text(question_item.get("answer") or entry.get("correct_answer") or entry.get("answer")),
    )
    card["first_seen"] = _clean_text(entry.get("first_seen")) or _min_non_empty(
        entry.get("first_wrong", ""),
        entry.get("first_correct", ""),
    )
    card["last_wrong"] = _clean_text(entry.get("last_wrong"))
    card["last_correct"] = _clean_text(entry.get("last_correct"))
    card["last_seen"] = _max_non_empty(
        entry.get("last_seen", ""),
        card["last_wrong"],
        card["last_correct"],
    )
    card["last_updated"] = card["last_seen"]
    card["last_mcq_id"] = qid
    card["quiz_question_id"] = qid
    card["question"] = _clean_text(question_item.get("question") or entry.get("question"))
    card["last_question"] = card["question"]
    card["choices"] = _normalize_choices(question_item.get("choices") or entry.get("choices"))
    answer = _clean_text(question_item.get("answer") or entry.get("correct_answer") or entry.get("answer")).upper()
    card["answer"] = answer
    card["correct_answer"] = answer
    card["explanation"] = _clean_text(
        question_item.get("explanation")
        or question_item.get("brief_explanation")
        or entry.get("explanation")
    )
    card["last_chosen_answer"] = _clean_text(entry.get("last_chosen_answer") or entry.get("chosen_answer")).upper()
    card["last_is_correct"] = bool(entry.get("last_is_correct", False))
    card["item_type"] = "mcq"
    card["origin"] = "quiz_mcq"
    card["version"] = 2
    return card


def _merge_cards(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing or {})

    merged["total_attempts"] = _as_int(merged.get("total_attempts")) + _as_int(incoming.get("total_attempts"))
    merged["wrong_attempts"] = _as_int(merged.get("wrong_attempts")) + _as_int(incoming.get("wrong_attempts"))
    merged["correct_attempts"] = _as_int(merged.get("correct_attempts")) + _as_int(incoming.get("correct_attempts"))
    merged_last_qid = _max_non_empty(
        merged.get("last_mcq_id", ""),
        merged.get("quiz_question_id", ""),
        incoming.get("last_mcq_id", ""),
        incoming.get("quiz_question_id", ""),
    )
    merged_question = _max_non_empty(
        merged.get("question", ""),
        merged.get("last_question", ""),
        incoming.get("question", ""),
        incoming.get("last_question", ""),
    )
    merged["mcq_history"] = _coerce_mcq_history(
        list(merged.get("mcq_history") or []) + list(incoming.get("mcq_history") or []),
        card_question=merged_question,
        last_qid=merged_last_qid,
        default_choices=merged.get("choices"),
        default_answer=_clean_text(merged.get("correct_answer") or merged.get("answer")),
    )
    merged["first_seen"] = _min_non_empty(merged.get("first_seen", ""), incoming.get("first_seen", ""))
    merged["last_wrong"] = _max_non_empty(merged.get("last_wrong", ""), incoming.get("last_wrong", ""))
    merged["last_correct"] = _max_non_empty(merged.get("last_correct", ""), incoming.get("last_correct", ""))
    merged["last_seen"] = _max_non_empty(merged.get("last_seen", ""), incoming.get("last_seen", ""))
    merged["last_updated"] = _max_non_empty(merged.get("last_updated", ""), incoming.get("last_updated", ""))

    for field in ("stem", "doc_id", "chunk_id", "card_id", "store_key", "item_type", "origin", "version"):
        if _clean_text(incoming.get(field)) and not _clean_text(merged.get(field)):
            merged[field] = incoming.get(field)

    for field in ("concept_label", "concept_id", "source_chunk", "source_chunk_preview", "title", "concept"):
        if _clean_text(incoming.get(field)) and not _clean_text(merged.get(field)):
            merged[field] = incoming.get(field)

    existing_snapshot = _max_non_empty(merged.get("last_updated", ""), merged.get("last_seen", ""))
    incoming_snapshot = _max_non_empty(incoming.get("last_updated", ""), incoming.get("last_seen", ""))
    if incoming_snapshot >= existing_snapshot:
        merged["last_mcq_id"] = _clean_text(incoming.get("last_mcq_id") or incoming.get("quiz_question_id"))
        merged["quiz_question_id"] = merged["last_mcq_id"]
        merged["question"] = _clean_text(incoming.get("question") or incoming.get("last_question"))
        merged["last_question"] = merged["question"]
        merged["choices"] = _normalize_choices(incoming.get("choices"))
        answer = _clean_text(incoming.get("correct_answer") or incoming.get("answer")).upper()
        merged["answer"] = answer
        merged["correct_answer"] = answer
        merged["explanation"] = _clean_text(incoming.get("explanation"))
        merged["last_chosen_answer"] = _clean_text(incoming.get("last_chosen_answer")).upper()
        merged["last_is_correct"] = bool(incoming.get("last_is_correct", False))

    merged["linked_chunk_ids"] = _linked_chunks(
        merged.get("linked_chunk_ids"),
        incoming.get("linked_chunk_ids"),
        merged.get("chunk_id"),
        incoming.get("chunk_id"),
    )
    if not _clean_text(merged.get("chunk_id")) and merged["linked_chunk_ids"]:
        merged["chunk_id"] = merged["linked_chunk_ids"][0]
    return merged


def _normalize_store(raw_data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    normalized: Dict[str, Any] = {}
    changed = False

    for store_key, raw_entry in (raw_data or {}).items():
        if not isinstance(raw_entry, dict):
            changed = True
            continue

        entry = dict(raw_entry)
        is_card_entry = bool(_clean_text(entry.get("card_id"))) or any(
            key in entry for key in ("total_attempts", "wrong_attempts", "mcq_history")
        )
        card = (
            _normalize_card_entry(entry, store_key)
            if is_card_entry
            else _build_card_from_legacy_entry(entry, store_key)
        )
        if not card:
            changed = True
            continue

        card_key = card.get("card_id") or build_chunk_card_id(card.get("chunk_id", ""))
        if not card_key:
            changed = True
            continue

        if card_key in normalized:
            normalized[card_key] = _merge_cards(normalized[card_key], card)
        else:
            normalized[card_key] = card

        if store_key != card_key or not is_card_entry:
            changed = True

    if sorted(normalized.keys()) != sorted((raw_data or {}).keys()):
        changed = True
    return normalized, changed


def _decorate_card(card: Dict[str, Any]) -> Dict[str, Any]:
    decorated = dict(card or {})
    wrong_attempts = max(_as_int(decorated.get("wrong_attempts")), 0)
    total_attempts = max(_as_int(decorated.get("total_attempts")), wrong_attempts)
    correct_attempts = max(_as_int(decorated.get("correct_attempts")), total_attempts - wrong_attempts)
    error_rate = (wrong_attempts / total_attempts) if total_attempts else 0.0
    score = (wrong_attempts + 1.0) / (total_attempts + 2.0)
    final_score = score * math.log(1 + total_attempts) if total_attempts > 0 else 0.0

    decorated["total_attempts"] = total_attempts
    decorated["wrong_attempts"] = wrong_attempts
    decorated["correct_attempts"] = correct_attempts
    decorated["wrong_count"] = wrong_attempts
    decorated["correct_count"] = correct_attempts
    decorated["error_count"] = wrong_attempts
    decorated["error_rate"] = error_rate
    decorated["score"] = score
    decorated["final_score"] = final_score
    decorated["title"] = _card_title(
        decorated.get("concept_label", ""),
        decorated.get("source_chunk_preview", "") or decorated.get("source_chunk", ""),
        decorated.get("chunk_id", ""),
    )
    decorated["concept"] = decorated["title"]
    decorated["store_key"] = decorated.get("store_key") or decorated.get("card_id") or ""
    decorated["last_seen"] = _max_non_empty(
        decorated.get("last_seen", ""),
        decorated.get("last_updated", ""),
        decorated.get("last_wrong", ""),
        decorated.get("last_correct", ""),
    )
    decorated["last_updated"] = decorated["last_seen"]
    decorated["question"] = _clean_text(decorated.get("question") or decorated.get("last_question"))
    decorated["last_question"] = decorated["question"]
    decorated["quiz_question_id"] = _clean_text(
        decorated.get("quiz_question_id") or decorated.get("last_mcq_id")
    )
    decorated["choices"] = _normalize_choices(decorated.get("choices"))
    decorated["answer"] = _clean_text(decorated.get("answer") or decorated.get("correct_answer")).upper()
    decorated["correct_answer"] = decorated["answer"]
    last_qid = _clean_text(decorated.get("quiz_question_id") or decorated.get("last_mcq_id"))
    decorated["mcq_history"] = _coerce_mcq_history(
        decorated.get("mcq_history"),
        card_question=decorated.get("question") or decorated.get("last_question"),
        last_qid=last_qid,
        default_choices=decorated.get("choices"),
        default_answer=decorated.get("answer"),
    )
    decorated["linked_chunk_ids"] = _linked_chunks(
        decorated.get("linked_chunk_ids"),
        decorated.get("chunk_id"),
    )
    return decorated


def _persist_confusion_remotely(path: Optional[Path] = None) -> bool:
    if not use_remote_store():
        return False
    if path is None:
        return True
    try:
        return Path(path).resolve() == _default_path().resolve()
    except Exception:
        return False


def _load_confusion_raw(path: Optional[Path] = None) -> Dict[str, Any]:
    if _persist_confusion_remotely(path):
        from backend.supabase_store import load_namespace_dict

        raw = load_namespace_dict(NAMESPACE_CONFUSION)
        return raw if isinstance(raw, dict) else {}

    p = Path(path or _default_path())
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def load_confusion(path: Optional[Path] = None) -> Dict[str, Any]:
    raw = _load_confusion_raw(path)
    if not raw:
        return {}

    normalized, changed = _normalize_store(raw)
    if changed:
        try:
            save_confusion(normalized, path)
        except Exception:
            pass
    return normalized


def save_confusion(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    if _persist_confusion_remotely(path):
        from backend.supabase_store import save_namespace_dict

        save_namespace_dict(NAMESPACE_CONFUSION, data)
        return

    p = Path(path or _default_path())
    _ensure_parent(p)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


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
    Record that the user answered a quiz item.

    Confusion cards are keyed by lecture concept and keep chunk ids as evidence.
    """
    p = Path(path or _default_path())
    data = load_confusion(p)
    t = _now_iso()

    quiz_item = load_quiz_item_by_id(qid) if qid else None
    mcq_source = dict(quiz_item or {})
    for key, value in (question_item or {}).items():
        if key not in mcq_source or value not in ("", None, [], {}):
            mcq_source[key] = value

    if qid and not _clean_text(mcq_source.get("id")):
        mcq_source["id"] = qid
    if qid and not _clean_text(mcq_source.get("mcq_id")):
        mcq_source["mcq_id"] = qid
    if question and not _clean_text(mcq_source.get("question")):
        mcq_source["question"] = question
    if stem and not _clean_text(mcq_source.get("stem")):
        mcq_source["stem"] = stem
    if doc_id and not _clean_text(mcq_source.get("doc_id")):
        mcq_source["doc_id"] = doc_id

    effective_label = _clean_text(concept_label or concept)
    if _clean_text(concept_label):
        mcq_source["concept_label"] = _clean_text(concept_label)
    if _clean_text(concept_id):
        mcq_source["concept_id"] = _clean_text(concept_id).lower()

    chunk_meta = resolve_concept_bucket(
        qid=qid,
        stem=stem,
        question=question or mcq_source.get("question", ""),
        question_item=mcq_source,
        existing={
            **mcq_source,
            "stem": stem or mcq_source.get("stem", ""),
            "doc_id": doc_id or mcq_source.get("doc_id", ""),
            "concept_id": concept_id or mcq_source.get("concept_id", ""),
            "concept_label": effective_label or mcq_source.get("concept_label", ""),
        },
    )
    chunk_id = _clean_text(chunk_meta.get("source_chunk_id"))
    if not chunk_id:
        raise ValueError("Quiz result is missing a chunk_id/source chunk anchor.")

    card_key = _clean_text(chunk_meta.get("concept_bucket_key")) or build_chunk_card_id(chunk_id)
    card = _normalize_card_entry(data.get(card_key, {}), card_key) or _new_card(
        chunk_id=chunk_id,
        stem=stem,
        doc_id=doc_id,
        card_id=card_key,
    )
    _apply_chunk_meta(card, chunk_meta)

    question_text = _clean_text(question or mcq_source.get("question"))
    mcq_id = _clean_text(qid or mcq_source.get("mcq_id") or mcq_source.get("id"))
    choices = _normalize_choices(mcq_source.get("choices"))
    answer = _clean_text(mcq_source.get("answer") or mcq_source.get("correct_answer")).upper()
    explanation = _clean_text(
        mcq_source.get("explanation")
        or mcq_source.get("brief_explanation")
        or card.get("explanation")
    )

    card["stem"] = _normalized_stem(stem or card.get("stem", ""))
    card["doc_id"] = _clean_text(doc_id or card.get("doc_id", ""))
    card["store_key"] = card_key
    card["card_id"] = card_key
    if chunk_id and not _clean_text(card.get("chunk_id")):
        card["chunk_id"] = chunk_id
    card["linked_chunk_ids"] = _linked_chunks(card.get("linked_chunk_ids"), chunk_id)
    card["last_mcq_id"] = mcq_id
    card["quiz_question_id"] = mcq_id
    if question_text:
        card["question"] = question_text
        card["last_question"] = question_text
    if choices:
        card["choices"] = choices
    if answer:
        card["answer"] = answer
        card["correct_answer"] = answer
    if explanation:
        card["explanation"] = explanation
    if chosen_answer:
        card["last_chosen_answer"] = _clean_text(chosen_answer).upper()
    card["last_is_correct"] = bool(is_correct)
    card["item_type"] = "mcq"
    card["origin"] = "quiz_mcq"
    card["version"] = 2

    card["total_attempts"] = _as_int(card.get("total_attempts")) + 1
    if is_correct:
        card["correct_attempts"] = _as_int(card.get("correct_attempts")) + 1
        card["last_correct"] = t
    else:
        card["wrong_attempts"] = _as_int(card.get("wrong_attempts")) + 1
        if mcq_id:
            evidence_entry = {
                "qid": mcq_id,
                "question": question_text,
                "choices": choices,
                "answer": answer,
                "chosen": _clean_text(chosen_answer).upper() if chosen_answer else "",
            }
            card["mcq_history"] = list(card.get("mcq_history") or []) + [evidence_entry]
        card["last_wrong"] = t

    card["last_seen"] = t
    card["last_updated"] = t
    if not _clean_text(card.get("first_seen")):
        card["first_seen"] = t

    data[card_key] = card
    save_confusion(data, p)


def get_top_confusions(
    limit: Optional[int] = 10,
    path: Optional[Path] = None,
    stem: str = "",
    doc_id: str = "",
    only_wrong: bool = False,
) -> List[Dict[str, Any]]:
    """
    Return persisted concept-based confusion cards ranked by normalized weakness.
    """
    p = Path(path or _default_path())
    data = load_confusion(p)
    items: List[Dict[str, Any]] = []

    for _, entry in data.items():
        item = _decorate_card(entry)
        if stem and not _entry_matches_stem(item, stem):
            continue
        if only_wrong and _as_int(item.get("wrong_attempts")) <= 0:
            continue
        entry_doc_id = _clean_text(item.get("doc_id"))
        if doc_id and entry_doc_id and entry_doc_id != doc_id:
            continue
        items.append(item)

    items.sort(
        key=lambda it: (
            float(it.get("final_score", 0.0)),
            _as_int(it.get("wrong_attempts")),
            _as_int(it.get("total_attempts")),
            _clean_text(it.get("last_updated")),
        ),
        reverse=True,
    )
    if limit is None:
        return items
    return items[: max(int(limit or 0), 0)]


def delete_confusion_entries(keys: List[str], path: Optional[Path] = None) -> int:
    """
    Delete confusion cards by their persisted store keys. Returns number removed.
    """
    p = Path(path or _default_path())
    data = load_confusion(p)
    if not data:
        return 0

    normalized_keys = set()
    for key in keys or []:
        clean = _clean_text(key)
        if not clean:
            continue
        normalized_keys.add(clean)
        if not clean.startswith("chunk:"):
            normalized_keys.add(build_chunk_card_id(clean))

    removed = 0
    for key in normalized_keys:
        if key in data:
            del data[key]
            removed += 1

    if removed:
        save_confusion(data, p)
    return removed
