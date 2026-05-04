# backend/confusion_analysis.py
from typing import List, Dict, Any, Optional
import re
from collections import defaultdict

from backend.concept_policy import resolve_concept_bucket

def _shorten(text: str, n: int = 200) -> str:
    if not text:
        return ""
    txt = " ".join(text.split())
    return txt if len(txt) <= n else txt[: n - 3].rsplit(" ", 1)[0] + "..."

def analyze_confusion(
    history: Optional[List[Dict[str, str]]] = None,
    quiz_submissions: Optional[List[Dict[str, Any]]] = None,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    top_n: int = 6,
    llm_call = None,
) -> List[Dict[str, Any]]:
    """
    Returns candidate confusion points ranked by signal strength.

    Output item fields:
      - concept: short label
      - status: "confused"|"shaky"|"clear"
      - reason: human-friendly reason
      - evidence: list of evidences
      - signal_strength: integer (higher = stronger evidence)

    Heuristics:
      - Repeated incorrect quiz answers -> strong signal (count-based)
      - Single incorrect quiz answer -> medium signal
      - Recent chat messages expressing confusion -> weak signal
      - If nothing found, returns a single 'clear' hint.
    """
    history = history or []
    quiz_submissions = quiz_submissions or []
    retrieved_chunks = retrieved_chunks or []

    # 1) Aggregate quiz signals by deterministic concept bucket.
    aggregated = defaultdict(lambda: {
        "count_wrong": 0,
        "examples": [],
        "concept_id": "",
        "concept_label": "",
        "source_chunk_id": "",
    })
    for sub in quiz_submissions:
        qtext = (sub.get("question") or "").strip()
        qid = sub.get("id") or ""
        is_correct = bool(sub.get("is_correct", False))
        if not is_correct:
            concept_meta = resolve_concept_bucket(
                qid=qid,
                stem=sub.get("stem", ""),
                question=qtext,
                question_item=sub,
                existing=sub,
            )
            key = concept_meta["concept_bucket_key"] or (qid or qtext or "unknown")
            aggregated[key]["count_wrong"] += 1
            aggregated[key]["concept_id"] = concept_meta["concept_id"] or aggregated[key]["concept_id"]
            aggregated[key]["concept_label"] = concept_meta["concept_label"] or aggregated[key]["concept_label"]
            aggregated[key]["source_chunk_id"] = concept_meta["source_chunk_id"] or aggregated[key]["source_chunk_id"]
            aggregated[key]["examples"].append({
                "type": "quiz",
                "id": qid,
                "question": qtext,
                "concept_id": concept_meta["concept_id"],
                "source_chunk_id": concept_meta["source_chunk_id"],
            })

    # 2) Conversation signals (weaker)
    confusion_terms = ["confus", "don't understand", "dont understand", "unclear", "why", "how", "hard", "difficult", "i'm stuck", "i'm lost", "can't", "cant", "help", "explain"]
    user_msgs = [turn.get("content", "") for turn in history if turn.get("role", "").lower().startswith("user")]
    chat_evidences = []
    for msg in user_msgs[-20:]:
        low = (msg or "").lower()
        if any(t in low for t in confusion_terms):
            chat_evidences.append({"type": "chat", "message": _shorten(msg, 400)})

    results: List[Dict[str, Any]] = []
    # create results from aggregated quiz signals
    for key, meta in aggregated.items():
        cnt = meta["count_wrong"]
        examples = meta["examples"]
        # status: confused if repeated mistakes (>=2), shaky if single mistake
        status = "confused" if cnt >= 2 else "shaky"
        reason = f"{'Repeated incorrect answers' if cnt >= 2 else 'Incorrect answer'} on quiz items (count={cnt})."
        label = meta.get("concept_label") or _shorten((examples[0] or {}).get("question", "") if examples else key, 160)
        results.append({
            "concept": label,
            "concept_id": meta.get("concept_id", ""),
            "concept_label": meta.get("concept_label", ""),
            "source_chunk_id": meta.get("source_chunk_id", ""),
            "status": status,
            "reason": reason,
            "evidence": examples,
            "signal_strength": cnt,
        })

    # add chat-based hints with low strength if nothing or to complement
    for ce in chat_evidences:
        concept = _shorten(ce.get("message", ""), 140)
        # only add if not already present
        if not any(r["concept"] == concept for r in results):
            results.append({
                "concept": concept,
                "status": "shaky",
                "reason": "User expressed confusion in chat or asked for clarification.",
                "evidence": [ce],
                "signal_strength": 1,
            })

    # optional: add retrieved chunks as low-priority hints if results empty
    if not results and retrieved_chunks:
        for idx, c in enumerate(retrieved_chunks[:top_n]):
            text = c.get("text") if isinstance(c, dict) else str(c)
            results.append({
                "concept": _shorten(text.split("\n", 1)[0], 140),
                "status": "shaky",
                "reason": "Chunk retrieved frequently and may contain dense concepts worth review.",
                "evidence": [{"type": "chunk", "id": c.get("id") if isinstance(c, dict) else None, "preview": _shorten(text, 300)}],
                "signal_strength": 1,
            })

    # fallback
    if not results:
        results.append({
            "concept": "No strong confusion signals detected",
            "status": "clear",
            "reason": "No recent incorrect quiz answers or explicit confusion in chat were found.",
            "evidence": [],
            "signal_strength": 0,
        })

    # Sort by signal_strength desc, then by status (confused > shaky > clear)
    status_rank = {"confused": 0, "shaky": 1, "clear": 2}
    results_sorted = sorted(results, key=lambda r: (-int(r.get("signal_strength", 0)), status_rank.get(r.get("status", "shaky"), 1)))

    return results_sorted[:top_n]
