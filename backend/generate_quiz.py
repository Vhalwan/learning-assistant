# backend/generate_quiz.py
"""
generate_quiz.py — Hybrid MCQ generator (strict extractor + LLM writer)

v6 changes vs v5  (goal: v4 speed + reliability, v5 diversity):
  ─────────────────────────────────────────────────────────────────
  SPEED / 503 FIXES
  ─────────────────────────────────────────────────────────────────
  1. Concept extraction now targets a lecture-level map of about 5-10 topics
     instead of treating chunks as the student-facing unit.

  2. BACKOFF_503 reduced from 10 s → 4 s, and the multiplier is now capped:
       wait = min(BACKOFF_503 * attempt, BACKOFF_503_MAX)   (max 12 s)
     v5 could wait 10 s, 20 s, 30 s — that killed perceived latency.

  3. Question-plan block in the batch prompt is now compact (one line per row,
     no extra whitespace) to reduce prompt token count.

  4. Concept extraction is skipped entirely when concepts are passed in by
     the caller (was already the case but now explicitly fast-pathed).

  5. Per-chunk fallback: plan_idx calculation was off-by-one in v5 (used
     len(out_mcqs) twice); fixed to use a simple counter.

  ─────────────────────────────────────────────────────────────────
  DIVERSITY KEPT FROM v5
  ─────────────────────────────────────────────────────────────────
  - _build_question_plan() unchanged.
  - _QUESTION_TYPES unchanged.
  - batch prompt still receives the question_plan block.
  - per-chunk fallback still receives concept + type guidance.
  - question_type field preserved on every output item.

  ─────────────────────────────────────────────────────────────────
  EVERYTHING ELSE UNCHANGED
  ─────────────────────────────────────────────────────────────────
  - All explanation fields (brief_explanation, detailed_explanation).
  - Validation logic.
  - generate_quiz_from_context() backwards-compatible wrapper.
"""

import json
import hashlib
import random
import re
import time
import logging
from typing import Callable, Dict, List, Optional, Set

from backend.concept_policy import canonicalize_chunk_id, derive_source_chunk_id
from backend.helpers import chunk_text as build_embedding_chunks

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
BACKOFF_BASE = 0.4
BACKOFF_503 = 4.0        # FIX v6: was 10.0 — smaller base keeps 503 retries fast
BACKOFF_503_MAX = 12.0   # FIX v6: cap so wait never exceeds 12 s (v5 could hit 30 s)

_MIN_SENT_WORDS = 8
_MIN_CHOICE_LEN = 8
_MIN_QUESTION_LEN = 12

# Display limits — large enough that quiz UI rarely shows "..." truncation.
_QUIZ_QUESTION_MAX_LEN = 1800
_QUIZ_CHOICE_MAX_LEN = 420

# Session-only diversity: max MCQs anchored to the same chunk_id in one quiz sitting.
_SESSION_MAX_MCQS_PER_CHUNK = 2

_BAD_PHRASES = [
    "unrelated concept", "is an unrelated concept", "not asked here",
    "not asked in the lecture", "not supported by", "This statement is not supported"
]

_DEFAULT_CONCEPTS = [
    {
        "concept_id": "unlabeled_concept",
        "concept_label": "Unlabeled concept",
    }
]

# ── Question types: core angles (definition / application / trap / compare) + extras ──
_CORE_QUESTION_TYPES = [
    ("definition",    "Definition — ask what the term or idea IS or means; identify the best lecture-faithful wording."),
    ("application",   "Application — give a short scenario or example; which option is the best lecture-supported judgment?"),
    ("misconception", "Misconception trap — one option should be a common wrong belief; the correct answer must match the lecture, not folk wisdom."),
    ("comparison",    "Compare two concepts from the lecture — which statement correctly distinguishes or relates them?"),
]
_EXTRA_QUESTION_TYPES = [
    ("mechanism",     "Ask HOW something works or WHY it happens (mechanism / reasoning)"),
    ("not_true",      "Ask which option is FALSE or NOT supported by the lecture"),
    ("consequence",   "Ask what result or implication follows from a claim in the lecture"),
    ("property",      "Ask about a specific property, characteristic, or requirement"),
    ("criticism",     "Ask about a limitation, flaw, or caveat the lecture mentions"),
]
_QUESTION_TYPES = _CORE_QUESTION_TYPES + _EXTRA_QUESTION_TYPES

_CONCEPT_PROMPT_TEMPLATE = """
You are an expert educator. Extract 5 to {max_concepts} canonical concepts from the lecture content when possible.
Aim for meaningful lecture topics that can span multiple chunks/slides. Use fewer than 5 only when the content truly has fewer distinct topics.
Return JSON ONLY (no extra text). Output must be a JSON array of objects with keys:
  - concept_id (snake_case, short, stable)
  - concept_label (human-readable)

Lecture content:
\"\"\"{content}\"\"\"
"""

# FIX v6: question_plan block is now formatted compactly (fewer tokens)
_BATCH_MCQ_PROMPT_TEMPLATE = """
You are an expert quiz writer. Generate exactly {n} high-quality multiple-choice questions (MCQs)
that test a learner's understanding of the lecture content below.

QUESTION PLAN (follow exactly, one row per question):
{question_plan}

Each row: concept = topic to focus on | type = cognitive structure to use.

Cognitive angle hints (follow the type id literally; do not collapse every question into generic recall):
  • definition — precise meaning / identification, not a rephrased duplicate of another question on the same concept.
  • application — novel mini-scenario; wrong answers plausible in real life but wrong per the lecture.
  • misconception — explicitly target a tempting error; correct option is the lecture view.
  • comparison — two ideas from the lecture contrasted; avoid repeating a definition-only stem.

Return JSON ONLY (no extra text, no markdown fences).
Output must be a JSON array of exactly {n} objects, each with keys:
  - id                  (string, e.g. "q1", "q2" ...)
  - question            (string)
  - choices             (object with keys A, B, C, D)
  - answer              (string, one of: A B C D)
  - brief_explanation   (string, 1-2 sentences max — plain statement of why the answer is correct,
                         NO label prefix like "Explanation:")
  - detailed_explanation (object with keys:
      why_correct   : string — full reasoning for the correct choice (2-4 sentences)
      why_wrong     : object with keys A, B, C, D — one sentence each explaining why
                      each option is incorrect (skip the correct letter or set it to "")
      source_chunk  : string — the verbatim sentence(s) from the lecture that most directly
                      support the correct answer (copy exactly, ≤ 3 sentences))

IMPORTANT:
- Randomise which letter (A/B/C/D) holds the correct answer across questions.
- Do NOT bias toward C, B, or any particular letter.
- All four options must be plausible; only one should be correct.
- Distractors must be clearly distinct from each other.
- Every question must cover a DIFFERENT concept from the others.

Lecture content:
\"\"\"{content}\"\"\"
"""

_PER_CHUNK_PROMPT_TEMPLATE = """
You are an expert quiz writer. Produce ONE high-quality multiple-choice question (MCQ) that tests
a learner's understanding of the provided paragraph. Return JSON ONLY (no extra text).

Question requirements:
  • Concept to focus on : {concept_label}
  • Question type       : {type_instruction}

JSON must be an object with keys:
  - id
  - question
  - choices             (object with keys A, B, C, D)
  - answer              (one of: A B C D)
  - brief_explanation   (1-2 sentences, plain, no label prefix)
  - detailed_explanation (object with keys:
      why_correct   : string
      why_wrong     : object with keys A, B, C, D (set correct letter to "")
      source_chunk  : verbatim sentence(s) from the paragraph that support the answer, ≤ 3 sentences)

IMPORTANT: Randomise which letter is correct. Do NOT bias toward C or any particular letter.
All four options must be plausible but only one correct.

Paragraph:
\"\"\"{paragraph}\"\"\"
"""


# ─────────────────────────────────────────────────────────────────────────────
# Diversity plan builder  (unchanged from v5)
# ─────────────────────────────────────────────────────────────────────────────

def _build_question_plan(concepts: List[Dict[str, str]], n: int) -> List[Dict[str, str]]:
    """
    Build a list of n dicts, each specifying:
      concept_id, concept_label, type_id, type_instruction

    Rules:
    1. Every question gets a DIFFERENT concept (cycle if n > len(concepts),
       never repeat the same concept back-to-back).
    2. Question types are shuffled and rotated — no two adjacent questions
       share a type, full set spread before repeating.
    3. Deterministic-random (shuffled each call) so reruns differ.
    """
    if not concepts:
        concepts = list(_DEFAULT_CONCEPTS)

    shuffled_concepts = concepts[:]
    random.shuffle(shuffled_concepts)

    concept_seq: List[Dict] = []
    last_id = None
    pool = shuffled_concepts[:]
    while len(concept_seq) < n:
        candidates = [c for c in pool if c["concept_id"] != last_id]
        if not candidates:
            pool = shuffled_concepts[:]
            random.shuffle(pool)
            candidates = [c for c in pool if c["concept_id"] != last_id] or pool
        chosen = candidates[0]
        concept_seq.append(chosen)
        pool.remove(chosen)
        last_id = chosen["concept_id"]
        if not pool:
            pool = shuffled_concepts[:]
            random.shuffle(pool)

    def _fresh_type_pool() -> List[tuple]:
        c = _CORE_QUESTION_TYPES[:]
        random.shuffle(c)
        e = _EXTRA_QUESTION_TYPES[:]
        random.shuffle(e)
        return c + e

    type_pool = _fresh_type_pool()
    type_seq: List[tuple] = []
    last_type = None
    pool_t = type_pool[:]
    while len(type_seq) < n:
        candidates_t = [t for t in pool_t if t[0] != last_type]
        if not candidates_t:
            type_pool = _fresh_type_pool()
            pool_t = type_pool[:]
            candidates_t = [t for t in pool_t if t[0] != last_type] or pool_t
        chosen_t = candidates_t[0]
        type_seq.append(chosen_t)
        pool_t.remove(chosen_t)
        last_type = chosen_t[0]
        if not pool_t:
            type_pool = _fresh_type_pool()
            pool_t = type_pool[:]

    plan = []
    for i in range(n):
        concept = concept_seq[i]
        qtype = type_seq[i]
        plan.append({
            "concept_id": concept["concept_id"],
            "concept_label": concept["concept_label"],
            "type_id": qtype[0],
            "type_instruction": qtype[1],
        })
    return plan


def _format_question_plan(plan: List[Dict[str, str]]) -> str:
    """
    FIX v6: compact one-liner per row (fewer tokens than v5's multi-field format).
    Example:  Q1: concept='Homo Economicus' | type=definition
    """
    lines = []
    for i, row in enumerate(plan, start=1):
        lines.append(
            f"Q{i}: concept='{row['concept_label']}' | type={row['type_id']} — {row['type_instruction']}"
        )
    return "\n".join(lines)


# ── unchanged helpers ──────────────────────────────────────────────────────

def safe_json_load(s):
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        raise ValueError("llm_call returned unsupported type")

    text = s.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    def find_balanced(t):
        start_idx = None
        opener = None
        for i, ch in enumerate(t):
            if ch in ("{", "["):
                start_idx = i
                opener = ch
                break
        if start_idx is None:
            return None
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        for i in range(start_idx, len(t)):
            ch = t[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"' and not esc:
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return t[start_idx:i+1]
        return None

    sub = find_balanced(text)
    if sub:
        try:
            return json.loads(sub)
        except Exception:
            try:
                alt = sub.replace("'", '"')
                return json.loads(alt)
            except Exception:
                pass

    raise ValueError("Could not parse JSON from LLM output")


def _shorten(s: str, max_len=240) -> str:
    if not s:
        return s
    s = s.strip()
    if len(s) <= max_len:
        return s
    cut = s[:max_len]
    last_p = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if last_p > max_len - 80:
        return cut[:last_p + 1].strip()
    return cut[:max_len - 3].rstrip() + "..."


def _slugify_concept_id(label: str, fallback: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "_", (label or "").strip().lower()).strip("_")
    return base or fallback


def _is_rate_limit_error(e: Exception) -> bool:
    s = str(e)
    return (
        "429" in s
        or "RESOURCE_EXHAUSTED" in s
        or "resource_exhausted" in s.lower()
        or "quota" in s.lower()
        or "rate limit" in s.lower()
        or "rate_limit" in s.lower()
    )


def _is_overload_error(e: Exception) -> bool:
    s = str(e)
    return "503" in s or "UNAVAILABLE" in s or "unavailable" in s.lower()


def _extract_retry_delay(e: Exception, default: float = 5.0) -> float:
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", str(e))
    if m:
        return float(m.group(1))
    return default


def _normalize_concepts(raw, max_concepts: int = 8) -> List[Dict[str, str]]:
    max_concepts = int(max_concepts or 1)
    max_concepts = max(1, min(max_concepts, 20))
    concepts: List[Dict[str, str]] = []
    used_ids = set()

    if raw is None:
        raw_list = []
    elif isinstance(raw, dict):
        if "concepts" in raw and isinstance(raw.get("concepts"), list):
            raw_list = raw.get("concepts") or []
        else:
            raw_list = [raw]
    elif isinstance(raw, list):
        raw_list = raw
    else:
        raw_list = []

    for idx, item in enumerate(raw_list):
        if not isinstance(item, dict):
            continue
        label = str(item.get("concept_label") or item.get("label") or item.get("concept") or "").strip()
        concept_id = str(item.get("concept_id") or item.get("id") or "").strip()
        if not label and not concept_id:
            continue
        if not label:
            label = concept_id
        if not concept_id:
            concept_id = _slugify_concept_id(label, f"concept_{len(concepts) + 1}")
        concept_id = re.sub(r"[^a-z0-9_]", "_", concept_id.lower())
        concept_id = re.sub(r"_+", "_", concept_id).strip("_") or f"concept_{len(concepts) + 1}"
        base_id = concept_id
        suffix = 2
        while concept_id in used_ids:
            concept_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(concept_id)
        concepts.append({"concept_id": concept_id, "concept_label": label})
        if len(concepts) >= max_concepts:
            break

    if not concepts:
        concepts = list(_DEFAULT_CONCEPTS)
    return concepts


def generate_concepts_from_context(
    context_text: str,
    max_concepts: int = 8,
    llm_call: Optional[Callable[[str], str]] = None,
) -> List[Dict[str, str]]:
    if llm_call is None:
        chunks = _paragraph_chunks(context_text or "", approx_words=80)
        concepts = []
        for chunk in chunks:
            if len(concepts) >= int(max_concepts or 5):
                break
            first = re.split(r"[.!?]\s+", chunk.strip(), maxsplit=1)[0] if chunk else ""
            label = " ".join((first or chunk).split()[:8]).strip()
            if len(label) < 3:
                continue
            concepts.append({"concept_label": label})
        return _normalize_concepts(concepts, max_concepts=max_concepts)

    prompt = _CONCEPT_PROMPT_TEMPLATE.format(
        max_concepts=int(max_concepts or 5),
        content=_shorten(context_text or "", max_len=4000),
    )
    try:
        raw = llm_call(prompt)
        parsed = safe_json_load(raw)
        return _normalize_concepts(parsed, max_concepts=max_concepts)
    except Exception:
        return list(_DEFAULT_CONCEPTS)


def _token_overlap_fraction(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", (a or "").lower()))
    tb = set(re.findall(r"\w+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    inter = ta.intersection(tb)
    return len(inter) / max(len(ta), len(tb))


def _looks_like_filler(s: str) -> bool:
    if not s:
        return True
    ls = s.lower()
    for p in _BAD_PHRASES:
        if p in ls:
            return True
    if len(s.split()) < 2 and len(s) < 6:
        return True
    return False


def _paragraph_chunks(text: str, approx_words: int = 80) -> List[str]:
    if not text:
        return []
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    chunks = []
    cur = []
    cur_words = 0
    for s in sents:
        w = len(s.split())
        if cur and (cur_words + w > approx_words) and len(cur) > 0:
            chunks.append(" ".join(cur))
            cur = [s]
            cur_words = w
        else:
            cur.append(s)
            cur_words += w
    if cur:
        chunks.append(" ".join(cur))
    chunks = [c for c in chunks if len(c.split()) >= _MIN_SENT_WORDS]
    return chunks


def _clean_ws(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _context_doc_id(context_text: str) -> str:
    return hashlib.sha1((context_text or "").encode("utf-8")).hexdigest()[:12]


def _build_context_chunk_rows(context_text: str) -> List[Dict[str, str]]:
    doc_id = _context_doc_id(context_text)
    raw_chunks = build_embedding_chunks(context_text or "", max_chars=2000, overlap=200)
    rows: List[Dict[str, str]] = []
    for idx, raw_chunk in enumerate(raw_chunks):
        chunk_value = _clean_ws(raw_chunk)
        if not chunk_value:
            continue
        rows.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}:uploaded_chunk_{idx}",
            "embedding_chunk_id": f"uploaded_chunk_{idx}",
            "chunk_text": chunk_value,
        })
    if not rows:
        fallback_chunk = _clean_ws(context_text)
        if fallback_chunk:
            rows.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}:uploaded_chunk_0",
                "embedding_chunk_id": "uploaded_chunk_0",
                "chunk_text": fallback_chunk,
            })
    return rows


def _find_matching_context_chunk(anchor_text: str, chunk_rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    anchor = _clean_ws(anchor_text)
    if not chunk_rows:
        return None
    if len(chunk_rows) == 1:
        return chunk_rows[0]
    if not anchor:
        return None

    anchor_low = anchor.lower()
    best_row: Optional[Dict[str, str]] = None
    best_score = -1.0
    for row in chunk_rows:
        chunk_value = row.get("chunk_text", "")
        chunk_low = chunk_value.lower()
        if anchor_low and anchor_low in chunk_low:
            return row
        if chunk_low and chunk_low in anchor_low:
            score = 0.95
        else:
            score = _token_overlap_fraction(anchor_low, chunk_low)
        if score > best_score:
            best_score = score
            best_row = row
    if best_score >= 0.08:
        return best_row
    return None


def _attach_chunk_metadata(item: Dict, context_chunk_rows: List[Dict[str, str]]) -> Dict:
    enriched = dict(item or {})
    detailed = enriched.get("detailed_explanation") or {}
    if not isinstance(detailed, dict):
        detailed = {}

    raw_chunk_id = _clean_ws(enriched.get("chunk_id") or enriched.get("source_chunk_id"))
    source_chunk = _clean_ws(detailed.get("source_chunk") or enriched.get("source_chunk"))
    chunk_hint = _clean_ws(enriched.get("_chunk_text_hint"))

    matched_row = None
    if raw_chunk_id:
        for row in context_chunk_rows:
            if raw_chunk_id in (row.get("chunk_id", ""), row.get("embedding_chunk_id", "")):
                matched_row = row
                break
    if matched_row is None:
        matched_row = _find_matching_context_chunk(source_chunk or chunk_hint, context_chunk_rows)

    doc_id = (
        (matched_row or {}).get("doc_id")
        or (context_chunk_rows[0].get("doc_id") if context_chunk_rows else "")
    )
    chunk_text_value = _clean_ws((matched_row or {}).get("chunk_text") or source_chunk or chunk_hint)
    chunk_id = _clean_ws((matched_row or {}).get("chunk_id") or "")
    if not chunk_id and raw_chunk_id:
        chunk_id = canonicalize_chunk_id(raw_chunk_id, doc_id=doc_id)
    if not chunk_id and chunk_text_value:
        chunk_id = canonicalize_chunk_id(derive_source_chunk_id(chunk_text_value), doc_id=doc_id)

    enriched["doc_id"] = doc_id
    enriched["mcq_id"] = enriched.get("mcq_id") or enriched.get("id", "")
    enriched["chunk_id"] = chunk_id
    enriched["source_chunk_id"] = chunk_id
    if chunk_text_value:
        enriched["chunk_text"] = chunk_text_value
        enriched["source_chunk"] = source_chunk or chunk_text_value
        enriched["source_chunk_preview"] = _shorten(chunk_text_value, max_len=180)
    enriched["detailed_explanation"] = detailed
    enriched.pop("_chunk_text_hint", None)
    return enriched


def _validate_mcq_obj(obj: dict, paragraph: str = "") -> bool:
    try:
        if not isinstance(obj, dict):
            return False
        if "question" not in obj or "choices" not in obj or "answer" not in obj:
            return False
        q = str(obj["question"]).strip()
        if len(q) < _MIN_QUESTION_LEN:
            return False
        choices = obj["choices"]
        if not isinstance(choices, dict):
            return False
        keys = ["A", "B", "C", "D"]
        if any(k not in choices for k in keys):
            return False
        seen_texts = set()
        for k in keys:
            ch = str(choices[k]).strip()
            if len(ch) < _MIN_CHOICE_LEN:
                return False
            if _looks_like_filler(ch):
                return False
            low = re.sub(r'\s+', ' ', ch.lower())
            if low in seen_texts:
                return False
            seen_texts.add(low)
        ans = str(obj["answer"]).strip().upper()
        if ans not in keys:
            return False
        if paragraph:
            correct_text = str(choices[ans])
            overlap = _token_overlap_fraction(correct_text, paragraph)
            if overlap < 0.05 and overlap <= 0.0:
                return False
        for k in keys:
            if k == ans:
                continue
            if _token_overlap_fraction(str(choices[k]), str(choices[ans])) > 0.9:
                return False
        if "explanation" in obj:
            ex = str(obj["explanation"]).strip()
            obj["explanation"] = ex
        return True
    except Exception:
        return False


def _normalise_detailed_explanation(raw_detail, answer_letter: str) -> Dict:
    empty = {
        "why_correct": "",
        "why_wrong": {"A": "", "B": "", "C": "", "D": ""},
        "source_chunk": "",
    }
    if not isinstance(raw_detail, dict):
        return empty

    why_correct = str(raw_detail.get("why_correct") or "").strip()
    source_chunk = str(raw_detail.get("source_chunk") or "").strip()

    raw_ww = raw_detail.get("why_wrong") or {}
    why_wrong: Dict[str, str] = {}
    for letter in ["A", "B", "C", "D"]:
        if letter == (answer_letter or "").upper():
            why_wrong[letter] = ""
        else:
            why_wrong[letter] = str(raw_ww.get(letter) or "").strip()

    return {
        "why_correct": why_correct,
        "why_wrong": why_wrong,
        "source_chunk": source_chunk,
    }


def _extract_explanation_fields(obj: dict, answer_letter: str) -> Dict:
    brief = (
        str(obj.get("brief_explanation") or obj.get("explanation") or "").strip()
    )
    raw_detail = obj.get("detailed_explanation")
    detailed = _normalise_detailed_explanation(raw_detail, answer_letter)
    return {
        "brief_explanation": brief,
        "detailed_explanation": detailed,
    }


def _normalize_session_chunk_counts(raw: Optional[Dict[str, int]]) -> Dict[str, int]:
    if not raw:
        return {}
    out: Dict[str, int] = {}
    for k, v in raw.items():
        ck = _clean_ws(str(k)).lower()
        if not ck:
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            iv = 0
        out[ck] = out.get(ck, 0) + max(0, iv)
    return out


def _normalize_question_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _chunk_key(item: Dict) -> str:
    return _clean_ws(str(item.get("chunk_id") or item.get("source_chunk_id") or "")).lower()


def _pick_with_chunk_budget(
    candidates: List[Dict],
    n: int,
    session_counts: Dict[str, int],
    max_per_chunk: int,
) -> List[Dict]:
    """
    Greedy selection up to n items, preferring chunks with lower combined usage.
    """
    counts = dict(session_counts)
    picked: List[Dict] = []
    remaining = list(candidates)
    stall_guard = 0
    while len(picked) < n and remaining:
        remaining.sort(key=lambda it: (counts.get(_chunk_key(it), 0), random.random()))
        item = remaining.pop(0)
        cid = _chunk_key(item)
        if counts.get(cid, 0) >= max_per_chunk:
            remaining.append(item)
            stall_guard += 1
            if stall_guard > max(len(remaining) * 4, 16):
                break
            continue
        stall_guard = 0
        picked.append(item)
        counts[cid] = counts.get(cid, 0) + 1
    return picked


def _ordered_embedding_rows(
    chunk_rows: List[Dict[str, str]],
    usage_counts: Dict[str, int],
) -> List[Dict[str, str]]:
    """Prefer chunks with fewer uses in this session (stable tie-break: earlier index)."""
    indexed = list(enumerate(chunk_rows))
    indexed.sort(key=lambda ix_row: (usage_counts.get(_clean_ws(ix_row[1].get("chunk_id") or "").lower(), 0), ix_row[0]))
    return [row for _, row in indexed]


def _assemble_and_attach(items: List[Dict], context_chunk_rows: List[Dict[str, str]]) -> List[Dict]:
    cleaned: List[Dict] = []
    for q in items:
        base = {
            "id": q["id"],
            "question": q["question"],
            "choices": q["choices"],
            "answer": q["answer"],
            "brief_explanation": q.get("brief_explanation", ""),
            "detailed_explanation": q.get("detailed_explanation", {
                "why_correct": "",
                "why_wrong": {"A": "", "B": "", "C": "", "D": ""},
                "source_chunk": "",
            }),
            "concept_id": q.get("concept_id", ""),
            "concept_label": q.get("concept_label", ""),
            "question_type": q.get("question_type", ""),
            "_chunk_text_hint": q.get("_chunk_text_hint", ""),
        }
        cleaned.append(_attach_chunk_metadata(base, context_chunk_rows))
    return cleaned


def _one_chunk_mcq_llm(
    llm_call: Callable[[str], str],
    chunk: str,
    plan_row: Dict[str, str],
    next_num: int,
    used_questions_fb: Set[str],
) -> Optional[Dict]:
    """Returns one MCQ dict (pre-metadata) or None if all retries fail."""
    prompt = _PER_CHUNK_PROMPT_TEMPLATE.format(
        paragraph=_shorten(chunk, max_len=2000),
        concept_label=plan_row["concept_label"],
        type_instruction=plan_row["type_instruction"],
    )
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            logger.info("Per-chunk MCQ LLM call — attempt %s", attempt)
            raw = llm_call(prompt)
            if raw is None:
                raise RuntimeError("LLM returned None")

            parsed = safe_json_load(raw)

            if isinstance(parsed, list) and parsed:
                obj = parsed[0]
            elif isinstance(parsed, dict) and "questions" in parsed:
                obj = (parsed["questions"] or [None])[0]
            elif isinstance(parsed, dict):
                obj = parsed
            else:
                raise ValueError("Unexpected shape from per-chunk LLM response")

            if not obj or not isinstance(obj, dict):
                raise ValueError("No MCQ object in response")
            if not obj.get("id"):
                obj["id"] = f"q{next_num}"
            if "answer" in obj:
                obj["answer"] = str(obj["answer"]).strip().upper()

            if not _validate_mcq_obj(obj, chunk):
                raise ValueError("MCQ failed validation")

            q_norm = _normalize_question_key(obj["question"])
            if q_norm in used_questions_fb:
                return None
            used_questions_fb.add(q_norm)

            obj["question"] = _shorten(str(obj["question"]), max_len=_QUIZ_QUESTION_MAX_LEN)
            for k in list(obj.get("choices", {}).keys()):
                obj["choices"][k] = _shorten(str(obj["choices"][k]), max_len=_QUIZ_CHOICE_MAX_LEN)

            expl_fields = _extract_explanation_fields(obj, obj["answer"])

            return {
                "id": obj["id"],
                "question": obj["question"],
                "choices": obj["choices"],
                "answer": obj["answer"],
                **expl_fields,
                "concept_id": plan_row["concept_id"],
                "concept_label": plan_row["concept_label"],
                "question_type": plan_row["type_id"],
                "_chunk_text_hint": chunk,
            }

        except Exception as e:
            if _is_rate_limit_error(e):
                delay = _extract_retry_delay(e, default=15.0)
                logger.error("Quota exhausted in per-chunk MCQ — waiting %.1fs. Error: %s", delay, e)
                time.sleep(delay)
                return None
            if _is_overload_error(e):
                wait = min(BACKOFF_503 * attempt, BACKOFF_503_MAX)
                logger.warning("Model overloaded (503) in per-chunk MCQ — waiting %.1fs.", wait)
                if attempt <= MAX_RETRIES:
                    time.sleep(wait)
                    continue
                return None
            logger.warning("Per-chunk MCQ attempt %s failed: %s", attempt, e)
            if attempt <= MAX_RETRIES:
                time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))
                continue
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core generator  (v6 — v4 speed + v5 diversity)
# ─────────────────────────────────────────────────────────────────────────────
def generate_mcq_from_context(
    context_text: str,
    n: int = 5,
    llm_call: Optional[Callable[[str], str]] = None,
    chunk_word_target: int = 90,
    concepts: Optional[List[Dict[str, str]]] = None,
    session_chunk_counts: Optional[Dict[str, int]] = None,
    max_questions_per_chunk_session: int = _SESSION_MAX_MCQS_PER_CHUNK,
) -> List[Dict]:
    context_chunk_rows = _build_context_chunk_rows(context_text or "")

    # ── 1. Extract concepts ──────────────────────────────────────────────
    # Concepts are lecture-level topics, not storage chunks. Preserve up to 10
    # saved concepts, and ask for about 8 when extracting from this lecture.
    concept_list = (
        _normalize_concepts(concepts, max_concepts=10)
        if concepts is not None
        else generate_concepts_from_context(
            context_text, max_concepts=8, llm_call=llm_call
        )
    )

    # ── 2. Build diversity plan BEFORE prompting ─────────────────────────
    plan = _build_question_plan(concept_list, n)

    # ── Deterministic path (no LLM) ─────────────────────────────────────
    if llm_call is None:
        chunks = _paragraph_chunks(context_text or "", approx_words=chunk_word_target) or [
            "Placeholder paragraph about " + ("the topic" if not context_text else context_text[:60])
        ]
        out = []
        for i, ch in enumerate(chunks[:n]):
            row = plan[i]
            qid = f"q{i+1}"
            out.append({
                "id": qid,
                "question": f"Placeholder ({row['type_id']}): question about {row['concept_label']}?",
                "choices": {
                    "A": "A placeholder plausible choice about the paragraph.",
                    "B": "Another placeholder distractor that looks plausible.",
                    "C": "A third placeholder distractor that is plausible here.",
                    "D": "The correct placeholder answer is this one.",
                },
                "answer": "D",
                "brief_explanation": "Placeholder brief explanation.",
                "detailed_explanation": {
                    "why_correct": "Placeholder: D is correct because it best matches the paragraph.",
                    "why_wrong": {
                        "A": "Placeholder: A is a distractor.",
                        "B": "Placeholder: B is a distractor.",
                        "C": "Placeholder: C is a distractor.",
                        "D": "",
                    },
                    "source_chunk": ch[:300],
                },
                "concept_id": row["concept_id"],
                "concept_label": row["concept_label"],
                "question_type": row["type_id"],
                "_chunk_text_hint": ch,
            })
        return [_attach_chunk_metadata(item, context_chunk_rows) for item in out[:n]]

    session_counts = _normalize_session_chunk_counts(session_chunk_counts)
    max_cap = max(1, int(max_questions_per_chunk_session or _SESSION_MAX_MCQS_PER_CHUNK))

    # ── LLM path — BATCH ────────────────────────────────────────────────
    content_for_prompt = _shorten(context_text or "", max_len=6000)
    question_plan_str = _format_question_plan(plan)
    batch_prompt = _BATCH_MCQ_PROMPT_TEMPLATE.format(
        n=n,
        question_plan=question_plan_str,
        content=content_for_prompt,
    )

    out_mcqs: List[Dict] = []
    batch_succeeded = False

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            logger.info("Batch MCQ call — attempt %s of %s", attempt, MAX_RETRIES + 1)
            raw = llm_call(batch_prompt)
            if raw is None:
                raise RuntimeError("LLM returned None")

            parsed = safe_json_load(raw)

            if isinstance(parsed, list):
                mcq_list = parsed
            elif isinstance(parsed, dict):
                mcq_list = parsed.get("questions") or parsed.get("mcqs") or [parsed]
            else:
                raise ValueError(f"Unexpected batch response type: {type(parsed)}")

            used_questions: set = set()
            for i, obj in enumerate(mcq_list):
                if len(out_mcqs) >= n:
                    break
                if not isinstance(obj, dict):
                    continue
                if not obj.get("id"):
                    obj["id"] = f"q{len(out_mcqs) + 1}"
                if "answer" in obj:
                    obj["answer"] = str(obj["answer"]).strip().upper()

                if not _validate_mcq_obj(obj):
                    logger.warning("Batch item %s failed validation — skipping", i)
                    continue

                q_text_norm = re.sub(r'\s+', ' ', obj["question"].strip().lower())
                if q_text_norm in used_questions:
                    continue
                used_questions.add(q_text_norm)

                obj["question"] = _shorten(str(obj["question"]), max_len=_QUIZ_QUESTION_MAX_LEN)
                for k in list(obj.get("choices", {}).keys()):
                    obj["choices"][k] = _shorten(str(obj["choices"][k]), max_len=_QUIZ_CHOICE_MAX_LEN)

                expl_fields = _extract_explanation_fields(obj, obj["answer"])

                plan_row = plan[len(out_mcqs)] if len(out_mcqs) < len(plan) else plan[-1]

                out_mcqs.append({
                    "id": obj["id"],
                    "question": obj["question"],
                    "choices": obj["choices"],
                    "answer": obj["answer"],
                    **expl_fields,
                    "concept_id": plan_row["concept_id"],
                    "concept_label": plan_row["concept_label"],
                    "question_type": plan_row["type_id"],
                })

            if out_mcqs:
                batch_succeeded = True
                break

            raise ValueError("Batch response contained no valid MCQ items")

        except Exception as e:
            if _is_rate_limit_error(e):
                delay = _extract_retry_delay(e, default=15.0)
                logger.error("Quota exhausted — waiting %.1fs then aborting. Error: %s", delay, e)
                time.sleep(delay)
                break

            if _is_overload_error(e):
                # FIX v6: cap the 503 wait so it never snowballs (was * attempt with no cap)
                wait = min(BACKOFF_503 * attempt, BACKOFF_503_MAX)
                logger.warning("Model overloaded (503) on batch attempt %s — backing off %.1fs.", attempt, wait)
                if attempt <= MAX_RETRIES:
                    time.sleep(wait)
                    continue
                else:
                    break

            logger.warning("Batch attempt %s failed: %s", attempt, e)
            if attempt <= MAX_RETRIES:
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
                logger.info("Backing off %.1fs before retry.", wait)
                time.sleep(wait)

    # ── Fallback: per-chunk (embedding chunks first — aligns with chunk_id diversity) ──
    if len(out_mcqs) < n:
        if not batch_succeeded:
            logger.info("Batch failed — falling back to per-chunk for remaining %s items.", n - len(out_mcqs))
        else:
            logger.info("Batch short — per-chunk fill for remaining %s item(s).", n - len(out_mcqs))

        usage_for_order: Dict[str, int] = dict(session_counts)
        for q in out_mcqs:
            tmp = _assemble_and_attach([q], context_chunk_rows)
            if tmp:
                ck = _chunk_key(tmp[0])
                if ck:
                    usage_for_order[ck] = usage_for_order.get(ck, 0) + 1

        ordered_rows = _ordered_embedding_rows(context_chunk_rows, usage_for_order)
        para_chunks = _paragraph_chunks(context_text or "", approx_words=chunk_word_target)
        used_questions_fb: Set[str] = {_normalize_question_key(q["question"]) for q in out_mcqs}

        fallback_q_index = len(out_mcqs)
        needed = n - len(out_mcqs)
        max_attempts = max(needed * 3, 15)
        attempts_used = 0

        def _embedding_pass(rows: List[Dict[str, str]]) -> None:
            nonlocal fallback_q_index, attempts_used
            for row in rows:
                if len(out_mcqs) >= n or attempts_used >= max_attempts:
                    return
                chunk = _clean_ws(row.get("chunk_text") or "")
                if not chunk:
                    continue
                cid_pre = _clean_ws(row.get("chunk_id") or "").lower()
                if cid_pre and usage_for_order.get(cid_pre, 0) >= max_cap:
                    continue
                plan_row = plan[min(fallback_q_index, len(plan) - 1)]
                attempts_used += 1
                got = _one_chunk_mcq_llm(
                    llm_call, chunk, plan_row, len(out_mcqs) + 1, used_questions_fb,
                )
                if not got:
                    continue
                tmp_att = _assemble_and_attach([got], context_chunk_rows)
                cid_a = _chunk_key(tmp_att[0]) if tmp_att else ""
                if cid_a and usage_for_order.get(cid_a, 0) >= max_cap:
                    continue
                out_mcqs.append(got)
                if cid_a:
                    usage_for_order[cid_a] = usage_for_order.get(cid_a, 0) + 1
                fallback_q_index += 1

        _embedding_pass(ordered_rows)

        for chunk in para_chunks:
            if len(out_mcqs) >= n or attempts_used >= max_attempts:
                break
            plan_row = plan[min(fallback_q_index, len(plan) - 1)]
            attempts_used += 1
            got = _one_chunk_mcq_llm(
                llm_call, chunk, plan_row, len(out_mcqs) + 1, used_questions_fb,
            )
            if not got:
                continue
            tmp_att = _assemble_and_attach([got], context_chunk_rows)
            cid_a = _chunk_key(tmp_att[0]) if tmp_att else ""
            if cid_a and usage_for_order.get(cid_a, 0) >= max_cap:
                continue
            out_mcqs.append(got)
            if cid_a:
                usage_for_order[cid_a] = usage_for_order.get(cid_a, 0) + 1
            fallback_q_index += 1

    # ── Attach metadata + session chunk budget (prefer unused chunks first) ───────────
    cleaned_all = _assemble_and_attach(out_mcqs, context_chunk_rows)
    selected = _pick_with_chunk_budget(cleaned_all, n, session_counts, max_cap)

    deficit = n - len(selected)
    if deficit > 0 and llm_call is not None:
        usage_live = dict(session_counts)
        for it in selected:
            ck = _chunk_key(it)
            if ck:
                usage_live[ck] = usage_live.get(ck, 0) + 1
        used_qnorm = {_normalize_question_key(it.get("question") or "") for it in cleaned_all}
        fb_idx = len(selected)

        ordered_rows2 = _ordered_embedding_rows(context_chunk_rows, usage_live)
        max_attempts2 = max(deficit * 4, 12)
        attempts2 = 0
        extras: List[Dict] = []

        def _topup_pass(rows: List[Dict[str, str]]) -> None:
            nonlocal fb_idx, attempts2
            for row in rows:
                if len(extras) >= deficit or attempts2 >= max_attempts2:
                    return
                chunk = _clean_ws(row.get("chunk_text") or "")
                if not chunk:
                    continue
                cid_pre = _clean_ws(row.get("chunk_id") or "").lower()
                if cid_pre and usage_live.get(cid_pre, 0) >= max_cap:
                    continue
                plan_row = plan[min(fb_idx, len(plan) - 1)]
                attempts2 += 1
                raw_got = _one_chunk_mcq_llm(
                    llm_call, chunk, plan_row, len(selected) + len(extras) + 1, used_qnorm,
                )
                if not raw_got:
                    continue
                attached_list = _assemble_and_attach([raw_got], context_chunk_rows)
                if not attached_list:
                    continue
                att = attached_list[0]
                ck_a = _chunk_key(att)
                if ck_a and usage_live.get(ck_a, 0) >= max_cap:
                    continue
                extras.append(att)
                if ck_a:
                    usage_live[ck_a] = usage_live.get(ck_a, 0) + 1
                qn = _normalize_question_key(att.get("question") or "")
                if qn:
                    used_qnorm.add(qn)
                fb_idx += 1

        _topup_pass(ordered_rows2)

        for ch in _paragraph_chunks(context_text or "", approx_words=chunk_word_target):
            if len(extras) >= deficit or attempts2 >= max_attempts2:
                break
            plan_row = plan[min(fb_idx, len(plan) - 1)]
            attempts2 += 1
            raw_got = _one_chunk_mcq_llm(
                llm_call, ch, plan_row, len(selected) + len(extras) + 1, used_qnorm,
            )
            if not raw_got:
                continue
            attached_list = _assemble_and_attach([raw_got], context_chunk_rows)
            if not attached_list:
                continue
            att = attached_list[0]
            ck_a = _chunk_key(att)
            if ck_a and usage_live.get(ck_a, 0) >= max_cap:
                continue
            extras.append(att)
            if ck_a:
                usage_live[ck_a] = usage_live.get(ck_a, 0) + 1
            qn = _normalize_question_key(att.get("question") or "")
            if qn:
                used_qnorm.add(qn)
            fb_idx += 1

        selected = selected + extras[:deficit]

    return selected[:n]


# ── backwards-compatible wrapper (unchanged signature) ───────────────────────
def generate_quiz_from_context(
    stem: str,
    context_text: str,
    n: int = 5,
    llm_call: Optional[Callable[[str], str]] = None,
    retries: int = MAX_RETRIES,
) -> Dict:
    try:
        mcqs = generate_mcq_from_context(context_text, n=n, llm_call=llm_call)
    except TypeError:
        mcqs = generate_mcq_from_context(context_text, n=n)

    questions = []
    for i, q in enumerate(mcqs):
        qid = q.get("id") or f"{stem}_q{i+1}"
        question_text = q.get("question", "")
        answer_label = q.get("answer", "")
        choices = q.get("choices", {}) or {}
        answer_text = choices.get(answer_label, "") or q.get("answer", "")
        difficulty = q.get("difficulty", "medium")
        questions.append({
            "id": qid,
            "mcq_id": q.get("mcq_id") or qid,
            "chunk_id": q.get("chunk_id", ""),
            "question": question_text,
            "answer": answer_text,
            "difficulty": difficulty,
        })
    return {"stem": stem, "questions": questions}


if __name__ == "__main__":
    import json

    def fake_llm(prompt: str) -> str:
        if "canonical concepts" in prompt:
            return json.dumps([
                {"concept_id": "field_definition",    "concept_label": "Definition of Psychology & Economics"},
                {"concept_id": "homo_economicus",     "concept_label": "Homo Economicus assumptions"},
                {"concept_id": "rationality_critique","concept_label": "Critique of rationality"},
                {"concept_id": "real_world_examples", "concept_label": "Real-world examples (GlowCaps)"},
                {"concept_id": "model_properties",    "concept_label": "Properties of a good model"},
            ])
        demo = [
            {
                "id": "q1", "question": "What is the primary goal of Psychology and Economics?",
                "choices": {"A": "To replace economics with psychology.", "B": "To use insights from other fields to improve economic models.", "C": "To study only psychological factors.", "D": "To prove rationality is always correct."},
                "answer": "B",
                "brief_explanation": "The field combines both disciplines to improve predictive power.",
                "detailed_explanation": {"why_correct": "B is correct.", "why_wrong": {"A": "Wrong.", "B": "", "C": "Wrong.", "D": "Wrong."}, "source_chunk": "use insights from other fields"},
            },
            {
                "id": "q2", "question": "Which assumption does Homo Economicus NOT make?",
                "choices": {"A": "Maximises expected utility.", "B": "Processes info as a Bayesian.", "C": "Is driven by preferences over changes.", "D": "Is perfectly selfless."},
                "answer": "D",
                "brief_explanation": "Homo Economicus is self-interested, not selfless.",
                "detailed_explanation": {"why_correct": "D is correct.", "why_wrong": {"A": "Wrong.", "B": "Wrong.", "C": "Wrong.", "D": ""}, "source_chunk": "self-interested"},
            },
            {
                "id": "q3", "question": "GlowCaps medication reminders illustrate a challenge to which assumption?",
                "choices": {"A": "Self-interest.", "B": "Optimal information processing.", "C": "Perfect willpower.", "D": "Stable preferences."},
                "answer": "C",
                "brief_explanation": "GlowCaps exist because people lack perfect self-control.",
                "detailed_explanation": {"why_correct": "C is correct.", "why_wrong": {"A": "Wrong.", "B": "Wrong.", "C": "", "D": "Wrong."}, "source_chunk": "willpower"},
            },
            {
                "id": "q4", "question": "Why do researchers consider the classical economic model too extreme?",
                "choices": {"A": "It assumes people are too selfless.", "B": "It overestimates rationality.", "C": "It ignores economic incentives.", "D": "It focuses on sociology."},
                "answer": "B",
                "brief_explanation": "People make predictable mistakes that the model ignores.",
                "detailed_explanation": {"why_correct": "B is correct.", "why_wrong": {"A": "Wrong.", "B": "", "C": "Wrong.", "D": "Wrong."}, "source_chunk": "predictable mistakes"},
            },
            {
                "id": "q5", "question": "What property of a good model does Gabaix and Laibson (2008) emphasise?",
                "choices": {"A": "Complexity.", "B": "Exact replication of reality.", "C": "Exclusive use of psychology.", "D": "Parsimony."},
                "answer": "D",
                "brief_explanation": "A good model is parsimonious — simple yet powerful.",
                "detailed_explanation": {"why_correct": "D is correct.", "why_wrong": {"A": "Wrong.", "B": "Wrong.", "C": "Wrong.", "D": ""}, "source_chunk": "parsimony"},
            },
        ]
        return json.dumps(demo)

    sample = (
        "Psychology and Economics is the field that studies the joint influence of psychological "
        "and economic factors on behaviour. Its goal is to use insights from other fields to make "
        "economic models more realistic. The classical model assumes Homo Economicus: a perfectly "
        "rational, self-interested agent who maximises expected utility. Critics argue this is too "
        "extreme because real people make predictable mistakes. GlowCaps remind users to take "
        "medication, illustrating imperfect willpower. Gabaix and Laibson (2008) argue good models "
        "require parsimony."
    )

    print("=== Diversity plan ===")
    concepts = [
        {"concept_id": "field_definition",    "concept_label": "Definition of Psychology & Economics"},
        {"concept_id": "homo_economicus",     "concept_label": "Homo Economicus assumptions"},
        {"concept_id": "rationality_critique","concept_label": "Critique of rationality"},
        {"concept_id": "real_world_examples", "concept_label": "Real-world examples (GlowCaps)"},
        {"concept_id": "model_properties",    "concept_label": "Properties of a good model"},
    ]
    plan = _build_question_plan(concepts, n=5)
    for i, row in enumerate(plan, 1):
        print(f"  Q{i}: [{row['type_id']:12s}] {row['concept_label']}")

    print("\n=== Generated MCQs ===")
    result = generate_mcq_from_context(sample, n=5, llm_call=fake_llm)
    for q in result:
        print(f"  {q['id']} [{q.get('question_type','?'):12s}] ({q.get('concept_label','?')}) — {q['question'][:70]}")
