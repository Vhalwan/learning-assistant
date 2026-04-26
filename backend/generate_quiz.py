# backend/generate_quiz.py
"""
generate_quiz.py — Hybrid MCQ generator (strict extractor + LLM writer)

Changes vs v3:
  - _BATCH_MCQ_PROMPT_TEMPLATE now requests two explanation fields:
      brief_explanation  : 1-2 sentence plain summary (shown inline after answer)
      detailed_explanation: {why_correct, why_wrong: {A,B,C,D}, source_chunk}
  - _PER_CHUNK_PROMPT (fallback) mirrors the same new fields.
  - source_chunk is tracked per-item through the batch and fallback paths.
  - Final assembly preserves all new fields.
  - All other logic (retry, validation, quota handling) is unchanged.
"""

import json
import random
import re
import time
import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
BACKOFF_BASE = 0.4
BACKOFF_503 = 10.0

_MIN_SENT_WORDS = 8
_MIN_CHOICE_LEN = 8
_MIN_QUESTION_LEN = 12

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

_CONCEPT_PROMPT_TEMPLATE = """
You are an expert educator. Extract 2 to {max_concepts} canonical concepts from the lecture content when possible.
Return JSON ONLY (no extra text). Output must be a JSON array of objects with keys:
  - concept_id (snake_case, short, stable)
  - concept_label (human-readable)

Lecture content:
\"\"\"{content}\"\"\"
"""

# ── CHANGED: batch prompt now requests brief + detailed explanation fields ──
_BATCH_MCQ_PROMPT_TEMPLATE = """
You are an expert quiz writer. Generate exactly {n} high-quality multiple-choice questions (MCQs)
that test a learner's understanding of the lecture content below.

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

Lecture content:
\"\"\"{content}\"\"\"
"""

# ── CHANGED: per-chunk fallback prompt mirrors new fields ──
_PER_CHUNK_PROMPT_TEMPLATE = """
You are an expert quiz writer. Produce ONE high-quality multiple-choice question (MCQ) that tests
a learner's understanding of the provided paragraph. Return JSON ONLY (no extra text).
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


def _normalize_concepts(raw, max_concepts: int = 5) -> List[Dict[str, str]]:
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
    max_concepts: int = 5,
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


# ── NEW: helper to normalise the detailed_explanation block from the LLM ──
def _normalise_detailed_explanation(raw_detail, answer_letter: str) -> Dict:
    """
    Coerce whatever the LLM returned for detailed_explanation into a clean dict:
      {why_correct, why_wrong: {A,B,C,D}, source_chunk}
    Gracefully handles missing / malformed values.
    """
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
            why_wrong[letter] = ""   # correct letter — no "why wrong" needed
        else:
            why_wrong[letter] = str(raw_ww.get(letter) or "").strip()

    return {
        "why_correct": why_correct,
        "why_wrong": why_wrong,
        "source_chunk": source_chunk,
    }


# ── NEW: helper to extract & clean all explanation fields from one MCQ obj ──
def _extract_explanation_fields(obj: dict, answer_letter: str) -> Dict:
    """
    Returns dict with:
      brief_explanation    : str  (falls back to legacy 'explanation' key)
      detailed_explanation : dict (normalised)
    """
    # brief — prefer new key, fall back to legacy 'explanation'
    brief = (
        str(obj.get("brief_explanation") or obj.get("explanation") or "").strip()
    )
    # detailed
    raw_detail = obj.get("detailed_explanation")
    detailed = _normalise_detailed_explanation(raw_detail, answer_letter)

    # NOTE: do NOT promote brief into why_correct.
    # brief_explanation is shown inline (outside the expander).
    # detailed_explanation lives exclusively inside the expander.
    # Keeping them separate enforces the strict outside/inside boundary.

    return {
        "brief_explanation": brief,
        "detailed_explanation": detailed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core generator  (v4 — new explanation fields, otherwise identical to v3)
# ─────────────────────────────────────────────────────────────────────────────
def generate_mcq_from_context(
    context_text: str,
    n: int = 5,
    llm_call: Optional[Callable[[str], str]] = None,
    chunk_word_target: int = 90,
    concepts: Optional[List[Dict[str, str]]] = None,
) -> List[Dict]:
    concept_list = (
        _normalize_concepts(concepts, max_concepts=5)
        if concepts is not None
        else generate_concepts_from_context(context_text, max_concepts=5, llm_call=llm_call)
    )

    # ── Deterministic path (no LLM) ─────────────────────────────────────
    if llm_call is None:
        chunks = _paragraph_chunks(context_text or "", approx_words=chunk_word_target) or [
            "Placeholder paragraph about " + ("the topic" if not context_text else context_text[:60])
        ]
        out = []
        for i, ch in enumerate(chunks[:n]):
            qid = f"q{i+1}"
            out.append({
                "id": qid,
                "question": "Placeholder: which statement best summarizes the paragraph?",
                "choices": {
                    "A": "A placeholder plausible choice about the paragraph.",
                    "B": "Another placeholder distractor that looks plausible.",
                    "C": "A third placeholder distractor that is plausible here.",
                    "D": "The correct placeholder answer is this one.",
                },
                "answer": "D",
                # ── new fields with placeholder content ──
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
            })
        for idx, itm in enumerate(out):
            concept = concept_list[idx % len(concept_list)]
            itm["concept_id"] = concept["concept_id"]
            itm["concept_label"] = concept["concept_label"]
        return out[:n]

    # ── LLM path — BATCH ────────────────────────────────────────────────
    content_for_prompt = _shorten(context_text or "", max_len=6000)
    batch_prompt = _BATCH_MCQ_PROMPT_TEMPLATE.format(n=n, content=content_for_prompt)

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

                obj["question"] = _shorten(str(obj["question"]), max_len=240)
                for k in list(obj.get("choices", {}).keys()):
                    obj["choices"][k] = _shorten(str(obj["choices"][k]), max_len=180)

                # ── extract new explanation fields ──
                expl_fields = _extract_explanation_fields(obj, obj["answer"])

                out_mcqs.append({
                    "id": obj["id"],
                    "question": obj["question"],
                    "choices": obj["choices"],
                    "answer": obj["answer"],
                    **expl_fields,   # brief_explanation + detailed_explanation
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
                logger.warning("Model overloaded (503) on batch attempt %s — backing off.", attempt)
                if attempt <= MAX_RETRIES:
                    time.sleep(BACKOFF_503 * attempt)
                    continue
                else:
                    break

            logger.warning("Batch attempt %s failed: %s", attempt, e)
            if attempt <= MAX_RETRIES:
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
                logger.info("Backing off %.1fs before retry.", wait)
                time.sleep(wait)

    # ── Fallback: per-chunk ──────────────────────────────────────────────
    if not batch_succeeded and len(out_mcqs) < n:
        logger.info("Batch failed — falling back to per-chunk for remaining %s items.", n - len(out_mcqs))
        chunks = _paragraph_chunks(context_text or "", approx_words=chunk_word_target)
        needed = n - len(out_mcqs)
        max_attempts = needed * 2
        attempts_used = 0
        used_questions_fb: set = {re.sub(r'\s+', ' ', q["question"].strip().lower()) for q in out_mcqs}

        for chunk in chunks:
            if len(out_mcqs) >= n or attempts_used >= max_attempts:
                break

            prompt = _PER_CHUNK_PROMPT_TEMPLATE.format(paragraph=_shorten(chunk, max_len=2000))
            for attempt in range(1, MAX_RETRIES + 2):
                attempts_used += 1
                try:
                    logger.info("Fallback chunk call — attempt %s", attempts_used)
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
                        obj["id"] = f"q{len(out_mcqs) + 1}"
                    if "answer" in obj:
                        obj["answer"] = str(obj["answer"]).strip().upper()

                    if not _validate_mcq_obj(obj, chunk):
                        raise ValueError("MCQ failed validation")

                    q_norm = re.sub(r'\s+', ' ', obj["question"].strip().lower())
                    if q_norm in used_questions_fb:
                        break
                    used_questions_fb.add(q_norm)

                    obj["question"] = _shorten(str(obj["question"]), max_len=240)
                    for k in list(obj.get("choices", {}).keys()):
                        obj["choices"][k] = _shorten(str(obj["choices"][k]), max_len=180)

                    # ── extract new explanation fields ──
                    expl_fields = _extract_explanation_fields(obj, obj["answer"])

                    out_mcqs.append({
                        "id": obj["id"],
                        "question": obj["question"],
                        "choices": obj["choices"],
                        "answer": obj["answer"],
                        **expl_fields,
                    })
                    break

                except Exception as e:
                    if _is_rate_limit_error(e):
                        delay = _extract_retry_delay(e, default=15.0)
                        logger.error("Quota exhausted in fallback — waiting %.1fs. Error: %s", delay, e)
                        time.sleep(delay)
                        chunks = []
                        break
                    if _is_overload_error(e):
                        logger.warning("Model overloaded (503) in fallback attempt %s.", attempts_used)
                        if attempt <= MAX_RETRIES:
                            time.sleep(BACKOFF_503 * attempt)
                            continue
                        break
                    logger.warning("Fallback attempt %s failed: %s", attempts_used, e)
                    if attempt <= MAX_RETRIES:
                        time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))
                    break

    # ── Final assembly ───────────────────────────────────────────────────
    cleaned = []
    for q in out_mcqs[:n]:
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
        }
        cleaned.append(base)

    for idx, itm in enumerate(cleaned):
        concept = concept_list[idx % len(concept_list)]
        itm["concept_id"] = concept["concept_id"]
        itm["concept_label"] = concept["concept_label"]

    return cleaned


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
            "question": question_text,
            "answer": answer_text,
            "difficulty": difficulty,
        })
    return {"stem": stem, "questions": questions}


if __name__ == "__main__":
    def fake_llm(prompt: str) -> str:
        demo = [
            {
                "id": "q1",
                "question": "What is the primary benefit of batch MCQ generation?",
                "choices": {
                    "A": "It uses more API calls for better quality.",
                    "B": "It reduces API calls to a single request per quiz.",
                    "C": "It generates questions one paragraph at a time.",
                    "D": "It avoids using any LLM at all.",
                },
                "answer": "B",
                "brief_explanation": "Batch generation sends one prompt for all questions, cutting API usage dramatically.",
                "detailed_explanation": {
                    "why_correct": "Option B is correct because the batch prompt requests all n questions in a single LLM call, reducing quota consumption from O(n) to O(1).",
                    "why_wrong": {
                        "A": "A is wrong — more API calls would increase, not reduce, quota usage.",
                        "B": "",
                        "C": "C describes the old per-chunk fallback strategy, not the primary batch approach.",
                        "D": "D is wrong — the batch path still uses the LLM; it just uses it once.",
                    },
                    "source_chunk": "all n MCQs requested in a SINGLE LLM call instead of one call per chunk.",
                },
            }
        ]
        return json.dumps(demo)

    sample = (
        "Information infrastructure is the set of systems that underpin digital services. "
        "The introduction chapter summarizes the key concepts of behavioral economics, "
        "including how people deviate from rational decision-making."
    )
    result = generate_mcq_from_context(sample, n=1, llm_call=fake_llm)
    print(json.dumps(result, indent=2))