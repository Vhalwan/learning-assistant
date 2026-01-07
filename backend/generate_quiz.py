# backend/generate_quiz.py
"""
generate_quiz.py — Hybrid MCQ generator (strict extractor + LLM writer)
This file expects an llm_call(prompt)->str callable to be passed in (or None to
use deterministic placeholders).
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

_MIN_SENT_WORDS = 8
_MIN_CHOICE_LEN = 8
_MIN_QUESTION_LEN = 12

_BAD_PHRASES = [
    "unrelated concept", "is an unrelated concept", "not asked here",
    "not asked in the lecture", "not supported by", "This statement is not supported"
]


# ---------- robust JSON loader ----------
def safe_json_load(s):
    """
    Robustly extract and parse the *first* JSON object/array in `s`.
    Accepts dict/list input and returns parsed object.
    Raises ValueError if no JSON can be parsed.
    """
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        raise ValueError("llm_call returned unsupported type")

    text = s.strip()

    # strip code fences commonly added by LLMs
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # find first balanced { ... } or [ ... ] while respecting quoted strings and escapes
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
            # last-ditch: replace single quotes with double quotes and try (dangerous but sometimes helpful)
            try:
                alt = sub.replace("'", '"')
                return json.loads(alt)
            except Exception:
                pass

    raise ValueError("Could not parse JSON from LLM output")


# ---------- small helpers ----------
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


_MCQ_PROMPT_TEMPLATE = """
You are an expert quiz writer. Produce ONE high-quality multiple-choice question (MCQ) that tests
a learner's understanding of the provided paragraph. Return JSON ONLY (no extra text).
JSON must be an object with keys: id, question, choices (A-D), answer (A-D), explanation.

Paragraph:
\"\"\"{paragraph}\"\"\"
"""


def _validate_mcq_obj(obj: dict, paragraph: str) -> bool:
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
        correct_text = str(choices[ans])
        overlap = _token_overlap_fraction(correct_text, paragraph)
        if overlap < 0.05 and overlap <= 0.0:
            return False
        for k in keys:
            if k == ans:
                continue
            if _token_overlap_fraction(str(choices[k]), correct_text) > 0.9:
                return False
        if "explanation" in obj:
            ex = str(obj["explanation"]).strip()
            if len(ex) > 500:
                return False
        return True
    except Exception:
        return False


# ---------- core generator ----------
def generate_mcq_from_context(context_text: str, n: int = 5,
                              llm_call: Optional[Callable[[str], str]] = None,
                              chunk_word_target: int = 90) -> List[Dict]:
    """
    Generate up to n MCQs. If llm_call is None -> deterministic placeholders.
    If llm_call provided -> use it for each chunk, validate JSON, and collect MCQs.
    """
    if llm_call is None:
        # deterministic placeholders for local dev/testing
        chunks = _paragraph_chunks(context_text or "", approx_words=chunk_word_target) or [
            "Placeholder paragraph about " + ("the topic" if not context_text else context_text[:60])
        ]
        out = []
        for i, ch in enumerate(chunks[:n]):
            qid = f"q{i+1}"
            q = "Placeholder: which statement best summarizes the paragraph?"
            choices = {
                "A": "A placeholder plausible choice about the paragraph.",
                "B": "Another placeholder distractor that looks plausible.",
                "C": "A third placeholder distractor that is plausible here.",
                "D": "The correct placeholder answer is this one."
            }
            out.append({"id": qid, "question": q, "choices": choices, "answer": "D", "explanation": "Placeholder explanation."})
        return [{"id": it["id"], "question": it["question"], "choices": it["choices"], "answer": it["answer"]} for it in out]

    # Use LLM
    chunks = _paragraph_chunks(context_text or "", approx_words=chunk_word_target)
    if not chunks:
        return []

    out_mcqs: List[Dict] = []
    used_questions = set()
    used_choice_texts = set()
    attempts = 0
    max_attempts = max(len(chunks) * (MAX_RETRIES + 1), n * 4)

    for i_chunk, chunk in enumerate(chunks):
        if len(out_mcqs) >= n:
            break
        prompt = _MCQ_PROMPT_TEMPLATE.format(paragraph=_shorten(chunk, max_len=2000))
        attempt = 0
        while attempt <= MAX_RETRIES:
            attempt += 1
            attempts += 1
            try:
                logger.info("Calling LLM for chunk %s (attempt %s)", i_chunk, attempt)
                raw = llm_call(prompt)
                if raw is None:
                    raise RuntimeError("LLM returned None")

                # log short preview for debugging (avoid huge logs)
                preview = str(raw)[:1000]
                logger.debug("LLM raw preview: %s", preview.replace("\n", " ")[:1000])

                # try to parse JSON
                parsed = safe_json_load(raw)

                # normalize possible shapes
                if isinstance(parsed, dict) and "question" in parsed and "choices" in parsed and "answer" in parsed:
                    mcq_obj = parsed
                else:
                    if isinstance(parsed, dict) and "questions" in parsed and isinstance(parsed["questions"], list) and parsed["questions"]:
                        mcq_obj = parsed["questions"][0]
                    elif isinstance(parsed, list) and parsed:
                        mcq_obj = parsed[0]
                    else:
                        raise ValueError("unexpected LLM JSON shape")

                if not mcq_obj.get("id"):
                    mcq_obj["id"] = f"q{len(out_mcqs) + 1}"

                if not _validate_mcq_obj(mcq_obj, chunk):
                    logger.warning("LLM returned MCQ failed validation for chunk %s on attempt %s: %s", i_chunk, attempt, mcq_obj)
                    if attempt <= MAX_RETRIES:
                        time.sleep(BACKOFF_BASE * attempt)
                        continue
                    else:
                        break

                q_text = re.sub(r'\s+', ' ', mcq_obj["question"].strip().lower())
                if q_text in used_questions:
                    logger.info("Duplicate question detected; skipping.")
                    break

                choice_concat = " ".join([re.sub(r'\s+', ' ', str(v).strip().lower()) for v in mcq_obj["choices"].values()])
                if choice_concat in used_choice_texts:
                    logger.info("Duplicate choices across quiz; skipping.")
                    break

                # trim
                mcq_obj["question"] = _shorten(str(mcq_obj["question"]), max_len=240)
                for k, v in mcq_obj["choices"].items():
                    mcq_obj["choices"][k] = _shorten(str(v), max_len=180)
                if "explanation" in mcq_obj:
                    mcq_obj["explanation"] = _shorten(str(mcq_obj["explanation"]), max_len=240)

                out_mcqs.append({
                    "id": mcq_obj["id"],
                    "question": mcq_obj["question"],
                    "choices": mcq_obj["choices"],
                    "answer": mcq_obj["answer"],
                    "explanation": mcq_obj.get("explanation", "")
                })
                used_questions.add(q_text)
                used_choice_texts.add(choice_concat)
                break

            except ValueError as e:
                logger.exception("Parsing/validation error on LLM output for chunk %s (attempt %s): %s", i_chunk, attempt, e)
                if attempt > MAX_RETRIES:
                    break
                time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))
                continue
            except Exception as e:
                logger.exception("LLM call error for chunk %s (attempt %s): %s", i_chunk, attempt, e)
                if attempt > MAX_RETRIES:
                    break
                time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))
                continue

        if attempts >= max_attempts:
            logger.warning("Reached max attempts (%s) while generating MCQs.", max_attempts)
            break

    # final normalize
    cleaned = []
    for q in out_mcqs[:n]:
        base = {
            "id": q["id"],
            "question": q["question"],
            "choices": q["choices"],
            "answer": q["answer"]
        }
        if q.get("explanation"):
            base["explanation"] = q["explanation"]
        cleaned.append(base)
    return cleaned


# Backwards-compatible wrapper
def generate_quiz_from_context(stem: str, context_text: str, n: int = 5,
                               llm_call: Optional[Callable[[str], str]] = None,
                               retries: int = MAX_RETRIES) -> Dict:
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
        answer_text = ""
        if isinstance(choices, dict) and answer_label and answer_label in choices:
            answer_text = choices.get(answer_label)
        else:
            answer_text = q.get("answer", "") or q.get("solution", "")
        difficulty = q.get("difficulty", "medium")
        questions.append({
            "id": qid,
            "question": question_text,
            "answer": answer_text,
            "difficulty": difficulty
        })
    return {"stem": stem, "questions": questions}


# quick demo
if __name__ == "__main__":
    def fake_llm(prompt: str) -> str:
        demo = {
            "id": "q1",
            "question": "Demo question?",
            "choices": {"A": "one", "B": "two", "C": "three", "D": "four"},
            "answer": "A",
            "explanation": "demo"
        }
        return json.dumps(demo)

    s = ("Information infrastructure is the set of systems ... "
         "The introduction chapter summarizes the concepts.")
    print(generate_mcq_from_context(s, n=2, llm_call=fake_llm))
