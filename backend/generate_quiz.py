# backend/generate_quiz.py
import json
import os
import time
from typing import List, Dict, Callable, Optional

DEFAULT_OUTPUT_DIR = "data/processed"
MAX_RETRIES = 2
BACKOFF_BASE = 0.5

def generate_quiz_from_context(stem: str, context_text: str, n: int = 5,
                               llm_call: Optional[Callable[[str], str]] = None,
                               retries: int = MAX_RETRIES) -> Dict:
    """
    Backwards-compatible quiz generator kept for local / legacy usage.
    NOTE: This function no longer writes files to disk (Quiz v1 requirement).

    Returns a dict: {"stem": <stem>, "questions": [ {id, question, answer, difficulty}, ... ] }
    This is a simple, deterministic placeholder when llm_call is None; otherwise it attempts
    to parse the LLM response the way the original implementation did — but still does NOT write files.
    """
    if llm_call is None:
        # local deterministic placeholder (no disk writes)
        quiz = {"stem": stem, "questions": []}
        for i in range(n):
            quiz["questions"].append({
                "id": f"{stem}_q{i+1}",
                "question": f"Placeholder question {i+1} about {stem}",
                "answer": f"Placeholder answer {i+1}",
                "difficulty": "medium"
            })
    else:
        prompt = (
            f"Create {n} short quiz items (question + answer) from the following context.\n\n"
            "Return valid JSON with a top-level key 'questions' which is a list of objects "
            "each having: id (string), question (string), answer (string), difficulty (easy|medium|hard).\n\n"
            f"Context:\n{context_text}\n\nReturn JSON ONLY."
        )
        attempt = 0
        parsed = None
        last_exc = None
        while attempt <= retries:
            try:
                resp = llm_call(prompt)
                if isinstance(resp, dict):
                    parsed = resp
                elif isinstance(resp, str):
                    parsed = json.loads(resp)
                else:
                    raise ValueError("llm_call returned unsupported type")
                # basic schema validation
                questions = parsed.get("questions")
                if not isinstance(questions, list):
                    raise ValueError("Response missing 'questions' list")
                # enforce n or fewer items
                if len(questions) > n:
                    questions = questions[:n]
                # normalize items
                out_qs = []
                for i, q in enumerate(questions):
                    if not isinstance(q, dict):
                        continue
                    out_qs.append({
                        "id": q.get("id") or f"{stem}_q{i+1}",
                        "question": q.get("question") or q.get("prompt") or "",
                        "answer": q.get("answer") or q.get("solution") or "",
                        "difficulty": q.get("difficulty") or "medium"
                    })
                quiz = {"stem": stem, "questions": out_qs}
                break
            except Exception as e:
                last_exc = e
                attempt += 1
                if attempt > retries:
                    raise RuntimeError(f"LLM quiz generation failed after {retries} retries: {e}")
                sleep = BACKOFF_BASE * (2 ** (attempt - 1))
                time.sleep(sleep)

    # Return quiz dict (no file writing)
    return quiz


# --- New function required by Quiz v1 spec ---
import random
import re

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "these", "those", "from", "which", "what",
    "when", "where", "were", "have", "has", "had", "are", "is", "was", "been", "but", "not",
    "their", "they", "them", "can", "could", "would", "should", "will", "shall", "about", "into",
    "over", "under", "between", "within", "also", "such", "other", "than", "then", "there", "here"
}


def _words_from_text(text: str) -> List[str]:
    # return words of length >=3, preserve order, simple normalization
    found = re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", text)
    out = []
    seen = set()
    for w in found:
        lw = w.strip()
        if lw.lower() in _STOPWORDS:
            continue
        if lw.lower() in seen:
            continue
        seen.add(lw.lower())
        out.append(lw)
    return out


def generate_mcq_from_context(context_text: str, n: int = 5) -> List[Dict]:
    """
    Generate up to `n` multiple-choice questions (MCQs) from the provided context_text.

    Each MCQ is a dict:
    {
      "id": "q1",
      "question": "...",
      "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer": "B"   # letter of correct choice
    }

    This implementation is deterministic: it uses the context to extract sentences and
    keywords and produces statement-style MCQs. It's simple and offline-friendly (no LLM calls).
    """
    # split into candidate sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context_text) if len(s.strip()) >= 30]
    if not sentences:
        # fallback: use whole context as single sentence
        sentences = [context_text.strip()[:400]]

    words = _words_from_text(context_text)
    # ensure at least some filler words for distractors
    if len(words) < 4:
        # fall back to naive splitting to get tokens
        tokens = re.findall(r"\b[A-Za-z]{3,}\b", context_text)
        for t in tokens:
            if t.lower() not in _STOPWORDS and t not in words:
                words.append(t)
            if len(words) >= 8:
                break

    out_questions: List[Dict] = []
    # use a deterministic seed derived from the context so repeated runs are stable
    base_seed = abs(hash(context_text)) % (2 ** 32)
    for i in range(n):
        q_idx = i + 1
        # pick sentence deterministically
        sent = sentences[i % len(sentences)]
        # choose a keyword from the sentence (first non-stopword token)
        sent_words = _words_from_text(sent)
        if sent_words:
            key = sent_words[0]
        elif words:
            key = words[i % len(words)]
        else:
            key = None

        # construct the correct statement (trim to reasonable length)
        correct_stmt = sent if len(sent) <= 300 else sent[:297] + "..."
        # create up to 3 distractors by replacing the keyword with other content words when possible
        candidates = [w for w in words if (not key or w.lower() != key.lower())]
        rng = random.Random(base_seed + i)  # deterministic per question
        rng.shuffle(candidates)

        distractors: List[str] = []
        for j in range(3):
            if j < len(candidates) and key:
                # replace first occurrence of key (case-insensitive)
                repl = candidates[j]
                # simple safe replacement
                pattern = re.compile(re.escape(key), flags=re.IGNORECASE)
                new_stmt = pattern.sub(repl, correct_stmt, count=1)
                # avoid producing identical to correct
                if new_stmt.strip() == correct_stmt.strip():
                    new_stmt = f"{correct_stmt} (not {repl})"
                distractors.append(new_stmt)
            elif j < len(candidates):
                distractors.append(f"{correct_stmt} (incorrect variant: {candidates[j]})")
            else:
                distractors.append("This statement is not supported by the lecture text.")

        # assemble choices and shuffle deterministically
        choices_list = [correct_stmt] + distractors[:3]
        rng.shuffle(choices_list)
        labels = ["A", "B", "C", "D"]
        choices = {label: text for label, text in zip(labels, choices_list)}

        # find correct label
        correct_label = next((k for k, v in choices.items() if v == correct_stmt), "A")

        out_questions.append({
            "id": f"q{q_idx}",
            "question": "Which of the following statements is correct according to the context?",
            "choices": choices,
            "answer": correct_label,
        })

    return out_questions
