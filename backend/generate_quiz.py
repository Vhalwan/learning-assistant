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
    Generate a quiz (list of flashcards) from context_text.

    - stem: filename-stem used for saving (e.g. 'lecture2')
    - context_text: text to summarize / create questions from
    - n: number of questions
    - llm_call: optional callable(prompt) -> JSON-string | dict. If provided, it's used
      to obtain quiz content. If None, produce a deterministic placeholder quiz.
    - returns quiz dict and writes file to data/processed/<stem>_quiz.json
    """
    if llm_call is None:
        # local deterministic placeholder
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

    # persist
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{stem}_quiz.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(quiz, f, ensure_ascii=False, indent=2)

    return quiz
