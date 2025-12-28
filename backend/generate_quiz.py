# backend/generate_quiz.py
import json
import os
from typing import List, Dict, Callable, Optional

DEFAULT_OUTPUT_DIR = "data/processed"

def generate_quiz_from_context(stem: str, context_text: str, n: int = 5, llm_call: Optional[Callable[[str], str]] = None) -> Dict:
    """
    Generate a quiz (list of flashcards) from context_text.

    - stem: filename-stem used for saving (e.g. 'lecture2')
    - context_text: text to summarize / create questions from
    - n: number of questions
    - llm_call: optional callable(prompt) -> JSON-string | dict. If provided, it's used
      to obtain quiz content. If None, produce a deterministic placeholder quiz.

    Returns the quiz dict and writes file to data/processed/<stem>_quiz.json
    """
    if llm_call is None:
        # Placeholder for quiz offline / tests
        quiz = {"stem": stem, "questions": []}
        for i in range(n):
            quiz["questions"].append({
                "id": f"{stem}_q{i+1}",
                "question": f"Placeholder question {i+1} about {stem}",
                "answer": f"Placeholder answer {i+1}"
            })
    else:
        prompt = f"Create {n} short quiz items (question + answer) from the following context:\n\n{context_text}\n\nReturn JSON with shape {{'questions': [{{'id':..., 'question':..., 'answer':...}}]}}."
        resp = llm_call(prompt)
        if isinstance(resp, str):
            parsed = json.loads(resp)
        elif isinstance(resp, dict):
            parsed = resp
        else:
            raise ValueError("llm_call must return JSON string or dict")
        # basic validation
        questions = parsed.get("questions")
        if not isinstance(questions, list):
            raise ValueError("llm_call response missing 'questions' list")
        quiz = {"stem": stem, "questions": questions}

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{stem}_quiz.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(quiz, f, ensure_ascii=False, indent=2)

    return quiz
