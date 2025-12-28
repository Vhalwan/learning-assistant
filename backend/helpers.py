# backend/helpers.py
import json
import os
from typing import Any

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200):
    """
    Break text into overlapping chunks.
    - max_chars: target chunk size (chars)
    - overlap: number of chars to overlap between chunks
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        # try to backtrack to last space for a better split
        if end < n:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # advance start safely
        if end >= n:
            break
        new_start = end - overlap
        if new_start <= start:
            start = end
        else:
            start = new_start

    return chunks
