# backend/helpers.py
def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200):
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
