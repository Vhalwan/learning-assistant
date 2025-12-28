# backend/create_embeddings.py
import os
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import numpy as np

from backend.helpers import chunk_text

load_dotenv()

# Config and defaults
INPUT_TXT = "data/processed/lecture1.txt"
OUTPUT_EMB = "data/processed/lecture1_embeddings.json"
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

# Imports the provider factory and deterministic_vector from embeddings_provider
from backend.embeddings_provider import get_embedding_provider, deterministic_vector

def create_embeddings_for_text(
    text: str,
    output_path: str,
    max_chars: int = 2000,
    overlap: int = 200,
    dim: int = EMBED_DIM,
    batch_size: int = 32
) -> List[Dict]:
    """
    Chunk text, generate embeddings (safe-mode or real API via provider), and save JSON.
    Returns list of rows: [{"id":..., "text":..., "embedding":...}]
    """
    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
    ids = []
    texts = []
    for i, c in enumerate(chunks):
        ids.append(f"uploaded_chunk_{i}")
        texts.append(c)

    provider = get_embedding_provider()

    # provider.embed returns normalized vectors as list[list[float]]
    vecs = provider.embed(texts, dim=dim, batch_size=batch_size)
    if len(vecs) != len(texts):
        raise RuntimeError("Embedding provider returned mismatched length")

    rows = []
    for id_, txt, emb in zip(ids, texts, vecs):
        # make sure embed is python list of floats
        rows.append({"id": id_, "text": txt, "embedding": [float(x) for x in emb]})

    outdir = os.path.dirname(output_path) or "."
    os.makedirs(outdir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows


def load_embeddings(path: str) -> Tuple[List[str], List[str], np.ndarray]:
    """Load embeddings JSON into ids, texts, vectors (np.ndarray)."""
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    vecs = np.array([r["embedding"] for r in rows], dtype=np.float32)
    return ids, texts, vecs


if __name__ == "__main__":
    if not os.path.exists(INPUT_TXT):
        raise SystemExit(f"Input text not found: {INPUT_TXT}. Run the PDF extractor first.")

    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        text = f.read()

    rows = create_embeddings_for_text(text, OUTPUT_EMB)
    print(f"Saved {len(rows)} embeddings to {OUTPUT_EMB}")
