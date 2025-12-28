# build_faiss_index.py
from backend.vectorstore.faiss_store import build_faiss_index, load_faiss_index, search_faiss
from backend.embeddings_provider import deterministic_vector
import os

emb_path = os.path.join("data", "processed", "lecture2_embeddings.json")
index_path = os.path.join("data", "processed", "lecture2_embeddings.index")

print("Building FAISS index from:", emb_path)
build_faiss_index(emb_path, index_path)
print("Built index:", index_path)

# Quick smoke test
index, metas = load_faiss_index(index_path)
q = deterministic_vector("sample query")
results = search_faiss(index, metas, q, k=3)
print("Top results (score, id):")
for score, meta, idx in results:
    print(score, meta.get("id"))
