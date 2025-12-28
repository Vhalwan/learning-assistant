# Learning Assistant — Gemini RAG + FAISS Study Tool

A learning assistant that helps students understand lecture PDFs using Retrieval-Augmented Generation (RAG), embeddings, and FAISS.  
Upload a lecture → extract and chunk text → embed → index → ask questions → get grounded AI answers.  
Future phases will add quiz generation and spaced-repetition support.

---

## Features
- Upload lecture PDFs
- Extract and chunk text
- Create embeddings (safe local or real Gemini)
- FAISS semantic index for fast search
- NumPy fallback retrieval
- Retrieval-Augmented Q&A showing retrieved chunks for transparency
- Works offline in SAFE mode (no API key required)
- Fully testable with mocks
- Future: quiz generation and SRS spaced repetition

---

## Tech Stack
- Python
- FAISS (optional, for fast vector search)
- NumPy
- (Optional) Google Gemini APIs for embeddings and LLMs
- PyPDF (or other PDF extraction)
- Streamlit (frontend)

---

## Project Structure
.env
requirements.txt
build_faiss_index.py

data/
raw/ # uploaded PDFs
processed/ # text chunks, embeddings JSON, index files, quizzes, study progress

backend/
init.py
extract_pdf.py # PDF -> plain text
helpers.py # chunk_text, save/load helpers
summarize_file.py # LLM summarization (Gemini)
summarize_file_safe.py # offline summarizer fallback
create_embeddings.py # chunk -> embeddings JSON (calls provider)
embeddings_provider.py # Safe + Gemini provider (Day-5)
rag_query.py # RAG orchestration (retrieval + LLM prompt)
generate_quiz.py # quiz generation scaffold
study_srs.py # spaced repetition scaffold
vectorstore/
faiss_store.py # build/load/search/rebuild/status helpers

test_create_embeddings.py
test_embeddings_provider.py
test_vectorstore.py
test_rag_query.py

frontend/
app.py # Streamlit UI (upload, embed, build index, query, quiz)

yaml
Copy code

---

## Environment (.env example)
Create a `.env` file at repo root with these variables (do not commit it):

Embedding mode
USE_SAFE_EMBEDDINGS=1 # 1 or 0 (default: 1). When 1 uses deterministic local embeddings.

Real provider keys (optional)
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_google_api_key_here # alternative env var fallback

Embedding model & dims
EMBED_MODEL=embedding-001
EMBED_DIM=1536

Optional custom endpoint for testing
EMBEDDING_API_URL=https://example.com/v1/embeddings

yaml
Copy code

---

## Installation

1. Create and activate a virtual environment:
python -m venv venv

Windows
venv\Scripts\activate

Mac/Linux
source venv/bin/activate

markdown
Copy code

2. Install dependencies:
pip install -r requirements.txt

sql
Copy code

3. (Optional) Install FAISS for local similarity search:
pip install faiss-cpu

yaml
Copy code

---

## Running the app

Start the Streamlit UI:
streamlit run frontend/app.py

yaml
Copy code

Typical workflow in the UI:
1. Upload PDF (data/raw/)
2. Extract preview and chunk
3. Toggle SAFE vs REAL embeddings
4. Create embeddings file (data/processed/<stem>_embeddings.json)
5. Build FAISS index (optional)
6. Ask a question (RAG) — retrieved chunks + LLM answer displayed
7. (Optional) Generate quiz and use SRS

---

## Tests

Run the test suite:
pytest -q

yaml
Copy code

Notes:
- Tests include provider normalization checks, FAISS build/search correctness, and RAG behavior with mocks.
- If FAISS is not installed, FAISS tests will skip; CI attempts to install `faiss-cpu` but tolerates failure.

---

## How the pipeline works (high level)
1. PDF -> extract plain text (`backend/extract_pdf.py`)
2. Text -> chunking (`backend/helpers.py`)
3. Chunks -> embeddings (`backend/create_embeddings.py` uses `embeddings_provider.py`)
4. (Optional) Embeddings -> FAISS index (`backend/vectorstore/faiss_store.py`)
5. Query -> embed query -> retrieve top-k chunks (FAISS or NumPy)
6. Combine retrieved chunks + question into a prompt -> send to LLM (`backend/rag_query.py`)
7. Show LLM answer and the supporting chunks for provenance

---
