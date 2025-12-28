# Learning Assistant — Gemini RAG + FAISS Study Tool

A learning assistant that helps students understand lecture PDFs using Retrieval-Augmented Generation (RAG), embeddings, and FAISS.  
Upload a lecture → extract and chunk text → embed → index → ask questions → get grounded AI answers.  
Future phases will add quiz generation and spaced-repetition support.

---

## Features
- Upload lecture PDFs
- Extract and chunk text
- Create embeddings (safe local or real Gemini)
- FAISS semantic index (fast search)
- NumPy fallback retrieval
- Retrieval-Augmented Question Answering
- Displays retrieved chunks for transparency
- Works offline in SAFE mode
- Fully testable without real API keys
- Future planned:
  - Quiz generation
  - SRS spaced repetition review system

---

## Tech Stack
- Python
- FAISS
- NumPy
- Google Gemini API (optional)
- PyPDF
- Streamlit

---

## Project Structure
.env
requirements.txt
build_faiss_index.py

data/
raw/
processed/

backend/
extract_pdf.py
helpers.py
summarize_file.py
summarize_file_safe.py
create_embeddings.py
embeddings_provider.py
rag_query.py
generate_quiz.py
study_srs.py

vectorstore/
faiss_store.py

test_create_embeddings.py
test_embeddings_provider.py
test_vectorstore.py
test_rag_query.py

frontend/
app.py

yaml
Copy code

---

## Installation

Create virtual environment:
python -m venv venv
venv\Scripts\activate # Windows
source venv/bin/activate # Mac/Linux

yaml
Copy code

Install dependencies:
pip install -r requirements.txt

r
Copy code

Install FAISS if missing:
pip install faiss-cpu

yaml
Copy code

---

## Environment Variables

Create `.env` file:
USE_SAFE_EMBEDDINGS=1
GEMINI_API_KEY=your_key_here
EMBED_MODEL=embedding-001
EMBED_DIM=1536

yaml
Copy code

Mode behavior summary:

Mode | Description
------|-----------
SAFE | Deterministic offline embeddings
REAL | Uses Gemini embeddings (requires API key)

---

## Running the App
streamlit run frontend/app.py

yaml
Copy code

---

## Tests
pytest -q

yaml
Copy code

Test coverage:
- Embedding provider behavior
- SAFE vs REAL mode
- FAISS index search logic
- NumPy retrieval consistency
- RAG pipeline correctness

---

## How It Works (Pipeline)
1. Upload PDF  
2. Extract text  
3. Chunk text into manageable segments  
4. Generate embeddings  
5. Optional: Build FAISS index  
6. Ask a question  
7. Retrieve similar chunks  
8. LLM answers using retrieved context  
9. User sees answer + supporting text chunks  

---

## UI Controls
- Upload PDF
- Toggle SAFE vs REAL embeddings
- Create embeddings file
- Build FAISS index
- Toggle FAISS vs NumPy retrieval
- View retrieved chunks
- Optional similarity scores
- Status messages for files & modes

---

## Current Status
- Stable RAG functionality
- Indexed retrieval working
- Deterministic test mode
- FAISS rebuild logic implemented
- Quiz + SRS scaffolds added
- CI-friendly testable design

---

## Common Issues

Problem | Solution
--------|---------
FAISS missing | `pip install faiss-cpu`
Streamlit cannot import | run from project root
SAFE mode stuck | set `USE_SAFE_EMBEDDINGS=0`
Windows path issues | restart terminal after enabling venv
No API key | System stays in SAFE mode automatically

---

## Roadmap
- Complete quiz generation
- Implement SRS spaced repetition
- Better UI
- Incremental FAISS updating
- Optional REST API mode

---

## Intended Users
Students who want reliable, source-backed AI help instead of hallucinating answers.

---

## Contributing
- Do not commit `.env`
- Ensure tests pass
- Keep structure clean and documented





You said:
ARE YOU TROLLING ME I SAID ONE BLOCK I CAN COPY


ChatGPT said:
markdown
Copy code
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

## Day-5 / Current status (concise)
- Embedding provider: Safe mode + Gemini scaffold with batching, retries, L2 normalization
- FAISS: build/load/search + rebuild_if_needed() + get_index_status() + richer metadata
- Tests: provider and FAISS tests added; `pytest` passes locally (e.g., "7 passed")
- Quiz & SRS: minimal scaffolds added so UI can integrate later
- CI: basic GitHub Actions workflow to run tests

---

## Common issues & fixes
- FAISS not available: run `pip install faiss-cpu` (or run tests which skip FAISS)
- Streamlit import errors: run commands from project root so Python path resolves
- Embedding dimension mismatch: ensure `EMBED_DIM` matches provider output or assert/check
- API keys: if not set, app runs in SAFE deterministic mode (no network calls)

---

## Development tips
- Run Streamlit from project root so `backend` package imports resolve.
- Use `USE_SAFE_EMBEDDINGS=1` for fast, reproducible local development.
- When adding real Gemini calls, mock `_call_api()` in tests to avoid relying on live keys.
- Use `rebuild_index_if_needed()` to programmatically ensure index freshness.

---

## Roadmap (next steps)
- Day-6: Harden RAG, wire quiz generation to UI, SRS integration
- Day-7: Full quiz question parsing, study sessions, performance improvements
- Future: REST API, incremental FAISS indexing, streaming responses

---

## Contributing
- Do not commit `.env` or secrets
- Add tests for new features
- Keep functions small and testable
- Ensure `pytest -q` passes before opening PRs

---