# Learning Assistant — Gemini RAG + FAISS Study Tool

A study app for lecture PDFs that turns course material into an interactive learning loop.

Upload a lecture PDF, extract and chunk the text, build embeddings, and ask grounded questions through Retrieval-Augmented Generation (RAG). The app also supports quiz generation, structured explanations, confusion tracking, and spaced repetition (SRS) so students can test themselves, review weak spots, and retain knowledge over time.

## Features

- Upload and process lecture PDFs
- Extract and chunk lecture text
- Create embeddings in:
  - SAFE local mode (deterministic, offline, no API key required)
  - Gemini-powered mode
- Optional FAISS semantic index for faster retrieval
- NumPy fallback retrieval when FAISS is unavailable
- Retrieval-Augmented Q&A with transparent source chunks
- Conversational chat with lecture context and history
- Quiz generation (MCQs) grounded in lecture content
- Structured quiz feedback:
  - correct answer explanation
  - why incorrect options are wrong
  - source/chunk grounding
- Confused concepts tracking for repeated mistakes
- Spaced Repetition System (SRS) for long-term retention
- Lecture-scoped review workflow
- Fully testable with mocks

## Tech Stack

- Python
- Streamlit
- NumPy
- FAISS (optional, for fast vector search)
- Google Gemini APIs (optional, for embeddings and LLMs)
- pdfplumber / PyPDF (PDF extraction)
- pytest (testing)

## Project Structure

```text
.env
requirements.txt

data/
  raw/                  # uploaded PDFs
  processed/            # chunks, embeddings JSON, FAISS indexes, quizzes, study progress

backend/
  __init__.py
  extract_pdf.py        # PDF → plain text
  helpers.py            # chunking and IO helpers
  summarize_file.py     # LLM summarization (Gemini)
  summarize_file_safe.py# offline summarizer fallback
  create_embeddings.py  # chunk → embeddings JSON
  embeddings_provider.py # SAFE + Gemini providers
  rag_query.py          # RAG orchestration
  generate_quiz.py      # quiz generation
  study_srs.py          # spaced repetition system
  confusion_store.py    # persisted confusion tracking
  vectorstore/
    faiss_store.py      # FAISS build/load/search helpers

frontend/
  app.py                # main Streamlit app
  handlers.py           # UI ↔ backend glue
  sections/
    chat.py             # chat UI
    quiz.py             # quiz UI
    confused.py         # confused concepts UI
    srs.py              # SRS review UI

tests/
  test_create_embeddings.py
  test_embeddings_provider.py
  test_vectorstore.py
  test_rag_query.py
```

## Environment Variables

Create a `.env` file at the repo root and do **not** commit it.

```env
# Embedding mode
USE_SAFE_EMBEDDINGS=1
# 1 = SAFE deterministic local embeddings (default)
# 0 = REAL provider embeddings

# Real provider keys (optional)
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here

# Embedding model & dimensions
EMBED_MODEL=embedding-001
EMBED_DIM=1536

# Optional custom endpoints
EMBEDDING_API_URL=https://example.com/v1/embeddings
API_BASE=http://localhost:8000
API_TOKEN=
```

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Mac / Linux:

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Optional: install FAISS

```bash
pip install faiss-cpu
```

## Running the App

Start the Streamlit UI:

```bash
streamlit run frontend/app.py
```

### Typical workflow

1. Upload a lecture PDF
2. Extract and preview the text
3. Create embeddings
4. Optionally build a FAISS index
5. Ask questions using RAG
6. Chat with lecture context
7. Generate quizzes to test understanding
8. Review confused concepts
9. Add missed concepts to SRS
10. Review due cards over time

## Learning Loop

**Test → Fix → Review**

1. **Quiz** — test understanding with generated MCQs
2. **Confused** — revisit concepts you missed repeatedly
3. **SRS** — reinforce knowledge with spaced repetition

## Tests

Run all tests:

```bash
pytest -q
```

### Notes

- Tests cover embeddings, FAISS indexing/search, and RAG behavior
- FAISS tests are skipped if `faiss-cpu` is not installed
- SAFE mode is recommended for deterministic CI testing

## How the Pipeline Works

1. PDF → extract plain text (`backend/extract_pdf.py`)
2. Text → chunking (`backend/helpers.py`)
3. Chunks → embeddings (`backend/create_embeddings.py`)
4. Embeddings → FAISS index (optional)
5. Query → embed → retrieve top-k chunks
6. Retrieved chunks + question → LLM prompt
7. Display answer with supporting chunks

## Notes for Developers

- Streamlit widgets must use **unique keys**, especially in loops
- SRS cards should store question text at creation time
- Confused and SRS sections are lecture-scoped
- SAFE mode is ideal for offline work and reproducible testing
- The UI favors clarity over hidden state

## Project Status

- Core RAG pipeline complete
- Quiz, Confused, and SRS learning loop implemented
- Quiz explanations include grounded reasoning and source references
- UI/UX is actively refined for clarity, readability, and scale
- Designed to feel like a real study web app, not a demo
