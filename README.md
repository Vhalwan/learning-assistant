# Learning Assistant — Gemini RAG + FAISS Study Tool

A learning assistant that helps students study lecture PDFs using Retrieval-Augmented Generation (RAG), embeddings, and optional FAISS indexing.  
Upload a lecture → extract and chunk text → embed → index → ask questions → get grounded AI answers.

The app also includes **quiz generation**, **confusion tracking**, and a **spaced-repetition (SRS)** learning loop to help students test, fix, and retain knowledge.

---

## Features

- Upload and process lecture PDFs
- Extract and chunk lecture text
- Create embeddings (SAFE local mode or real Gemini embeddings)
- Optional FAISS semantic index for fast retrieval
- NumPy fallback retrieval when FAISS is unavailable
- Retrieval-Augmented Q&A with transparent source chunks
- Offline SAFE mode (no API key required)
- Conversational chat with history
- Quiz generation (MCQs) from lecture content
- Confused concepts list (prioritized by repeated mistakes)
- Spaced Repetition System (SRS) for long-term retention
- Lecture-specific or cross-lecture review
- Fully testable with mocks

---

## Tech Stack

- Python
- Streamlit (frontend)
- NumPy
- FAISS (optional, for fast vector search)
- Google Gemini APIs (optional, for embeddings and LLMs)
- pdfplumber / PyPDF (PDF extraction)
- pytest (testing)

---

## Project Structure

.env  
requirements.txt  

data/  
&nbsp;&nbsp;raw/         # uploaded PDFs  
&nbsp;&nbsp;processed/   # chunks, embeddings JSON, FAISS indexes, quizzes, study progress  

backend/  
&nbsp;&nbsp;__init__.py  
&nbsp;&nbsp;extract_pdf.py         # PDF → plain text  
&nbsp;&nbsp;helpers.py             # chunking and IO helpers  
&nbsp;&nbsp;summarize_file.py      # LLM summarization (Gemini)  
&nbsp;&nbsp;summarize_file_safe.py # offline summarizer fallback  
&nbsp;&nbsp;create_embeddings.py   # chunk → embeddings JSON  
&nbsp;&nbsp;embeddings_provider.py # SAFE + Gemini providers  
&nbsp;&nbsp;rag_query.py           # RAG orchestration  
&nbsp;&nbsp;generate_quiz.py       # quiz generation  
&nbsp;&nbsp;study_srs.py           # spaced repetition system  
&nbsp;&nbsp;confusion_store.py     # persisted confusion tracking  
&nbsp;&nbsp;vectorstore/  
&nbsp;&nbsp;&nbsp;&nbsp;faiss_store.py # FAISS build/load/search helpers  

frontend/  
&nbsp;&nbsp;app.py                 # main Streamlit app  
&nbsp;&nbsp;handlers.py            # UI ↔ backend glue  
&nbsp;&nbsp;sections/  
&nbsp;&nbsp;&nbsp;&nbsp;chat.py        # chat UI  
&nbsp;&nbsp;&nbsp;&nbsp;quiz.py        # quiz UI  
&nbsp;&nbsp;&nbsp;&nbsp;confused.py    # confused concepts UI  
&nbsp;&nbsp;&nbsp;&nbsp;srs.py         # SRS review UI  

tests/  
&nbsp;&nbsp;test_create_embeddings.py  
&nbsp;&nbsp;test_embeddings_provider.py  
&nbsp;&nbsp;test_vectorstore.py  
&nbsp;&nbsp;test_rag_query.py  

---

## Environment (.env example)

Create a `.env` file at the repo root (do **not** commit it):

Embedding mode  
USE_SAFE_EMBEDDINGS=1  
# 1 = SAFE deterministic local embeddings (default)  
# 0 = REAL provider embeddings  

Real provider keys (optional)  
GEMINI_API_KEY=your_gemini_api_key_here  
GOOGLE_API_KEY=your_google_api_key_here  

Embedding model & dimensions  
EMBED_MODEL=embedding-001  
EMBED_DIM=1536  

Optional custom endpoints  
EMBEDDING_API_URL=https://example.com/v1/embeddings  
API_BASE=http://localhost:8000  
API_TOKEN=

---

## Installation

1. Create and activate a virtual environment:

python -m venv venv  

Windows:  
venv\Scripts\activate  

Mac / Linux:  
source venv/bin/activate  

2. Install dependencies:

pip install -r requirements.txt  

3. (Optional) Install FAISS:

pip install faiss-cpu  

---

## Running the App

Start the Streamlit UI:

streamlit run frontend/app.py  

Typical workflow:

1. Upload a lecture PDF (saved to `data/raw/`)
2. Extract and preview text
3. Toggle SAFE vs REAL embeddings
4. Create embeddings (`data/processed/<stem>_embeddings.json`)
5. Optionally build a FAISS index
6. Ask questions using RAG (see retrieved chunks + answer)
7. Chat conversationally with lecture context
8. Generate quizzes to test understanding
9. Review confused concepts
10. Add important items to SRS and review due cards

---

## Learning Loop (Conceptual)

**Test → Fix → Review**

1. **Quiz** — test understanding with generated MCQs  
2. **Confused** — review concepts you missed repeatedly  
3. **SRS** — lock knowledge into long-term memory  

---

## Tests

Run all tests:

pytest -q  

Notes:
- Tests cover embeddings, FAISS indexing/search, and RAG behavior
- FAISS tests are skipped if `faiss-cpu` is not installed
- SAFE mode is recommended for deterministic CI testing

---

## How the Pipeline Works (High Level)

1. PDF → extract plain text (`backend/extract_pdf.py`)
2. Text → chunking (`backend/helpers.py`)
3. Chunks → embeddings (`backend/create_embeddings.py`)
4. Embeddings → FAISS index (optional)
5. Query → embed → retrieve top-k chunks
6. Retrieved chunks + question → LLM prompt
7. Display answer with supporting chunks (provenance)

---

## Notes for Developers

- Streamlit widgets must use **unique keys** (especially in loops)
- SRS cards should ideally store question text at creation time
- Confused and SRS sections support lecture-specific filtering
- SAFE mode is ideal for offline work and reproducible testing
- UI favors clarity over hidden state: most learning content is visible by default

---

## Project Status

- Core RAG pipeline complete
- Quiz, Confused, and SRS learning loop implemented
- UI/UX actively refined for clarity, readability, and scale
- Designed to feel like a real study web app, not a demo

---
