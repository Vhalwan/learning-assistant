# frontend/app.py
import os
import sys
from pathlib import Path

# Make sure the project root is on sys.path so "backend" is importable when Streamlit launches
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pdfplumber
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Backend helpers
from backend.summarize_file import summarize_with_gemini
from backend.create_embeddings import (
    create_embeddings_for_text,
    load_embeddings,
    EMBED_DIM
)
from backend.embeddings_provider import deterministic_vector, get_embedding_provider
from backend.rag_query import rag_answer_from_embeddings

# generate quiz + SRS
from backend.generate_quiz import generate_quiz_from_context
from backend.study_srs import SRSManager

# faiss builder
try:
    from backend.vectorstore.faiss_store import build_faiss_index
    _faiss_builder_available = True
except Exception:
    _faiss_builder_available = False

st.set_page_config(page_title="Learning Assistant", layout="centered")
st.title("Learning Assistant — Gemini Demo")

st.markdown(
    """
Upload a lecture PDF. Controls:
- Toggle SAFE vs REAL embeddings (affects embedding creation).
- (Re)create embeddings file.
- Build FAISS index (optional).
- Toggle FAISS vs NumPy search for retrieval.
"""
)

# Upload + controls
uploaded = st.file_uploader("Upload a lecture PDF", type=["pdf"])
if uploaded:

    # Save uploaded file
    tmp_pdf = Path("data/raw") / uploaded.name
    tmp_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp_pdf.write_bytes(uploaded.read())

    # Extract text preview
    with pdfplumber.open(tmp_pdf) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    text = "\n\n".join(pages)

    st.subheader("Extracted preview")
    st.write(text[:1000] + ("..." if len(text) > 1000 else ""))

    # derive the stem and paths
    stem = tmp_pdf.stem
    embeddings_path = Path(f"data/processed/{stem}_embeddings.json")
    index_path = Path(f"data/processed/{stem}_embeddings.index")

    # UI control
    col1, col2, col3 = st.columns(3)
    with col1:
        use_safe_toggle = st.checkbox("Use SAFE embeddings (deterministic)", value=os.getenv("USE_SAFE_EMBEDDINGS", "1").lower() in ("1", "true", "yes"))
    with col2:
        recreate_btn = st.button("(Re)create embeddings")
    with col3:
        build_index_btn = st.button("Build FAISS index")

    # Toggle for search the backend
    use_faiss_search = st.checkbox("Use FAISS for retrieval (if available)", value=False)

    os.environ["USE_SAFE_EMBEDDINGS"] = "1" if use_safe_toggle else "0"

    # Create embeddings when missing or when they are requested
    if (not embeddings_path.exists()) or recreate_btn:
        try:
            with st.spinner("Creating embeddings..."):
                rows = create_embeddings_for_text(text, str(embeddings_path), dim=EMBED_DIM)
            st.success(f"Embeddings created: {embeddings_path} ({len(rows)} chunks)")
        except Exception as e:
            st.error("Failed to create embeddings.")
            st.exception(e)

    # Show embeddings status
    try:
        ids, texts, vecs = load_embeddings(str(embeddings_path))
        st.info(f"Embeddings file: {embeddings_path} — {len(ids)} chunks — Mode: {'SAFE' if os.environ.get('USE_SAFE_EMBEDDINGS','1') in ('1','true') else 'REAL'}")
    except Exception:
        ids, texts, vecs = [], [], np.array([])
        st.warning("Embeddings file not found. Click (Re)create embeddings.")

    # Build FAISS index if it is requested
    if build_index_btn:
        if not embeddings_path.exists():
            st.error("Cannot build FAISS index — embeddings file missing. Create embeddings first.")
        elif not _faiss_builder_available:
            st.error("FAISS builder not available (faiss-cpu not installed). Install: pip install faiss-cpu")
        else:
            try:
                with st.spinner("Building FAISS index..."):
                    build_faiss_index(str(embeddings_path), str(index_path))
                st.success(f"FAISS index built: {index_path}")
            except Exception as e:
                st.error("FAISS index build failed.")
                st.exception(e)

    st.markdown("---")
    st.subheader("Ask a question (Retrieval-Augmented)")

    question = st.text_input("Your question (based on uploaded lecture)")
    k = st.number_input("Top-k chunks to retrieve", min_value=1, max_value=10, value=3, step=1)

    if st.button("Ask (RAG)"):
        if len(ids) == 0:
            st.error("Embeddings not loaded. Create embeddings first.")
        elif not question or not question.strip():
            st.warning("Please enter a question.")
        else:

            # Determine faiss index path candidate and pass it to rag_query
            candidate_index_path = str(index_path) if index_path.exists() else None
            with st.spinner("Retrieving relevant chunks and asking Gemini..."):
                try:

                    # Request rich metadata so we can show scores & provenance
                    ans, retrieved_chunks, prompt_used, provenance = rag_answer_from_embeddings(
                        question,
                        str(embeddings_path),
                        top_k=int(k),
                        use_faiss=bool(use_faiss_search),
                        faiss_index_path=candidate_index_path,
                        use_safe=(True if os.environ.get("USE_SAFE_EMBEDDINGS","1") in ("1","true","yes") else False),
                        use_query_expansion=False,
                        return_meta=True,
                        llm_call=None
                    )

                    st.subheader("Top retrieved chunks (expand to read)")
                    if not retrieved_chunks:
                        st.info("No chunks retrieved.")
                    else:
                        for i, chunk in enumerate(retrieved_chunks, start=1):
                            score = chunk.get("score", 0.0)
                            cid = chunk.get("id")
                            pos = chunk.get("pos")
                            with st.expander(f"{i}. chunk (id={cid} pos={pos} score={score:.4f})"):
                                st.write(chunk.get("text", "")[:2000] + ("..." if len(chunk.get("text","")) > 2000 else ""))

                    st.subheader("Answer")
                    st.write(ans)

                    st.subheader("Provenance (sentence -> chunk)")
                    if provenance and provenance.get("sentences"):
                        for s in provenance.get("sentences", []):
                            st.write(f"- \"{s['sentence']}\"  → chunk_id={s.get('chunk_id')} (score={s.get('score'):.3f})")
                    else:
                        st.info("No provenance available.")

                    with st.expander("Prompt used (debug)"):
                        st.code(prompt_used[:4000])

                except FileNotFoundError as fe:
                    st.error("File missing for retrieval.")
                    st.exception(fe)
                except Exception as e:
                    st.error("RAG failed — see details.")
                    st.exception(e)

    st.markdown("---")
    st.subheader("Study/Quiz")
    n_q = st.number_input("Number of quiz items", min_value=1, max_value=20, value=5)
    if st.button("Generate Quiz from lecture"):
        if not text:
            st.warning("No document text extracted.")
        else:
            with st.spinner("Generating quiz..."):
                try:
                    quiz = generate_quiz_from_context(stem, text, n=int(n_q), llm_call=None)
                    st.success(f"Quiz generated: {len(quiz.get('questions', []))} items.")
                    # display flashcards
                    for q in quiz.get("questions", []):
                        with st.expander(f"Q: {q.get('question')[:120]}"):
                            st.write("Answer:", q.get("answer"))
                            st.write("ID:", q.get("id"))
                            if st.button(f"Start SRS for {q.get('id')}", key=f"srs_{q.get('id')}"):
                                mgr = SRSManager()
                                mgr.ensure_card(q.get("id"))
                                st.info(f"Registered card {q.get('id')} in SRS.")
                except Exception as e:
                    st.error("Quiz generation failed.")
                    st.exception(e)

    st.markdown("---")
    st.caption("Tip: For reproducible tests set USE_SAFE_EMBEDDINGS=1 and build FAISS index to compare results with NumPy search.")
