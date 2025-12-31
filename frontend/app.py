import os
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pdfplumber
import numpy as np
from dotenv import load_dotenv
import requests

load_dotenv()


# Backend helpers
from backend.summarize_file import summarize_with_gemini
from backend.create_embeddings import (
    create_embeddings_for_text,
    load_embeddings,
    EMBED_DIM,
)
from backend.embeddings_provider import deterministic_vector, get_embedding_provider
from backend.rag_query import rag_answer_from_embeddings, rag_generate_summary_from_embeddings

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

Two primary modes:
- Ask a Question (fast, targeted) — default
- Generate Summary (document-level overview) — explicit
"""
)

# ----------------------------
# API mode toggle + token UI
# ----------------------------
API_DEFAULT = os.getenv("API_BASE", "http://localhost:8000")

if "use_api_mode" not in st.session_state:
    st.session_state.use_api_mode = False

# read default token from env var, allow override in UI
default_token = os.getenv("API_TOKEN", "") or ""
if "api_token" not in st.session_state:
    st.session_state.api_token = default_token

col_api, col_token = st.columns([1, 2])
with col_api:
    st.session_state.use_api_mode = st.checkbox("Use API mode", value=st.session_state.use_api_mode)
with col_token:
    st.session_state.api_token = st.text_input(
        "API Token (override)",
        value=st.session_state.api_token,
        type="password",
        help="Reads default from environment; typing here overrides for this session",
    )


def call_query_api(
    question: str,
    embeddings_path: str,
    top_k: int,
    use_faiss: bool,
    faiss_index_path: Optional[str] = None,
    api_base: str = API_DEFAULT,
    token: str = "",
) -> Dict[str, Any]:
    """
    Call the FastAPI /query endpoint and return parsed JSON.
    Raises requests.HTTPError on non-2xx responses.
    """
    url = f"{api_base.rstrip('/')}/query"
    payload = {
        "question": question,
        "embeddings_path": embeddings_path,
        "top_k": top_k,
        "use_faiss": use_faiss,
        "faiss_index_path": faiss_index_path,
        "use_query_expansion": False,
    }
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def call_summarize_api(
    embeddings_path: str,
    summary_type: str = "brief",
    top_k: Optional[int] = None,
    api_base: str = API_DEFAULT,
    token: str = "",
) -> Dict[str, Any]:
    """
    Call the FastAPI /summarize endpoint and return parsed JSON.
    """
    url = f"{api_base.rstrip('/')}/summarize"
    payload = {
        "embeddings_path": embeddings_path,
        "summary_type": summary_type,
        "top_k": top_k,
    }
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


# presentation helpers
def _strip_key_concepts_from_answer(answer: str) -> str:
    """
    Remove any trailing 'Key Concepts' style block from an LLM answer for the Q&A UI.
    Keeps the main answer concise.
    """
    if not isinstance(answer, str):
        return str(answer)
    m = re.search(r"\n+(?:\d+\s*)?Key\s+Concepts?\b", answer, flags=re.IGNORECASE)
    if m:
        return answer[:m.start()].strip()
    # also handle 'Key concepts / highlights' phrasing
    m2 = re.search(r"\n+Key\s+concepts\b", answer, flags=re.IGNORECASE)
    if m2:
        return answer[:m2.start()].strip()
    return answer.strip()

def _clean_summary_text(summary: str) -> str:
    """
    Remove leading boilerplate like "Here's a concise summary..." if present (cosmetic).
    """
    if not isinstance(summary, str):
        return str(summary)
    # remove a common leading phrase
    cleaned = re.sub(r"^\s*Here(?:'|’)s (?:a|an) (?:concise|brief) summary[^\n]*\n*[:\-]*\s*", "", summary, flags=re.IGNORECASE)
    return cleaned.strip()


# ----------------------------
# Upload + controls
# ----------------------------
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
        use_safe_toggle = st.checkbox(
            "Use SAFE embeddings (deterministic)",
            value=os.getenv("USE_SAFE_EMBEDDINGS", "1").lower() in ("1", "true", "yes"),
        )
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
        st.info(
            f"Embeddings file: {embeddings_path} — {len(ids)} chunks — Mode: {'SAFE' if os.environ.get('USE_SAFE_EMBEDDINGS','1') in ('1','true') else 'REAL'}"
        )
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
    # Two tabs: Ask a Question (default) and Generate Summary
    tab1, tab2 = st.tabs(["Ask a Question", "Generate Summary"])

    # -------------------
    # Tab: Ask a Question
    # -------------------
    with tab1:
        st.subheader("Ask a question (Retrieval-Augmented)")

        question = st.text_input("Your question (based on uploaded lecture)", key="qa_question")
        k = st.number_input("Top-k chunks to retrieve", min_value=1, max_value=10, value=3, step=1, key="qa_k")

        if st.button("Ask (RAG)"):
            if len(ids) == 0:
                st.error("Embeddings not loaded. Create embeddings first.")
            elif not question or not question.strip():
                st.warning("Please enter a question.")
            else:

                candidate_index_path = str(index_path) if index_path.exists() else None

                with st.spinner("Retrieving relevant chunks and asking Gemini..."):
                    try:
                        if st.session_state.use_api_mode:
                            api_base = os.getenv("API_BASE", API_DEFAULT)
                            token = st.session_state.api_token or ""
                            resp = call_query_api(
                                question=question,
                                embeddings_path=str(embeddings_path),
                                top_k=int(k),
                                use_faiss=bool(use_faiss_search),
                                faiss_index_path=candidate_index_path,
                                api_base=api_base,
                                token=token,
                            )
                            ans = resp.get("answer")
                            retrieved_chunks = resp.get("retrieved", [])
                            prompt_used = resp.get("prompt")
                            provenance = resp.get("provenance")
                            latency = resp.get("latency_s", None)
                        else:
                            ans, retrieved_chunks, prompt_used, provenance = rag_answer_from_embeddings(
                                question,
                                str(embeddings_path),
                                top_k=int(k),
                                use_faiss=bool(use_faiss_search),
                                faiss_index_path=candidate_index_path,
                                use_safe=(True if os.environ.get("USE_SAFE_EMBEDDINGS", "1") in ("1", "true", "yes") else False),
                                use_query_expansion=False,
                                return_meta=True,
                                llm_call=None,
                            )
                            latency = None

                        # strip any appended Key Concepts block from Q&A answers for concise display
                        display_answer = _strip_key_concepts_from_answer(ans or "")

                        # Display retrieved chunks
                        st.subheader("Top retrieved chunks (expand to read)")
                        if not retrieved_chunks:
                            st.info("No chunks retrieved.")
                        else:
                            for i, chunk in enumerate(retrieved_chunks, start=1):
                                score = chunk.get("score", 0.0)
                                cid = chunk.get("id")
                                pos = chunk.get("pos")
                                with st.expander(f"{i}. chunk (id={cid} pos={pos} score={score:.4f})"):
                                    st.write(chunk.get("text", "")[:2000] + ("..." if len(chunk.get("text", "")) > 2000 else ""))

                        st.subheader("Answer")
                        st.write(display_answer or "")

                        if latency is not None:
                            st.caption(f"Latency: {latency:.3f}s")

                        st.subheader("Provenance (sentence -> chunk)")
                        if provenance and provenance.get("sentences"):
                            for s in provenance.get("sentences", []):
                                st.write(f"- \"{s['sentence']}\"  → chunk_id={s.get('chunk_id')} (score={s.get('score'):.3f})")
                        else:
                            st.info("No provenance available.")

                        with st.expander("Prompt used (debug)"):
                            if prompt_used:
                                st.code(prompt_used[:4000])
                            else:
                                st.info("No prompt captured.")

                    except requests.HTTPError as he:
                        try:
                            err_json = he.response.json()
                            detail = err_json.get("detail", str(he))
                        except Exception:
                            detail = str(he)
                        st.error(f"API request failed: {detail}")
                        st.exception(he)
                    except FileNotFoundError as fe:
                        st.error("File missing for retrieval.")
                        st.exception(fe)
                    except Exception as e:
                        st.error("RAG failed — see details.")
                        st.exception(e)

    # -----------------------
    # Tab: Generate Summary
    # -----------------------
    with tab2:
        st.subheader("Generate document-level summary (explicit)")

        summary_type = st.selectbox("Summary detail level", ["brief", "detailed"], index=0)
        summary_top_k = st.number_input("Limit to first N chunks (leave blank / 0 to use all)", min_value=0, max_value=10000, value=0, step=1)
        if st.button("Generate Summary"):
            if len(ids) == 0:
                st.error("Embeddings not loaded. Create embeddings first.")
            else:
                candidate_index_path = str(index_path) if index_path.exists() else None
                with st.spinner("Generating summary..."):
                    try:
                        if st.session_state.use_api_mode:
                            token = st.session_state.api_token or ""
                            resp = call_summarize_api(
                                embeddings_path=str(embeddings_path),
                                summary_type=summary_type,
                                top_k=(int(summary_top_k) if summary_top_k > 0 else None),
                                api_base=os.getenv("API_BASE", API_DEFAULT),
                                token=token,
                            )
                            out = resp.get("out", {}) or {}
                            summary = out.get("summary") or ""
                            key_concepts = out.get("key_concepts", []) or []
                            used_chunks = resp.get("used_chunks", []) or []
                        else:
                            out, used_chunks = rag_generate_summary_from_embeddings(
                                str(embeddings_path),
                                summary_type=summary_type,
                                top_k=(int(summary_top_k) if summary_top_k > 0 else None),
                                use_safe=(True if os.environ.get("USE_SAFE_EMBEDDINGS", "1") in ("1", "true", "yes") else False),
                                return_meta=False,
                                llm_call=None,
                            )
                            # out is a dict with 'summary' and 'key_concepts'
                            summary = out.get("summary", "")
                            key_concepts = out.get("key_concepts", [])
                            # used_chunks is a list of chunk text strings (or in other modes could be dicts)
                        # cosmetic cleaning
                        summary_display = _clean_summary_text(summary)

                        st.subheader("Summary")
                        st.write(summary_display or "")

                        st.subheader("Key concepts / highlights")
                        if key_concepts:
                            # variable-length set — display joined and count
                            st.write(f"{len(key_concepts)} items — " + ", ".join(key_concepts))
                        else:
                            st.info("No key concepts extracted.")

                        with st.expander("Show used chunks (preview)"):
                            if not used_chunks:
                                st.info("No chunks available.")
                            else:
                                # be resilient: each element might be a dict with id/pos/text OR a plain text string
                                for c in used_chunks:
                                    if isinstance(c, dict):
                                        # dict path (id/pos/text)
                                        cid = c.get("id", "<no-id>")
                                        pos = c.get("pos", "<no-pos>")
                                        txt = c.get("text", "")
                                        st.write(f"- id={cid} pos={pos}: {txt[:300]}{'...' if len(txt) > 300 else ''}")
                                    elif isinstance(c, str):
                                        # plain string path (preview)
                                        st.write(f"- chunk text preview: {c[:300]}{'...' if len(c) > 300 else ''}")
                                    else:
                                        # unknown type — just str()
                                        s = str(c)
                                        st.write(f"- {s[:300]}{'...' if len(s) > 300 else ''}")

                    except requests.HTTPError as he:
                        try:
                            detail = he.response.json().get("detail", str(he))
                        except Exception:
                            detail = str(he)
                        st.error(f"API summarize failed: {detail}")
                        st.exception(he)
                    except Exception as e:
                        st.error("Summary generation failed.")
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
                    if st.session_state.use_api_mode:
                        api_base = os.getenv("API_BASE", API_DEFAULT)
                        token = st.session_state.api_token or ""
                        url = f"{api_base.rstrip('/')}/generate_quiz"
                        payload = {"stem": stem, "context_text": text, "n": int(n_q)}
                        headers = {}
                        if token:
                            headers["Authorization"] = f"Bearer {token}"
                        resp = requests.post(url, json=payload, headers=headers, timeout=60)
                        resp.raise_for_status()
                        out = resp.json()
                        quiz = out.get("quiz", {})
                        out_path = out.get("out_path")
                        st.success(f"Quiz generated: {len(quiz.get('questions', []))} items. (wrote: {out_path})")
                    else:
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
                except requests.HTTPError as he:
                    try:
                        detail = he.response.json().get("detail", str(he))
                    except Exception:
                        detail = str(he)
                    st.error(f"API quiz generation failed: {detail}")
                    st.exception(he)
                except Exception as e:
                    st.error("Quiz generation failed.")
                    st.exception(e)

    st.markdown("---")
    st.caption("Tip: For reproducible tests set USE_SAFE_EMBEDDINGS=1 and build FAISS index to compare results with NumPy search.")
