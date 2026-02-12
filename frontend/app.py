# frontend/app.py
import os
import sys
import re
import json
import html as html_mod
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables once at startup
load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import streamlit.components.v1 as components
import pdfplumber
import numpy as np
import requests
from frontend.sections.confused import render as render_confused
from frontend.sections.quiz import render as render_quiz
from frontend.sections.srs import render as render_srs
from frontend.sections.chat import render as render_chat

# ------------------------
# New imports (refactored)
# ------------------------
from frontend.ui_helpers import (
    strip_key_concepts_from_answer,
    clean_summary_text,
    trim_history_to_max_turns,
    render_assistant_html,
)
from frontend.handlers import (
    init_llm,
    create_embeddings_if_needed,
    load_embeddings_wrapper,
    perform_query,
    perform_summary,
    perform_chat,
    generate_quiz,
    build_index,
    save_quiz_to_disk,
    load_all_quiz_items_wrapper,
    load_quiz_item_by_id_wrapper,
    perform_confusion_analysis,
    record_quiz_result,
)
st.set_page_config(
    page_title="Learning Assistant",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
header[data-testid="stHeader"] {
    height: 0px;
}
</style>
""", unsafe_allow_html=True)
# Backend helpers (unchanged)
from backend.create_embeddings import EMBED_DIM, create_embeddings_for_text, load_embeddings
from backend.vectorstore.faiss_store import build_faiss_index  # optional; handlers uses it too
from backend.generate_quiz import generate_mcq_from_context
from backend.study_srs import SRSManager, INTERVALS
from backend.quiz_storage import save_quiz_items, load_quiz_item_by_id, load_all_quiz_items

# initialize LLM (this mirrors existing behaviour)
llm = init_llm()
# We avoid calling st.* until after set_page_config — but keep same behavior messages
# We'll store messages in a list and display after page config
_startup_msgs = []
if llm is None:
    _startup_msgs.append(("warning", "LLM not available — using placeholders"))
else:
    _startup_msgs.append(("info", "LLM ready — will generate real MCQs"))

# Check FAISS builder availability (handlers also has this; keep for early error messages)
try:
    from backend.vectorstore.faiss_store import build_faiss_index as _maybe_build_faiss
    _faiss_builder_available = True
except Exception:
    _faiss_builder_available = False
st.markdown(
    """
    <style>
      :root {
        --accent: #4f46e5;
        --accent-soft: #eef2ff;
        --card-bg: #f8fafc;
        --card-border: #e2e8f0;
        --success-bg: #ecfdf3;
        --error-bg: #fef2f2;
        --warning-bg: #fff7ed;
      }

      /* Keep main content spacing */
      div.block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
      }

      /* Typography and headings */
      h1, h2, h3, h4 {
        color: #111827;
      }
      h2 {
        border-left: 4px solid var(--accent);
        padding-left: 0.5rem;
      }

      /* Metric / alert visuals */
      div[data-testid="stMetric"] {
        background: var(--accent-soft);
        padding: 0.75rem;
        border-radius: 12px;
        border: 1px solid #e0e7ff;
      }
      div[data-testid="stAlert"] {
        border-radius: 12px;
      }
      div[data-testid="stAlert"][data-baseweb="notification"] {
        border-left: 4px solid var(--accent);
      }
      div[data-testid="stAlert"][data-alert-type="success"] {
        background: var(--success-bg);
      }
      div[data-testid="stAlert"][data-alert-type="error"] {
        background: var(--error-bg);
      }
      div[data-testid="stAlert"][data-alert-type="warning"] {
        background: var(--warning-bg);
      }

      /* Card-like blocks (non-invasive selectors) */
      div[data-testid="stVerticalBlock"] > div.la-card,
      div[data-testid="stVerticalBlock"]:has(> div.la-card) {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        padding: 1rem 1.1rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
      }
      div.la-card {
        display: none;
      }
      div[data-testid="stVerticalBlock"]:has(> div.la-action-bar) {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        padding: 0.7rem 0.9rem;
        border-radius: 12px;
        margin: 0.6rem 0;
      }
      div.la-action-bar {
        display: none;
      }

      /* App header visuals */
      .app-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.25rem;
        border-radius: 18px;
        background: linear-gradient(135deg, var(--accent), #6d28d9);
        color: #fff;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.25);
      }
      .app-logo {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
      }
      .app-title {
        font-size: 1.6rem;
        font-weight: 700;
      }
      .app-subtitle {
        font-size: 0.95rem;
        opacity: 0.9;
      }

      /* Sticky bottom form area (keeps chat input visible) */
      div[data-testid="stForm"] {
        position: sticky;
        bottom: 0;
        background: #ffffff;
        border-top: 1px solid #e5e7eb;
        padding-top: 0.5rem;
        z-index: 10;
      }
      .chat-fade-top {
        height: 12px;
        background: linear-gradient(to bottom, rgba(255,255,255,0.95), rgba(255,255,255,0));
      }

      /* NOTE: intentionally do NOT target section[data-testid="stSidebar"]
         or the toggle button. Modifying those breaks Streamlit's internal
         layout/toggle logic. */
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="app-header">
      <div class="app-logo">LA</div>
      <div>
        <div class="app-title">Learning Assistant</div>
        <div class="app-subtitle">Turn lectures into actionable study sessions</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# show startup msgs
for lvl, msg in _startup_msgs:
    if lvl == "warning":
        st.warning(msg)
    else:
        st.info(msg)

st.markdown(
    """
**Upload a lecture PDF and study it interactively.**  

Controls:
- Toggle SAFE vs REAL embeddings.
- (Re)create embeddings file.
- Build FAISS index (optional).
- Toggle FAISS vs NumPy search.

Primary study modes:
1. **Ask a Question (RAG)** — targeted answers using retrieved chunks.  
2. **Generate Summary** — quick lecture overview.  
3. **Chat** — conversational mode with history.

Learning Enhancements:
- **Quiz Generation (MCQs)** — self-test knowledge.
- **Spaced Repetition (SRS)** — review cards efficiently.
- **Transparent Retrieval** — see chunks used for answers.
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

with st.sidebar:
    st.markdown("### 📄 Upload & Settings")
    uploaded = st.file_uploader("Upload a lecture PDF", type=["pdf"])
    st.markdown("#### API Settings")
    st.session_state.use_api_mode = st.checkbox("Use API mode", value=st.session_state.use_api_mode)
    st.session_state.api_token = st.text_input(
        "API Token (override)", value=st.session_state.api_token, type="password",
        help="Reads default from environment; typing here overrides for this session",
    )
    st.markdown("#### Navigation")
    st.markdown(
        """
        - [Setup](#setup-your-lecture)
        - [Study modes](#study-modes)
        - [Quiz](#study-quiz-mcq-v1)
        - [SRS](#spaced-repetition-review)
        - [Confused?](#confused-quick-prioritized-list)
        """,
        unsafe_allow_html=True,
    )

st.markdown('<a id="setup-your-lecture"></a>', unsafe_allow_html=True)
st.markdown("## 1️⃣ Setup your lecture")
st.write(
    "Upload a single lecture PDF on the left side bar and create embeddings for that lecture. "
    "This is usually a one-time step per file — once embeddings are created you can use the Study tools below."
)

# ----------------------------
# Upload + controls (unchanged logic)
# ----------------------------
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
    st.session_state["current_stem"] = stem
    current_label = stem.replace("_", " ").title()

    st.info(f"📄 Current lecture: {current_label}")
    # Group embedding controls into a neat card-like area
    with st.container():
        st.markdown("#### Embeddings & Indexing")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            use_safe_toggle = st.checkbox(
                "Use SAFE embeddings (filter sensitive or risky outputs)",
                value=os.getenv("USE_SAFE_EMBEDDINGS", "1").lower() in ("1", "true", "yes"),
            )
        with col2:
            recreate_btn = st.button("Recreate embeddings")
        with col3:
            build_index_btn = st.button("Build FAISS index")
        use_faiss_search = st.checkbox("Use FAISS for retrieval (faster retrieval for large documents)",
                                       value=False)

    os.environ["USE_SAFE_EMBEDDINGS"] = "1" if use_safe_toggle else "0"
    st.session_state["use_faiss_search"] = use_faiss_search
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

    # Build FAISS index if requested
    if build_index_btn:
        if not embeddings_path.exists():
            st.error("Cannot build FAISS index — embeddings file missing. Create embeddings first.")
        elif not _faiss_builder_available:
            st.error("FAISS builder not available (faiss-cpu not installed). Install: pip install faiss-cpu")
        else:
            try:
                with st.spinner("Building FAISS index..."):
                    build_index(str(embeddings_path), str(index_path))
                    st.success(f"FAISS index built: {index_path}")
            except Exception as e:
                st.error("FAISS index build failed.")
                st.exception(e)

    # ----------------------------
    # 2️⃣ Study modes (UI-only guidance)
    # ----------------------------
    st.markdown("---")
    st.markdown('<a id="study-modes"></a>', unsafe_allow_html=True)
    st.markdown("## 2️⃣ Study modes")
    st.write(
        "Use the tabs below to explore the lecture: ask targeted questions, generate summaries, or chat conversationally. "
        "Below the tabs you'll find Quiz and Review tools to test and retain what you learn."
    )

    # Use slightly larger tabs with descriptions to improve discoverability
    tab1, tab2, tab3 = st.tabs(
        [
            "Ask a Question — RAG",
            "Generate Summary",
            "Chat (conversational RAG)",
        ]
    )

    # Hidden defaults: keep top-k fixed for QA and Chat (3), and summary default 0
    if "qa_k" not in st.session_state:
        st.session_state["qa_k"] = 3
    if "summary_top_k" not in st.session_state:
        st.session_state["summary_top_k"] = 0

    # -------------------
    # Tab: Ask a Question
    # -------------------
    with tab1:
        st.subheader("Ask a question (Retrieval-Augmented)")
        st.markdown("Type a focused question about the uploaded lecture. The assistant will retrieve supporting chunks.")
        qa_col1, qa_col2 = st.columns([4, 1])
        with qa_col1:
            question = st.text_input("Your question (based on uploaded lecture)", key="qa_question")
        # hide top-k input from users; keep value in session_state
        k = st.session_state.get("qa_k", 3)

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
                            resp = perform_query(
                                question=question,
                                embeddings_path=str(embeddings_path),
                                top_k=int(k),
                                use_faiss=bool(use_faiss_search),
                                faiss_index_path=candidate_index_path,
                                use_api_mode=True,
                                api_base=os.getenv("API_BASE", API_DEFAULT),
                                token=st.session_state.api_token or "",
                                llm_call=None,
                            )
                        else:
                            resp = perform_query(
                                question=question,
                                embeddings_path=str(embeddings_path),
                                top_k=int(k),
                                use_faiss=bool(use_faiss_search),
                                faiss_index_path=candidate_index_path,
                                use_api_mode=False,
                                llm_call=llm,
                            )

                        ans = resp.get("answer")
                        retrieved_chunks = resp.get("retrieved", [])
                        prompt_used = resp.get("prompt")
                        provenance = resp.get("provenance")
                        latency = resp.get("latency", None)

                        display_answer = strip_key_concepts_from_answer(ans or "")

                        # Show retrieval + answer in two-column layout for clarity
                        out_col_left, out_col_right = st.columns([2, 3])
                        with out_col_left:
                            st.subheader("Top retrieved chunks")
                            if not retrieved_chunks:
                                st.info("No chunks retrieved.")
                            else:
                                for i, chunk in enumerate(retrieved_chunks, start=1):
                                    score = chunk.get("score", 0.0)
                                    cid = chunk.get("id")
                                    pos = chunk.get("pos")
                                    with st.expander(f"{i}. chunk — id={cid} pos={pos} score={score:.4f}"):
                                        st.write(chunk.get("text", "")[:2000] + ("..." if len(chunk.get("text", "")) > 2000 else ""))
                        with out_col_right:
                            st.subheader("Answer")
                            st.write(display_answer or "")
                            if latency is not None:
                                st.caption(f"Latency: {latency:.3f}s")
                            st.subheader("Provenance (sentence → chunk)")
                            if provenance and provenance.get("sentences"):
                                for s in provenance.get("sentences", []):
                                    st.write(f"- \"{s['sentence']}\" → chunk_id={s.get('chunk_id')} (score={s.get('score'):.3f})")
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
        st.markdown("Choose summary length and optionally limit the chunks used.")
        s_col1, s_col2 = st.columns([2, 1])
        with s_col1:
            summary_type = st.selectbox("Summary detail level", ["brief", "detailed"], index=0)
        # hide explicit summary chunk limit from users; compute automatically for large docs
        summary_top_k = st.session_state.get("summary_top_k", 0)

        if st.button("Generate Summary"):
            if len(ids) == 0:
                st.error("Embeddings not loaded. Create embeddings first.")
            else:
                candidate_index_path = str(index_path) if index_path.exists() else None
                with st.spinner("Generating summary..."):
                    try:
                        # determine effective top_k for summarization: usually all (None),
                        # but for very large documents use a limited number of chunks silently
                        effective_top_k = None
                        try:
                            num_chunks = int(len(ids))
                        except Exception:
                            num_chunks = 0
                        if summary_top_k and int(summary_top_k) > 0:
                            effective_top_k = int(summary_top_k)
                        else:
                            # automatic heuristic: if many chunks, restrict to a fraction
                            if num_chunks > 300:
                                effective_top_k = min(500, max(200, int(num_chunks * 0.25)))
                            else:
                                effective_top_k = None

                        if st.session_state.use_api_mode:
                            resp = perform_summary(
                                embeddings_path=str(embeddings_path),
                                summary_type=summary_type,
                                top_k=effective_top_k,
                                use_api_mode=True,
                                api_base=os.getenv("API_BASE", API_DEFAULT),
                                token=st.session_state.api_token or "",
                            )
                        else:
                            resp = perform_summary(
                                embeddings_path=str(embeddings_path),
                                summary_type=summary_type,
                                top_k=effective_top_k,
                                use_api_mode=False,
                                llm_call=None,
                            )

                        summary = resp.get("summary", "")
                        key_concepts = resp.get("key_concepts", []) or []
                        used_chunks = resp.get("used_chunks", []) or []
                        summary_display = clean_summary_text(summary)

                        st.subheader("Summary")
                        if summary_display:
                            st.write(summary_display)
                        else:
                            st.info("No summary returned.")
                        st.subheader("Key concepts / highlights")
                        if key_concepts:
                            st.write(f"{len(key_concepts)} items — " + ", ".join(key_concepts))
                        else:
                            st.info("No key concepts extracted.")

                        with st.expander("Show used chunks (preview)"):
                            if not used_chunks:
                                st.info("No chunks available.")
                            else:
                                for c in used_chunks:
                                    if isinstance(c, dict):
                                        cid = c.get("id", "<no-id>")
                                        pos = c.get("pos", "<no-pos>")
                                        txt = c.get("text", "")
                                        st.write(f"- id={cid} pos={pos}: {txt[:300]}{'...' if len(txt) > 300 else ''}")
                                    elif isinstance(c, str):
                                        st.write(f"- chunk text preview: {c[:300]}{'...' if len(c) > 300 else ''}")
                                    else:
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

    # -----------------------
    # Tab: Chat
    # -----------------------
    with tab3:
        render_chat(st=st, stem=stem, llm=llm)

    st.markdown("---")
    hist_key = f"chat_history_{stem}"

    # ----------------------------
    # Learning Loop: Quiz → Confused → SRS (UI-only grouping)
    # ----------------------------
    st.markdown("## 🔁 Learning Loop — Test → Fix → Review")
    st.write(
        "A simple study workflow: 1) take a short quiz to test yourself, "
        "2) review the concepts you repeatedly missed, and 3) add important items to your spaced repetition (SRS) for long-term retention."
    )

    # Step 1: Quiz (unchanged behavior — just a short intro)
    st.markdown("### 1) Test yourself — Quiz")
    st.write("Generate a quick set of multiple-choice questions from the lecture and check your understanding.")
    # Render quiz directly (removed outer expander)
    render_quiz(st=st, stem=stem, text=text, llm=llm, hist_key=hist_key)

    st.markdown("---")

    # Step 2: Confused (prioritized list of things you missed)
    st.markdown("### 2) Review what confused you")
    st.write("These are concepts you missed multiple times. Click 'Explain simply' for a short explanation or add items to SRS.")
    # Render confused directly (removed outer expander)
    render_confused(
        st=st,
        stem=stem,
        embeddings_path=embeddings_path,
        index_path=index_path,
        use_faiss_search=use_faiss_search,
        llm=llm,
    )

    st.markdown("---")

    # Step 3: Spaced Repetition (SRS)
    st.markdown("### 3) Lock it in — Spaced Repetition (SRS)")
    st.write("Add items you want to retain and review due cards here. This helps move knowledge into long-term memory.")
    # Render SRS directly (removed outer expander)
    render_srs(st, stem=stem)

    st.markdown("---")
    st.caption("Tip: For reproducible tests set USE_SAFE_EMBEDDINGS=1 and build FAISS index to compare results with NumPy search.")
