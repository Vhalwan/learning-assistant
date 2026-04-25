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
from frontend.sections.srs import render_srs_section
from frontend.sections.chat import render as render_chat

# ------------------------
# New imports (refactored)
# ------------------------
from frontend.runtime_ui_helpers import (
    strip_key_concepts_from_answer,
    strip_retrieval_artifacts,
    clean_summary_text,
	clean_key_concepts_list,
	derive_key_concepts_from_summary_text,
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
        --accent: #0f766e;
        --accent-strong: #115e59;
        --accent-soft: #ecfeff;
        --accent-warm: #f59e0b;
        --card-bg: rgba(255, 255, 255, 0.88);
        --card-border: #d7e5e4;
        --text: #12212a;
        --muted: #546571;
        --success-bg: #ecfdf3;
        --error-bg: #fef2f2;
        --warning-bg: #fff7ed;
      }

      /* Keep main content spacing */
      div[data-testid="stAppViewContainer"] {
        background:
          radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 30%),
          radial-gradient(circle at top right, rgba(245, 158, 11, 0.12), transparent 24%),
          linear-gradient(180deg, #f7fbfb 0%, #eef4f5 100%);
      }
      div.block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1080px;
      }

      /* Typography and headings */
      html, body, [class*="css"] {
        color: var(--text);
      }
      h1, h2, h3, h4 {
        color: var(--text);
      }
      h2 {
        border-left: 5px solid var(--accent);
        padding-left: 0.75rem;
        margin-top: 0.4rem;
      }
      h3 {
        margin-top: 0.2rem;
      }
      p, li, label, .stMarkdown, .stCaption {
        color: var(--text);
      }

      /* Metric / alert visuals */
      div[data-testid="stMetric"] {
        background: var(--accent-soft);
        padding: 0.85rem 0.9rem;
        border-radius: 16px;
        border: 1px solid #cce7e3;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
      }
      div[data-testid="stAlert"] {
        border-radius: 14px;
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

      div[data-testid="stTextInput"] input,
      div[data-testid="stTextArea"] textarea,
      div[data-testid="stNumberInput"] input,
      div[data-baseweb="select"] > div,
      div[data-baseweb="base-input"] > div {
        border-radius: 14px !important;
        border-color: #d2dfdf !important;
        background: rgba(255, 255, 255, 0.96) !important;
      }
      div[data-testid="stTextInput"] input:focus,
      div[data-testid="stTextArea"] textarea:focus,
      div[data-testid="stNumberInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px rgba(15, 118, 110, 0.18) !important;
      }

      div.stButton > button,
      div[data-testid="stFormSubmitButton"] > button {
        border-radius: 14px;
        border: 1px solid #c8d8d8;
        background: rgba(255, 255, 255, 0.96);
        color: var(--text);
        font-weight: 600;
        min-height: 2.95rem;
        line-height: 1.25;
        white-space: normal;
        padding: 0.6rem 1rem;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
        transition: border-color 0.16s ease, transform 0.16s ease, box-shadow 0.16s ease;
      }
      div.stButton > button:hover,
      div[data-testid="stFormSubmitButton"] > button:hover {
        border-color: var(--accent);
        box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
        transform: translateY(-1px);
      }
      div.stButton > button[kind="primary"],
      div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent), var(--accent-strong));
        color: #ffffff;
        border-color: transparent;
      }
      div[data-testid="stCheckbox"] {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid #d6e1e1;
        border-radius: 14px;
        padding: 0.2rem 0.75rem;
      }

      /* Card-like blocks (non-invasive selectors) */
      div[data-testid="stVerticalBlock"] > div.la-card,
      div[data-testid="stVerticalBlock"]:has(> div.la-card) {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        padding: 1.1rem 1.2rem;
        border-radius: 22px;
        margin-bottom: 1.2rem;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
        backdrop-filter: blur(6px);
      }
      div.la-card {
        display: none;
      }
      div[data-testid="stVerticalBlock"]:has(> div.la-action-bar) {
        background: linear-gradient(180deg, rgba(236, 254, 255, 0.95), rgba(248, 250, 252, 0.95));
        border: 1px solid #cfe3e2;
        padding: 0.85rem 1rem;
        border-radius: 18px;
        margin: 0.8rem 0 0.2rem;
      }
      div.la-action-bar {
        display: none;
      }

      div[data-testid="stTabs"] button[data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.55rem 1rem;
        background: rgba(255, 255, 255, 0.72);
      }
      div[data-testid="stTabs"] button[aria-selected="true"] {
        background: #ffffff;
        color: var(--accent-strong);
      }

      /* App header visuals */
      .app-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.15rem 1.35rem;
        border-radius: 24px;
        background:
          radial-gradient(circle at top right, rgba(245, 158, 11, 0.32), transparent 24%),
          linear-gradient(135deg, #0f766e, #164e63 72%);
        color: #fff;
        margin-bottom: 1.5rem;
        box-shadow: 0 18px 38px rgba(15, 23, 42, 0.14);
      }
      .app-logo {
        width: 54px;
        height: 54px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.16);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.08em;
        border: 1px solid rgba(255, 255, 255, 0.2);
      }
      .app-eyebrow {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        opacity: 0.86;
        margin-bottom: 0.2rem;
      }
      .app-title {
        font-size: 1.8rem;
        font-weight: 700;
      }
      .app-subtitle {
        font-size: 0.98rem;
        opacity: 0.92;
        max-width: 42rem;
      }

      .la-overview {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.85rem;
        margin: 0.25rem 0 1.3rem;
      }
      .la-overview-card,
      .la-step-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #d9e6e6;
        border-radius: 20px;
        padding: 1rem 1.05rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
      }
      .la-overview-label,
      .la-step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 2rem;
        min-height: 2rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent-strong);
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.65rem;
        padding: 0 0.55rem;
      }
      .la-overview-title,
      .la-step-title {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.35rem;
      }
      .la-overview-copy,
      .la-step-copy {
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.45;
        margin: 0;
      }
      .la-step-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.85rem;
        margin: 0.35rem 0 1rem;
      }
      .la-inline-banner {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid #d6e3e3;
        border-radius: 18px;
        padding: 0.95rem 1rem;
        margin: 0.4rem 0 1rem;
      }
      .la-inline-banner strong {
        display: block;
        margin-bottom: 0.25rem;
      }

      @media (max-width: 720px) {
        .app-header {
          align-items: flex-start;
        }
        div.block-container {
          padding-top: 1rem;
        }
      }

      /* Sticky bottom form area (keeps chat input visible) */
      div[data-testid="stForm"] {
        position: sticky;
        bottom: 0;
        background: rgba(255, 255, 255, 0.96);
        border-top: 1px solid #dce6e6;
        padding-top: 0.55rem;
        z-index: 10;
        backdrop-filter: blur(10px);
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
        <div class="app-eyebrow">Study workflow</div>
        <div class="app-title">Learning Assistant</div>
        <div class="app-subtitle">Turn one lecture into a clean loop: understand it, test yourself, fix weak spots, and review later.</div>
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

Options:
- SAFE embeddings are used by default.
- Recreate embeddings or build FAISS only when needed.

Study modes:
- **Ask** for focused lecture questions.
- **Summary** for quick overviews.
- **Chat** for back-and-forth understanding.

Learning tools:
- **Quiz** to test understanding.
- **Confused** to review weak concepts.

Review tools:
- **SRS** to revisit important items over time.
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
        - [Quiz](#study-quiz)
        - [Confused?](#confused-quick-prioritized-list)
        - [SRS](#spaced-repetition-review)
        """,
        unsafe_allow_html=True,
    )

st.markdown('<a id="setup-your-lecture"></a>', unsafe_allow_html=True)
st.markdown("## 1. Setup your lecture")
st.write(
    "Upload one lecture PDF from the sidebar, then create embeddings for it. "
    "Once that is ready, every study tool below works off the same lecture."
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
    st.markdown(
        """
        <div class="la-inline-banner">
          <strong>Upload lecture</strong>
          Your lecture is uploaded and processing happens automatically with SAFE embeddings.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(f"📄 Current lecture: {current_label}")

    # SAFE is the default behavior; keep it as a status, not a user choice.
    os.environ["USE_SAFE_EMBEDDINGS"] = "1"
    recreate_btn = False
    build_index_btn = False

    # Keep FAISS as the only optional performance choice.
    use_faiss_search = st.checkbox(
        "Enable fast search (FAISS) for larger lectures",
        value=bool(st.session_state.get("use_faiss_search", False)),
        help="Optional performance boost for retrieval. Regular search still works without this.",
    )
    st.session_state["use_faiss_search"] = use_faiss_search
    # Create embeddings when missing or when they are requested
    if (not embeddings_path.exists()) or recreate_btn:
        try:
            with st.spinner("Creating embeddings..."):
                create_embeddings_for_text(text, str(embeddings_path), dim=EMBED_DIM)
        except Exception as e:
            st.error("Failed to create embeddings.")
            st.exception(e)

    # Show a single completion status when setup is ready.
    try:
        ids, texts, vecs = load_embeddings(str(embeddings_path))
        if len(ids) > 0:
            st.success("Lecture is ready to study")
    except Exception:
        ids, texts, vecs = [], [], np.array([])
        st.warning("Processing not ready yet. Open Advanced reset tools if you need to recreate embeddings.")

    # Advanced/reset tools are hidden from the default setup flow.
    with st.expander("Advanced reset tools"):
        st.caption("Use these only when troubleshooting or resetting processed data.")
        adv_col1, adv_col2 = st.columns([1, 1])
        with adv_col1:
            recreate_btn = st.button("Recreate embeddings")
        with adv_col2:
            build_index_btn = st.button("Build FAISS index")

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
    st.markdown("## 2. Study modes")
    st.write(
        "Use the tabs below to understand the lecture in the way that fits best right now. "
        "The learning loop underneath turns that understanding into practice and review."
    )

    # Use slightly larger tabs with descriptions to improve discoverability
    tab1, tab2, tab3 = st.tabs(
        [
            "Ask",
            "Summary",
            "Chat",
        ]
    )
    if st.session_state.pop(f"open_chat_tab_{stem}", False):
        components.html(
            """
            <script>
            const clickChatTab = () => {
              const tabs = Array.from(parent.document.querySelectorAll('button[data-baseweb="tab"]'));
              const chatTab = tabs.find((btn) => (btn.innerText || "").trim().toLowerCase() === "chat");
              if (chatTab) {
                chatTab.click();
              }
            };
            setTimeout(clickChatTab, 0);
            setTimeout(clickChatTab, 120);
            </script>
            """,
            height=0,
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
        st.subheader("Ask a question")
        st.markdown("Type a focused question about the uploaded lecture.")
        question = st.text_input("Your question (based on uploaded lecture)", key="qa_question")
        # hide top-k input from users; keep value in session_state
        k = st.session_state.get("qa_k", 3)

        if st.button("Ask"):
            if len(ids) == 0:
                st.error("Embeddings not loaded. Create embeddings first.")
            elif not question or not question.strip():
                st.warning("Please enter a question.")
            else:
                candidate_index_path = str(index_path) if index_path.exists() else None
                with st.spinner("Searching and preparing answer..."):
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
                        latency = resp.get("latency", None)

                        display_answer = strip_retrieval_artifacts(strip_key_concepts_from_answer(ans or ""))

                        st.subheader("Answer")
                        st.write(display_answer or "")
                        if latency is not None:
                            st.caption(f"Latency: {latency:.3f}s")

                        with st.expander("Sources"):
                            if not retrieved_chunks:
                                st.info("No supporting context was retrieved.")
                            else:
                                for i, chunk in enumerate(retrieved_chunks, start=1):
                                    snippet = strip_retrieval_artifacts((chunk.get("text") or "").strip())
                                    if snippet:
                                        st.write(f"{i}. {snippet[:260]}{'...' if len(snippet) > 260 else ''}")

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
        st.subheader("Generate document-level summary")
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
                        summary_display = clean_summary_text(summary, summary_type=summary_type)
                        key_concepts = clean_key_concepts_list(resp.get("key_concepts", []) or [], summary_type=summary_type)
                        if not key_concepts:
                            key_concepts = derive_key_concepts_from_summary_text(summary_display, summary_type=summary_type)
                        used_chunks = resp.get("used_chunks", []) or []

                        st.subheader("Summary")
                        if summary_display:
                            st.markdown(summary_display)
                        else:
                            st.info("No summary returned.")
                        st.subheader("Key concepts / highlights")
                        if key_concepts:
                            st.markdown("\n".join(f"- {item}" for item in key_concepts))
                        else:
                            st.caption("The detailed summary above already includes the main concepts.")

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
    st.markdown("## 3. Learning loop")
    st.write(
        "Work straight down the page: test yourself, fix the concepts that keep tripping you up, then move the important ones into spaced repetition."
    )
    st.markdown(
        """
        <div class="la-step-grid">
          <div class="la-step-card">
            <div class="la-step-number">1</div>
            <div class="la-step-title">Quiz</div>
            <p class="la-step-copy">Generate a short MCQ set and see where your understanding is strongest or weakest.</p>
          </div>
          <div class="la-step-card">
            <div class="la-step-number">2</div>
            <div class="la-step-title">Confused</div>
            <p class="la-step-copy">Review the concepts you missed repeatedly and launch follow-up help without rewriting the prompt yourself.</p>
          </div>
          <div class="la-step-card">
            <div class="la-step-number">3</div>
            <div class="la-step-title">SRS</div>
            <p class="la-step-copy">Save the concepts worth retaining and review them on a lighter, more predictable cadence.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Step 1: Quiz (unchanged behavior — just a short intro)
    st.markdown("### 3.1 Test yourself")
    st.write("Generate a quick set of multiple-choice questions from the lecture and check your understanding.")
    # Render quiz directly (removed outer expander)
    render_quiz(st=st, stem=stem, text=text, llm=llm, hist_key=hist_key)

    st.markdown("---")

    # Step 2: Confused (prioritized list of things you missed)
    st.markdown("### 3.2 Review what confused you")
    st.write("This section surfaces the concepts you missed repeatedly, with quick actions for explanation, chat follow-up, and SRS.")
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
    st.markdown("### 3.3 Spaced repetition")
    st.write("Review due cards to move knowledge into long-term memory, or switch to Browse to view and manage all saved cards.")
    render_srs_section(st, stem=stem)

    st.markdown("---")
    st.caption("Tip: For reproducible tests set USE_SAFE_EMBEDDINGS=1 and build FAISS index to compare results with NumPy search.")
