import os
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

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
from backend.rag_query import (
    rag_answer_from_embeddings,
    rag_generate_summary_from_embeddings,
    rag_chat_answer,
)

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

Three primary modes:
- Ask a Question (fast, targeted) — default
- Generate Summary (document-level overview) — explicit
- Chat (conversational, maintains history) — new
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
    url = f"{api_base.rstrip('/')}/summarize"
    payload = {"embeddings_path": embeddings_path, "summary_type": summary_type, "top_k": top_k}
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


def call_chat_api(
    question: str,
    embeddings_path: str,
    history: Optional[List[Dict[str, str]]],
    top_k: int,
    use_faiss: bool,
    faiss_index_path: Optional[str] = None,
    api_base: str = API_DEFAULT,
    token: str = "",
) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/chat"
    payload = {
        "question": question,
        "embeddings_path": embeddings_path,
        "history": history or [],
        "top_k": top_k,
        "use_faiss": use_faiss,
        "faiss_index_path": faiss_index_path,
        "use_query_expansion": False,
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
    Remove any trailing 'Key Concepts' style block from an LLM answer for the Q&A / Chat UI.
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
    if not isinstance(summary, str):
        return str(summary)
    cleaned = re.sub(r"^\s*Here(?:'|’)s (?:a|an) (?:concise|brief) summary[^\n]*\n*[:\-]*\s*", "", summary, flags=re.IGNORECASE)
    return cleaned.strip()


# ----------------------------
# Chat-history trimming helper
# ----------------------------
# CHANGE: added a helper to silently trim chat history to a max number of turns (pairs)
MAX_CHAT_TURNS = 6  # maximum number of user+assistant pairs to retain (silent)

def _trim_history_to_max_turns(history: List[Dict[str, Any]], max_turns: int = MAX_CHAT_TURNS) -> List[Dict[str, Any]]:
    """
    Trim the provided history to the last `max_turns` user+assistant pairs.
    This function is robust to imperfect alternation: it groups messages into pairs where possible,
    then returns the last max_turns pairs flattened back into a list of messages in original order.
    """
    if not history:
        return history

    pairs: List[List[Dict[str, Any]]] = []
    i = 0
    n = len(history)
    while i < n:
        role = (history[i].get("role") or "").lower()
        if role.startswith("user"):
            user_msg = history[i]
            # look ahead for next assistant
            if i + 1 < n and (history[i + 1].get("role") or "").lower().startswith("assistant"):
                assistant_msg = history[i + 1]
                pairs.append([user_msg, assistant_msg])
                i += 2
            else:
                # incomplete pair (user only)
                pairs.append([user_msg])
                i += 1
        elif role.startswith("assistant"):
            # assistant without preceding user
            pairs.append([history[i]])
            i += 1
        else:
            # unknown role - keep as single
            pairs.append([history[i]])
            i += 1

    # keep only the last max_turns pairs
    if len(pairs) <= max_turns:
        # nothing to trim
        flattened = [m for p in pairs for m in p]
        return flattened

    kept_pairs = pairs[-max_turns:]
    flattened = [m for p in kept_pairs for m in p]
    return flattened


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

    # UI controls
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

    # Build FAISS index if requested
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
    tab1, tab2, tab3 = st.tabs(["Ask a Question", "Generate Summary", "Chat"])

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

                        display_answer = _strip_key_concepts_from_answer(ans or "")

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
                            summary = out.get("summary", "")
                            key_concepts = out.get("key_concepts", [])
                        summary_display = _clean_summary_text(summary)

                        st.subheader("Summary")
                        st.write(summary_display or "")

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
    # Tab: Chat (conversation)
    # -----------------------
    with tab3:
        st.subheader("Chat with the lecture (conversational)")

        # ensure a per-document chat history in session state
        hist_key = f"chat_history_{stem}"
        if hist_key not in st.session_state:
            st.session_state[hist_key] = []  # list of {"role": "user"/"assistant", "content": "...", "meta": {...}}

        # Chat controls (keep above the form)
        chat_k = st.number_input("Top-k chunks to retrieve for each turn", min_value=1, max_value=10, value=3, step=1, key="chat_k")
        # CHANGE: renamed button label to "Clear chat" per UX request
        if st.button("Clear chat"):
            # Reset only this document's chat history (does not affect other tabs)
            st.session_state[hist_key] = []
            st.success("Chat history cleared for this document.")

        # --- Input form comes first now ---
        with st.form(key=f"chat_form_{stem}"):
            user_msg = st.text_input("Message to assistant", key=f"chat_input_{stem}")
            send_pressed = st.form_submit_button("Send")

        # If the user submitted, process and update session_state before rendering the history
        if send_pressed:
            if len(ids) == 0:
                st.error("Embeddings not loaded. Create embeddings first.")
            elif not user_msg or not user_msg.strip():
                st.warning("Please enter a message.")
            else:
                candidate_index_path = str(index_path) if index_path.exists() else None
                payload_history = [{"role": h.get("role"), "content": h.get("content")} for h in st.session_state[hist_key]]

                with st.spinner("Running conversational RAG..."):
                    try:
                        if st.session_state.use_api_mode:
                            api_base = os.getenv("API_BASE", API_DEFAULT)
                            token = st.session_state.api_token or ""
                            resp = call_chat_api(
                                question=user_msg,
                                embeddings_path=str(embeddings_path),
                                history=payload_history,
                                top_k=int(chat_k),
                                use_faiss=bool(use_faiss_search),
                                faiss_index_path=candidate_index_path,
                                api_base=api_base,
                                token=token,
                            )
                            ans = resp.get("answer")
                            updated_history = resp.get("history", None)
                            retrieved = resp.get("retrieved", [])
                            prompt_used = resp.get("prompt")
                            provenance = resp.get("provenance")
                        else:
                            qa_out = rag_chat_answer(
                                user_msg,
                                str(embeddings_path),
                                history=payload_history,
                                top_k=int(chat_k),
                                use_faiss=bool(use_faiss_search),
                                faiss_index_path=candidate_index_path,
                                use_safe=(True if os.environ.get("USE_SAFE_EMBEDDINGS", "1") in ("1", "true", "yes") else False),
                                use_query_expansion=False,
                                return_meta=True,
                                llm_call=None,
                            )
                            if isinstance(qa_out, (tuple, list)):
                                ans = qa_out[0]
                                updated_history = qa_out[1] if len(qa_out) >= 2 else None
                                retrieved = qa_out[2] if len(qa_out) >= 3 else []
                                prompt_used = qa_out[3] if len(qa_out) >= 4 else None
                                provenance = qa_out[4] if len(qa_out) >= 5 else None
                            else:
                                ans = str(qa_out)
                                updated_history = None
                                retrieved = []
                                prompt_used = None
                                provenance = None

                        display_answer = _strip_key_concepts_from_answer(ans or "")

                        if updated_history:
                            # Convert backend history into local structure and sanitize assistant text
                            new_hist: List[Dict[str, Any]] = []
                            for h in updated_history:
                                role = h.get("role", "user")
                                content_raw = h.get("content", "") or ""
                                if role and role.lower().startswith("assistant"):
                                    content = _strip_key_concepts_from_answer(content_raw)
                                else:
                                    content = content_raw
                                new_hist.append({"role": role, "content": content})
                            # attach meta to last assistant turn
                            if new_hist and new_hist[-1].get("role") and new_hist[-1]["role"].lower().startswith("assistant"):
                                new_hist[-1]["meta"] = {"retrieved": retrieved or [], "prompt": prompt_used, "provenance": provenance}
                            # CHANGE: enforce silent max history trimming immediately after updating history
                            st.session_state[hist_key] = _trim_history_to_max_turns(new_hist, max_turns=MAX_CHAT_TURNS)
                        else:
                            # Append locally (user + assistant); assistant content already sanitized
                            st.session_state[hist_key].append({"role": "user", "content": user_msg})
                            st.session_state[hist_key].append(
                                {"role": "assistant", "content": display_answer, "meta": {"retrieved": retrieved or [], "prompt": prompt_used, "provenance": provenance}}
                            )
                            # CHANGE: enforce silent max history trimming immediately after appending new messages
                            st.session_state[hist_key] = _trim_history_to_max_turns(st.session_state[hist_key], max_turns=MAX_CHAT_TURNS)

                        # Provide immediate feedback
                        st.success("Assistant replied — see chat above.")
                    except Exception as e:
                        st.error("Conversational RAG failed.")
                        st.exception(e)

        # --- Now render the chat history (after any update above) ---
        st.markdown("----")
        chat_history: List[Dict[str, Any]] = st.session_state[hist_key]
        if not chat_history:
            st.info("No messages yet. Start by asking a question below.")
        else:
            for turn in chat_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                # Always present sanitized assistant content at render time as well (safety)
                if role and role.lower().startswith("assistant"):
                    content = _strip_key_concepts_from_answer(content or "")
                meta = turn.get("meta", {}) or {}
                if role == "user":
                    st.markdown(f"**You:**  \n{content}")
                else:
                    st.markdown(f"**Assistant:**  \n{content}")
                    if meta:
                        retrieved = meta.get("retrieved", []) or []
                        prov = meta.get("provenance")
                        prompt_used = meta.get("prompt")
                        if retrieved:
                            with st.expander("Retrieved chunks (turn)"):
                                for c in retrieved:
                                    try:
                                        st.write(f"- id={c.get('id')} pos={c.get('pos')} score={c.get('score'):.4f}")
                                    except Exception:
                                        st.write(f"- {c}")
                        if prov and prov.get("sentences"):
                            with st.expander("Provenance (this turn)"):
                                for s in prov.get("sentences", []):
                                    st.write(f"- \"{s.get('sentence')}\" → chunk={s.get('chunk_id')} (score={s.get('score'):.3f})")
                        if prompt_used:
                            with st.expander("Prompt used (debug)"):
                                st.code(str(prompt_used)[:4000])

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
