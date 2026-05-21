# frontend/sections/chat.py
"""
Chat / RAG UI extracted from frontend/app.py — UX-optimized presentation only.

Expose:
    def render(st, stem, llm)

Behavior & state keys are unchanged — only presentation/layout has been improved.
This file preserves all logic, call signatures, and session_state keys exactly.
"""

import os
import uuid
import re
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

import streamlit as st
import streamlit.components.v1 as components

# handlers + ui helpers (same as used previously in app.py)
from frontend.handlers import perform_chat, perform_query
from frontend.runtime_ui_helpers import (
    strip_key_concepts_from_answer,
    strip_retrieval_artifacts,
    trim_history_to_max_turns,
    build_chat_html,
    scroll_to_anchor,
)

CHAT_SECTION_ANCHOR = "la-chat-section-anchor"

API_DEFAULT = os.getenv("API_BASE", "http://localhost:8000")

# --------------------------
# Quiz helpers (chat mode)
# --------------------------
_QUESTION_PREFIXES = ("q:", "question:", "new question:", "ask:")
_ANSWER_PREFIXES = ("a:", "answer:", "ans:")
_QUESTION_STARTERS = (
    "what", "why", "how", "when", "where", "who", "which",
    "explain", "define", "describe", "tell me", "give me", "list",
    "compare", "contrast",
)


def _strip_inline_instructions(text: str) -> str:
    """Remove auto-appended Instructions block from stored user messages."""
    if not text:
        return ""
    parts = str(text).split("\n\nInstructions:", 1)
    return parts[0].strip()


def _normalize_user_msg(text: str) -> Tuple[str, str]:
    """Return (clean_text, mode) where mode is 'question', 'answer', or 'unknown'."""
    raw = (text or "").strip()
    low = raw.lower()
    for p in _QUESTION_PREFIXES:
        if low.startswith(p):
            return raw[len(p):].lstrip(), "question"
    for p in _ANSWER_PREFIXES:
        if low.startswith(p):
            return raw[len(p):].lstrip(), "answer"
    return raw, "unknown"


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if "?" in t:
        return True
    low = t.lower()
    for starter in _QUESTION_STARTERS:
        if low.startswith(starter + " ") or low == starter:
            return True
    return False


def _extract_last_question(text: str) -> str:
    """Return the last question (ending with '?') if present; otherwise empty string."""
    if not text:
        return ""
    idx = text.rfind("?")
    if idx < 0:
        return ""
    # Find a sensible start boundary
    start = max(text.rfind("\n", 0, idx), text.rfind(".", 0, idx), text.rfind("!", 0, idx))
    start = 0 if start < 0 else start + 1
    q = text[start:idx + 1].strip()
    q = re.sub(r"^(quiz|question|q)\s*[:\-]\s*", "", q, flags=re.IGNORECASE)
    # Require a minimal length to avoid capturing stray '?'
    if len(q.split()) < 3:
        return ""
    if not q.endswith("?"):
        q = q + "?"
    return q


def _fallback_quiz_question(retrieved: List[Dict[str, Any]]) -> str:
    """Heuristic fallback to ensure a quiz question exists when the model omits one."""
    if not retrieved:
        return "What is one key idea discussed in the lecture?"
    text = (retrieved[0].get("text") or "").strip()
    if not text:
        return "What is one key idea discussed in the lecture?"
    # Take the first sentence as a seed
    first = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    if not first:
        return "What is one key idea discussed in the lecture?"
    # Pattern: "X stands for Y" -> "What does X stand for?"
    m = re.match(r"(.+?)\s+stands\s+for\s+.+", first, flags=re.IGNORECASE)
    if m:
        subj = m.group(1).strip()
        if subj:
            return f"What does {subj} stand for?"
    # Pattern: "X is/are Y" -> "What is/are X?"
    m = re.match(r"(.+?)\s+(is|are)\s+.+", first, flags=re.IGNORECASE)
    if m:
        subj = m.group(1).strip()
        verb = m.group(2).lower()
        if subj and len(subj.split()) <= 8:
            return f"What {verb} {subj}?"
    return "What is one key idea discussed in the lecture?"


def _format_saved_timestamp(value: str) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(str(value))
        return dt.strftime("%b %d, %Y %I:%M %p")
    except Exception:
        return str(value)


def _derive_default_chat_title(history: List[Dict[str, Any]], fallback_index: int) -> str:
    first_user = ""
    for turn in history or []:
        if str(turn.get("role", "")).lower().startswith("user"):
            first_user = (turn.get("content") or "").strip()
            break
    first_user = re.sub(r"\s+", " ", first_user).strip()
    if first_user:
        words = first_user.split()
        short = " ".join(words[:7]).strip()
        if len(words) > 7:
            short += "..."
        return short
    return f"Saved chat {fallback_index}"


def _submit_chat_message(
    st: Any,
    *,
    user_msg: str,
    stem: str,
    hist_key: str,
    embeddings_path: Path,
    index_path: Path,
    embeddings_ready: bool,
    chat_k: int,
    explain_new: bool,
    include_example: bool,
    turn_quiz: bool,
    llm,
    mod_reset_key: str,
    clear_input_key: str,
    from_confusion_followup: bool = False,
    show_spinner: bool = True,
) -> bool:
    """Run the chat send pipeline. Returns True if a message was sent."""
    loading_key = f"chat_followup_loading_{stem}"
    scroll_key = f"scroll_to_chat_{stem}"
    scroll_align_key = f"chat_scroll_align_{stem}"
    try:
        return _submit_chat_message_impl(
            st,
            user_msg=user_msg,
            stem=stem,
            hist_key=hist_key,
            embeddings_path=embeddings_path,
            index_path=index_path,
            embeddings_ready=embeddings_ready,
            chat_k=chat_k,
            explain_new=explain_new,
            include_example=include_example,
            turn_quiz=turn_quiz,
            llm=llm,
            mod_reset_key=mod_reset_key,
            clear_input_key=clear_input_key,
            from_confusion_followup=from_confusion_followup,
            show_spinner=show_spinner,
            scroll_key=scroll_key,
            scroll_align_key=scroll_align_key,
        )
    finally:
        st.session_state.pop(loading_key, None)


def _submit_chat_message_impl(
    st: Any,
    *,
    user_msg: str,
    stem: str,
    hist_key: str,
    embeddings_path: Path,
    index_path: Path,
    embeddings_ready: bool,
    chat_k: int,
    explain_new: bool,
    include_example: bool,
    turn_quiz: bool,
    llm,
    mod_reset_key: str,
    clear_input_key: str,
    from_confusion_followup: bool,
    show_spinner: bool,
    scroll_key: str,
    scroll_align_key: str,
) -> bool:
    """Inner send pipeline (loading flag cleared by wrapper)."""
    if not embeddings_ready:
        st.error("Embeddings not loaded. Create embeddings first.")
        return False
    if not user_msg or not str(user_msg).strip():
        st.warning("Please enter a message.")
        return False

    pending_quiz_key = f"chat_pending_quiz_{stem}"
    pending_quiz = st.session_state.get(pending_quiz_key)
    pending_question = ""
    if isinstance(pending_quiz, dict):
        pending_question = str(pending_quiz.get("question") or "").strip()
    elif isinstance(pending_quiz, str):
        pending_question = pending_quiz.strip()

    user_msg_clean, msg_mode = _normalize_user_msg(user_msg)
    force_question = msg_mode == "question"
    force_answer = msg_mode == "answer"
    force_flag_key = f"chat_force_question_{stem}"
    force_flag = bool(st.session_state.pop(force_flag_key, False))
    if force_flag and not force_answer:
        force_question = True
    looks_question = _looks_like_question(user_msg_clean)

    answer_mode = False
    if pending_question and not force_question and (force_answer or not looks_question):
        answer_mode = True
    else:
        if pending_question:
            st.session_state[pending_quiz_key] = None

    payload_history = []
    for h in st.session_state.get(hist_key, []) or []:
        role = h.get("role")
        content = h.get("content", "") or ""
        if role and str(role).lower().startswith("user"):
            content = _strip_inline_instructions(content)
        payload_history.append({"role": role, "content": content})

    modifiers = []
    if explain_new:
        modifiers.append("Explain as if the learner is new to the topic.")
    if include_example:
        modifiers.append("Include one concrete example.")
    if turn_quiz:
        modifiers.append(
            "End with one quiz question (single sentence ending with '?'). "
            "Do not add any text after the question."
        )

    if answer_mode:
        prompt_question = (
            "You are grading a student's answer to a quiz question about the lecture. "
            "Use ONLY the provided context.\n\n"
            f"Quiz question:\n{pending_question}\n\n"
            f"Student answer:\n{user_msg_clean}\n\n"
            "Provide a brief verdict (Correct/Partially correct/Incorrect) and a short explanation. "
            "If incorrect or incomplete, state the correct answer succinctly."
        )
    else:
        prompt_question = user_msg_clean

    if modifiers:
        prompt_question = f"{prompt_question}\n\nInstructions: " + " ".join(modifiers)
    candidate_index_path = str(index_path) if index_path.exists() else None
    use_faiss_search = bool(st.session_state.get("use_faiss_search", False))

    spinner_msg = "Searching and generating response..."
    spinner_ctx = st.spinner(spinner_msg) if show_spinner else nullcontext()
    with spinner_ctx:
        try:
            if st.session_state.get("use_api_mode", False):
                resp = perform_chat(
                    question=prompt_question,
                    embeddings_path=str(embeddings_path),
                    history=payload_history,
                    top_k=int(chat_k),
                    use_faiss=bool(use_faiss_search),
                    faiss_index_path=candidate_index_path,
                    use_api_mode=True,
                    api_base=os.getenv("API_BASE", API_DEFAULT),
                    token=st.session_state.get("api_token", "") or "",
                    llm_call=None,
                )
            else:
                resp = perform_chat(
                    question=prompt_question,
                    embeddings_path=str(embeddings_path),
                    history=payload_history,
                    top_k=int(chat_k),
                    use_faiss=bool(use_faiss_search),
                    faiss_index_path=candidate_index_path,
                    use_api_mode=False,
                    llm_call=llm,
                )

            ans = resp.get("answer")
            updated_history = resp.get("history", None)
            retrieved = resp.get("retrieved", []) or []
            prompt_used = resp.get("prompt")
            provenance = resp.get("provenance")
            display_answer = strip_retrieval_artifacts(strip_key_concepts_from_answer(ans or ""))

            quiz_question = ""
            if turn_quiz:
                quiz_question = _extract_last_question(display_answer)
                if not quiz_question:
                    fallback_q = _fallback_quiz_question(retrieved)
                    if fallback_q:
                        display_answer = display_answer.strip()
                        if display_answer:
                            display_answer = f"{display_answer}\n\n{fallback_q}"
                        else:
                            display_answer = fallback_q
                        quiz_question = _extract_last_question(display_answer) or fallback_q

            if updated_history:
                new_hist: List[Dict[str, str]] = []
                for h in updated_history:
                    role = h.get("role", "user")
                    content_raw = h.get("content", "") or ""
                    if role and role.lower().startswith("assistant"):
                        content = strip_key_concepts_from_answer(content_raw)
                    else:
                        content = _strip_inline_instructions(content_raw)
                    new_hist.append({"role": role, "content": content})
                for i in range(len(new_hist) - 1, -1, -1):
                    if (new_hist[i].get("role") or "").lower().startswith("user"):
                        new_hist[i]["content"] = user_msg_clean
                        break
                for i in range(len(new_hist) - 1, -1, -1):
                    if (new_hist[i].get("role") or "").lower().startswith("assistant"):
                        new_hist[i]["content"] = display_answer
                        break
                st.session_state[hist_key] = trim_history_to_max_turns(new_hist, max_turns=60)
            else:
                st.session_state[hist_key].append({"role": "user", "content": user_msg_clean})
                st.session_state[hist_key].append({
                    "role": "assistant",
                    "content": display_answer,
                    "meta": {"retrieved": retrieved or [], "prompt": prompt_used, "provenance": provenance},
                })
                st.session_state[hist_key] = trim_history_to_max_turns(st.session_state[hist_key], max_turns=60)

            st.session_state[mod_reset_key] = True
            st.session_state[clear_input_key] = True

            if turn_quiz and quiz_question:
                st.session_state[pending_quiz_key] = {"question": quiz_question}
            else:
                st.session_state[pending_quiz_key] = None

            st.session_state[scroll_align_key] = "last_user"
            if from_confusion_followup:
                st.session_state[scroll_key] = True
                st.session_state[f"open_chat_tab_{stem}"] = True

            st.success("Assistant replied — chat updated.")
            try:
                st.rerun()
            except Exception:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass
            return True
        except Exception as e:
            st.error("Conversational RAG failed.")
            st.exception(e)
            return False


def render(
    st: Any,
    stem: str,
    llm,
    embeddings_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
    embeddings_ready: Optional[bool] = None,
    use_faiss_search: Optional[bool] = None,
):
    """
    Render the Chat / conversational RAG UI for the given document stem.

    Args:
        st: streamlit module (passed from app.py)
        stem: document stem (string) used to build session keys
        llm: local llm object or None
        embeddings_path: user-scoped embeddings JSON path
        index_path: user-scoped FAISS index path
        embeddings_ready: whether lecture embeddings are loaded
        use_faiss_search: whether to use FAISS for retrieval
    """
    from backend.user_context import get_embeddings_path, get_index_path

    if embeddings_path is None:
        embeddings_path = get_embeddings_path(stem)
    else:
        embeddings_path = Path(embeddings_path)
    if index_path is None:
        index_path = get_index_path(stem)
    else:
        index_path = Path(index_path)
    if embeddings_ready is None:
        embeddings_ready = embeddings_path.exists()
    if use_faiss_search is not None:
        st.session_state["use_faiss_search"] = bool(use_faiss_search)

    # session keys (same naming as app.py)
    hist_key = f"chat_history_{stem}"
    saved_chats_key = f"saved_chats_{stem}"
    history_toggle_key = f"show_history_{stem}"

    # initialize session state entries if missing (same as before)
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []
    if saved_chats_key not in st.session_state:
        st.session_state[saved_chats_key] = []
    if history_toggle_key not in st.session_state:
        st.session_state[history_toggle_key] = False

    st.markdown(
        f'<div id="{CHAT_SECTION_ANCHOR}" style="scroll-margin-top:5.5rem;"></div>',
        unsafe_allow_html=True,
    )
    st.subheader("💬 Chat with the lecture (conversational)")

    # Apply any pending modifier reset BEFORE widgets are instantiated
    mod_reset_key = f"chat_mod_reset_{stem}"
    if st.session_state.pop(mod_reset_key, False):
        st.session_state[f"chat_mod_new_{stem}"] = False
        st.session_state[f"chat_mod_example_{stem}"] = False
        st.session_state[f"chat_mod_quiz_{stem}"] = False
    response_style = st.selectbox(
        "Response style",
        [
            "Standard",
            "Beginner friendly",
            "Include a practical example",
            "Beginner friendly + knowledge check",
        ],
        key=f"chat_style_{stem}",
        help="Choose how the assistant structures its reply.",
    )
    explain_new = response_style in ("Beginner friendly", "Beginner friendly + knowledge check")
    include_example = response_style == "Include a practical example"
    turn_quiz = response_style == "Beginner friendly + knowledge check"

    # --------------------------
    # Top controls: save, history buttons, clear
    # NOTE: keep top-k internal/default but hide input from users
    # (preserve behavior but do not show the number input)
    # --------------------------
    chat_k_key = f"chat_k_{stem}"
    if chat_k_key not in st.session_state:
        st.session_state[chat_k_key] = 3
    chat_k = st.session_state[chat_k_key]

    save_title_key = f"save_title_{stem}"
    st.text_input("Save conversation as (optional)", key=save_title_key, placeholder="Title (optional)")

    # Cleaner action row: save, history toggle, clear
    action_cols = st.columns(3)
    # Save current conversation
    with action_cols[0]:
        if st.button("💾 Save conversation", key=f"save_conv_{stem}", use_container_width=True):
            current = st.session_state.get(hist_key, []) or []
            if not current:
                st.warning("Nothing to save — conversation is empty.")
            else:
                title_raw = (st.session_state.get(save_title_key) or "").strip()
                inferred_title = _derive_default_chat_title(current, fallback_index=len(st.session_state[saved_chats_key]) + 1)
                new_item = {
                    "id": str(uuid.uuid4()),
                    "title": title_raw or inferred_title,
                    "history": current,
                    "created": datetime.utcnow().isoformat(),
                }
                st.session_state[saved_chats_key].insert(0, new_item)
                st.success("Conversation saved.")

    # Single history toggle button
    with action_cols[1]:
        history_open = bool(st.session_state.get(history_toggle_key, False))
        if st.button("📚 Open history" if not history_open else "📚 Close history", key=f"toggle_history_btn_{stem}", use_container_width=True):
            st.session_state[history_toggle_key] = not history_open

    # Clear chat
    with action_cols[2]:
        if st.button("🧹 Clear chat", key=f"clear_chat_{stem}", use_container_width=True):
            st.session_state[hist_key] = []
            st.success("Chat cleared.")

    st.markdown("---")

    # --------------------------
    # History view (when toggled on): history-only UI, hides chat & input
    # --------------------------
    if st.session_state.get(history_toggle_key):
        st.markdown("## Saved conversations (this document)")
        if st.button("Back to active chat", key=f"history_back_to_chat_{stem}", use_container_width=True):
            st.session_state[history_toggle_key] = False
            try:
                st.rerun()
            except Exception:
                pass
        saved = st.session_state.get(saved_chats_key, []) or []
        if not saved:
            st.info("No saved conversations for this document yet.")
        else:
            # Render saved conversations with improved spacing and clear buttons
            for idx, item in enumerate(saved):
                title = item.get("title", f"Saved chat {idx+1}")
                created = _format_saved_timestamp(item.get("created", ""))
                with st.expander(f"{title}" + (f" · {created}" if created else ""), expanded=False):
                    preview = item.get("history", []) or []
                    if not preview:
                        st.write("_(empty conversation)_")
                    else:
                        # Show preview in readable blocks with badges for role
                        for turn in preview:
                            role = (turn.get("role") or "user").lower()
                            content = turn.get("content", "") or ""
                            if role.startswith("assistant"):
                                st.markdown(f"**Assistant:**  \n{content}")
                            else:
                                st.markdown(f"**You:**  \n{content}")

                    btns = st.columns(2)
                    # Load and return to chat view
                    if btns[0].button("Load in Chat", key=f"load_saved_{stem}_{idx}", use_container_width=True):
                        st.session_state[hist_key] = item.get("history", []) or []
                        st.session_state[history_toggle_key] = False
                        st.success(f"Loaded: {item.get('title')}")
                        try:
                            st.rerun()
                        except Exception:
                            pass
                    # Delete
                    if btns[1].button("Delete", key=f"del_saved_{stem}_{idx}", use_container_width=True):
                        st.session_state[saved_chats_key].pop(idx)
                        st.success("Deleted saved conversation.")
                        try:
                            st.rerun()
                        except Exception:
                            pass

        st.markdown("---")
        st.info("History mode: select 'Load in Chat' to return to chat mode with that conversation loaded.")
        # Skip rendering the main chat & input while in history mode
        return

    # --------------------------
    # Main chat view: render after form handling so updated history shows immediately
    # --------------------------
    chat_container = st.container()
    clear_input_key = f"chat_clear_input_{stem}"

    pending_loaded = False

    # Clear input on next run after a successful send
    if st.session_state.pop(clear_input_key, False) and not pending_loaded:
        st.session_state[f"chat_input_{stem}"] = ""

    # Input form (clear_on_submit=True so Streamlit clears the input automatically)
    with st.form(key=f"chat_form_{stem}", clear_on_submit=True):
        # Use text_area for multi-line input display; pending/clear prompts set chat_input_* above
        user_msg = st.text_area(
            "Message to assistant",
            key=f"chat_input_{stem}",
            placeholder="Type your message here...",
            height=100,
        )
        if st.session_state.pop(f"chat_focus_input_{stem}", False):
            components.html(
                """
                <script>
                const focusChat = () => {
                  const nodes = parent.document.querySelectorAll('textarea');
                  const target = Array.from(nodes).find((el) => (el.getAttribute('aria-label') || '').toLowerCase().includes('message to assistant'));
                  if (target) {
                    target.focus();
                    target.selectionStart = target.value.length;
                    target.selectionEnd = target.value.length;
                  }
                };
                setTimeout(focusChat, 0);
                setTimeout(focusChat, 120);
                </script>
                """,
                height=0,
            )
        send_pressed = st.form_submit_button("Send")

        if send_pressed:
            _submit_chat_message(
                st,
                user_msg=user_msg,
                stem=stem,
                hist_key=hist_key,
                embeddings_path=embeddings_path,
                index_path=index_path,
                embeddings_ready=bool(embeddings_ready),
                chat_k=int(chat_k),
                explain_new=explain_new,
                include_example=include_example,
                turn_quiz=turn_quiz,
                llm=llm,
                mod_reset_key=mod_reset_key,
                clear_input_key=clear_input_key,
            )

    # Render the chat AFTER handling the form so latest reply is visible immediately
    chat_history = st.session_state.get(hist_key, []) or []

    display_history = []
    for turn in chat_history:
        role = turn.get("role", "user")
        content = turn.get("content", "") or ""
        if str(role).lower().startswith("assistant"):
            content = strip_retrieval_artifacts(content)
        display_history.append({"role": role, "content": content})

    scroll_align = st.session_state.pop(f"chat_scroll_align_{stem}", "bottom")
    chat_html, chat_height = build_chat_html(
        display_history,
        max_height=720,
        scroll_align=scroll_align,
    )

    with chat_container:
        st.markdown('<div class="chat-fade-top"></div>', unsafe_allow_html=True)
        # Surround chat with a subtle header that shows count & quick actions
        header_cols = st.columns([3, 1])
        with header_cols[0]:
            st.markdown(f"**Conversation — {len(chat_history)} turns**")
        with header_cols[1]:
            st.caption(
                "Latest message at top"
                if scroll_align == "last_user"
                else "Auto-scroll enabled"
            )

        components.html(chat_html, height=chat_height, scrolling=False)

    if st.session_state.pop(f"scroll_to_chat_{stem}", False):
        scroll_to_anchor(CHAT_SECTION_ANCHOR)
