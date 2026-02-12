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
from pathlib import Path
from typing import Any, List, Dict

import streamlit as st
import streamlit.components.v1 as components

# handlers + ui helpers (same as used previously in app.py)
from frontend.handlers import perform_chat, perform_query
from frontend.ui_helpers import (
    strip_key_concepts_from_answer,
    trim_history_to_max_turns,
    build_chat_html,
)

API_DEFAULT = os.getenv("API_BASE", "http://localhost:8000")


def render(st: Any, stem: str, llm):
    """
    Render the Chat / conversational RAG UI for the given document stem.

    Args:
        st: streamlit module (passed from app.py)
        stem: document stem (string) used to build session keys
        llm: local llm object or None
    """
    # Re-derive embeddings/index paths (same logic as in app.py)
    embeddings_path = Path(f"data/processed/{stem}_embeddings.json")
    index_path = Path(f"data/processed/{stem}_embeddings.index")

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

    st.subheader("💬 Chat with the lecture (conversational)")
    mod_cols = st.columns(3)
    with mod_cols[0]:
        explain_new = st.checkbox("Explain like I'm new to this", key=f"chat_mod_new_{stem}")
    with mod_cols[1]:
        include_example = st.checkbox("Give me an example", key=f"chat_mod_example_{stem}")
    with mod_cols[2]:
        turn_quiz = st.checkbox("Turn this into a quiz question", key=f"chat_mod_quiz_{stem}")

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

    # Better layout for the action buttons: grouped with icons and compact descriptions
    action_cols = st.columns([1.5, 1, 1, 1])
    # Save current conversation
    with action_cols[0]:
        if st.button("💾 Save conversation", key=f"save_conv_{stem}"):
            current = st.session_state.get(hist_key, []) or []
            if not current:
                st.warning("Nothing to save — conversation is empty.")
            else:
                new_item = {
                    "id": str(uuid.uuid4()),
                    "title": st.session_state.get(save_title_key) or f"Chat {st.session_state.get('save_title_time', '') or ''}{uuid.uuid4()}",
                    "history": current,
                    "created": __import__("datetime").datetime.utcnow().isoformat(),
                }
                st.session_state[saved_chats_key].insert(0, new_item)
                st.success("Conversation saved.")

    # Show / Hide history buttons (stable behavior)
    with action_cols[1]:
        if st.button("📚 Show history", key=f"show_history_btn_{stem}"):
            st.session_state[history_toggle_key] = True
    with action_cols[2]:
        if st.button("🗂️ Hide history", key=f"hide_history_btn_{stem}"):
            st.session_state[history_toggle_key] = False

    # Clear chat
    with action_cols[3]:
        if st.button("🧹 Clear chat", key=f"clear_chat_{stem}"):
            st.session_state[hist_key] = []
            st.success("Chat cleared.")

    st.markdown("---")

    # --------------------------
    # History view (when toggled on): history-only UI, hides chat & input
    # --------------------------
    if st.session_state.get(history_toggle_key):
        st.markdown("## Saved conversations (this document)")
        saved = st.session_state.get(saved_chats_key, []) or []
        if not saved:
            st.info("No saved conversations for this document yet.")
        else:
            # Render saved conversations with improved spacing and clear buttons
            for idx, item in enumerate(saved):
                title = item.get("title", f"Saved chat {idx+1}")
                created = item.get("created", "")
                with st.expander(f"{title} — saved {created}", expanded=False):
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

                    btns = st.columns([1, 1, 1])
                    # Load (replace) and return to chat view
                    if btns[0].button("Load (replace) → open chat", key=f"load_saved_{stem}_{idx}"):
                        st.session_state[hist_key] = item.get("history", []) or []
                        st.session_state[history_toggle_key] = False
                        st.success(f"Loaded: {item.get('title')} — switching to chat view.")
                    # Append and return to chat view
                    if btns[1].button("Append → open chat", key=f"append_saved_{stem}_{idx}"):
                        st.session_state[hist_key].extend(item.get("history", []) or [])
                        st.session_state[hist_key] = trim_history_to_max_turns(st.session_state[hist_key], max_turns=60)
                        st.session_state[history_toggle_key] = False
                        st.success(f"Appended: {item.get('title')} — switching to chat view.")
                    # Delete
                    if btns[2].button("Delete", key=f"del_saved_{stem}_{idx}"):
                        st.session_state[saved_chats_key].pop(idx)
                        st.success("Deleted saved conversation.")

        st.markdown("---")
        st.info("History mode: select 'Load' or 'Append' to return to chat mode with that conversation loaded.")
        # Skip rendering the main chat & input while in history mode
        return

    # --------------------------
    # Main chat view: render after form handling so updated history shows immediately
    # --------------------------
    chat_container = st.container()

    # Optional UI-only control: collapse older messages (doesn't modify stored history)
    collapse_key = f"chat_collapse_{stem}"
    if collapse_key not in st.session_state:
        st.session_state[collapse_key] = False
    col_collapse = st.columns([4, 1])
    with col_collapse[1]:
        if st.button("Toggle collapse old", key=f"{collapse_key}_btn"):
            st.session_state[collapse_key] = not st.session_state[collapse_key]
    
    # Check for pending input from Confused section
    pending_input_key = f"chat_pending_input_{stem}"
    pending_input = st.session_state.pop(pending_input_key, None)
    if pending_input:
        st.session_state[f"chat_input_{stem}"] = pending_input
        st.success("✅ Follow-up prompt loaded! Edit and send below.")

    # Input form (clear_on_submit=True so Streamlit clears the input automatically)
    with st.form(key=f"chat_form_{stem}", clear_on_submit=True):
        # Use text_area for multi-line input display
        # Explicitly set value from session_state to show pending follow-up prompts
        user_msg = st.text_area(
            "Message to assistant",
            value=st.session_state.get(f"chat_input_{stem}", ""),
            key=f"chat_input_{stem}",
            placeholder="Type your message here...",
            height=100,
        )
        send_pressed = st.form_submit_button("Send")

        if send_pressed:
            # Basic validation (preserve behavior)
            if not embeddings_path.exists():
                st.error("Embeddings not loaded. Create embeddings first.")
            elif not user_msg or not user_msg.strip():
                st.warning("Please enter a message.")
            else:
                payload_history = [{"role": h.get("role"), "content": h.get("content")} for h in st.session_state.get(hist_key, [])]
                modifiers = []
                if explain_new:
                    modifiers.append("Explain as if the learner is new to the topic.")
                if include_example:
                    modifiers.append("Include one concrete example.")
                if turn_quiz:
                    modifiers.append("End with one quiz question.")
                prompt_question = user_msg.strip()
                if modifiers:
                    prompt_question = f"{prompt_question}\n\nInstructions: " + " ".join(modifiers)
                candidate_index_path = str(index_path) if index_path.exists() else None

                # decide use_faiss_search from session (app.py must set this before calling render)
                use_faiss_search = bool(st.session_state.get("use_faiss_search", False))

                with st.spinner("Running conversational RAG..."):
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
                        display_answer = strip_key_concepts_from_answer(ans or "")

                        # prefer backend-provided updated history; otherwise append user+assistant
                        if updated_history:
                            new_hist: List[Dict[str, str]] = []
                            for h in updated_history:
                                role = h.get("role", "user")
                                content_raw = h.get("content", "") or ""
                                if role and role.lower().startswith("assistant"):
                                    content = strip_key_concepts_from_answer(content_raw)
                                else:
                                    content = content_raw
                                new_hist.append({"role": role, "content": content})
                            st.session_state[hist_key] = trim_history_to_max_turns(new_hist, max_turns=60)
                        else:
                            st.session_state[hist_key].append({"role": "user", "content": user_msg})
                            st.session_state[hist_key].append({
                                "role": "assistant",
                                "content": display_answer,
                                "meta": {"retrieved": retrieved or [], "prompt": prompt_used, "provenance": provenance}
                            })
                            st.session_state[hist_key] = trim_history_to_max_turns(st.session_state[hist_key], max_turns=60)

                        st.success("Assistant replied — chat updated.")

                    except Exception as e:
                        st.error("Conversational RAG failed.")
                        st.exception(e)

    # Render the chat AFTER handling the form so latest reply is visible immediately
    chat_history = st.session_state.get(hist_key, []) or []

    # If collapse toggle is active, show only the last N messages for readability
    if st.session_state.get(collapse_key, False) and len(chat_history) > 10:
        display_history = chat_history[-10:]
        # add a small notice
        with chat_container:
            st.info("Older messages collapsed for readability — toggle to show more.")
    else:
        display_history = chat_history

    # use a taller max height so chat uses more vertical space
    chat_html, chat_height = build_chat_html(display_history, max_height=720)

    with chat_container:
        st.markdown('<div class="chat-fade-top"></div>', unsafe_allow_html=True)
        # Surround chat with a subtle header that shows count & quick actions
        header_cols = st.columns([3, 1])
        with header_cols[0]:
            st.markdown(f"**Conversation — {len(chat_history)} turns**")
        with header_cols[1]:
            # quick "scroll to bottom" affordance (just an informative hint; actual scrolling handled by component)
            st.caption("Auto-scroll enabled")

        components.html(chat_html, height=chat_height, scrolling=True)