# frontend/app.py
import os
import sys
import re
import json
import hashlib
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
    parse_summary_sections,
    trim_history_to_max_turns,
    render_assistant_html,
)
from frontend.handlers import (
    init_llm,
    create_embeddings_if_needed,
    ensure_concepts_for_lecture,
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
from backend.confusion_store import get_top_confusions
from backend.quiz_session_log import recent_sessions


def _sidebar_lecture_feedback(stem: str) -> Dict[str, Any]:
    """Aggregate sidebar-ready lecture feedback from confusion history."""
    empty = {
        "questions_answered": 0,
        "accuracy_pct": None,
    }
    if not (stem or "").strip():
        return empty
    try:
        items = get_top_confusions(limit=None, stem=stem)
    except Exception:
        return empty
    if not items:
        return empty

    total_attempts = sum(int(x.get("total_attempts") or 0) for x in items)
    total_correct = sum(int(x.get("correct_attempts") or 0) for x in items)
    accuracy_pct: Optional[float] = None
    if total_attempts > 0:
        accuracy_pct = (total_correct / total_attempts) * 100.0

    return {
        "questions_answered": total_attempts,
        "accuracy_pct": accuracy_pct,
    }


def _sidebar_recent_performance(stem: str, limit: int = 3) -> tuple[List[Dict[str, Any]], str]:
    rows = recent_sessions(stem, limit)
    if not rows:
        return ([], "No recent quiz rounds yet.")

    accuracies: List[float] = []
    for row in rows:
        try:
            accuracies.append(float(row.get("accuracy_pct", 0)))
        except Exception:
            accuracies.append(0.0)

    if len(accuracies) == 1:
        score = accuracies[0]
        if score < 60:
            return (rows, "Your recent results are lower — reviewing now will help.")
        return (rows, "Recent results look steady — keep going.")

    first_score = accuracies[0]
    last_score = accuracies[-1]
    recent_average = sum(accuracies) / len(accuracies)

    if recent_average < 60 or last_score < first_score:
        message = "Recent results are lower — a quick review will help."
    elif last_score >= first_score + 10:
        message = "Recent results are improving."
    else:
        message = "Recent results look steady — keep going."
    return (rows, message)


def _sidebar_topic_label(item: Dict[str, Any], limit: int = 28) -> str:
    raw = (
        (item.get("title") or "").strip()
        or (item.get("concept_label") or "").strip()
        or (item.get("source_chunk_preview") or "").strip()
        or (item.get("source_chunk") or "").strip()
    )
    label = " ".join(raw.split())
    if not label:
        return ""
    if len(label) <= limit:
        return label
    return label[: limit - 3].rsplit(" ", 1)[0] + "..."


def _sidebar_card_stem(card_id: str, meta: Dict[str, Any] | None) -> str:
    if meta and meta.get("stem"):
        return str(meta.get("stem") or "").strip()
    if card_id and "_" in card_id:
        return card_id.rsplit("_", 1)[0]
    return ""


def _sidebar_bar_percentages(rows: List[Dict[str, int]]) -> List[float]:
    bars: List[float] = []
    for row in rows:
        total = max(int(row.get("total", 0) or 0), 0)
        correct = max(int(row.get("correct", 0) or 0), 0)
        if total <= 0:
            bars.append(0.0)
            continue
        bars.append((correct / total) * 100.0)
    return bars


def _sidebar_progress_snapshot(stem: str, limit: int = 5) -> Dict[str, Any]:
    rows = recent_sessions(stem, limit)
    empty = {
        "score_text": "—",
        "score_subtext": "No completed quiz session yet",
        "trend_text": "Waiting for data",
        "trend_class": "steady",
        "bar_percentages": [],
    }
    if not rows:
        return empty

    cleaned_rows: List[Dict[str, Any]] = []
    for row in rows:
        try:
            correct = int(row.get("correct", 0))
        except Exception:
            correct = 0
        try:
            total = int(row.get("total", 0))
        except Exception:
            total = 0
        cleaned_rows.append({"correct": correct, "total": total})

    last_row = cleaned_rows[-1]
    bar_percentages = _sidebar_bar_percentages(cleaned_rows)
    trend_text = "Steady →"
    trend_class = "steady"
    if len(bar_percentages) >= 2:
        prior_average = sum(bar_percentages[:-1]) / max(len(bar_percentages) - 1, 1)
        last_accuracy = bar_percentages[-1]
        if last_accuracy >= prior_average + 8:
            trend_text = "Improving ↑"
            trend_class = "up"
        elif last_accuracy <= prior_average - 8:
            trend_text = "Lower ↓"
            trend_class = "down"

    return {
        "score_text": f"{last_row['correct']} / {last_row['total']}" if last_row["total"] > 0 else "—",
        "score_subtext": "last session",
        "trend_text": trend_text,
        "trend_class": trend_class,
        "bar_percentages": bar_percentages,
    }


def _sidebar_recent_weak_topics(stem: str, limit: int = 2) -> List[str]:
    if not (stem or "").strip():
        return []
    try:
        items = get_top_confusions(limit=None, stem=stem)
    except Exception:
        return []
    if not items:
        return []

    def recent_key(item: Dict[str, Any]) -> str:
        return max(
            str(item.get("last_wrong") or "").strip(),
            str(item.get("last_seen") or "").strip(),
            str(item.get("last_updated") or "").strip(),
        )

    recent_wrong = [item for item in items if int(item.get("wrong_attempts") or 0) > 0]
    recent_wrong.sort(key=recent_key, reverse=True)

    topics: List[str] = []
    seen = set()
    for item in recent_wrong:
        label = _sidebar_topic_label(item)
        if not label:
            continue
        normalized = label.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        topics.append(label)
        if len(topics) >= limit:
            break
    return topics


def _sidebar_due_srs_count(stem: str) -> int:
    if not (stem or "").strip():
        return 0
    try:
        srs_mgr = SRSManager()
        due_cards = srs_mgr.get_due_cards()
    except Exception:
        return 0
    count = 0
    for card_id in due_cards:
        meta = srs_mgr.get_card_meta(card_id) or {}
        if _sidebar_card_stem(card_id, meta) == stem:
            count += 1
    return count


# initialize LLM (this mirrors existing behaviour)
llm = init_llm()
# We avoid calling st.* until after set_page_config — but keep same behavior messages
# We'll store messages in a list and display after page config
_startup_msgs = []
if llm is None:
    _startup_msgs.append(("warning", "LLM not available — using placeholders"))

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

      .la-learn-rail {
        display: flex;
        flex-wrap: wrap;
        align-items: stretch;
        gap: 0.35rem 0.5rem;
        margin: 0.5rem 0 1.25rem;
        padding: 1rem 1.1rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(236, 254, 255, 0.95), rgba(255, 255, 255, 0.92));
        border: 1px solid #c5e3df;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
      }
      .la-flow-node {
        flex: 1 1 140px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.35rem;
        padding: 0.75rem 0.85rem;
        border-radius: 16px;
        background: #fff;
        border: 1px solid #d5e8e6;
        text-decoration: none;
        color: var(--text);
        min-height: 4.5rem;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
      }
      .la-flow-node:hover {
        border-color: var(--accent);
        box-shadow: 0 8px 20px rgba(15, 118, 110, 0.12);
      }
      .la-flow-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.85rem;
        height: 1.85rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent-strong);
        font-weight: 700;
        font-size: 0.9rem;
      }
      .la-flow-t {
        font-weight: 700;
        font-size: 0.98rem;
      }
      .la-flow-d {
        font-size: 0.82rem;
        color: var(--muted);
        line-height: 1.35;
      }
      .la-flow-arrow {
        align-self: center;
        font-size: 1.25rem;
        color: var(--accent);
        font-weight: 600;
        user-select: none;
        padding: 0 0.15rem;
      }
      .la-sidebar-card {
        margin: 0.65rem 0;
        padding: 0.9rem 0.95rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #d5e8e6;
        box-shadow: 0 10px 24px rgba(15, 118, 110, 0.08);
      }
      .la-sidebar-kicker {
        margin-bottom: 0.45rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
      }
      .la-sidebar-score {
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.05;
        color: var(--text);
      }
      .la-sidebar-sub {
        margin-top: 0.2rem;
        font-size: 0.82rem;
        color: var(--muted);
      }
      .la-sidebar-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        margin-top: 0.6rem;
      }
      .la-sidebar-badge.up {
        background: #ecfdf3;
        color: #166534;
      }
      .la-sidebar-badge.down {
        background: #fef2f2;
        color: #b91c1c;
      }
      .la-sidebar-badge.steady {
        background: var(--accent-soft);
        color: var(--accent-strong);
      }
      .la-sidebar-spark {
        margin-top: 0.8rem;
        display: flex;
        align-items: flex-end;
        gap: 0.32rem;
        height: 4.75rem;
      }
      .la-sidebar-spark-bar {
        flex: 1 1 0;
        min-width: 0.42rem;
        border-radius: 6px 6px 0 0;
        background: linear-gradient(180deg, #14b8a6 0%, #0f766e 100%);
        min-height: 4px;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.24);
      }
      .la-sidebar-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.38rem;
      }
      .la-sidebar-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.58rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent-strong);
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1.2;
      }
      .la-sidebar-chip-muted {
        background: #f1f5f9;
        color: #475569;
      }
      .la-sidebar-due {
        margin-top: 0.15rem;
        display: flex;
        align-items: baseline;
        gap: 0.35rem;
      }
      .la-sidebar-due-count {
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1;
        color: var(--text);
      }
      .la-sidebar-due-label {
        font-size: 0.82rem;
        color: var(--muted);
      }
      @media (max-width: 720px) {
        .la-flow-arrow { display: none; }
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
      <div class="app-logo">LV</div>
      <div>
        <div class="app-title">Lectova</div>
        <div class="app-subtitle">One lecture. Understand it, test it, remember it.</div>
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

# ----------------------------
# API mode toggle + token UI (defaults; widgets live in sidebar Advanced)
# ----------------------------
API_DEFAULT = os.getenv("API_BASE", "http://localhost:8000")
if "use_api_mode" not in st.session_state:
    st.session_state.use_api_mode = False

# read default token from env var, allow override in UI
default_token = os.getenv("API_TOKEN", "") or ""
if "api_token" not in st.session_state:
    st.session_state.api_token = default_token

if "use_faiss_search" not in st.session_state:
    st.session_state["use_faiss_search"] = False

with st.sidebar:
    st.markdown("### Lectova")
    st.markdown("##### Upload lecture PDF")
    uploaded = st.file_uploader(
        "Lecture PDF",
        type=["pdf"],
        help="Drag and drop a file here, or click Browse files.",
    )
    sidebar_pdf_name: Optional[str] = None
    sidebar_stem: Optional[str] = None
    if uploaded is not None:
        sidebar_pdf_name = uploaded.name
        sidebar_stem = Path(uploaded.name).stem
    elif st.session_state.get("current_stem"):
        sidebar_stem = str(st.session_state.get("current_stem") or "").strip() or None
        sidebar_pdf_name = (st.session_state.get("current_pdf_filename") or "").strip() or None
        if sidebar_stem and not sidebar_pdf_name:
            sidebar_pdf_name = f"{sidebar_stem}.pdf"

    # Clear session state if no upload but session has old data
    if uploaded is None and st.session_state.get("current_stem"):
        st.session_state["current_stem"] = None
        st.session_state["current_pdf_filename"] = None
        st.rerun()

    if sidebar_pdf_name:
        st.caption(f"Lecture loaded: **{sidebar_pdf_name}**")
        st.caption("Upload another PDF above anytime to replace it.")
    else:
        st.caption("No lecture uploaded")

    st.markdown("---")
    st.markdown("##### Study tools")
    st.markdown(
        """
- [Study](#study-modes)
- [Test yourself](#test-yourself)
- [Weak topics](#weak-topics)
- [Spaced repetition (SRS)](#srs)
        """,
        unsafe_allow_html=True,
    )

    if sidebar_stem:
        progress = _sidebar_progress_snapshot(sidebar_stem, limit=5)
        weak_topics = _sidebar_recent_weak_topics(sidebar_stem, limit=2)
        due_count = _sidebar_due_srs_count(sidebar_stem)
        spark_bars = "".join(
            f'<span class="la-sidebar-spark-bar" style="height: max(4px, {max(0.0, min(100.0, float(bar or 0.0))):.1f}%);"></span>'
            for bar in progress.get("bar_percentages", [])
        )
        sparkline_html = ""
        if spark_bars:
            sparkline_html = f'<div class="la-sidebar-spark">{spark_bars}</div>'
        weak_topics_html = "".join(
            f'<span class="la-sidebar-chip">{html_mod.escape(topic)}</span>'
            for topic in weak_topics
        ) or '<span class="la-sidebar-chip la-sidebar-chip-muted">No recent misses</span>'
        due_label = "card due now" if due_count == 1 else "cards due now"
        st.markdown("---")
        st.markdown("##### This lecture")
        st.markdown(
            f"""
            <div class="la-sidebar-card">
              <div class="la-sidebar-kicker">Progress</div>
              <div class="la-sidebar-score">{html_mod.escape(progress["score_text"])}</div>
              <div class="la-sidebar-sub">{html_mod.escape(progress["score_subtext"])}</div>
              <div class="la-sidebar-badge {html_mod.escape(progress["trend_class"])}">{html_mod.escape(progress["trend_text"])}</div>
              {sparkline_html}
            </div>
            <div class="la-sidebar-card">
              <div class="la-sidebar-kicker">Struggling with</div>
              <div class="la-sidebar-chip-row">{weak_topics_html}</div>
              <div class="la-sidebar-sub">Based on recent wrong answers</div>
            </div>
            <div class="la-sidebar-card">
              <div class="la-sidebar-kicker">SRS cards due</div>
              <div class="la-sidebar-due">
                <div class="la-sidebar-due-count">{due_count}</div>
                <div class="la-sidebar-due-label">{html_mod.escape(due_label)}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    with st.expander("Advanced", expanded=False):
        st.caption("Optional technical settings. You can ignore these for normal study.")
        st.session_state.use_api_mode = st.checkbox(
            "Use study server instead of local",
            value=st.session_state.use_api_mode,
            help="When enabled, requests go to your configured API base URL.",
        )
        st.session_state.api_token = st.text_input(
            "Use your own API key (optional)",
            value=st.session_state.api_token,
            type="password",
            help="Defaults from your environment; entering a value overrides for this session only.",
        )
        st.session_state["use_faiss_search"] = st.checkbox(
            "Faster search for long PDFs",
            value=bool(st.session_state.get("use_faiss_search", False)),
            help="Optional index for quicker retrieval on large lectures. Ordinary search still works if off.",
        )

# Show instruction only if no file is uploaded or loaded
if not uploaded and not st.session_state.get("current_stem"):
    st.markdown(
        "Upload a lecture PDF from the sidebar to get started."
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

    # derive the stem and paths
    stem = tmp_pdf.stem
    doc_id = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]
    embeddings_path = Path(f"data/processed/{stem}_embeddings.json")
    index_path = Path(f"data/processed/{stem}_embeddings.index")
    st.session_state["current_stem"] = stem
    st.session_state["current_pdf_filename"] = uploaded.name
    current_label = stem.replace("_", " ").title()

    # SAFE is the default behavior; keep it as a status, not a user choice.
    os.environ["USE_SAFE_EMBEDDINGS"] = "1"
    use_faiss_search = bool(st.session_state.get("use_faiss_search", False))

    # Check if lecture is already ready
    lecture_is_ready = False
    ids, texts, vecs = [], [], np.array([])
    try:
        ids, texts, vecs = load_embeddings(str(embeddings_path))
        lecture_is_ready = len(ids) > 0
    except Exception:
        pass

    # ========== COLLAPSED UI: Lecture is ready ==========
    if lecture_is_ready:
        st.markdown('<a id="setup-your-lecture"></a>', unsafe_allow_html=True)
        st.markdown(f"<span style='color: var(--muted); font-size: 0.9em;'>✓ Lecture ready: {current_label}</span>", unsafe_allow_html=True)

        # Extracted preview accordion (collapsed)
        with st.expander("Extracted preview", expanded=False):
            st.write(text[:1000] + ("..." if len(text) > 1000 else ""))

        # Advanced reset tools accordion (collapsed)
        with st.expander("Advanced reset tools", expanded=False):
            st.caption("Use these only when troubleshooting or resetting processed data.")
            recreate_btn = False
            build_index_btn = False
            adv_col1, adv_col2 = st.columns([1, 1])
            with adv_col1:
                recreate_btn = st.button("Recreate embeddings", key="recreate_ready")
            with adv_col2:
                build_index_btn = st.button("Build FAISS index", key="build_index_ready")

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

            # Recreate embeddings if requested
            if recreate_btn:
                try:
                    with st.spinner("Creating embeddings..."):
                        create_embeddings_for_text(text, str(embeddings_path), dim=EMBED_DIM)
                    st.success("Embeddings recreated. Refresh the page.")
                except Exception as e:
                    st.error("Failed to recreate embeddings.")
                    st.exception(e)

    # ========== FULL SETUP UI: Lecture not yet ready ==========
    else:
        st.markdown('<a id="setup-your-lecture"></a>', unsafe_allow_html=True)
        st.markdown("## 1. Setup your lecture")
        st.write(
            "Upload one lecture PDF from the sidebar, then create embeddings for it. "
            "Once that is ready, every study tool below works off the same lecture."
        )

        st.subheader("Extracted preview")
        st.write(text[:1000] + ("..." if len(text) > 1000 else ""))

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

        recreate_btn = False
        build_index_btn = False

        # Create embeddings when missing or when they are requested
        if (not embeddings_path.exists()):
            try:
                with st.spinner("Creating embeddings..."):
                    create_embeddings_for_text(text, str(embeddings_path), dim=EMBED_DIM)
            except Exception as e:
                st.error("Failed to create embeddings.")
                st.exception(e)

        try:
            with st.spinner("Extracting lecture concepts..."):
                lecture_concepts, concepts_created = ensure_concepts_for_lecture(
                    stem=stem,
                    text=text,
                    llm_call=llm,
                    doc_id=doc_id,
                    max_concepts=8,
                )
            if concepts_created and lecture_concepts:
                st.caption(f"Concept map ready: {len(lecture_concepts)} lecture concepts.")
        except Exception as e:
            st.warning(f"Lecture concepts could not be extracted yet: {e}")

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
                recreate_btn = st.button("Recreate embeddings", key="recreate_setup")
            with adv_col2:
                build_index_btn = st.button("Build FAISS index", key="build_index_setup")

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
    st.markdown("## 2. Study")
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
                            sections = parse_summary_sections(summary_display)
                            if sections.get("has_structure"):
                                sec_defs = [
                                    ("Key ideas", "key_ideas"),
                                    ("Definitions", "definitions"),
                                    ("Exam traps", "exam_traps"),
                                    ("Recap (3 bullets)", "recap"),
                                ]
                                for title, sk in sec_defs:
                                    body = (sections.get(sk) or "").strip()
                                    if body:
                                        st.markdown(f"### {title}")
                                        st.markdown(body)
                                rem = (sections.get("remainder") or "").strip()
                                if rem:
                                    st.markdown("### Also covered")
                                    st.markdown(rem)
                            else:
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
        "Follow the path: test yourself, clear up weak topics, then promote what matters into spaced repetition."
    )
    st.markdown(
        """
        <div class="la-learn-rail" role="navigation" aria-label="Learning loop steps">
          <a class="la-flow-node" href="#study-quiz">
            <span class="la-flow-num">1</span>
            <span class="la-flow-t">Quiz</span>
            <span class="la-flow-d">Mixed angles — definition, scenario, trap, compare.</span>
          </a>
          <span class="la-flow-arrow" aria-hidden="true">→</span>
          <a class="la-flow-node" href="#weak-topics">
            <span class="la-flow-num">2</span>
            <span class="la-flow-t">Weak topics</span>
            <span class="la-flow-d">See what you miss and fix it in one place.</span>
          </a>
          <span class="la-flow-arrow" aria-hidden="true">→</span>
          <a class="la-flow-node" href="#srs">
            <span class="la-flow-num">3</span>
            <span class="la-flow-t">SRS</span>
            <span class="la-flow-d">Save cards and review on a schedule.</span>
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Step 1: Quiz (unchanged behavior — just a short intro)
    st.markdown('<a id="test-yourself"></a>', unsafe_allow_html=True)
    st.markdown("### 3.1 Test yourself")
    st.write("Generate a quick set of multiple-choice questions from the lecture and check your understanding.")
    # Render quiz directly (removed outer expander)
    render_quiz(st=st, stem=stem, text=text, llm=llm, hist_key=hist_key)

    st.markdown("---")

    # Step 2: Weak topics (prioritized list of things you missed)
    st.markdown('<a id="weak-topics"></a>', unsafe_allow_html=True)
    st.markdown("### 3.2 Weak topics")
    st.write("Concepts you missed show up here, with quick actions for explanation and chat follow-up.")
    # Render confused directly (removed outer expander)
    render_confused(
        st=st,
        stem=stem,
        embeddings_path=embeddings_path,
        index_path=index_path,
        use_faiss_search=use_faiss_search,
        llm=llm,
    )

    # Step 3: Spaced Repetition (SRS)
    st.markdown('<a id="srs"></a>', unsafe_allow_html=True)
    st.markdown("### 3.3 Spaced repetition (SRS)")
    st.caption("Review due cards, or browse and manage all saved cards.")
    render_srs_section(st, stem=stem)
