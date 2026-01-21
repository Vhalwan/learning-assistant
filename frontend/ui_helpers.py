# frontend/ui_helpers.py
import os
import re
import json
import html as html_mod
import requests
from typing import Optional, List, Dict, Any
import uuid
# frontend/ui_helpers.py
# ----------------------
# Add these functions to this file (append at end or near other UI helper functions).

from typing import List, Dict, Tuple
import html as html_mod
import time

def _escape_html(s: str) -> str:
    if s is None:
        return ""
    # keep basic newlines -> <br> for readability
    return html_mod.escape(str(s)).replace("\n", "<br>")

def build_chat_html(
    chat_history: List[Dict[str, str]],
    max_height: int = 520,
    width: str = "100%",
    container_id: str = None,
) -> Tuple[str, int]:
    """
    Build an HTML string that contains the whole chat in a single scrollable container.
    Returns (html_string, height_px). The generated HTML includes JS to auto-scroll to bottom.
    """
    # container id to avoid collisions
    cid = container_id or f"chat_{int(time.time() * 1000)}"
    # simple style: left = assistant, right = user
    msg_blocks = []
    for turn in chat_history or []:
        role = (turn.get("role") or "user").lower()
        content = turn.get("content", "") or ""
        # If content seems already HTML, still escape for safety
        escaped = _escape_html(content)
        if role.startswith("assistant"):
            block = f'''
            <div class="msg bot">
              <div class="bubble">{escaped}</div>
            </div>
            '''
        else:
            block = f'''
            <div class="msg user">
              <div class="bubble">{escaped}</div>
            </div>
            '''
        msg_blocks.append(block)

    # compute height (cap)
    height = min(max(120 + 80 * max(0, len(chat_history) - 1), 200), max_height)

    html = f"""
    <style>
      /* container */
      #{cid}_wrap {{
        width: {width};
        height: {height}px;
        border: 1px solid #eee;
        border-radius: 8px;
        padding: 12px;
        overflow-y: auto;
        background: #fafafa;
      }}
      /* message rows */
      .msg {{
        margin: 6px 0;
        display: flex;
        clear: both;
      }}
      .msg .bubble {{
        max-width: 78%;
        padding: 10px 12px;
        border-radius: 12px;
        line-height: 1.4;
        white-space: pre-wrap;
        word-wrap: break-word;
      }}
      .msg.user {{
        justify-content: flex-end;
      }}
      .msg.user .bubble {{
        background: #d1e7dd;
        border-top-right-radius: 6px;
      }}
      .msg.bot {{
        justify-content: flex-start;
      }}
      .msg.bot .bubble {{
        background: #ffffff;
        border: 1px solid #e6e6e6;
        border-top-left-radius: 6px;
      }}
    </style>

    <div id="{cid}_wrap">
      {"".join(msg_blocks) if msg_blocks else '<div style="color:#666">No messages yet — start the conversation below.</div>'}
    </div>

    <script>
      // auto-scroll to bottom after render
      const container = document.getElementById("{cid}_wrap");
      if (container) {{
          container.scrollTop = container.scrollHeight;
      }}
      // expose a function to scroll later if needed
      window["{cid}_scrollToBottom"] = function() {{
          const c = document.getElementById("{cid}_wrap");
          if(c) c.scrollTop = c.scrollHeight;
      }};
    </script>
    """
    return html, height

# -----------------------
# API wrappers (used by handlers or app)
# -----------------------
API_DEFAULT = os.getenv("API_BASE", "http://localhost:8000")


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


# -----------------------
# Small presentation helpers
# -----------------------
def strip_key_concepts_from_answer(answer: str) -> str:
    """ Remove any trailing 'Key Concepts' style block from an LLM answer for the Q&A / Chat UI. """
    if not isinstance(answer, str):
        return str(answer)
    m = re.search(r"\n+(?:\d+\s*)?Key\s+Concepts?\b", answer, flags=re.IGNORECASE)
    if m:
        return answer[:m.start()].strip()
    m2 = re.search(r"\n+Key\s+concepts\b", answer, flags=re.IGNORECASE)
    if m2:
        return answer[:m2.start()].strip()
    return answer.strip()


def clean_summary_text(summary: str) -> str:
    if not isinstance(summary, str):
        return str(summary)
    cleaned = re.sub(
        r"^\s*Here(?:'|’)s (?:a|an) (?:concise|brief) summary[^\n]*\n*[:\-]*\s*",
        "",
        summary,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def trim_history_to_max_turns(history: List[Dict[str, Any]], max_turns: int = 6) -> List[Dict[str, Any]]:
    """ Trim the provided history to the last max_turns user+assistant pairs. """
    if not history:
        return history
    pairs = []
    i = 0
    n = len(history)
    while i < n:
        role = (history[i].get("role") or "").lower()
        if role.startswith("user"):
            user_msg = history[i]
            if i + 1 < n and (history[i + 1].get("role") or "").lower().startswith("assistant"):
                assistant_msg = history[i + 1]
                pairs.append([user_msg, assistant_msg])
                i += 2
            else:
                pairs.append([user_msg])
                i += 1
        elif role.startswith("assistant"):
            pairs.append([history[i]])
            i += 1
        else:
            pairs.append([history[i]])
            i += 1
    if len(pairs) <= max_turns:
        return [m for p in pairs for m in p]
    kept_pairs = pairs[-max_turns:]
    return [m for p in kept_pairs for m in p]


# -----------------------
# Helper to render assistant bubble HTML (returns html and height)
# -----------------------
def render_assistant_html(content: str) -> Dict[str, Any]:
    uid = str(uuid.uuid4()).replace("-", "")[:12]
    escaped_content_html = html_mod.escape(content).replace("\n", "<br>")
    js_text = json.dumps(content)
    lines = max(1, content.count("\n") + 1)
    height_px = min(650, max(110, 22 * lines + 80))
    assistant_html = f"""
<div style="border-radius:10px;padding:10px;margin:8px 0;max-width:90%;background:#f7fafc;border:1px solid #e6eef5;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
    <strong>Assistant</strong>
  </div>
  <div style="white-space:pre-wrap;font-family:inherit;font-size:14px;line-height:1.4;">{escaped_content_html}</div>
  <div style="display:flex;justify-content:flex-end;margin-top:8px;">
    <button id="copy_{uid}" style="border:none;padding:6px 10px;border-radius:8px;cursor:pointer;background:#eef2f7;font-size:13px;">
      Copy
    </button>
  </div>
</div>

<script>
const btn = document.getElementById("copy_{uid}");
if (btn) {{
    btn.addEventListener("click", function() {{
        navigator.clipboard.writeText({js_text}).then(function() {{
            const old = btn.innerText;
            btn.innerText = 'Copied';
            setTimeout(() => {{ btn.innerText = old; }}, 1200);
        }});
    }});
}}
</script>
"""
    return {"html": assistant_html, "height": height_px}
