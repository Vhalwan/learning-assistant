import html as html_mod
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests


API_DEFAULT = os.getenv("API_BASE", "http://localhost:8000")


def _escape_html(value: str) -> str:
    if value is None:
        return ""
    return html_mod.escape(str(value)).replace("\n", "<br>")


def build_chat_html(
    chat_history: List[Dict[str, str]],
    max_height: int = 520,
    width: str = "100%",
    container_id: Optional[str] = None,
) -> Tuple[str, int]:
    """Build a scrollable chat transcript HTML block."""
    cid = container_id or f"chat_{int(time.time() * 1000)}"
    msg_blocks: List[str] = []
    for turn in chat_history or []:
        role = (turn.get("role") or "user").lower()
        content = _escape_html(turn.get("content", "") or "")
        if role.startswith("assistant"):
            block = f"""
            <div class="msg bot">
              <div class="bubble">{content}</div>
            </div>
            """
        else:
            block = f"""
            <div class="msg user">
              <div class="bubble">{content}</div>
            </div>
            """
        msg_blocks.append(block)

    height = min(max(120 + 80 * max(0, len(chat_history) - 1), 200), max_height)
    html = f"""
    <style>
      #{cid}_wrap {{
        width: {width};
        height: {height}px;
        border: 1px solid #eee;
        border-radius: 8px;
        padding: 12px;
        overflow-y: auto;
        background: #fafafa;
      }}
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
      {"".join(msg_blocks) if msg_blocks else '<div style="color:#666">No messages yet - start the conversation below.</div>'}
    </div>

    <script>
      const container = document.getElementById("{cid}_wrap");
      if (container) {{
          container.scrollTop = container.scrollHeight;
      }}
      window["{cid}_scrollToBottom"] = function() {{
          const c = document.getElementById("{cid}_wrap");
          if (c) c.scrollTop = c.scrollHeight;
      }};
    </script>
    """
    return html, height


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
    headers: Dict[str, str] = {}
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
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            body = exc.response.text if exc.response is not None else ""
            transient = status in (429, 502, 503, 504) or "503" in body or "UNAVAILABLE" in body
            if attempt >= 3 or not transient:
                raise
            time.sleep(2.5 * attempt)
        except requests.Timeout as exc:
            last_exc = exc
            if attempt >= 3:
                raise
            time.sleep(2.5 * attempt)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Summarize API request failed")


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
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()


def strip_key_concepts_from_answer(answer: str) -> str:
    if not isinstance(answer, str):
        return str(answer)
    match = re.search(r"\n+(?:\d+\s*)?Key\s+Concepts?\b", answer, flags=re.IGNORECASE)
    if match:
        return answer[:match.start()].strip()
    match = re.search(r"\n+Key\s+concepts\b", answer, flags=re.IGNORECASE)
    if match:
        return answer[:match.start()].strip()
    return answer.strip()


def strip_retrieval_artifacts(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    cleaned = re.sub(r"\s*\[chunk:[^\]]+\]\s*", " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_summary_text(summary: str, summary_type: str = "brief") -> str:
    if not isinstance(summary, str):
        return str(summary)
    cleaned = re.sub(
        r"^\s*Here(?:'|â€™)s (?:a|an) (?:concise|brief) summary[^\n]*\n*[:\-]*\s*",
        "",
        summary,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*(?:\*\*)?(?:summary|overview|brief summary|detailed summary)(?:\*\*)?[ \t]*[:\-]?[ \t]*(?:\r?\n+)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*Here(?:'|’)s (?:a|an) (?:concise|brief) summary[^\n]*\n*[:\-]*\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if str(summary_type).strip().lower() == "detailed":
        marker_match = re.search(r"DETAILED[, ]+STRUCTURED SUMMARY|DETAILED SUMMARY", cleaned, flags=re.IGNORECASE)
        if marker_match:
            cleaned = cleaned[marker_match.start():]
        cleaned = re.sub(
            r"^\s*(?:concise|brief)\s+summary\s*:.*?(?=\n\s*(?:detailed|full)\s+summary\s*:)",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = re.sub(r"^\s*(?:detailed|full)[, ]+structured\s+summary\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*(?:detailed|full)\s+summary\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cutoff = re.search(r"\n+\s*(?:5\s+Key\s+Concepts|Key\s+concepts\s*/\s*highlights)\b", cleaned, flags=re.IGNORECASE)
        if cutoff:
            cleaned = cleaned[:cutoff.start()]
    cleaned = re.sub(
        r"\n+\s*(?:\d+\s+)?Key\s+concepts?(?:\s*/\s*highlights)?\s*:.*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"^\s*(?:\*\*)?(?:overview|summary)(?:\*\*)?[ \t]*[:\-]?[ \t]*(?:\r?\n+)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if cleaned.startswith("**") and cleaned.count("**") % 2 == 1:
        cleaned = cleaned[2:].lstrip()
    return strip_retrieval_artifacts(cleaned)


def clean_key_concepts_list(key_concepts: List[str], summary_type: str = "brief") -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in key_concepts or []:
        item = re.sub(r"\s+", " ", str(raw or "")).strip(" -:\n\t")
        if not item:
            continue
        if len(item) > 140:
            continue
        if re.search(
            r"concise explanation|detailed.*summary|^summary$|^overview$|^takeaways?$|^highlights?$|^5 key concepts?$",
            item,
            flags=re.IGNORECASE,
        ):
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    cap = 8 if str(summary_type).strip().lower() == "detailed" else 12
    return out[:cap]


def derive_key_concepts_from_summary_text(summary: str, summary_type: str = "brief") -> List[str]:
    if not isinstance(summary, str):
        return []

    candidates: List[str] = []
    for match in re.findall(r"\*\*([^*\n]{3,80})\*\*", summary):
        candidates.append(match)

    for raw_line in summary.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        plain = re.sub(r"^[#>\-\*\u2022\s]+", "", stripped)
        plain = re.sub(r"^\d+\.\s*", "", plain).strip().rstrip(":")
        plain = plain.strip("* ").strip()
        if not plain:
            continue
        if stripped.lstrip().startswith(("-", "*", "•")) or raw_line.strip().endswith(":"):
            candidates.append(plain)

    return clean_key_concepts_list(candidates, summary_type=summary_type)


def trim_history_to_max_turns(history: List[Dict[str, Any]], max_turns: int = 6) -> List[Dict[str, Any]]:
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
