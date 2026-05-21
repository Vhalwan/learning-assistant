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


def _format_chat_inline_html(escaped_text: str) -> str:
    """Apply bold/code to already HTML-escaped text."""
    text = escaped_text
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def format_chat_assistant_html(text: str) -> str:
    """Render common markdown patterns in assistant chat bubbles as safe HTML."""
    if not text:
        return ""
    raw = str(text).replace("\r\n", "\n")
    lines = raw.split("\n")
    blocks: List[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        bullet = re.match(r"^[\*\-]\s+(.+)$", stripped)
        if bullet:
            items: List[str] = []
            while i < len(lines):
                s = lines[i].strip()
                m = re.match(r"^[\*\-]\s+(.+)$", s)
                if not m:
                    break
                items.append(_format_chat_inline_html(html_mod.escape(m.group(1))))
                i += 1
            blocks.append("<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>")
            continue
        heading = re.match(r"^#{1,4}\s+(.+)$", stripped)
        if heading:
            title = _format_chat_inline_html(html_mod.escape(heading.group(1)))
            blocks.append(f'<p class="chat-h">{title}</p>')
            i += 1
            continue
        if not stripped:
            i += 1
            continue
        para_lines: List[str] = []
        while i < len(lines):
            s = lines[i].strip()
            if not s:
                break
            if re.match(r"^[\*\-]\s+", s) or re.match(r"^#{1,4}\s+", s):
                break
            para_lines.append(lines[i])
            i += 1
        if para_lines:
            body = html_mod.escape("\n".join(para_lines))
            body = _format_chat_inline_html(body.replace("\n", "<br>"))
            blocks.append(f"<p>{body}</p>")
    if blocks:
        return "".join(blocks)
    return _format_chat_inline_html(html_mod.escape(raw).replace("\n", "<br>"))


def scroll_to_anchor(anchor_id: str) -> None:
    """Best-effort scroll to an anchor in the Streamlit parent document."""
    import streamlit.components.v1 as st_components

    aid = json.dumps(anchor_id)
    st_components.html(
        f"""
        <script>
        const ID = {aid};
        function findEl() {{
            const byId = (doc) => (doc && doc.getElementById ? doc.getElementById(ID) : null);
            let el = byId(window.parent.document);
            if (el) return el;
            const scanRoots = [window.parent.document, document];
            for (const root of scanRoots) {{
                if (!root || !root.querySelectorAll) continue;
                const frames = root.querySelectorAll("iframe");
                for (const fr of frames) {{
                    try {{
                        const d = fr.contentDocument;
                        el = byId(d);
                        if (el) return el;
                    }} catch (e) {{}}
                }}
            }}
            return byId(document);
        }}
        function go() {{
            const el = findEl();
            if (el) el.scrollIntoView({{ behavior: "smooth", block: "start" }});
        }}
        setTimeout(go, 0);
        setTimeout(go, 120);
        setTimeout(go, 400);
        </script>
        """,
        height=0,
        width=0,
    )


def build_chat_html(
    chat_history: List[Dict[str, str]],
    max_height: int = 520,
    width: str = "100%",
    container_id: Optional[str] = None,
    scroll_align: str = "bottom",
) -> Tuple[str, int]:
    """Build a scrollable chat transcript HTML block.

    scroll_align:
      - "bottom": show the end of the transcript (default)
      - "last_user": align the latest user bubble to the top so the reply reads downward
    """
    cid = container_id or f"chat_{int(time.time() * 1000)}"
    msg_blocks: List[str] = []
    for turn in chat_history or []:
        role = (turn.get("role") or "user").lower()
        raw_content = turn.get("content", "") or ""
        if role.startswith("assistant"):
            content = format_chat_assistant_html(raw_content)
            block = f"""
            <div class="msg bot">
              <div class="bubble bot-rich">{content}</div>
            </div>
            """
        else:
            content = _escape_html(raw_content)
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
        overflow-x: hidden;
        overflow-y: auto;
        background: #fafafa;
        box-sizing: border-box;
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
        overflow-wrap: anywhere;
        word-break: break-word;
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
      .msg.bot .bubble.bot-rich {{
        white-space: normal;
      }}
      .msg.bot .bubble.bot-rich p {{
        margin: 0.35em 0;
      }}
      .msg.bot .bubble.bot-rich p.chat-h {{
        margin: 0.6em 0 0.35em;
        font-weight: 600;
      }}
      .msg.bot .bubble.bot-rich ul {{
        margin: 0.35em 0 0.5em;
        padding-left: 1.25em;
      }}
      .msg.bot .bubble.bot-rich li {{
        margin: 0.2em 0;
      }}
      .msg.bot .bubble.bot-rich strong {{
        font-weight: 600;
      }}
      .msg.bot .bubble.bot-rich code {{
        font-family: ui-monospace, monospace;
        font-size: 0.92em;
        background: #f0f4f8;
        padding: 0.1em 0.35em;
        border-radius: 4px;
      }}
    </style>

    <div id="{cid}_wrap">
      {"".join(msg_blocks) if msg_blocks else '<div style="color:#666">No messages yet - start the conversation below.</div>'}
    </div>

    <script>
      const container = document.getElementById("{cid}_wrap");
      const scrollAlign = {json.dumps(scroll_align)};
      function applyChatScroll() {{
          if (!container) return;
          if (scrollAlign === "last_user") {{
              const users = container.querySelectorAll(".msg.user");
              if (users.length) {{
                  const lastUser = users[users.length - 1];
                  const pad = 6;
                  container.scrollTop = Math.max(0, lastUser.offsetTop - pad);
                  return;
              }}
          }}
          container.scrollTop = container.scrollHeight;
      }}
      applyChatScroll();
      setTimeout(applyChatScroll, 0);
      setTimeout(applyChatScroll, 80);
      setTimeout(applyChatScroll, 200);
      window["{cid}_scrollToBottom"] = function() {{
          const c = document.getElementById("{cid}_wrap");
          if (c) c.scrollTop = c.scrollHeight;
      }};
      window["{cid}_scrollToLastUser"] = function() {{
          const c = document.getElementById("{cid}_wrap");
          if (!c) return;
          const users = c.querySelectorAll(".msg.user");
          if (!users.length) {{
              c.scrollTop = c.scrollHeight;
              return;
          }}
          const lastUser = users[users.length - 1];
          c.scrollTop = Math.max(0, lastUser.offsetTop - 6);
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


def parse_summary_sections(markdown: str) -> Dict[str, Any]:
    """
    Split a model-written summary that uses ### Key ideas / Definitions / Exam traps / Recap.
    Returns dict with section bodies, leftover text, and has_structure if enough headings matched.
    """
    if not isinstance(markdown, str) or not markdown.strip():
        return {
            "has_structure": False,
            "key_ideas": "",
            "definitions": "",
            "exam_traps": "",
            "recap": "",
            "remainder": "",
        }

    text = markdown.strip()
    # Heading pattern: ## or ### optional whitespace + title
    heading_re = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)
    matches = list(heading_re.finditer(text))
    if not matches:
        return {
            "has_structure": False,
            "key_ideas": "",
            "definitions": "",
            "exam_traps": "",
            "recap": "",
            "remainder": text,
        }

    def _norm_title(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    canon = {
        "key ideas": "key_ideas",
        "definitions": "definitions",
        "exam traps": "exam_traps",
        "recap": "recap",
    }

    sections: Dict[str, str] = {k: "" for k in ("key_ideas", "definitions", "exam_traps", "recap")}
    spans: List[Tuple[int, int, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw_title = m.group(2).strip()
        key = canon.get(_norm_title(raw_title))
        if key:
            body = text[start:end].strip()
            sections[key] = body
            spans.append((m.start(), end, key))

    hit = sum(1 for k in sections if (sections[k] or "").strip())
    has_structure = hit >= 2

    if not spans:
        return {
            "has_structure": False,
            "key_ideas": "",
            "definitions": "",
            "exam_traps": "",
            "recap": "",
            "remainder": text,
        }

    covered = sorted(spans, key=lambda x: x[0])
    remainder_parts: List[str] = []
    pos = 0
    for a, b, _ in covered:
        if a > pos:
            chunk = text[pos:a].strip()
            if chunk:
                remainder_parts.append(chunk)
        pos = max(pos, b)
    if pos < len(text):
        tail = text[pos:].strip()
        if tail:
            remainder_parts.append(tail)

    return {
        "has_structure": has_structure,
        "key_ideas": sections["key_ideas"],
        "definitions": sections["definitions"],
        "exam_traps": sections["exam_traps"],
        "recap": sections["recap"],
        "remainder": "\n\n".join(remainder_parts).strip(),
    }


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
