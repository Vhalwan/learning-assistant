"""Compatibility wrapper for legacy imports.

Active code now imports from ``frontend.runtime_ui_helpers``.
This shim keeps older references working without duplicating logic.
"""

from frontend.runtime_ui_helpers import (  # noqa: F401
    API_DEFAULT,
    build_chat_html,
    call_chat_api,
    call_query_api,
    call_summarize_api,
    clean_key_concepts_list,
    clean_summary_text,
    derive_key_concepts_from_summary_text,
    render_assistant_html,
    strip_key_concepts_from_answer,
    strip_retrieval_artifacts,
    trim_history_to_max_turns,
)
