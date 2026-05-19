"""Remote vs local persistence switch for user-scoped JSON blobs."""
from __future__ import annotations

from backend.user_context import USE_REMOTE_STORAGE, get_user_id

NAMESPACE_QUIZ = "quiz"
NAMESPACE_CONCEPTS = "concepts"
NAMESPACE_CONFUSION = "confusion"
NAMESPACE_SRS = "srs"


def use_remote_store() -> bool:
    return USE_REMOTE_STORAGE and bool(get_user_id())
