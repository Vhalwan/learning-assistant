# backend/llm_client.py
import os
import re
import time
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

INVALID_GEMINI_API_KEY_MESSAGE = "Invalid API key — please check and try again"

_last_gemini_api_key_error: Optional[str] = None


def is_invalid_gemini_api_key_error(e: Exception) -> bool:
    """True when Gemini rejected the API key (not quota/transient failures)."""
    text = f"{e} {getattr(e, 'message', '')}".lower()
    if any(
        marker in text
        for marker in (
            "api_key_invalid",
            "api key not valid",
            "invalid api key",
            "invalid_api_key",
            "api key is invalid",
            "incorrect api key",
            "invalid authentication credentials",
        )
    ):
        return True
    if "permission_denied" in text and "api" in text:
        return True
    if "unauthenticated" in text and "api" in text:
        return True
    code = getattr(e, "code", None)
    if code in (401, 403, "401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED"):
        return "api" in text or "key" in text or "credential" in text
    return False


def _record_gemini_api_key_error() -> None:
    global _last_gemini_api_key_error
    _last_gemini_api_key_error = INVALID_GEMINI_API_KEY_MESSAGE


def pop_gemini_api_key_error() -> Optional[str]:
    """Return and clear the latest invalid-API-key message, if any."""
    global _last_gemini_api_key_error
    msg = _last_gemini_api_key_error
    _last_gemini_api_key_error = None
    return msg


def _is_transient_gemini_error(e: Exception) -> bool:
    s = str(e)
    return (
        "429" in s
        or "RESOURCE_EXHAUSTED" in s
        or "resource_exhausted" in s.lower()
        or "503" in s
        or "UNAVAILABLE" in s
        or "unavailable" in s.lower()
        or "502" in s
        or "504" in s
        or "deadline" in s.lower()
        or "timeout" in s.lower()
    )


def _retry_delay_from_error(e: Exception, attempt: int, status_hint: str = "") -> float:
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", str(e))
    if m:
        return float(m.group(1))
    if "503" in status_hint or "503" in str(e):
        return min(4.0 * attempt, 12.0)
    if "429" in status_hint or "429" in str(e):
        return min(8.0 * attempt, 30.0)
    return min(2.0 * attempt, 8.0)


def get_llm_call(api_key: Optional[str] = None) -> Optional[Callable[[str], str]]:
    """
    Return a function llm_call(prompt)->str using the GEMINI_API_KEY.
    Retries transient 429/503/timeouts; uses JSON response mode for quiz-style prompts.

    If api_key is provided (non-empty), it overrides GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
    """
    gemini_key = (api_key or "").strip() or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        logger.info("No GEMINI_API_KEY / GOOGLE_API_KEY found -> using placeholders.")
        return None

    try:
        import google.genai as genai
        client = genai.Client(api_key=gemini_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        max_retries = max(1, int(os.getenv("GEMINI_MAX_RETRIES", "4")))
        temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.35"))

        def _genai_call(prompt: str) -> str:
            use_json = "Return JSON ONLY" in (prompt or "")
            last_err: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                try:
                    kwargs = {"model": model_name, "contents": prompt}
                    if use_json:
                        try:
                            from google.genai import types
                            kwargs["config"] = types.GenerateContentConfig(
                                response_mime_type="application/json",
                                temperature=temperature,
                            )
                        except Exception:
                            pass
                    resp = client.models.generate_content(**kwargs)
                    text = getattr(resp, "text", None)
                    if text:
                        return text
                    if hasattr(resp, "candidates") and resp.candidates:
                        parts = resp.candidates[0].content.parts
                        if parts and getattr(parts[0], "text", None):
                            return parts[0].text
                    return str(resp)
                except Exception as e:
                    last_err = e
                    if is_invalid_gemini_api_key_error(e):
                        _record_gemini_api_key_error()
                        raise
                    if attempt >= max_retries or not _is_transient_gemini_error(e):
                        raise
                    delay = _retry_delay_from_error(e, attempt)
                    logger.warning(
                        "Gemini transient error on attempt %s/%s; retrying in %.1fs: %s",
                        attempt,
                        max_retries,
                        delay,
                        e,
                    )
                    time.sleep(delay)
            if last_err is not None:
                raise last_err
            raise RuntimeError("Gemini call failed without a response")

        if (api_key or "").strip():
            logger.info("LLM callable ready, using user-provided API key.")
        else:
            logger.info("LLM callable ready, using GEMINI_API_KEY from environment.")
        return _genai_call

    except Exception as e:
        if is_invalid_gemini_api_key_error(e):
            _record_gemini_api_key_error()
        logger.warning("Failed to initialize google-genai client: %s", e)
        return None
