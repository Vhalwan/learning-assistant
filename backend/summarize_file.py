# backend/summarize_file.py
import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"
logger = logging.getLogger(__name__)

URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def _retry_delay_seconds(resp: requests.Response) -> float:
    try:
        payload = resp.json()
    except Exception:
        return 0.0

    try:
        details = (payload.get("error") or {}).get("details") or []
        retry_delay = str(details[0].get("retryDelay") or "").strip() if details else ""
    except Exception:
        retry_delay = ""

    if retry_delay.endswith("s"):
        retry_delay = retry_delay[:-1]
    try:
        return float(retry_delay)
    except Exception:
        return 0.0


def _call_gemini_api(prompt_text: str, timeout: int = 60, max_retries: int = 3) -> str:
    """
    Low-level call to Gemini. Returns the model text output.
    """
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt_text
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    params = {
        "key": API_KEY
    }

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(URL, headers=headers, params=params, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            # defensive: navigate response to pick first candidate text
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                # fallback to raw JSON if structure unexpected
                return str(data)
        except requests.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            transient = status in (429, 500, 502, 503, 504)
            if attempt >= max_retries or not transient:
                raise
            retry_delay = _retry_delay_seconds(exc.response) if exc.response is not None else 0.0
            if retry_delay <= 0:
                retry_delay = 8.0 * attempt if status == 503 else 3.0 * attempt
            logger.warning(
                "Gemini request failed with status %s on attempt %s/%s; retrying in %.1fs",
                status,
                attempt,
                max_retries,
                retry_delay,
            )
            time.sleep(retry_delay)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            retry_delay = 2.0 * attempt
            logger.warning(
                "Gemini request raised %s on attempt %s/%s; retrying in %.1fs",
                type(exc).__name__,
                attempt,
                max_retries,
                retry_delay,
            )
            time.sleep(retry_delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Gemini request failed without returning a response")


def generate_with_gemini(prompt: str) -> str:
    """
    General-purpose generator wrapper: send the exact prompt to Gemini and return text.
    Use this for Q&A and chat flows where you build the prompt yourself.
    """
    return _call_gemini_api(prompt)


def summarize_with_gemini(text: str) -> str:
    """
    Higher-level summarization helper kept for backwards compatibility.
    This helper builds a *summary-specific* prompt (including Request: list 5 key concepts)
    and uses the low-level API to obtain the summary + key concepts.
    """
    prompt = (
        "Summarize the following lecture notes into a concise explanation "
        "and list 5 key concepts:\n\n"
        + text
    )
    return _call_gemini_api(prompt)


if __name__ == "__main__":
    with open("data/processed/lecture1.txt", "r", encoding="utf-8") as f:
        txt = f.read()
    print(summarize_with_gemini(txt))
