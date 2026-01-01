# backend/summarize_file.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"

URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def _call_gemini_api(prompt_text: str, timeout: int = 60) -> str:
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

    resp = requests.post(URL, headers=headers, params=params, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # defensive: navigate response to pick first candidate text
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        # fallback to raw JSON if structure unexpected
        return str(data)


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
