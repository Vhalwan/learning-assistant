# backend/summarize_file.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"

URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def summarize_with_gemini(text: str) -> str:
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Summarize the following lecture notes into a concise explanation "
                            "and list 5 key concepts:\n\n"
                            + text
                        )
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

    resp = requests.post(URL, headers=headers, params=params, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    return data["candidates"][0]["content"]["parts"][0]["text"]


if __name__ == "__main__":
    with open("data/processed/lecture1.txt", "r", encoding="utf-8") as f:
        txt = f.read()
    print(summarize_with_gemini(txt))
