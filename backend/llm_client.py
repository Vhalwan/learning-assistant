import os
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

def get_llm_call() -> Optional[Callable[[str], str]]:
    """
    Return a function llm_call(prompt)->str using the GEMINI_API_KEY.
    This version **ignores QUIZ_FORCE_PLACEHOLDERS** and always uses the key if available.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        logger.info("No GEMINI_API_KEY / GOOGLE_API_KEY found -> using placeholders.")
        return None

    # ALWAYS use the key, ignore placeholders
    try:
        import google.genai as genai
        client = genai.Client(api_key=gemini_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        def _genai_call(prompt: str) -> str:
            resp = client.models.generate_content(model=model_name, contents=prompt)
            return getattr(resp, "text", str(resp))

        logger.info("LLM callable ready, using GEMINI_API_KEY from environment.")
        return _genai_call

    except Exception as e:
        logger.warning("Failed to initialize google-genai client: %s", e)
        return None
