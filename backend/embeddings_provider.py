# backend/embeddings_provider.py
from __future__ import annotations
import os
import time
import logging
from typing import List, Optional
import numpy as np
import requests

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
USE_SAFE = os.getenv("USE_SAFE_EMBEDDINGS", "1").lower() in ("1", "true", "yes")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embedding-001")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "").strip()

MAX_RETRIES = int(os.getenv("EMBED_RETRIES", "2"))
BACKOFF_BASE = float(os.getenv("EMBED_BACKOFF_BASE", "0.5"))


def deterministic_vector(text: str, dim: int = EMBED_DIM) -> List[float]:
    """Deterministic local pseudo-embedding (sha256 -> RNG)"""
    import hashlib
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big") % (2 ** 32)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype("float32")
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.tolist()


class BaseEmbeddingProvider:
    def embed(self, texts: List[str], dim: int = EMBED_DIM, batch_size: int = 32) -> List[List[float]]:
        raise NotImplementedError()


class SafeEmbeddingProvider(BaseEmbeddingProvider):
    """Local deterministic embeddings — used for testing & when no key present."""
    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def embed(self, texts: List[str], dim: int = None, batch_size: int = 32) -> List[List[float]]:
        dim = dim or self.dim
        return [deterministic_vector(t, dim=dim) for t in texts]


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """
    Uses google.generativeai client if available, otherwise does an HTTP POST to EMBEDDING_API_URL.
    All network I/O is inside _call_api so unit tests can monkeypatch it.
    """

    def __init__(self, api_key: str, model: str = EMBED_MODEL, api_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url or EMBEDDING_API_URL
        if not self.api_url:
            self.api_url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:embed"
            logger.debug("No EMBEDDING_API_URL set; using placeholder URL: %s", self.api_url)

        # Detect optional official client
        self._has_genai = False
        try:
            import google.generativeai as genai  # type: ignore
            self._genai = genai
            self._has_genai = True
            logger.debug("google.generativeai client detected; will prefer it if usable")
        except Exception:
            self._genai = None
            self._has_genai = False

    def embed(self, texts: List[str], dim: int = EMBED_DIM, batch_size: int = 32) -> List[List[float]]:
        all_vecs: List[List[float]] = []
        n = len(texts)
        batch_size = max(1, int(batch_size))
        for start in range(0, n, batch_size):
            batch = texts[start:start + batch_size]
            attempt = 0
            while True:
                try:
                    vecs = self._call_api(batch)
                    if not isinstance(vecs, list) or len(vecs) != len(batch):
                        raise ValueError("API returned unexpected shape/length")
                    all_vecs.extend(vecs)
                    break
                except requests.HTTPError as e:
                    status = getattr(e.response, "status_code", None)
                    attempt += 1
                    if status == 429 and attempt <= MAX_RETRIES:
                        sleep = BACKOFF_BASE * (2 ** (attempt - 1))
                        logger.warning("Rate limited (429). Backing off %.2fs (attempt %d/%d).", sleep, attempt, MAX_RETRIES)
                        time.sleep(sleep)
                        continue
                    logger.exception("HTTP error from embedding API (attempt=%d): %s", attempt, e)
                    raise
                except Exception as e:
                    attempt += 1
                    if attempt <= MAX_RETRIES:
                        sleep = BACKOFF_BASE * (2 ** (attempt - 1))
                        logger.warning("Transient embedding error. Retrying in %.2fs (attempt %d/%d).", sleep, attempt, MAX_RETRIES)
                        time.sleep(sleep)
                        continue
                    logger.exception("Embedding API failed after %d attempts.", attempt)
                    raise

        # We can normalize to L2 = 1 per vector
        arr = np.array(all_vecs, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] == 0:
            raise ValueError("Embedding provider returned invalid array shape")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        return arr.tolist()

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        Perform the low-level call. Tests should monkeypatch this method to return
        deterministic vectors. Prefer official client when available; otherwise, POST to api_url.
        Expected return: list of vectors (list[float]) of same length as input texts.
        """
        if self._has_genai and self._genai is not None:
            try:
                # configure the client if needed
                if self.api_key:
                    try:
                        self._genai.configure(api_key=self.api_key)
                    except Exception:
                        os.environ["GOOGLE_API_KEY"] = self.api_key
                # Attempt to call a common embeddings method
                resp = self._genai.get_embeddings(model=self.model, input=texts)
                if isinstance(resp, dict) and "data" in resp:
                    out = []
                    for item in resp["data"]:
                        v = item.get("embedding") or item.get("vector")
                        if isinstance(v, list):
                            out.append([float(x) for x in v])
                        else:
                            raise ValueError("Client returned non-list embedding")
                    return out
                if isinstance(resp, list):
                    out = []
                    for r in resp:
                        if isinstance(r, dict):
                            v = r.get("embedding") or r.get("vector")
                            out.append([float(x) for x in v])
                        else:
                            out.append([float(x) for x in r])
                    return out
            except Exception as e:
                logger.exception("google.generativeai client invocation failed; falling back to HTTP: %s", e)


        # HTTP fallback
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"model": self.model, "input": texts}
        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
        if resp.status_code >= 400:
            http_err = requests.HTTPError(f"Embedding API request failed: {resp.status_code} {resp.text}")
            http_err.response = resp
            raise http_err

        data = resp.json()
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            out = []
            for item in data["data"]:
                v = item.get("embedding") or item.get("vector") or item.get("embeddings")
                if isinstance(v, list):
                    out.append([float(x) for x in v])
                else:
                    raise ValueError("Unable to parse embedding vector from response item")
            return out

        if "embeddings" in data and isinstance(data["embeddings"], list):
            return [[float(x) for x in e] for e in data["embeddings"]]

        if "embedding" in data and isinstance(data["embedding"], list):
            # support [[...], [...]] or a single vector
            if isinstance(data["embedding"][0], list):
                return [[float(x) for x in e] for e in data["embedding"]]
            return [[float(x) for x in data["embedding"]]]

        raise ValueError("Embedding API response missing expected fields. Keys: %s" % list(data.keys()))


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Factory: respects USE_SAFE and missing key fallback."""
    if USE_SAFE:
        logger.info("USE_SAFE_EMBEDDINGS enabled -> using SafeEmbeddingProvider")
        return SafeEmbeddingProvider(dim=EMBED_DIM)

    if not GEMINI_API_KEY:
        logger.warning("No GEMINI_API_KEY found; falling back to SafeEmbeddingProvider")
        return SafeEmbeddingProvider(dim=EMBED_DIM)

    try:
        provider = GoogleEmbeddingProvider(api_key=GEMINI_API_KEY, model=EMBED_MODEL)
        logger.info("Using GoogleEmbeddingProvider (model=%s)", EMBED_MODEL)
        return provider
    except Exception as e:
        logger.exception("Failed to initialize GoogleEmbeddingProvider; falling back to SafeEmbeddingProvider: %s", e)
        return SafeEmbeddingProvider(dim=EMBED_DIM)
