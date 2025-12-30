# frontend/api.py
"""
FastAPI application exposing programmatic access to core Learning Assistant flows.

Endpoints:
 - POST /query         -> run RAG query against embeddings + optional FAISS index
 - POST /build_index   -> build (or rebuild) a FAISS index from embeddings JSON
 - POST /append_index  -> append new embeddings into an existing FAISS index
 - POST /generate_quiz -> generate a quiz from provided context (writes JSON file)
 - GET  /status        -> index status (exists, n_vectors, dim, created_at, index_mtime)

Notes:
 - This module favors using the deterministic "safe" embedding path by default
   via rag_answer_from_embeddings's default behavior (USE_SAFE_EMBEDDINGS=1).
 - Responses include latency_s where applicable.
"""
from __future__ import annotations
import time
import logging
import os
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel

from backend.logging_config import configure_logging

# core functions from your backend
from backend.rag_query import rag_answer_from_embeddings
from backend.vectorstore.faiss_store import (
    build_faiss_index,
    append_to_index,
    get_index_status,
    rebuild_index_if_needed,
)
from backend.generate_quiz import generate_quiz_from_context

# configure logging once
configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Learning Assistant API", version="0.1.0")


# Simple auth guard dependency
def check_token(authorization: Optional[str] = Header(None)) -> bool:
    """
    If API_TOKEN env var is set, require requests to send Authorization: Bearer <token>.
    If API_TOKEN is not set, allow requests (useful for local dev).
    """
    api_token = os.getenv("API_TOKEN", "") or ""
    if not api_token:
        # no token configured -> allow access
        return True

    if not authorization:
        logger.warning("Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        logger.warning("Malformed Authorization header")
        raise HTTPException(status_code=401, detail="Malformed Authorization header")

    token = authorization.split(" ", 1)[1].strip()
    if token != api_token:
        logger.warning("Invalid API token provided")
        raise HTTPException(status_code=401, detail="Invalid API token")

    # authorized
    return True


# Request / response models
class QueryRequest(BaseModel):
    question: str
    embeddings_path: str
    top_k: Optional[int] = 3
    use_faiss: Optional[bool] = False
    faiss_index_path: Optional[str] = None
    use_safe: Optional[bool] = None
    use_query_expansion: Optional[bool] = False


class BuildIndexRequest(BaseModel):
    embeddings_path: str
    index_path: Optional[str] = None
    force: Optional[bool] = False


class AppendIndexRequest(BaseModel):
    embeddings_path: str
    index_path: Optional[str] = None


class GenerateQuizRequest(BaseModel):
    stem: str
    context_text: str
    n: Optional[int] = 5


# Helpers
def _default_index_path_for_embeddings(embeddings_path: str) -> str:
    # default: strip .json and append .index
    if embeddings_path.endswith(".json"):
        return embeddings_path[:-5] + ".index"
    return embeddings_path + ".index"


# Endpoints
@app.post("/query")
def post_query(body: QueryRequest, _auth: Any = Depends(check_token)) -> Dict[str, Any]:
    """
    Run a RAG query and return the answer with retrieved chunks, prompt and provenance.
    """
    start = time.perf_counter()
    try:
        answer_tuple = rag_answer_from_embeddings(
            body.question,
            body.embeddings_path,
            top_k=body.top_k,
            use_faiss=bool(body.use_faiss),
            faiss_index_path=body.faiss_index_path
            or (_default_index_path_for_embeddings(body.embeddings_path) if body.use_faiss else None),
            use_safe=body.use_safe,
            use_query_expansion=bool(body.use_query_expansion),
            return_meta=True,
        )
    except FileNotFoundError as e:
        logger.warning("Query failed: %s", e)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Query failed unexpectedly: %s", e)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    elapsed = time.perf_counter() - start
    # rag_answer_from_embeddings returns (answer, retrieved_chunks, prompt, provenance)
    try:
        answer, retrieved_chunks, prompt_used, provenance = answer_tuple
    except Exception:
        if isinstance(answer_tuple, tuple) and len(answer_tuple) >= 2:
            answer = answer_tuple[0]
            retrieved_chunks = answer_tuple[1]
            prompt_used = None
            provenance = None
        else:
            raise HTTPException(status_code=500, detail="Unexpected response from RAG function")

    resp = {
        "answer": answer,
        "retrieved": retrieved_chunks,
        "prompt": prompt_used,
        "provenance": provenance,
        "latency_s": elapsed,
    }
    logger.info(
        "RAG query completed",
        extra={"latency_s": elapsed, "question_hash": hash(body.question)},
    )
    return resp


@app.post("/build_index")
def post_build_index(body: BuildIndexRequest, _auth: Any = Depends(check_token)) -> Dict[str, Any]:
    """
    Build or rebuild a FAISS index from embeddings JSON.
    If force=True, rebuilds always. Otherwise rebuilds only if needed.
    """
    index_path = body.index_path or _default_index_path_for_embeddings(body.embeddings_path)
    start = time.perf_counter()
    try:
        did = rebuild_index_if_needed(body.embeddings_path, index_path, force=bool(body.force))
    except FileNotFoundError as e:
        logger.warning("Build index failed: %s", e)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Index build failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Index build failed: {e}")

    elapsed = time.perf_counter() - start
    status = get_index_status(index_path)
    resp = {"built": bool(did), "index_path": index_path, "status": status, "latency_s": elapsed}
    logger.info("Index build finished", extra={"index_path": index_path, "built": bool(did), "time_s": elapsed})
    return resp


@app.post("/append_index")
def post_append_index(body: AppendIndexRequest, _auth: Any = Depends(check_token)) -> Dict[str, Any]:
    """
    Append embeddings from embeddings_path into an existing index (or create it if missing).
    Returns appended_count and new status.
    """
    index_path = body.index_path or _default_index_path_for_embeddings(body.embeddings_path)
    start = time.perf_counter()
    try:
        appended = append_to_index(body.embeddings_path, index_path)
    except FileNotFoundError as e:
        logger.warning("Append index failed: %s", e)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Append index failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Append failed: {e}")
    elapsed = time.perf_counter() - start
    status = get_index_status(index_path)
    resp = {"appended": int(appended), "index_path": index_path, "status": status, "latency_s": elapsed}
    logger.info("Index append finished", extra={"index_path": index_path, "appended": appended, "time_s": elapsed})
    return resp


@app.post("/generate_quiz")
def post_generate_quiz(body: GenerateQuizRequest, _auth: Any = Depends(check_token)) -> Dict[str, Any]:
    """
    Generate a quiz from context_text. Writes JSON file to data/processed/<stem>_quiz.json.
    """
    start = time.perf_counter()
    try:
        quiz = generate_quiz_from_context(body.stem, body.context_text, n=body.n)
    except Exception as e:
        logger.exception("Quiz generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {e}")
    elapsed = time.perf_counter() - start
    out_path = f"data/processed/{body.stem}_quiz.json"
    resp = {"quiz": quiz, "out_path": out_path, "latency_s": elapsed}
    logger.info("Quiz generated", extra={"stem": body.stem, "n": body.n, "time_s": elapsed})
    return resp


@app.get("/status")
def get_status(
    index_path: Optional[str] = None, embeddings_path: Optional[str] = None, _auth: Any = Depends(check_token)
) -> Dict[str, Any]:
    """
    Return index status. Provide either index_path or embeddings_path (index inferred).
    """
    if not index_path and not embeddings_path:
        raise HTTPException(status_code=400, detail="index_path or embeddings_path must be provided")
    idx = index_path or _default_index_path_for_embeddings(embeddings_path)
    try:
        status = get_index_status(idx)
    except Exception as e:
        logger.exception("Status check failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}")
    # Also report whether FAISS is available on this host
    faiss_available = True
    try:
        # lazy import check
        import backend.vectorstore.faiss_store as fs

        faiss_available = getattr(fs, "FAISS_AVAILABLE", True)
    except Exception:
        faiss_available = False
    resp = {"index_path": idx, "status": status, "faiss_available": faiss_available}
    return resp
