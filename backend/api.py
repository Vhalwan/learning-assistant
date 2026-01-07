"""
FastAPI application exposing programmatic access to core Learning Assistant flows.

Endpoints:
 - POST /query         -> run RAG query against embeddings + optional FAISS index (Q&A mode)
 - POST /summarize     -> produce document-level summary from embeddings (Summary mode)
 - POST /build_index   -> build (or rebuild) a FAISS index from embeddings JSON
 - POST /append_index  -> append new embeddings into an existing FAISS index
 - POST /generate_quiz -> generate a quiz from provided context (writes JSON file)
 - GET  /status        -> index status (exists, n_vectors, dim, created_at, index_mtime)
 - POST /chat          -> conversational chat mode with per-session history + RAG
"""
from __future__ import annotations
import time
import logging
import os
from typing import Optional, Any, Dict, List
from dotenv import load_dotenv
load_dotenv()
from backend.llm_client import get_llm_call

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel

from backend.logging_config import configure_logging

from backend.rag_query import (
    rag_answer_from_embeddings,
    rag_generate_summary_from_embeddings,
    rag_chat_answer,
)
from backend.vectorstore.faiss_store import (
    build_faiss_index,
    append_to_index,
    get_index_status,
    rebuild_index_if_needed,
)
from backend.generate_quiz import generate_quiz_from_context

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Learning Assistant API", version="0.2.0")


# Simple auth guard dependency
def check_token(authorization: Optional[str] = Header(None)) -> bool:
    """
    If API_TOKEN env var is set, require requests to send Authorization: Bearer <token>.
    If API_TOKEN is not set, allow requests (useful for local dev).
    """
    api_token = os.getenv("API_TOKEN", "") or ""
    if not api_token:
        # no token configured to allow access
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


class SummarizeRequest(BaseModel):
    embeddings_path: str
    summary_type: Optional[str] = "brief"  # "brief" or "detailed"
    top_k: Optional[int] = None
    use_safe: Optional[bool] = None


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
    type: Optional[str] = "mcq"


class ChatRequest(BaseModel):
    """
    Request body for conversational chat mode.
    history: optional list of {"role": "user" | "assistant", "content": "<text>"} entries.
    """
    question: str
    embeddings_path: str
    history: Optional[List[Dict[str, str]]] = None
    top_k: Optional[int] = 3
    use_faiss: Optional[bool] = False
    faiss_index_path: Optional[str] = None
    use_safe: Optional[bool] = None
    use_query_expansion: Optional[bool] = False


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
    Run a RAG query (Q&A mode) and return the answer with retrieved chunks, prompt and provenance.
    This endpoint intentionally does NOT produce a document-level summary.
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


@app.post("/chat")
def post_chat(body: ChatRequest, _auth: Any = Depends(check_token)) -> Dict[str, Any]:
    """
    Conversational chat endpoint.

    Accepts:
      - question: the user's new message
      - history: optional prior conversation (list of {"role": "user"/"assistant", "content": "..."})
      - embeddings_path: path to embeddings JSON
      - top_k / use_faiss / faiss_index_path etc.

    Behavior:
      - Calls backend.rag_query.rag_chat_answer(...) which is expected to
        build a prompt using the conversation history + retrieved chunks
        and return the assistant's reply and updated history plus metadata.
    """
    start = time.perf_counter()
    try:
        faiss_idx = body.faiss_index_path or (
            _default_index_path_for_embeddings(body.embeddings_path) if body.use_faiss else None
        )

        # Call the RAG chat helper. Expected signature:
        # rag_chat_answer(question, embeddings_path, history=None, top_k=3, use_faiss=False, faiss_index_path=None, use_safe=None, use_query_expansion=False, return_meta=True)
        chat_out = rag_chat_answer(
            body.question,
            body.embeddings_path,
            history=body.history,
            top_k=body.top_k,
            use_faiss=bool(body.use_faiss),
            faiss_index_path=faiss_idx,
            use_safe=body.use_safe,
            use_query_expansion=bool(body.use_query_expansion),
            return_meta=True,
        )
    except FileNotFoundError as e:
        logger.warning("Chat failed (missing file): %s", e)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Chat failed unexpectedly: %s", e)
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    elapsed = time.perf_counter() - start

    # Expected return shapes (defensive handling):
    # Preferred: (answer, updated_history, retrieved_chunks, prompt_used, provenance)
    # Or: (answer, updated_history)
    # Or: answer string
    answer = None
    updated_history = None
    retrieved_chunks = None
    prompt_used = None
    provenance = None

    try:
        # If it's a tuple/list, unpack defensively
        if isinstance(chat_out, (tuple, list)):
            if len(chat_out) >= 1:
                answer = chat_out[0]
            if len(chat_out) >= 2:
                updated_history = chat_out[1]
            if len(chat_out) >= 3:
                retrieved_chunks = chat_out[2]
            if len(chat_out) >= 4:
                prompt_used = chat_out[3]
            if len(chat_out) >= 5:
                provenance = chat_out[4]
        else:
            # single value (likely string)
            answer = chat_out
    except Exception as e:
        logger.exception("Failed to parse rag_chat_answer output: %s", e)
        raise HTTPException(status_code=500, detail=f"Unexpected chat response shape: {e}")

    # If rag helper didn't return history, build one locally (append user+assistant)
    try:
        if updated_history is None:
            # Start from provided history or empty, append latest turn
            base_hist = list(body.history or [])
            # ensure roles are set correctly
            base_hist.append({"role": "user", "content": body.question})
            base_hist.append({"role": "assistant", "content": answer})
            updated_history = base_hist
    except Exception:
        # fallback minimal history
        updated_history = [{"role": "user", "content": body.question}, {"role": "assistant", "content": answer}]

    resp = {
        "answer": answer,
        "history": updated_history,
        "retrieved": retrieved_chunks,
        "prompt": prompt_used,
        "provenance": provenance,
        "latency_s": elapsed,
    }

    logger.info(
        "Chat completed",
        extra={
            "latency_s": elapsed,
            "question_hash": hash(body.question),
            "history_len": len(updated_history) if updated_history else 0,
        },
    )
    return resp


@app.post("/summarize")
def post_summarize(body: SummarizeRequest, _auth: Any = Depends(check_token)) -> Dict[str, Any]:
    """
    Produce a document-level summary using the embeddings JSON as the source.
    Returns summary text and extracted key concepts and optionally the used chunks.
    """
    start = time.perf_counter()
    try:
        out, used_chunks = rag_generate_summary_from_embeddings(
            body.embeddings_path,
            summary_type=body.summary_type or "brief",
            top_k=body.top_k,
            use_safe=body.use_safe,
            return_meta=False,
        )
    except FileNotFoundError as e:
        logger.warning("Summarize failed: %s", e)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Summarize failed unexpectedly: %s", e)
        raise HTTPException(status_code=500, detail=f"Summarize failed: {e}")

    elapsed = time.perf_counter() - start
    resp = {
        "out": out,  # contains 'summary' and 'key_concepts'
        "used_chunks": used_chunks,
        "latency_s": elapsed,
    }
    logger.info("Summarize completed", extra={"embeddings_path": body.embeddings_path, "time_s": elapsed})
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


@app.post("/generate_quiz_live")
def post_generate_quiz_live(body: GenerateQuizRequest, _auth: Any = Depends(check_token)) -> Dict[str, Any]:
    """
    Live quiz generation using the configured LLM (requires GEMINI_API_KEY in env or .env).
    Returns:
      - quiz: list of {id, question, choices, answer, [explanation]}
      - raw_llm: list of raw LLM responses (one per chunk / attempt) for debugging
      - latency_s: total time
    """
    start = time.perf_counter()
    try:
        if (body.type or "mcq").lower() != "mcq":
            raise HTTPException(status_code=400, detail=f"Unsupported quiz type: {body.type}")

        # lazy import to avoid circulars
        from backend.generate_quiz import generate_mcq_from_context

        # get a working llm_call (or None)
        llm_call = get_llm_call()
        if llm_call is None:
            raise HTTPException(status_code=503, detail="LLM not initialized. Check GEMINI_API_KEY / google-genai installation.")

        # capture raw LLM responses for debugging / audit
        raw_outputs: List[str] = []
        def capturing_llm(prompt: str) -> str:
            out = llm_call(prompt)
            try:
                raw_outputs.append(out if isinstance(out, str) else str(out))
            except Exception:
                raw_outputs.append("<unable to stringify raw output>")
            return out

        # generate MCQs (will call the capturing_llm above)
        quiz_list = generate_mcq_from_context(body.context_text, n=body.n or 5, llm_call=capturing_llm)

        # prefix IDs with stem
        prefixed = []
        for item in quiz_list:
            item_id = item.get("id") or f"q{len(prefixed)+1}"
            item["id"] = f"{body.stem}_{item_id}"
            prefixed.append(item)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Live quiz generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Live quiz generation failed: {e}")

    elapsed = time.perf_counter() - start
    return {"quiz": prefixed, "raw_llm": raw_outputs, "latency_s": elapsed}


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
