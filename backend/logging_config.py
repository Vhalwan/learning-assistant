# backend/logging_config.py
"""
Simple structured-ish logging configuration used by the API and other entrypoints.

This keeps logs consistent and machine-friendly without pulling in an external
JSON logging package. Call configure_logging() early in your process (FastAPI
startup event does this automatically).
"""
import logging
import sys

DEFAULT_LEVEL = logging.INFO

def configure_logging(level: int = DEFAULT_LEVEL) -> None:
    """
    Configure root logger to emit structured-ish log lines (JSON-like).
    Keeps the implementation minimal and dependency-free.
    """
    root = logging.getLogger()
    if root.handlers:
        # don't reconfigure if already configured
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    fmt = (
        '{"ts":"%(asctime)s","lvl":"%(levelname)s","name":"%(name)s",'
        '"msg":"%(message)s","module":"%(module)s","func":"%(funcName)s"}'
    )
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)
