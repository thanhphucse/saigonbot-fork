#!/usr/bin/env python3
"""
Run the SaigonBot Chatbot API server.
Reads API_HOST and API_PORT from .env / environment variables.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import uvicorn

load_dotenv()

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def _resolve_bind() -> tuple[str, int]:
    host = (os.getenv("API_HOST") or "0.0.0.0").strip() or "0.0.0.0"
    raw_port = (os.getenv("API_PORT") or "8000").strip() or "8000"
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise SystemExit(f"Invalid API_PORT={raw_port!r}: must be an integer") from exc
    return host, port


if __name__ == "__main__":
    host, port = _resolve_bind()
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True  # Auto-reload on code changes for development
    )
