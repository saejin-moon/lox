"""
Minimal .env loader (no external dependency).

`python-dotenv` is not installed in this environment; scripts that relied on it
silently ran without the API key. This helper parses KEY=VALUE lines and sets
os.environ (without overriding an existing value unless override=True), matching
the manual fallback the working scripts already carried.
"""
from __future__ import annotations

import os


def load_env(path: str | None = None, override: bool = False) -> bool:
    """Loads KEY=VALUE pairs from .env. Returns True if the file was read."""
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if override or key not in os.environ:
                    os.environ[key] = value
        return True
    except OSError:
        return False