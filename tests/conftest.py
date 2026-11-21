"""Shared pytest fixtures and lightweight helpers for the LOX test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the LOX repository root."""
    return Path(__file__).resolve().parents[1]
