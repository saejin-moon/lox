#!/usr/bin/env bash
set -e
uv run python scripts/consolidate_duckdb.py "$@"
