#!/usr/bin/env bash
set -euo pipefail

echo "==> Building and installing lox-core PyO3 extension..."
uv run maturin develop --manifest-path crates/lox-core/Cargo.toml --release
echo "==> lox-core successfully built and installed into active environment!"
