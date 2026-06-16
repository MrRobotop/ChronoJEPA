#!/usr/bin/env bash
# Create the environment, install dependencies, lint, and run the tests.
# Future sessions run this to verify the project state in one command.
set -euo pipefail

# Run from the repository root regardless of where the script is invoked.
cd "$(dirname "$0")/.."

echo "==> uv sync"
uv sync

echo "==> ruff check"
uv run ruff check .

echo "==> ruff format --check"
uv run ruff format --check .

echo "==> pytest"
uv run pytest -q

echo "==> OK: environment ready, lint clean, tests green"
