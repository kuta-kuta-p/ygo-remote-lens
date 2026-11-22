#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

if command -v uv >/dev/null 2>&1; then
  exec uv run --with-requirements requirements.txt uvicorn server:app --host 0.0.0.0 --port 8000
else
  exec python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
fi
