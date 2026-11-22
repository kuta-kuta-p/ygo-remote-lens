#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

PORT="${PORT:-8000}"
CACHE_BIN=".cache/cloudflared"

# 1. cloudflared バイナリを確認（なければ .cache/cloudflared に自動取得・sudo不要・アカウント不要）
if command -v cloudflared >/dev/null 2>&1; then
  CF_BIN="$(command -v cloudflared)"
elif [ -x "$CACHE_BIN" ]; then
  CF_BIN="$CACHE_BIN"
else
  mkdir -p .cache
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64|amd64) CF_ASSET="cloudflared-linux-amd64" ;;
    aarch64|arm64) CF_ASSET="cloudflared-linux-arm64" ;;
    *) echo "未対応アーキテクチャ: $ARCH"; exit 1 ;;
  esac
  echo "⬇️ Cloudflare Quick Tunnel (cloudflared) を初回ダウンロード中..."
  curl -fL "https://github.com/cloudflare/cloudflared/releases/latest/download/${CF_ASSET}" -o "$CACHE_BIN"
  chmod +x "$CACHE_BIN"
  CF_BIN="$CACHE_BIN"
fi

SERVER_PID=""
cleanup() {
  echo ""
  echo "🛑 サーバーとトンネルを停止しています..."
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# 2. OpenCV SIFT サーバーをバックグラウンド起動
echo "🚀 OpenCV SIFT サーバーを起動中 (port: ${PORT})..."
if command -v uv >/dev/null 2>&1; then
  uv run --with-requirements requirements.txt uvicorn server:app --host 127.0.0.1 --port "$PORT" &
else
  python3 -m uvicorn server:app --host 127.0.0.1 --port "$PORT" &
fi
SERVER_PID=$!

# 起動完了待機
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

echo "✅ ローカルサーバー準備完了: http://localhost:${PORT}"
echo "🌐 友達共有用の一時HTTPS URLを発行中（Ctrl+C で終了）..."
echo "------------------------------------------------------------"

# 3. アカウント不要の Cloudflare Quick Tunnel 起動（WSL2/家庭用回線でのQUIC切断を防ぐため http2 を使用）
"$CF_BIN" tunnel --protocol http2 --url "http://127.0.0.1:${PORT}"
