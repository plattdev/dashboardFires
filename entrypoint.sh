#!/bin/sh
set -eu

# ═══════════════════════════════════════════════════════════════════
#  Entrypoint: bridge environment variables → Streamlit secrets.toml
#
#  This lets the SAME app.py code work unchanged on:
#    • Streamlit Community Cloud  (reads .streamlit/secrets.toml)
#    • Fly.io / Cloud Run / Docker (env vars → secrets.toml at boot)
# ═══════════════════════════════════════════════════════════════════

SECRETS_DIR="/app/.streamlit"
SECRETS_FILE="${SECRETS_DIR}/secrets.toml"

mkdir -p "${SECRETS_DIR}"

# Only overwrite if FIRMS_API_KEY is provided as an env var
if [ -n "${FIRMS_API_KEY:-}" ]; then
    cat > "${SECRETS_FILE}" <<EOF
FIRMS_API_KEY = "${FIRMS_API_KEY}"
EOF
    echo "entrypoint: wrote FIRMS_API_KEY to ${SECRETS_FILE}"
elif [ ! -f "${SECRETS_FILE}" ]; then
    echo "entrypoint: WARNING — FIRMS_API_KEY not set and no secrets.toml found"
fi

exec streamlit run app.py \
    --server.port=8080 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.runOnSave=false \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false \
    --client.toolbarMode=minimal

