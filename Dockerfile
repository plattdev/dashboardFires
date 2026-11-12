# syntax=docker/dockerfile:1
# ═══════════════════════════════════════════════════════════════════
#  Spanish Wildfire Tracker — Production Container (2026)
#  Multi-stage build: fast installs with uv, minimal runtime image.
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: Install Python dependencies ─────────────────────────
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /usr/local/bin/uv

WORKDIR /build

COPY requirements.txt .
RUN uv pip install \
        --system \
        --no-cache \
        --compile-bytecode \
        -r requirements.txt

# ── Stage 2: Lean runtime image ──────────────────────────────────
FROM python:3.12-slim AS runtime

# System libraries required for runtime, rasterio (libexpat1) and healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl tini libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the installed site-packages and console scripts from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Non-root user (CIS Docker Benchmark §4.1)
RUN groupadd -r appuser \
    && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy application code and data
COPY --chown=appuser:appuser . .

# Ensure the entrypoint is executable
RUN chmod +x /app/entrypoint.sh

USER appuser

EXPOSE 8080

# Streamlit exposes /_stcore/health — use it for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -fs http://localhost:8080/_stcore/health || exit 1

# tini ensures Streamlit receives SIGTERM correctly on shutdown
ENTRYPOINT ["tini", "--"]
CMD ["/app/entrypoint.sh"]

