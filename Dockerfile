# Healthcare Knowledge Navigator — RAG + CRAG on NVIDIA + Pinecone.
#
# Multi-stage build: builder installs deps into a venv, runtime copies just
# the venv + source, keeping the final image small.

# ─── builder ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build-time system deps needed by a few Python wheels (unstructured, pypdf).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt

# ─── runtime ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# Runtime system deps: libmagic for `unstructured`, poppler for PDF text
# extraction (used by pypdf in some code paths).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for the app.
RUN useradd --create-home --shell /bin/bash hkn
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=hkn:hkn server.py ingest.py list_models.py verify_pinecone.py ./
COPY --chown=hkn:hkn src ./src
COPY --chown=hkn:hkn static ./static
COPY --chown=hkn:hkn data ./data

USER hkn

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
