FROM docker.io/library/python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir uv==0.12.15
COPY pyproject.toml uv.lock ./
COPY backend ./backend
COPY python/nnmodelling-runtime ./python/nnmodelling-runtime
RUN uv sync --frozen --package nnmodelling-backend --group examples --no-dev

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
USER 65532:65532
