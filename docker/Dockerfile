# syntax=docker/dockerfile:1

# ---- the browser bundle -------------------------------------------------
FROM node:20-alpine AS web
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# vite.config.ts writes to ../sttt/web/static, i.e. /app/sttt/web/static
RUN npm run build

# ---- the native search --------------------------------------------------
FROM python:3.13-slim AS native
RUN apt-get update && apt-get install -y --no-install-recommends g++ make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY cpp/ ./cpp/
# -march=native is NOT usable here. cpp/Makefile and setup.py both default to
# it, which is correct for a machine building for itself and fatal for an image
# built on one CPU and pulled onto an older one -- it dies with SIGILL, for
# some users only. x86-64-v2 (SSE4.2, POPCNT) is safe on anything since ~2009.
# CI asserts the built module reports this rather than 'native'.
RUN make -C cpp sttt_cpp.so PYTHON=python3 \
    CXXFLAGS="-O3 -march=x86-64-v2 -Wall -Wextra -Werror -std=c++17 -fPIC -pthread -I./include"

# ---- what actually ships ------------------------------------------------
FROM python:3.13-slim
LABEL org.opencontainers.image.title="Super Tic-Tac-Toe" \
      org.opencontainers.image.description="Play Ultimate Tic-Tac-Toe against a self-play network" \
      org.opencontainers.image.licenses="MIT"

RUN useradd --create-home --uid 10001 player
WORKDIR /app

COPY docker/requirements-web.txt ./
RUN pip install --no-cache-dir --no-compile -r requirements-web.txt

COPY sttt/ ./sttt/
COPY configs/web/ ./configs/web/
COPY models/model-6000.onnx models/model-6000.onnx.json ./models/
COPY --from=native /app/cpp/sttt_cpp*.so ./cpp/
COPY --from=web /app/sttt/web/static ./sttt/web/static

ENV STTT_WEB_MODEL=/app/models/model-6000.onnx \
    STTT_WEB_THREADS=4 \
    STTT_WEB_MAX_SESSIONS=8 \
    STTT_WEB_SESSION_TTL=1800 \
    STTT_WEB_PORT=8000 \
    STTT_WEB_BACKEND=auto \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER player
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u, sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"
CMD ["python", "-m", "sttt.web.server"]
