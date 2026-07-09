# ---------- Stage 1: build wheels ----------
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---------- Stage 2: minimal runtime ----------
FROM python:3.12-slim AS runtime

# libgomp is required by xgboost's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /usr/sbin/nologin appuser
WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY src/ ./src/

USER appuser
ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--trials", "25"]
