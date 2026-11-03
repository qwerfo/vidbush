# Vidbush: API, UI and worker in one container.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIDBUSH_ROOT=/app

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg espeak-ng fonts-dejavu-core libgl1 libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY vidbush ./vidbush
RUN pip install ".[whisper]"

COPY config ./config
COPY assets ./assets
COPY fixtures ./fixtures
COPY ui/dist ./ui/dist
COPY docs ./docs
COPY .env.example ./

RUN mkdir -p /app/output /app/data
VOLUME ["/app/output", "/app/data", "/app/music"]
EXPOSE 8080
CMD ["python", "-m", "vidbush", "serve", "--host", "0.0.0.0", "--port", "8080"]
