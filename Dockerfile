# Vidbush: API, UI and worker in one container.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIDBUSH_ROOT=/app \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip ffmpeg espeak-ng \
      fonts-dejavu-core libgl1 libglib2.0-0t64 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv

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
