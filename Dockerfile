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
      fonts-dejavu-core libgl1 libglib2.0-0t64 ca-certificates curl unzip \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv

# yt-dlp needs a JavaScript runtime (deno) plus yt-dlp-ejs to solve YouTube player
# challenges; without them most YouTube formats are missing. Official release binary.
ARG DENO_VERSION=2.9.8
RUN case "$(dpkg --print-architecture)" in \
      amd64) DENO_ARCH=x86_64 ;; arm64) DENO_ARCH=aarch64 ;; *) echo "unsupported arch" && exit 1 ;; \
    esac \
 && curl -fsSL -o /tmp/deno.zip \
      "https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-${DENO_ARCH}-unknown-linux-gnu.zip" \
 && unzip -q /tmp/deno.zip -d /usr/local/bin \
 && chmod +x /usr/local/bin/deno \
 && rm /tmp/deno.zip \
 && deno --version

WORKDIR /app
COPY pyproject.toml ./
COPY vidbush ./vidbush
RUN pip install -e ".[whisper]"

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
