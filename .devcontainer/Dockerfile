FROM mcr.microsoft.com/devcontainers/python:3.9-bullseye

# Install system dependencies for OpenCV and audio processing
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libasound2-dev \
        espeak \
        ffmpeg \
        nodejs \
        npm \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install node version 20
RUN npm install -g n && n 20

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace
