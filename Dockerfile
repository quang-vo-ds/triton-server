FROM nvcr.io/nvidia/pytorch:24.10-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (no build-essential needed, PyTorch already installed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock README.md LICENSE ./

# Copy source code and config
COPY src/ ./src/
COPY .env ./

# Install uv and sync dependencies
RUN pip install uv && uv sync --no-dev --python 3.12

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Set the default command to run the server
CMD ["uv", "run", "python", "-m", "triton_server"]
