FROM nvcr.io/nvidia/pytorch:24.10-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add CUDA 11.8 repository and install libraries for PaddlePaddle compatibility
# PaddlePaddle 2.6.2 requires cuDNN 8.6.0 for CUDA 11.8
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-11-8 \
    libcudnn8=8.6.0.*-1+cuda11.8 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublas.so.11 \
              /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublas.so \
    && ln -sf /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublasLt.so.11 \
              /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcublasLt.so

# Ensure PaddlePaddle uses the CUDA 11.8 libraries
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock README.md LICENSE ./

# Copy source code and config
COPY src/ ./src/
COPY .env ./

# Install uv and sync dependencies
RUN pip install uv && uv sync --no-dev

# Set the default command to run the server
CMD ["uv", "run", "python", "-m", "src.triton_server"]
