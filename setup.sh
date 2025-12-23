#!/bin/bash
set -e

echo "=========================================="
echo "  Triton Server Setup Script"
echo "=========================================="

# Change to the directory where this script is located
cd "$(dirname "$0")"

# ==========================================
# 1. Install Python 3.12
# ==========================================
echo ""
echo "[1/5] Installing Python 3.12..."

if command -v python3.12 &> /dev/null; then
    echo "Python 3.12 is already installed: $(python3.12 --version)"
else
    echo "Installing Python 3.12..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    echo "Python 3.12 installed: $(python3.12 --version)"
fi

# ==========================================
# 2. Install NVIDIA Container Toolkit
# ==========================================
echo ""
echo "[2/5] Installing NVIDIA Container Toolkit..."

if command -v nvidia-ctk &> /dev/null; then
    echo "NVIDIA Container Toolkit is already installed"
else
    echo "Installing NVIDIA Container Toolkit..."
    
    # Add NVIDIA GPG key and repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker to use NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker || true
    
    echo "NVIDIA Container Toolkit installed successfully"
fi

# ==========================================
# 3. Install uv (if not already installed)
# ==========================================
echo ""
echo "[3/5] Checking uv installation..."

if command -v uv &> /dev/null; then
    echo "uv is already installed: $(uv --version)"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the shell config to get uv in PATH
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed: $(uv --version)"
fi

# Ensure uv is in PATH for this session
export PATH="$HOME/.local/bin:$PATH"

# ==========================================
# 4. Create .venv and sync with uv lock
# ==========================================
echo ""
echo "[4/5] Creating virtual environment and syncing dependencies..."

if [ -f "uv.lock" ]; then
    echo "Found uv.lock, syncing dependencies..."
    uv sync --python 3.12
    echo "Dependencies synced successfully"
else
    echo "No uv.lock found, creating new virtual environment..."
    uv venv --python 3.12
    uv sync
    echo "Virtual environment created and dependencies installed"
fi

# ==========================================
# 5. Download model checkpoints
# ==========================================
echo ""
echo "[5/5] Downloading model checkpoints..."

if [ -f "checkpoints/download.sh" ]; then
    bash checkpoints/download.sh
    echo "Checkpoints downloaded successfully"
else
    echo "Warning: checkpoints/download.sh not found, skipping checkpoint download"
fi

# ==========================================
# Done!
# ==========================================

