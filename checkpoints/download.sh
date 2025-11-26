#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Use either wget or curl to download the checkpoints
if command -v wget &> /dev/null; then
    CMD="wget"
    DOWNLOAD_CMD="wget -O"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
    DOWNLOAD_CMD="curl -L -o"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

echo "Setting up checkpoint directories..."

# Create directories for each model
mkdir -p sam
mkdir -p whisper
mkdir -p owlv2
mkdir -p paddle-ocr

# Define the URLs for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_t_url="${SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
sam2p1_hiera_s_url="${SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
sam2p1_hiera_b_plus_url="${SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"
sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

# SAM 2.1 checkpoints
echo "Downloading SAM 2.1 checkpoints..."
echo "Downloading sam2.1_hiera_tiny.pt..."
$DOWNLOAD_CMD sam/sam2.1_hiera_tiny.pt $sam2p1_hiera_t_url || { echo "Failed to download checkpoint from $sam2p1_hiera_t_url"; exit 1; }

# Whisper checkpoints using Hugging Face Hub
echo "Downloading whisper-tiny model..."
HF_HUB_ENABLE_HF_TRANSFER=1 hf download openai/whisper-tiny --local-dir whisper/whisper-tiny || { echo "Failed to download Whisper tiny model"; exit 1; }

# OWLv2 checkpoints using Hugging Face Hub
echo "Downloading owlv2-base-patch16 model..."
HF_HUB_ENABLE_HF_TRANSFER=1 hf download google/owlv2-base-patch16 --local-dir owlv2/owlv2-base-patch16 || { echo "Failed to download OWLv2 base model"; exit 1; }

# PaddleOCR checkpoints using Hugging Face Hub
echo "Downloading PP-OCRv5_server_det model..."
HF_HUB_ENABLE_HF_TRANSFER=1 hf download PaddlePaddle/PP-OCRv5_server_det --local-dir paddle-ocr/PP-OCRv5_server_det || { echo "Failed to download PP-OCRv5_server_det model"; exit 1; }

echo "Downloading PP-OCRv5_server_rec model..."
HF_HUB_ENABLE_HF_TRANSFER=1 hf download PaddlePaddle/PP-OCRv5_server_rec --local-dir paddle-ocr/PP-OCRv5_server_rec || { echo "Failed to download PP-OCRv5_server_rec model"; exit 1; }

echo "Download complete. Checkpoints organized in model-specific folders."