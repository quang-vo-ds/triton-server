#!/bin/bash

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Load .env file from project root if it exists
ENV_FILE="../.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# Get environment (default: dev)
APP_ENV="${APP_ENV:-dev}"
echo "Downloading models for environment: $APP_ENV"

# Config file path
CONFIG_FILE="../src/triton_server/configs/conf/${APP_ENV}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Function to extract value from YAML section
# Usage: get_yaml_section_value "section_name" "key"
get_yaml_section_value() {
    local section="$1"
    local key="$2"
    awk -v section="$section" -v key="$key" '
        $0 ~ "^"section":" { in_section=1; next }
        in_section && /^[a-zA-Z_]+:/ && $0 !~ "^[[:space:]]" { in_section=0 }
        in_section && $0 ~ "^[[:space:]]+"key":" {
            gsub(/.*:[[:space:]]*"?/, "")
            gsub(/".*/, "")
            print
            exit
        }
    ' "$CONFIG_FILE"
}

# Function to extract model name from path (e.g., "checkpoints/whisper/whisper-tiny" -> "whisper-tiny")
extract_model_name() {
    basename "$1"
}

# Read model names from config
WHISPER_MODEL=$(extract_model_name "$(get_yaml_section_value "whisper_settings" "MODEL_NAME")")
SAM_MODEL=$(get_yaml_section_value "sam_settings" "MODEL_NAME")
OWL_MODEL=$(extract_model_name "$(get_yaml_section_value "owl_settings" "MODEL_NAME")")
PADDLE_DET_MODEL=$(extract_model_name "$(get_yaml_section_value "paddle_settings" "DET_MODEL_NAME")")
PADDLE_REC_MODEL=$(extract_model_name "$(get_yaml_section_value "paddle_settings" "REC_MODEL_NAME")")

echo "Models to download:"
echo "  Whisper: $WHISPER_MODEL"
echo "  SAM: $SAM_MODEL"
echo "  OWL: $OWL_MODEL"
echo "  PaddleOCR: $PADDLE_DET_MODEL, $PADDLE_REC_MODEL"

# Helper function to check if model needs ONNX export
needs_onnx_export() {
    [[ "$1" == *_onnx ]]
}

# Helper function to get base model name (without _onnx suffix)
get_base_model() {
    echo "${1%_onnx}"
}

# Setup download command
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -O"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -L -o"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# Create directories
mkdir -p sam whisper owlv2 paddle-ocr

# Download SAM
echo "Downloading $SAM_MODEL..."
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/${SAM_MODEL}.pt"
$DOWNLOAD_CMD "sam/${SAM_MODEL}.pt" "$SAM_URL" || { echo "Failed to download SAM"; exit 1; }

# Download Whisper
if needs_onnx_export "$WHISPER_MODEL"; then
    WHISPER_BASE=$(get_base_model "$WHISPER_MODEL")
    echo "Downloading $WHISPER_BASE and exporting to ONNX..."
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "openai/$WHISPER_BASE" --local-dir "whisper/$WHISPER_BASE" || { echo "Failed to download Whisper"; exit 1; }
    ./export_onnx.sh "openai/$WHISPER_BASE" "whisper/$WHISPER_MODEL" whisper || { echo "Failed to export Whisper to ONNX"; exit 1; }
else
    echo "Downloading $WHISPER_MODEL..."
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "openai/$WHISPER_MODEL" --local-dir "whisper/$WHISPER_MODEL" || { echo "Failed to download Whisper"; exit 1; }
fi

# Download OWLv2
echo "Downloading $OWL_MODEL..."
HF_HUB_ENABLE_HF_TRANSFER=1 hf download "google/$OWL_MODEL" --local-dir "owlv2/$OWL_MODEL" || { echo "Failed to download OWLv2"; exit 1; }

# Download PaddleOCR
echo "Downloading $PADDLE_DET_MODEL and $PADDLE_REC_MODEL..."
HF_HUB_ENABLE_HF_TRANSFER=1 hf download "PaddlePaddle/$PADDLE_DET_MODEL" --local-dir "paddle-ocr/$PADDLE_DET_MODEL" || { echo "Failed to download $PADDLE_DET_MODEL"; exit 1; }
HF_HUB_ENABLE_HF_TRANSFER=1 hf download "PaddlePaddle/$PADDLE_REC_MODEL" --local-dir "paddle-ocr/$PADDLE_REC_MODEL" || { echo "Failed to download $PADDLE_REC_MODEL"; exit 1; }

echo "Download complete for $APP_ENV environment."
