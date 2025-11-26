#!/bin/bash

# Export Hugging Face models to ONNX format
# Usage: ./export_onnx.sh <model_name> <output_dir> [model_type]
# Example: ./export_onnx.sh openai/whisper-tiny whisper/whisper-tiny_onnx whisper

MODEL_NAME=${1:-"openai/whisper-tiny"}
OUTPUT_DIR=${2:-"onnx_out"}
MODEL_TYPE=${3:-"default"}

# Model-specific configurations
export_whisper() {
    optimum-cli export onnx --model "$MODEL_NAME" \
        "$OUTPUT_DIR" --no-post-process \
        --output_names encoder_model.onnx decoder_model.onnx \
        --opset 17
}

export_default() {
    optimum-cli export onnx --model "$MODEL_NAME" \
        "$OUTPUT_DIR" --opset 17
}

# Add more model types here as functions
# export_bert() { ... }
# export_llama() { ... }

echo "Exporting $MODEL_NAME to $OUTPUT_DIR (type: $MODEL_TYPE)..."

case "$MODEL_TYPE" in
    whisper)
        export_whisper
        ;;
    *)
        export_default
        ;;
esac

echo "✓ Export complete!"

