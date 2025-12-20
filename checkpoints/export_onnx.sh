#!/bin/bash

# Export Hugging Face models to ONNX format
# Usage: ./export_onnx.sh <model_name> <output_dir> [model_type]
# Example: ./export_onnx.sh openai/whisper-tiny whisper/whisper-tiny_onnx whisper

MODEL_NAME=${1:-"openai/whisper-tiny"}
OUTPUT_DIR=${2:-"onnx_out"}
MODEL_TYPE=${3:-"default"}

echo "Exporting $MODEL_NAME to $OUTPUT_DIR (type: $MODEL_TYPE)..."

# Use Python API for ONNX export since optimum 2.0 removed CLI export
python3 -c "
from transformers import AutoConfig
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import os

model_name = '$MODEL_NAME'
output_dir = '$OUTPUT_DIR'
model_type = '$MODEL_TYPE'

os.makedirs(output_dir, exist_ok=True)

if model_type == 'whisper':
    # Export Whisper model to ONNX
    model = ORTModelForSpeechSeq2Seq.from_pretrained(model_name, export=True)
    model.save_pretrained(output_dir)
else:
    # Generic export - try auto detection
    from optimum.onnxruntime import ORTModel
    model = ORTModel.from_pretrained(model_name, export=True)
    model.save_pretrained(output_dir)

print(f'Successfully exported {model_name} to {output_dir}')
"

echo "✓ Export complete!"

