# Triton Server

Multi-model inference server powered by PyTriton, supporting Whisper (speech recognition) and SAM 2.1 (image segmentation).

## Quick Start

First, download all model checkpoints:

```bash
bash checkpoints/download.sh
```

For ONNX models, export them using the export script:

```bash
# Export Whisper model to ONNX format
bash checkpoints/export_onnx.sh openai/whisper-tiny checkpoints/whisper/whisper-tiny_onnx whisper
```

Then start the server:

```bash
docker compose up --build
```

## Models

### Whisper (Speech-to-Text)
- **Input**: `audio_arr` - float32 array, shape `(-1,)` - Audio samples at 16kHz
- **Output**: `transcripts` - bytes, shape `(1,)` - JSON-encoded transcription chunks

### SAM 2.1 (Image Segmentation)
- **Input**: `image_arr` - uint8 array, shape `(-1, -1, 3)` - RGB image
- **Output**: `images` - bytes array, shape `(-1, -1, 3)` - Segmented image

## Usage Examples

See test files for client usage:
- Whisper: `tests/test_whisper_client.py`
- SAM: `tests/test_sam_client.py`

## Requirements

- Docker & Docker Compose
- Python 3.12+ (for development)

## License

See [LICENSE](LICENSE)
