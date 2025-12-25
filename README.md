# Triton Server

Multi-model inference server powered by PyTriton, supporting Whisper (speech recognition), Jina-CLIP (image/text embeddings), OWLv2 (object detection), and PaddleOCR (text recognition).

## Quick Start

### New Ubuntu Instance Setup

For a fresh Ubuntu instance, run the setup script to install all dependencies:

```bash
bash setup.sh
```

This will:

1. Install Python 3.12
2. Install NVIDIA Container Toolkit
3. Install uv package manager
4. Create virtual environment and sync dependencies
5. Download model checkpoints

### Manual Setup

First, download all model checkpoints:

```bash
bash checkpoints/download.sh
```

For ONNX models, export them using the export script:

```bash
# Export Whisper model to ONNX format
bash checkpoints/export_onnx.sh openai/whisper-base checkpoints/whisper/whisper-base_onnx whisper
```

Then start the server:

```bash
docker compose up --build
```

## Models

### Whisper (Speech-to-Text)

- **Input**: `audio_arr` - float32 array, shape `(batch, -1)` - Audio samples at 16kHz
- **Output**: `transcripts` - bytes, shape `(batch,)` - Transcription text

### Jina-CLIP (Image/Text Embeddings)

Two endpoints for multimodal embeddings:

**Text Encoding** (`jina-clip-v2-text`):

- **Input**: `texts` - bytes array, shape `(batch,)` - Text strings
- **Output**: `embeddings` - float32 array, shape `(batch, embedding_dim)` - Text embeddings

**Image Encoding** (`jina-clip-v2-image`):

- **Input**: `image_arr` - uint8 array, shape `(batch, H, W, 3)` - RGB images
- **Output**: `embeddings` - float32 array, shape `(batch, embedding_dim)` - Image embeddings

### OWLv2 (Object Detection)

Open-vocabulary object detection with text prompts:

- **Input**:
  - `image_arr` - uint8 array, shape `(batch, H, W, 3)` - RGB images
  - `prompts` - bytes array, shape `(batch, num_prompts)` - Text prompts for detection
- **Output**: `detections` - array, shape `(batch, num_detections, 6)` - Bounding boxes `[x1, y1, x2, y2, label, score]`

### PaddleOCR (Text Recognition)

OCR for text extraction from images:

- **Input**: `image_arr` - uint8 array, shape `(batch, H, W, 3)` - RGB images
- **Output**: `text` - bytes array, shape `(batch, 1)` - Extracted text per image

## Usage Examples

See test files for client usage:

- Whisper: `tests/test_whisper_client.py`
- Jina-CLIP: `tests/test_jina_clip_client.py`
- OWLv2: `tests/test_owl_client.py`
- PaddleOCR: `tests/test_paddle_client.py`

## Requirements

- Docker & Docker Compose
- Python 3.12+ (for development)
- NVIDIA GPU with CUDA support

## License

See [LICENSE](LICENSE)
