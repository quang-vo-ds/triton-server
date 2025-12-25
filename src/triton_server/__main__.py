import click
import json
import logging
from pathlib import Path

import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

from .engine import WhisperONNX, OwlViT, PaddleOCR, JinaClip
from .configs import settings

logger = logging.getLogger(__name__)
whisper_engine = WhisperONNX()
owl_engine = OwlViT()
paddle_engine = PaddleOCR()
jina_clip_engine = JinaClip()


@batch
def _whisper_infer_func(audio_arr: np.ndarray) -> dict:
    predictions = whisper_engine.transcribe(audio_arr)

    # Extract chunks from predictions
    all_chunks = []
    for prediction in predictions:
        prediction_json = json.dumps([chunk.model_dump() for chunk in prediction])
        all_chunks.append(np.array([prediction_json]))

    logger.info(f"Number of chunks: {len(all_chunks)}")

    return {"transcripts": np.char.encode(all_chunks, "utf-8")}


@batch
def _owl_infer_func(image_arr: np.ndarray, prompts: np.ndarray) -> dict:
    prompts_decoded = np.array(
        [[p.decode("utf-8") for p in batch] for batch in prompts]
    )
    detections = owl_engine.detect(image_arr, prompts_decoded)
    results = [np.array([[str(v) for v in det] for det in dets]) for dets in detections]
    return {"detections": np.char.encode(results, "utf-8")}


@batch
def _paddle_infer_func(image_arr: np.ndarray) -> dict:
    results = paddle_engine.recognize(image_arr)
    return {"text": np.char.encode(results, "utf-8")}


@batch
def _jina_clip_text_infer_func(texts: np.ndarray) -> dict:
    texts_decoded = np.array([t.decode("utf-8") for t in texts.flatten()])
    embeddings = jina_clip_engine.encode_text(texts_decoded)
    return {"embeddings": embeddings}


@batch
def _jina_clip_image_infer_func(image_arr: np.ndarray) -> dict:
    embeddings = jina_clip_engine.encode_image(image_arr)
    return {"embeddings": embeddings}


def main():
    with Triton(config=TritonConfig(strict_readiness=True)) as triton:
        triton.bind(
            model_name=Path(settings.whisper_settings.MODEL_NAME).name,
            infer_func=_whisper_infer_func,
            inputs=[
                Tensor(name="audio_arr", dtype=np.float32, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="transcripts", dtype=bytes, shape=(1,)),
            ],
            config=ModelConfig(
                max_batch_size=settings.triton_settings.TRITON_MAX_BATCH_SIZE,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=settings.triton_settings.TRITON_MAX_QUEUE_DELAY_MICROSECONDS
                ),
            ),
            strict=True,
        )
        logger.info("Serving Whisper Inference")

        triton.bind(
            model_name=Path(settings.owl_settings.MODEL_NAME).name,
            infer_func=_owl_infer_func,
            inputs=[
                Tensor(name="image_arr", dtype=np.uint8, shape=(-1, -1, 3)),
                Tensor(name="prompts", dtype=bytes, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="detections", dtype=bytes, shape=(-1, 6)),
            ],
            config=ModelConfig(
                max_batch_size=settings.triton_settings.TRITON_MAX_BATCH_SIZE,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=settings.triton_settings.TRITON_MAX_QUEUE_DELAY_MICROSECONDS
                ),
            ),
            strict=True,
        )
        logger.info("Serving OWL Inference")

        triton.bind(
            model_name=Path(settings.paddle_settings.DET_MODEL_NAME).parent.name,
            infer_func=_paddle_infer_func,
            inputs=[
                Tensor(name="image_arr", dtype=np.uint8, shape=(-1, -1, 3)),
            ],
            outputs=[
                Tensor(name="text", dtype=bytes, shape=(1,)),
            ],
            config=ModelConfig(
                max_batch_size=settings.triton_settings.TRITON_MAX_BATCH_SIZE,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=settings.triton_settings.TRITON_MAX_QUEUE_DELAY_MICROSECONDS
                ),
            ),
            strict=True,
        )
        logger.info("Serving PaddleOCR Inference")

        triton.bind(
            model_name=f"{Path(settings.jina_clip_settings.MODEL_NAME).name}-text",
            infer_func=_jina_clip_text_infer_func,
            inputs=[
                Tensor(name="texts", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="embeddings", dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(
                max_batch_size=settings.triton_settings.TRITON_MAX_BATCH_SIZE,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=settings.triton_settings.TRITON_MAX_QUEUE_DELAY_MICROSECONDS
                ),
            ),
            strict=True,
        )
        logger.info("Serving Jina CLIP Text Inference")

        triton.bind(
            model_name=f"{Path(settings.jina_clip_settings.MODEL_NAME).name}-image",
            infer_func=_jina_clip_image_infer_func,
            inputs=[
                Tensor(name="image_arr", dtype=np.uint8, shape=(-1, -1, 3)),
            ],
            outputs=[
                Tensor(name="embeddings", dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(
                max_batch_size=settings.triton_settings.TRITON_MAX_BATCH_SIZE,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=settings.triton_settings.TRITON_MAX_QUEUE_DELAY_MICROSECONDS
                ),
            ),
            strict=True,
        )
        logger.info("Serving Jina CLIP Image Inference")

        triton.serve()


if __name__ == "__main__":
    main()
