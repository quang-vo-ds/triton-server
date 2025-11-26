import click
import json
import logging
from pathlib import Path

import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

from .engine import WhisperONNX, SAMEngine
from .configs import settings

logger = logging.getLogger(__name__)
whisper_engine = WhisperONNX()
sam_engine = SAMEngine()


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
def _sam_infer_func(image_arr: np.ndarray) -> dict:
    predictions = sam_engine.predict(image_arr)

    logger.info(f"Number of predictions: {len(predictions)}")

    return {"images": predictions}


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
            model_name=Path(settings.sam_settings.MODEL_NAME).name,
            infer_func=_sam_infer_func,
            inputs=[
                Tensor(name="image_arr", dtype=np.uint8, shape=(-1, -1, 3)),
            ],
            outputs=[
                Tensor(name="images", dtype=np.uint8, shape=(-1, -1, 3)),
            ],
            config=ModelConfig(
                max_batch_size=settings.triton_settings.TRITON_MAX_BATCH_SIZE,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=settings.triton_settings.TRITON_MAX_QUEUE_DELAY_MICROSECONDS
                ),
            ),
            strict=True,
        )
        logger.info("Serving SAM Inference")

        triton.serve()


if __name__ == "__main__":
    main()
