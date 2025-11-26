import json
from typing import Any
import datetime

from pydantic import BaseModel, Field
import numpy as np
from transformers import pipeline

from .base import BaseWhisper
from .transcript import Transcript
from ...configs import settings


class WhisperTorch(BaseWhisper):
    def __init__(
        self,
        model_name: str = settings.whisper_settings.MODEL_NAME,
        device: str = settings.whisper_settings.DEVICE,
    ):
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
        )

    def transcribe(self, audio_arr: np.ndarray) -> list[list[Transcript]]:
        """
        Transcribe an audio file using the Whisper model.
        """

        # Transform inputs to batch
        if audio_arr.ndim == 2:
            batch = [audio_arr[i] for i in range(audio_arr.shape[0])]
        else:
            batch = [audio_arr]

        # Calculate batch size
        batch_size = min(len(batch), settings.whisper_settings.BATCH_SIZE)

        # Process batch
        result = self.pipe(
            batch,
            batch_size=batch_size,
            return_timestamps="word",
            generate_kwargs={"language": "english"},
        )

        chunks = [r.get("chunks", []) for r in result]

        return [self._process_segment(chunk, [".", "?", "!"], 3) for chunk in chunks]

    @staticmethod
    def _process_segment(
        chunks: list[dict[str, Any]],
        delimiter: list,
        min_sentence_length,
    ) -> list[Transcript]:
        """
        Process a segment of the audio file.
        """

        # If the segments is a list, process each segment
        final_transcript = []
        current_sentence = ""
        start_time = None
        end_time = None

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            word = chunk["text"]
            word_start = np.round(chunk["timestamp"][0])
            word_end = np.round(chunk["timestamp"][1])

            if start_time is None:
                start_time = word_start

            current_sentence += word
            end_time = word_end

            if (
                word.strip().endswith(tuple(delimiter))
                and len(current_sentence.split()) >= min_sentence_length
            ):
                final_transcript.append(
                    Transcript(
                        text=current_sentence.strip(),
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
                current_sentence = ""
                start_time = None
                end_time = None

        return final_transcript
