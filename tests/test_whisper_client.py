import numpy as np
import librosa
import time
from pytriton.client import ModelClient

AUDIO_FILE = "data/audio.wav"
SERVER_URL = "http://localhost:8000/"
MODEL_NAME = "whisper-tiny_onnx"
MODEL_VERSION = "1"

INIT_TIMEOUT_S = 300.0
INFERENCE_TIMEOUT_S = 300.0


def test_client():
    audio_arr, sampling_rate = librosa.load(AUDIO_FILE, sr=16000)
    audio_len = len(audio_arr)

    batch_1 = np.stack([audio_arr, audio_arr], axis=0)
    batch_2 = np.stack(
        [audio_arr[: audio_len // 2], audio_arr[: audio_len // 2]], axis=0
    )

    with ModelClient(
        SERVER_URL,
        MODEL_NAME,
        init_timeout_s=INIT_TIMEOUT_S,
        inference_timeout_s=INFERENCE_TIMEOUT_S,
    ) as client:
        # Send first request
        print("Sending request (1).")
        result_dict_1 = client.infer_batch(batch_1)

        # Wait 2 seconds immediately after sending first request
        print("Waiting 2 seconds...")
        time.sleep(2)

        # Send second request
        print("Sending request (2).")
        result_dict_2 = client.infer_batch(batch_2)

        # Process first result
        for output_name, output_data in result_dict_1.items():
            print(f"{output_name}: {output_data} for request (1).")

        # Process second result
        for output_name, output_data in result_dict_2.items():
            print(f"{output_name}: {output_data} for request (2).")


if __name__ == "__main__":
    test_client()
