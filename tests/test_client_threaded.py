import numpy as np
import librosa
import time
import threading
from pytriton.client import ModelClient

AUDIO_FILE = "data/audio.wav"
WHISPER_SERVER_URL = "http://localhost:8000"
MODEL_NAME = "whisper-tiny"
MODEL_VERSION = "1"


def send_batch(sample_data, sample_id):
    """Send a single sample in a thread."""
    with ModelClient(
        url=WHISPER_SERVER_URL,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        init_timeout_s=60,
        inference_timeout_s=60.0,
    ) as client:
        print(f"Sample {sample_id}: Sending request")
        start_time = time.time()

        result = client.infer_sample(sample_data)

        end_time = time.time()
        print(f"Sample {sample_id}: Completed in {end_time - start_time:.2f}s")

        return sample_id, result


def test_three_samples():
    """Send three samples with 1s delay between each."""
    # Load audio data
    audio_arr, sampling_rate = librosa.load(AUDIO_FILE, sr=16000)
    audio_len = len(audio_arr)

    # Create three different samples
    sample1 = audio_arr
    sample2 = audio_arr
    sample3 = audio_arr[: audio_len // 2]

    samples = [sample1, sample2, sample3]
    results = []
    threads = []

    print("Starting three samples with 1s delay between each...")
    start_time = time.time()

    # Create and start three threads with 1s delay between each
    for i, sample in enumerate(samples):
        thread = threading.Thread(
            target=lambda s=sample, idx=i: results.append(send_batch(s, idx + 1))
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    end_time = time.time()
    print(f"\nAll three samples completed in {end_time - start_time:.2f}s")
    print(f"Results: {len(results)} samples processed")

    return results


if __name__ == "__main__":
    test_three_samples()
